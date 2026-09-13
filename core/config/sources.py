"""Market settlement rules loaded from per-sport ``market_rules.yaml``.

Every sport declares its settlement semantics — ``settlement_scope`` /
``tie_allowed`` / ``push_allowed`` per market kind — in the versioned
``sports/<sport>/config/market_rules.yaml`` (spec §20: explicit,
configurable settlement; no universal formula). The sport adapter's
``settlement_config`` and the production runner's settlement gates read
them from here; no call site hardcodes a ``SettlementConfig``.

The loader fails loudly on a missing file, unknown keys, or invalid
values — an undeclared market kind never silently defaults (same
no-silent-defaults policy as the study YAML loader).

Note: ``core.contracts.settlement_rules`` is imported lazily inside
``load_market_rules`` because that module itself imports
``core.config.platform``; a top-level import here would make
``import core.contracts`` and ``import core.config`` mutually recursive.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import yaml

from core.config.platform import normalize_sport_key


class MarketRulesError(ValueError):
    """Raised when ``market_rules.yaml`` is missing or invalid."""


_MARKET_RULES_TOP_KEYS = {"sport", "market_rules_version", "settlement"}
_SETTLEMENT_FIELDS = {"settlement_scope", "tie_allowed", "push_allowed"}
_SCOPES = ("regulation", "full_game")


def _sport_config_dir(sport: str) -> Path:
    """``sports/<sport>/config/`` resolved from the repository root
    (``core/config/sources.py`` -> parents[2] is the repo root)."""
    return Path(__file__).resolve().parents[2] / "sports" / sport / "config"


@dataclass(frozen=True)
class SportMarketRules:
    """The declared settlement rules of one sport (immutable)."""

    sport: str
    market_rules_version: str
    settlement: Mapping[str, object]  # kind -> SettlementConfig
    source_path: str

    def settlement_config(self, market_kind: str):
        """The ``SettlementConfig`` for one declared market kind.

        Raises ``KeyError`` for an undeclared kind — the adapter/runner
        must never invent settlement semantics for a kind the config does
        not declare.
        """
        try:
            return self.settlement[market_kind]
        except KeyError:
            raise KeyError(
                f"unknown {self.sport.upper()} market kind {market_kind!r} "
                f"(declared kinds in {self.source_path}: "
                f"{sorted(self.settlement)})") from None


def _require_bool(file: Path, kind: str, field: str, value) -> bool:
    if not isinstance(value, bool):
        raise MarketRulesError(
            f"{file}: settlement.{kind}.{field} must be a boolean, "
            f"got {value!r}")
    return value


def load_market_rules(sport: str,
                      path: str | Path | None = None) -> SportMarketRules:
    """Load and validate one sport's ``market_rules.yaml``.

    Fails loudly (``MarketRulesError``) on: missing file, invalid YAML,
    unknown top-level keys, a ``sport`` mismatch, a missing/blank
    version, an empty or malformed ``settlement`` mapping, unknown
    settlement fields, or a non-boolean tie/push flag.
    """
    sport = normalize_sport_key(sport)
    file = (Path(path) if path is not None
            else _sport_config_dir(sport) / "market_rules.yaml")
    if not file.is_file():
        raise MarketRulesError(
            f"market_rules.yaml missing for {sport}: {file}")
    try:
        doc = yaml.safe_load(file.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise MarketRulesError(f"{file}: invalid YAML: {exc}") from exc
    if not isinstance(doc, dict):
        raise MarketRulesError(
            f"{file}: expected a top-level mapping, got {type(doc).__name__}")
    unknown = sorted(set(doc) - _MARKET_RULES_TOP_KEYS)
    if unknown:
        raise MarketRulesError(
            f"{file}: unknown top-level key(s) {unknown} "
            f"(allowed: {sorted(_MARKET_RULES_TOP_KEYS)})")
    if doc.get("sport") != sport:
        raise MarketRulesError(
            f"{file}: sport must be {sport!r}, got {doc.get('sport')!r}")
    version = doc.get("market_rules_version")
    if not isinstance(version, str) or not version.strip():
        raise MarketRulesError(
            f"{file}: market_rules_version must be a non-empty string")
    settlement_doc = doc.get("settlement")
    if not isinstance(settlement_doc, dict) or not settlement_doc:
        raise MarketRulesError(
            f"{file}: settlement must be a non-empty mapping of market kinds")

    # Imported lazily — see the module docstring for the cycle rationale.
    from core.contracts.settlement_rules import SettlementConfig, SettlementError

    settlement: dict[str, SettlementConfig] = {}
    for kind, cfg_doc in settlement_doc.items():
        if not isinstance(kind, str) or not kind.strip():
            raise MarketRulesError(
                f"{file}: settlement keys must be non-empty strings, "
                f"got {kind!r}")
        if not isinstance(cfg_doc, dict):
            raise MarketRulesError(
                f"{file}: settlement.{kind} must be a mapping, "
                f"got {type(cfg_doc).__name__}")
        unknown_fields = sorted(set(cfg_doc) - _SETTLEMENT_FIELDS)
        if unknown_fields:
            raise MarketRulesError(
                f"{file}: settlement.{kind}: unknown field(s) {unknown_fields} "
                f"(allowed: {sorted(_SETTLEMENT_FIELDS)})")
        scope = cfg_doc.get("settlement_scope")
        if scope not in _SCOPES:
            raise MarketRulesError(
                f"{file}: settlement.{kind}.settlement_scope must be one of "
                f"{_SCOPES}, got {scope!r}")
        try:
            settlement[kind] = SettlementConfig(
                settlement_scope=scope,
                tie_allowed=_require_bool(file, kind, "tie_allowed",
                                          cfg_doc.get("tie_allowed", False)),
                push_allowed=_require_bool(file, kind, "push_allowed",
                                           cfg_doc.get("push_allowed", False)),
            )
        except SettlementError as exc:
            raise MarketRulesError(
                f"{file}: settlement.{kind}: {exc}") from exc
    return SportMarketRules(
        sport=sport,
        market_rules_version=version,
        settlement=MappingProxyType(settlement),
        source_path=str(file),
    )
