"""Per-sport data-source policy loaded from ``data_sources.yaml``.

The versioned ``sports/<sport>/config/data_sources.yaml`` declares the
policy §5/§5.1 fixes in config rather than code: the external sources a
sport consumes, the payload columns each source must deliver (schema
drift fails loudly), and the transport policy (retries, backoff,
chunking, refresh tail). The ingestion modules read their policy from
here — the values stay configurable without touching transport code.

Fails loudly on a missing file, unknown keys, missing sources/required
columns, or invalid policy values (same no-silent-defaults policy as the
study and market-rules loaders).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import yaml

from core.config.platform import normalize_sport_key


class DataSourcesError(ValueError):
    """Raised when ``data_sources.yaml`` is missing or invalid."""


_TOP_KEYS = {"sport", "data_sources_version", "sources", "policy"}
_SOURCE_KEYS = {"source_id", "required_columns"}
_POLICY_KEYS = {"chunk_retries", "chunk_retry_base_ms",
                "refresh_tail_days", "request_pause_ms"}


class SourcePolicyError(DataSourcesError):
    """Kept for symmetry with the market-rules loader; the data-sources
    loader raises :class:`DataSourcesError` for every failure mode."""


def _sport_config_dir(sport: str) -> Path:
    """``sports/<sport>/config/`` resolved from the repository root."""
    return Path(__file__).resolve().parents[2] / "sports" / sport / "config"


@dataclass(frozen=True)
class SourcePolicy:
    """Transport policy shared by a sport's sources (immutable)."""

    chunk_retries: int
    chunk_retry_base_ms: int
    refresh_tail_days: int
    request_pause_ms: int = 0

    def __post_init__(self) -> None:
        for name in ("chunk_retries", "chunk_retry_base_ms",
                     "refresh_tail_days", "request_pause_ms"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise DataSourcesError(
                    f"data_sources policy.{name} must be a non-negative "
                    f"integer, got {v!r}")


@dataclass(frozen=True)
class SportDataSources:
    """One sport's declared source + schema policy (immutable)."""

    sport: str
    data_sources_version: str
    required_columns: Mapping[str, tuple[str, ...]]
    source_ids: Mapping[str, str]
    policy: SourcePolicy
    source_path: str


def load_data_sources(sport: str,
                      path: str | Path | None = None) -> SportDataSources:
    """Load and validate one sport's ``data_sources.yaml``.

    Fails loudly (``DataSourcesError``) on: missing file, invalid YAML,
    unknown top-level keys, a ``sport`` mismatch, a missing/blank
    version, an empty ``sources`` mapping, a source without required
    columns, or an invalid transport-policy value.
    """
    sport = normalize_sport_key(sport)
    file = (Path(path) if path is not None
            else _sport_config_dir(sport) / "data_sources.yaml")
    if not file.is_file():
        raise DataSourcesError(
            f"data_sources.yaml missing for {sport}: {file}")
    try:
        doc = yaml.safe_load(file.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise DataSourcesError(f"{file}: invalid YAML: {exc}") from exc
    if not isinstance(doc, dict):
        raise DataSourcesError(
            f"{file}: expected a top-level mapping, got {type(doc).__name__}")
    unknown = sorted(set(doc) - _TOP_KEYS)
    if unknown:
        raise DataSourcesError(
            f"{file}: unknown top-level key(s) {unknown} "
            f"(allowed: {sorted(_TOP_KEYS)})")
    if doc.get("sport") != sport:
        raise DataSourcesError(
            f"{file}: sport must be {sport!r}, got {doc.get('sport')!r}")
    version = doc.get("data_sources_version")
    if not isinstance(version, str) or not version.strip():
        raise DataSourcesError(
            f"{file}: data_sources_version must be a non-empty string")
    sources_doc = doc.get("sources")
    if not isinstance(sources_doc, dict) or not sources_doc:
        raise DataSourcesError(
            f"{file}: sources must be a non-empty mapping of source names")

    required: dict[str, tuple[str, ...]] = {}
    source_ids: dict[str, str] = {}
    for name, entry in sources_doc.items():
        if not isinstance(name, str) or not name.strip():
            raise DataSourcesError(
                f"{file}: source names must be non-empty strings, "
                f"got {name!r}")
        if not isinstance(entry, dict):
            raise DataSourcesError(
                f"{file}: sources.{name} must be a mapping, "
                f"got {type(entry).__name__}")
        unknown_fields = sorted(set(entry) - _SOURCE_KEYS)
        if unknown_fields:
            raise DataSourcesError(
                f"{file}: sources.{name}: unknown field(s) {unknown_fields} "
                f"(allowed: {sorted(_SOURCE_KEYS)})")
        cols = entry.get("required_columns")
        if not isinstance(cols, list) or not cols \
                or not all(isinstance(c, str) and c.strip() for c in cols):
            raise DataSourcesError(
                f"{file}: sources.{name}.required_columns must be a "
                f"non-empty list of column names")
        sid = entry.get("source_id")
        if sid is not None and (not isinstance(sid, str) or not sid.strip()):
            raise DataSourcesError(
                f"{file}: sources.{name}.source_id must be a non-empty "
                f"string when declared")
        required[name] = tuple(cols)
        if sid is not None:
            source_ids[name] = sid

    policy_doc = doc.get("policy") or {}
    if not isinstance(policy_doc, dict):
        raise DataSourcesError(
            f"{file}: policy must be a mapping, "
            f"got {type(policy_doc).__name__}")
    unknown_policy = sorted(set(policy_doc) - _POLICY_KEYS)
    if unknown_policy:
        raise DataSourcesError(
            f"{file}: policy: unknown field(s) {unknown_policy} "
            f"(allowed: {sorted(_POLICY_KEYS)})")
    try:
        policy = SourcePolicy(
            chunk_retries=policy_doc.get("chunk_retries", 3),
            chunk_retry_base_ms=policy_doc.get("chunk_retry_base_ms", 1000),
            refresh_tail_days=policy_doc.get("refresh_tail_days", 0),
            request_pause_ms=policy_doc.get("request_pause_ms", 0),
        )
    except DataSourcesError as exc:
        raise DataSourcesError(f"{file}: {exc}") from exc
    return SportDataSources(
        sport=sport,
        data_sources_version=version,
        required_columns=MappingProxyType(required),
        source_ids=MappingProxyType(source_ids),
        policy=policy,
        source_path=str(file),
    )
