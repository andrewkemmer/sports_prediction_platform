"""Feature-contract and artifact-contract interfaces.

These are the *semantic* interfaces every sport adapter implements. Market
schemas stay sport-specific (Phase 0 decision 5): MLB carries run-line
grids, NFL carries spread/total offers, and future NHL/NBA schemas will
differ again. The shared layer defines only what the frontend adapters and
the pipeline agree on — never one identical CSV shape across sports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from core.config.platform import PlatformConfig, normalize_sport_key
from core.study.date_resolution import artifact_date_from_filename
from core.contracts.settlement_rules import SettlementConfig

# ---------------------------------------------------------------------------
# Feature contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSpec:
    """One feature in the shared feature contract.

    Mirrors the reference ``features_metadata`` entry shape (name, summary,
    definition, formula, source, window, units, direction, members,
    tooltip) so the Model Monitor's metadata tooltips render unchanged.
    """

    name: str
    summary: str
    definition: str
    formula: str
    source: str
    window: str
    units: str
    direction: str
    members: tuple[str, ...] = ()
    tooltip: str = ""


@dataclass(frozen=True)
class FeatureContract:
    """The declared feature set for one sport's model."""

    sport: str
    version: str
    features: tuple[FeatureSpec, ...]

    def __post_init__(self) -> None:
        if not self.features:
            raise ValueError("feature contract must declare features")
        names = [f.name for f in self.features]
        if len(names) != len(set(names)):
            raise ValueError("duplicate feature names in contract")
        if any(not n for n in names):
            raise ValueError("feature names must be non-empty")

    def feature_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.features)

    def get(self, name: str) -> FeatureSpec:
        for f in self.features:
            if f.name == name:
                return f
        raise KeyError(f"feature {name!r} not in {self.sport} contract")

    def to_metadata_json(self, generated_for: str = "",
                         warnings: list | None = None) -> dict:
        """Serialize to the reference ``features_metadata_*.json`` shape.

        ``warnings`` carries the run's explicit unavailable-feature markers
        (never silently dropped columns): entirely-unavailable sources are
        routed per the missing-data policy and listed here verbatim.
        """
        return {
            "generated_for": generated_for,
            "n_features": len(self.features),
            "warnings": list(warnings or []),
            "features": {
                f.name: {
                    "name": f.name,
                    "summary": f.summary,
                    "definition": f.definition,
                    "formula": f.formula,
                    "source": f.source,
                    "window": f.window,
                    "units": f.units,
                    "direction": f.direction,
                    "members": list(f.members),
                    "tooltip": f.tooltip,
                }
                for f in self.features
            },
            "categorical_context": {},
        }


# ---------------------------------------------------------------------------
# Declared-contract <-> registry-metadata validation (shared interface)
# ---------------------------------------------------------------------------
# Sports declare their served columns as explicit contract tuples and keep
# the per-feature metadata in whatever model their registry already uses
# (MLB: ``FeatureEntry`` records; NFL/NHL/NBA: their ``_SPEC_DEFS`` mapping).
# These two checks are deliberately name-only so no sport is forced onto a
# parallel registry, and they are bidirectional so a declared column can
# never ship undocumented and a metadata entry can never become dead.


class FeatureContractError(ValueError):
    """Raised when declared contract tuples and registry metadata drift."""


def validate_columns_have_metadata(
        sport: str, scope: str, columns, metadata_names) -> None:
    """Every declared contract column must have registry metadata.

    A declared-but-undocumented column is a production defect: the pipeline
    would serve a feature with no ``features_metadata`` entry (the
    dashboard's tooltip source). Fails loud rather than tolerating drift.
    """
    documented = set(metadata_names)
    missing = [c for c in columns if c not in documented]
    if missing:
        raise FeatureContractError(
            f"{sport}/{scope}: declared columns missing registry metadata: "
            f"{missing}")


def validate_metadata_is_reachable(
        sport: str, metadata_names, declared_columns) -> None:
    """Every registry metadata entry must be claimed by a declared tuple.

    Undeclared metadata is a parallel, dead registry: it documents a feature
    no declared contract serves. Fails loud so the registry and the declared
    tuples cannot drift apart.
    """
    declared = set(declared_columns)
    orphans = sorted(set(metadata_names) - declared)
    if orphans:
        raise FeatureContractError(
            f"{sport}: registry metadata not reachable from any declared "
            f"contract tuple: {orphans}")


#: The per-feature field set of ``FeatureSpec`` — the shape every
#: ``features_metadata`` document must carry, one entry per cataloged
#: feature. Kept next to the dataclass it mirrors so the two cannot drift.
FEATURE_SPEC_FIELDS: tuple[str, ...] = ("name", "summary", "definition",
                                        "formula", "source", "window",
                                        "units", "direction", "members",
                                        "tooltip")


class FeaturesMetadataError(ValueError):
    """Raised when a sport is about to write a non-canonical metadata doc."""


def require_features_metadata(sport: str, payload) -> dict:
    """Validate a canonical ``features_metadata`` document pre-write.

    ``payload`` is the document ``FeatureContract.to_metadata_json`` produces
    and that every sport's model-monitor writer embeds (and that the durable
    ``<sport>_features_metadata`` / ``features_metadata`` artifact carries):
    the wrapper keys plus one entry per cataloged feature, each entry holding
    the full :data:`FEATURE_SPEC_FIELDS` set. This is the shared interface
    across all four sports; the per-sport METADATA MODEL is untouched — each
    registry still owns its own declaration (MLB ``FeatureEntry`` records,
    NNX ``_SPEC_DEFS``) and only the serialized shape is enforced here.

    Rejects an absent, empty, or partial document rather than writing it. A
    stub document is indistinguishable from "this sport declares no features"
    and silently strips the Model Monitor's per-feature metadata, so it is
    treated as a defect and fails loud before any bytes reach the sink.
    """
    if not isinstance(payload, dict):
        raise FeaturesMetadataError(
            f"{sport}: features_metadata must be a mapping, got "
            f"{type(payload).__name__}")
    for key in ("generated_for", "n_features", "warnings", "features"):
        if key not in payload:
            raise FeaturesMetadataError(
                f"{sport}: features_metadata missing wrapper key {key!r}")
    features = payload["features"]
    if not isinstance(features, dict) or not features:
        raise FeaturesMetadataError(
            f"{sport}: features_metadata carries no per-feature entries "
            f"(got {type(features).__name__} with "
            f"{len(features) if hasattr(features, '__len__') else 'n/a'} "
            f"entries) — a stub document is never written")
    expected = set(FEATURE_SPEC_FIELDS)
    for name, entry in features.items():
        if not isinstance(entry, dict):
            raise FeaturesMetadataError(
                f"{sport}: features_metadata[{name!r}] is not a mapping")
        missing = sorted(expected - set(entry))
        if missing:
            raise FeaturesMetadataError(
                f"{sport}: features_metadata[{name!r}] missing FeatureSpec "
                f"fields {missing}")
        if entry.get("name") != name:
            raise FeaturesMetadataError(
                f"{sport}: features_metadata[{name!r}] declares name "
                f"{entry.get('name')!r}")
    # NOTE: ``n_features`` is deliberately NOT cross-checked against
    # ``len(features)``. Sports differ on purpose — MLB serves its moneyline
    # width while documenting the full catalog — so the wrapper's count is
    # that sport's own declaration, not a shared invariant.
    return payload


# ---------------------------------------------------------------------------
# Artifact contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactFamily:
    """One frontend artifact family for a sport.

    ``pattern`` is the dated filename pattern (e.g. ``todays_games_*.csv``);
    ``date_stamped`` is False only for durable contract artifacts that are
    exempt from the 20-day retention policy (Phase 0 decision 7; window
    updated to 20 days by B-003, Phase 7.5d) — the
    frontend must either read them directly as durable state or not at all.
    """

    family: str
    pattern: str
    ext: str
    date_stamped: bool = True
    retention_exempt: bool = False
    description: str = ""

    def date_from_name(self, name: str) -> str | None:
        if not self.date_stamped:
            return None
        base = self.pattern
        if "*" in base:
            prefix = base.split("*")[0]
        else:
            prefix = base.rsplit("_", 1)[0] + "_"
        return artifact_date_from_filename(name, prefix, self.ext)


@dataclass(frozen=True)
class ArtifactContract:
    """The artifact families a sport publishes to its data_delivery sink."""

    sport: str
    families: tuple[ArtifactFamily, ...]

    def __post_init__(self) -> None:
        if not self.families:
            raise ValueError("artifact contract must declare families")
        names = [f.family for f in self.families]
        if len(names) != len(set(names)):
            raise ValueError("duplicate artifact families")
        undated = [f.family for f in self.families
                   if not f.date_stamped and not f.retention_exempt]
        if undated:
            raise ValueError(
                f"undated artifacts must be retention_exempt: {undated}")

    def family(self, name: str) -> ArtifactFamily:
        for f in self.families:
            if f.family == name:
                return f
        raise KeyError(f"artifact family {name!r} not in {self.sport} contract")

    def retention_candidates(self) -> tuple[ArtifactFamily, ...]:
        """Families subject to the 20-day rolling deletion policy."""
        return tuple(f for f in self.families
                     if f.date_stamped and not f.retention_exempt)


# ---------------------------------------------------------------------------
# Sport adapter protocol (the Phase 1 deliverable for Phase 2+ implementers)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParticipantBox:
    """One side of the per-team participant matchup panel.

    MLB: starting pitcher (name, ERA, K/9). NFL: quarterback (name, rating,
    TD/game). NHL: goalie (name, GAA, SV%). NBA: highest-scoring player
    (name, PPG, FG%). ``stats`` is an ordered (label, display) list so each
    sport renders its own two stats in the shared box chrome.
    """

    role: str
    name: str
    stats: tuple[tuple[str, str], ...] = ()
    is_tbd: bool = False

    def __post_init__(self) -> None:
        if self.is_tbd:
            return
        if not self.name:
            raise ValueError("participant name must be non-empty unless TBD")


@runtime_checkable
class SportAdapter(Protocol):
    """The semantic interface each sport module implements (Phase 2+).

    Implementations live in ``sports/<sport>/adapter.py``. They translate
    sport-specific artifact schemas into the shared semantic types the
    frontend renders. Nothing here forces a shared CSV schema.
    """

    @property
    def sport_key(self) -> str: ...

    def artifact_contract(self) -> ArtifactContract:
        """The artifact families this sport publishes."""
        ...

    def feature_contract(self) -> FeatureContract:
        """The feature contract this sport's model trains on."""
        ...

    def normalize_games(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Derive every display column the game cards expect.

        Must produce (when inputs allow): ``game_status`` (Final/Live/
        Scheduled), ``evening_game``, ``day_game``, ``*_team_name``,
        ``model_pick``, ``model_correct``, ``is_upset``, ``is_coin_flip``,
        and de-duplicate on the game id. Never fabricates outcomes.
        """
        ...

    def participant_box(self, game_row: pd.Series,
                        side: str) -> ParticipantBox:
        """The per-team participant box (pitcher/QB/goalie/top scorer).

        ``side`` is ``home`` or ``away``. Returns a TBD box when the
        participant is unannounced — never a fabricated name.
        """
        ...

    def market_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Normalize this sport's market artifact into its sport-specific
        frame (schema stays sport-specific; the frontend market adapter
        dispatches on ``sport_key``)."""
        ...

    def market_probs(self, market_row: pd.Series) -> dict[str, float]:
        """Semantic market probabilities for one game row.

        Keys: ``p_home``, ``p_away`` plus sport-specific lines (MLB:
        ``p_over_<line>``, ``p_home_cover_<line>``, ``p_rl_<line>_*``;
        NFL: ``p_cover_offered``, ``p_over_offered``; etc.).

        TIE and PUSH are distinct and both first-class (Phase 1 review
        decision 1): include ``p_tie`` when the settlement scope allows a
        game-level tie (NFL full-game, NHL regulation three-way markets —
        never force ``p_tie`` to null on a three-way regulation market)
        and ``p_push`` when a wager can land exactly on an integer line.
        Scalar two-way views renormalize via
        ``core.markets.normalize_push_probs`` / ``normalize_tie_probs`` —
        push/tie mass is never silently folded into a side.
        """
        ...

    def settlement_config(self, market_kind: str) -> "SettlementConfig":
        """The settlement config for one of this sport's market kinds.

        ``market_kind`` is a sport-specific label (e.g. ``moneyline``,
        ``total``, ``run_line``, ``spread``). Each sport decides its own
        ``settlement_scope`` (regulation vs full_game), ``tie_allowed``,
        and ``push_allowed`` per market kind.
        """
        ...


def load_artifact_contract(path: str | Path) -> ArtifactContract:
    """Load an artifact contract from JSON (sport-declared, file-backed)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    families = tuple(
        ArtifactFamily(
            family=f["family"],
            pattern=f["pattern"],
            ext=f["ext"],
            date_stamped=bool(f.get("date_stamped", True)),
            retention_exempt=bool(f.get("retention_exempt", False)),
            description=str(f.get("description", "")),
        )
        for f in data.get("families", [])
    )
    return ArtifactContract(sport=normalize_sport_key(data.get("sport")),
                            families=families)


def default_artifact_contract(sport: str,
                              config: PlatformConfig | None = None) -> ArtifactContract:
    """The artifact contract for a sport, pinned to real production names.

    Patterns are locked to the read-only reference repository's real
    ``data_delivery`` filenames (see ``tests/fixtures/``), not invented
    shapes: MLB uses the ``run_engine_`` prefixes for markets/monitor/OOF;
    NFL prefixes every family with ``nfl_``. Every family is date-stamped
    and retention-subject except the durable history/contract artifacts,
    which are explicitly retention-exempt (Phase 0 decision 7/8).
    """
    s = normalize_sport_key(sport)
    if s == "mlb":
        dated = [
            ("todays_games", "todays_games_*.csv", "csv"),
            ("power_rankings", "power_rankings_*.csv", "csv"),
            ("calibration", "calibration_*.json", "json"),
            ("model_monitor", "model_monitor_*.json", "json"),
            ("markets", "run_engine_markets_*.csv", "csv"),
            ("markets_monitor", "run_engine_monitor_*.json", "json"),
            ("shap_game", "shap_game_*.csv", "csv"),
            ("predictions_history", "predictions_history_*.csv", "csv"),
            ("feature_drift", "feature_drift_*.csv", "csv"),
            ("feature_coverage", "feature_coverage_*.csv", "csv"),
            ("rolling_brier", "rolling_brier_*.json", "json"),
            ("run_engine_oof", "run_engine_oof_*.csv", "csv"),
            # §22 Amendment 11 support artifact: decided-game frame for
            # board-finals reconciliation + markets team-name bridging.
            ("game_level_features", "game_level_features_*.csv", "csv"),
        ]
    else:
        dated = [
            ("todays_games", f"{s}_todays_games_*.csv", "csv"),
            ("power_rankings", f"{s}_power_rankings_*.csv", "csv"),
            ("calibration", f"{s}_calibration_*.json", "json"),
            ("model_monitor", f"{s}_model_monitor_*.json", "json"),
            ("markets", f"{s}_run_engine_markets_*.csv", "csv"),
            ("markets_monitor", f"{s}_run_engine_monitor_*.json", "json"),
            ("shap_game", f"{s}_shap_game_*.csv", "csv"),
            ("predictions_history", f"{s}_predictions_history_*.csv",
             "csv"),
            ("feature_drift", f"{s}_run_engine_feature_drift_*.csv",
             "csv"),
            ("feature_coverage", f"{s}_run_engine_feature_coverage_*.csv",
             "csv"),
            # §22.1 standalone monitor siblings (r5): same records MLB
            # persists, sport-prefixed filenames.
            ("rolling_brier", f"{s}_rolling_brier_*.json", "json"),
            ("features_metadata", f"{s}_features_metadata_*.json", "json"),
        ]
    durable = [
        ("model_history", "model_history.json", "json"),
        ("model_version_history", "model_version_history.json", "json"),
    ]
    if s == "mlb":
        # MLB keeps its durable features_metadata declaration (reviewed pin,
        # tests/test_mlb_artifact_schemas.py): the real repo emits the dated
        # copy, but the family is retention-exempt for MLB.
        durable.append(("features_metadata", "features_metadata.json", "json"))
    # NNX: the formerly declared-but-unemitted durable
    # ``{s}_features_metadata.json`` is replaced by the dated standalone
    # emission added above (r5 §22.1 migration).
    prefix = "" if s == "mlb" else f"{s}_"
    families = []
    for name, pattern, ext in dated:
        families.append(ArtifactFamily(
            family=name,
            pattern=pattern,
            ext=ext,
            description=f"{s} frontend artifact (20-day retention)",
        ))
    for name, pattern, ext in durable:
        families.append(ArtifactFamily(
            family=f"{prefix}{name}",
            pattern=f"{prefix}{pattern}",
            ext=ext,
            date_stamped=False,
            retention_exempt=True,
            description=f"{s} durable contract artifact (retention-exempt)",
        ))
    return ArtifactContract(sport=s, families=tuple(families))
