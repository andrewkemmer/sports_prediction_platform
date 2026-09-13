"""Deterministic candidate generation.

Every candidate derives ONLY from raw fields already present in the
sport's durable catalog/store, through documented-safe derivations:

* diffs (home − away over per-side field pairs, skipping candidates
  whose twin production diff is already served),
* per-team rolling means over the side columns (shift(1) before the
  window — strictly-prior values only, per team's own chronology),
* pairwise interactions only (a × b, never higher order).

Ordering is fully determined by sorted names — the same catalog and
bounds always produce the identical catalog (fingerprint-stable). No
fabricated or unavailable fields: a raw field absent from the catalog
can never become a candidate.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

INTERACTION_SEP = "__x__"
DIFF_SUFFIX = "_home_minus_away"
ROLL_SEP = "__rollmean"
ROLL_SUFFIX = "_diff"
SIDE_SUFFIXES = ("_home", "_away")


@dataclass(frozen=True, order=True)
class CandidateSpec:
    """One generated candidate feature."""

    name: str          # unique, deterministic derivation-encoding name
    derivation: str    # "incumbent" | "diff" | "rolling" | "interaction"
    base_field: str    # raw catalog field ("" for pure interactions)
    params: tuple = () # e.g. (window,) for rolling, (partner,) for interact


def split_side(field: str) -> tuple[str, str] | None:
    """``elo_home`` -> (\"elo\", \"home\"); None when not a side column."""
    for suffix in SIDE_SUFFIXES:
        if field.endswith(suffix):
            return field[: -len(suffix)], suffix.lstrip("_")
    return None


def _roll_name(stem: str, window: int) -> str:
    return f"{stem}{ROLL_SEP}{window}{ROLL_SUFFIX}"


def side_stems(raw_fields: tuple[str, ...]) -> tuple[str, ...]:
    """Stems with a COMPLETE home/away pair in the pool
    (``elo_home`` + ``elo_away`` -> "elo")."""
    sides = {f for f in raw_fields if split_side(f) is not None}
    stems = set()
    for f in sides:
        stem, side = split_side(f)  # type: ignore[misc]
        if side == "home" and f"{stem}_away" in sides:
            stems.add(stem)
    return tuple(sorted(stems))


def rolling_candidates(raw_fields: tuple[str, ...],
                       horizons: tuple[int, ...]) -> list[CandidateSpec]:
    """Per-team rolling-mean-difference candidates, one per side-pair
    stem per horizon: each team's OWN strictly-prior event stream is
    windowed (shift(1) before the mean) and the feature is the home
    team's rolled mean minus the away team's rolled mean — the temporal
    analogue of the served diff features. Game-level (non-side) fields
    have no team-chronology semantics and generate no rolling
    candidates."""
    out: list[CandidateSpec] = []
    for stem in side_stems(raw_fields):
        for w in sorted(horizons):
            out.append(CandidateSpec(_roll_name(stem, w), "rolling",
                                     f"{stem}_home", (int(w),)))
    return out


def diff_candidates(raw_fields: tuple[str, ...],
                    incumbent: tuple[str, ...] = ()) -> list[CandidateSpec]:
    """home − away diff candidates over complete side-column pairs.

    A candidate is skipped when its twin production diff (``<stem>_diff``)
    is already a served incumbent feature — the identical mathematical
    quantity must not re-enter the catalog as a duplicate.
    """
    inc = set(incumbent)
    out: list[CandidateSpec] = []
    for stem in side_stems(raw_fields):
        if f"{stem}_diff" in inc:
            continue  # twin already served in production
        out.append(CandidateSpec(f"{stem}{DIFF_SUFFIX}", "diff",
                                 f"{stem}_home", ()))
    return out


def interaction_candidates(fields: tuple[str, ...]) -> list[CandidateSpec]:
    """PAIRWISE-only interaction candidates (a × b), name-sorted, a < b.

    Higher-order products are structurally impossible here: the product
    is taken over exactly two distinct base fields, and interactions of
    interactions are never generated (bases must be catalog raw fields).
    """
    fs = sorted(set(fields))
    out: list[CandidateSpec] = []
    for i, a in enumerate(fs):
        for b in fs[i + 1:]:
            out.append(CandidateSpec(f"{a}{INTERACTION_SEP}{b}",
                                     "interaction", a, (b,)))
    return out


def build_candidates(raw_fields: tuple[str, ...],
                     bounds,                     # core.experiments.search_space.Bounds
                     *, incumbent: tuple[str, ...] = (),
                     ) -> tuple[list[CandidateSpec], list[str]]:
    """Deterministic candidate catalog for one sport/model scope.

    Returns (candidates, withheld). The catalog contains: incumbent
    features (unchanged base), diff candidates (twin-checked), per-team
    rolling lookbacks per horizon, then pairwise interactions — truncated
    to ``bounds.max_candidate_features`` by the deterministic priority
    order (incumbent → diffs → rolling → interactions), so the bounded
    run keeps a stable prefix. ``withheld`` lists names dropped by the
    bound (logged by the runner; nothing is silently discarded).
    """
    fields = tuple(sorted(set(raw_fields)))
    if not fields:
        raise ValueError("build_candidates: empty raw field pool")

    inc = sorted(set(incumbent))
    diffs = diff_candidates(fields, incumbent)
    rolls = rolling_candidates(fields, bounds.horizon_choices)
    inters = interaction_candidates(fields)

    budget = bounds.max_candidate_features
    ordered: list[CandidateSpec] = (
        [CandidateSpec(name, "incumbent", name, ()) for name in inc]
        + diffs + rolls + inters)
    if len(ordered) <= budget:
        return ordered, []
    # Deterministic truncation by the fixed priority order above: the
    # bounded run keeps a stable prefix; everything dropped is reported.
    kept = ordered[:budget]
    withheld = [c.name for c in ordered[budget:]]
    return kept, withheld


def candidate_catalog_fingerprint(candidates: list[CandidateSpec]) -> str:
    """SHA-256 over the sorted, canonical catalog serialization — the
    determinism proof for the candidate-generation layer."""
    canon = "|".join(f"{c.name}:{c.derivation}:{c.base_field}:"
                     f"{','.join(str(p) for p in c.params)}"
                     for c in sorted(candidates))
    return hashlib.sha256(canon.encode()).hexdigest()


def horizon_candidates(bounds) -> tuple[int, ...]:
    """The run's training-horizon/lookback candidates (5–10, frozen)."""
    return tuple(sorted(bounds.horizon_choices))


def spec_from_name(name: str,
                   raw_fields: tuple[str, ...]) -> CandidateSpec | None:
    """Reconstruct a candidate spec from its deterministic name (the
    adapter's matrix path resolves candidate columns this way). Returns
    None for incumbent features or unknown names."""
    if ROLL_SEP in name:
        stem, _, tail = name.partition(ROLL_SEP)
        if not tail.endswith(ROLL_SUFFIX):
            return None
        try:
            window = int(tail[: -len(ROLL_SUFFIX)])
        except ValueError:
            return None
        if f"{stem}_home" in raw_fields and f"{stem}_away" in raw_fields:
            return CandidateSpec(name, "rolling", f"{stem}_home", (window,))
        return None
    if DIFF_SUFFIX in name:
        stem = name[: -len(DIFF_SUFFIX)]
        if f"{stem}_home" in raw_fields and f"{stem}_away" in raw_fields:
            return CandidateSpec(name, "diff", f"{stem}_home", ())
        return None
    if INTERACTION_SEP in name:
        a, _, b = name.partition(INTERACTION_SEP)
        if a in raw_fields and b in raw_fields and a < b:
            return CandidateSpec(name, "interaction", a, (b,))
        return None
    return None  # incumbent features are columns, not specs


# ---------------------------------------------------------------------------
# Shared derivation evaluation (runner + adapters compute identically)
# ---------------------------------------------------------------------------
def derive_column(df, spec: CandidateSpec, *,
                  team_cols: tuple[str, str] = ("home_team", "away_team"),
                  date_col: str = "game_date"):
    """Evaluate one candidate derivation on a frame, PIT-safely.

    Rolling means are PER-TEAM: the side column's values are unpivoted to
    the team's own chronological event sequence, shifted by one event
    (the current row never enters its own feature — the same shift(1)
    discipline as the production engines), windowed, and mapped back to
    the home/away slots. When team/date columns are absent the derivation
    falls back to a plain shifted rolling mean. Interactions multiply two
    base columns row-wise (both already strictly-prior values).
    """
    import pandas as pd

    if spec.derivation == "incumbent":
        return df[spec.name] if spec.name in df.columns \
            else pd.Series(np.nan, index=df.index)
    if spec.derivation == "diff":
        stem, _ = split_side(spec.base_field)  # base is the *_home column
        home_col, away_col = f"{stem}_home", f"{stem}_away"
        home = df[home_col] if home_col in df.columns \
            else pd.Series(np.nan, index=df.index)
        away = df[away_col] if away_col in df.columns \
            else pd.Series(np.nan, index=df.index)
        return home - away
    if spec.derivation == "rolling":
        window = int(spec.params[0])
        ht, at = team_cols
        stem, _ = split_side(spec.base_field)  # base is the *_home column
        home_col, away_col = f"{stem}_home", f"{stem}_away"
        if (home_col in df.columns and away_col in df.columns
                and ht in df.columns and at in df.columns
                and date_col in df.columns):
            home_r, away_r = _team_rolling(df, home_col, away_col, window,
                                           ht, at, date_col)
            return home_r - away_r
        # Fallback (incomplete pair/identity): plain shifted rolling of
        # the base column.
        base = df[spec.base_field] if spec.base_field in df.columns \
            else pd.Series(np.nan, index=df.index)
        return base.shift(1).rolling(window, min_periods=1).mean()
    if spec.derivation == "interaction":
        partner = str(spec.params[0])
        a = df[spec.base_field] if spec.base_field in df.columns \
            else pd.Series(np.nan, index=df.index)
        b = df[partner] if partner in df.columns \
            else pd.Series(np.nan, index=df.index)
        return a * b
    raise ValueError(f"derive_column: unknown derivation {spec.derivation}")


def _team_rolling(df, home_col: str, away_col: str, window: int,
                  home_team_col: str, away_team_col: str,
                  date_col: str) -> tuple:
    """Per-team shift(1) rolling means over a twin-column pair.

    Unpivots BOTH slot columns into each team's own chronological event
    stream — the home slot's values belong to the home team's sequence,
    the away slot's values to the away team's — sorts by (team, date,
    row) deterministically, applies shift(1) then the rolling mean within
    each team's own sequence, and returns (home_team_rolled,
    away_team_rolled) mapped back to the original rows. Strictly prior
    by construction: the current row's own value never enters its own
    feature, and no opponent information leaks into a team's stream."""
    import pandas as pd

    n = len(df)
    home_vals = df[home_col].astype(float)
    away_vals = df[away_col].astype(float)
    dates = pd.to_datetime(df[date_col], errors="coerce")
    home_team = df[home_team_col]
    away_team = df[away_team_col]

    long = pd.DataFrame({
        "row": np.tile(np.arange(n), 2),
        "slot": np.repeat(np.array(["home", "away"], dtype=object), n),
        "team": pd.concat([home_team, away_team], ignore_index=True),
        "date": pd.concat([dates, dates], ignore_index=True),
        "value": pd.concat([home_vals, away_vals], ignore_index=True),
    })
    long = long[long["team"].notna() & long["date"].notna()]
    long = long.sort_values(["team", "date", "row"], kind="mergesort")
    long["rolled"] = long.groupby("team", sort=False)["value"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean())

    home_out = pd.Series(np.nan, index=df.index, dtype=float)
    away_out = pd.Series(np.nan, index=df.index, dtype=float)
    for slot, out in (("home", home_out), ("away", away_out)):
        block = long[long["slot"] == slot]
        if block.empty:
            continue
        vals = pd.Series(block["rolled"].to_numpy(),
                         index=df.index[block["row"].to_numpy()])
        out.loc[vals.index] = vals
    return home_out, away_out


def model_candidates(bounds) -> tuple[str, ...]:
    """The run's model candidates (5–8, frozen)."""
    return tuple(bounds.model_names)

# ---------------------------------------------------------------------------
# Frozen optimizer bounds and declarative run config (merged from the
# former core/optimization/models.py in the Phase-r1 restructure).
# Bounds are frozen, not tunable per run: <= 1,000 candidate features,
# 5-10 horizon candidates, 5-8 model candidates, pairwise interactions
# only, fixed study.yaml fold geometry, bounded wall clock, fixed seed.
# ---------------------------------------------------------------------------


MAX_CANDIDATE_FEATURES = 1_000
MIN_HORIZON_CANDIDATES = 5
MAX_HORIZON_CANDIDATES = 10
MIN_MODEL_CANDIDATES = 5
MAX_MODEL_CANDIDATES = 8
MAX_INTERACTION_ORDER = 2  # pairwise only
DEFAULT_SEED = 42
DEFAULT_MAX_SECONDS = 600.0
# Frozen per-scope sweep cap: the bounded walk-forward sweep evaluates at
# most this many candidate feature sets per scope. The runner truncates
# the deterministic candidate priority order to a stable prefix (every
# drop is logged), so the run stays inside its wall-clock budget AND
# remains fully reproducible — unlike a time-based cutoff alone.
DEFAULT_SWEEP_CAP = 60
# Parallel worker count for the candidate-set sweep (0/1 = sequential).
# Pure execution detail: each candidate set's walk-forward OOF is
# independent and returned in deterministic order, so results are
# bit-identical for any n_jobs.


@dataclass(frozen=True)
class Bounds:
    """The frozen per-run candidate bounds (validated)."""

    max_candidate_features: int = MAX_CANDIDATE_FEATURES
    horizon_choices: tuple[int, ...] = (3, 5, 8, 10, 12, 15)
    model_names: tuple[str, ...] = (
        "xgboost", "lightgbm", "logistic", "randomforest", "mlp",
    )
    interaction_order: int = MAX_INTERACTION_ORDER
    seed: int = DEFAULT_SEED
    max_seconds: float = DEFAULT_MAX_SECONDS
    sweep_cap: int = DEFAULT_SWEEP_CAP
    n_jobs: int = 1

    def __post_init__(self) -> None:
        if self.max_candidate_features < 1 or \
                self.max_candidate_features > MAX_CANDIDATE_FEATURES:
            raise ValueError(
                f"max_candidate_features must be in [1, {MAX_CANDIDATE_FEATURES}]")
        if not (MIN_HORIZON_CANDIDATES <= len(self.horizon_choices)
                <= MAX_HORIZON_CANDIDATES):
            raise ValueError(
                f"need {MIN_HORIZON_CANDIDATES}–{MAX_HORIZON_CANDIDATES} "
                "horizon/lookback candidates")
        if any(h < 1 for h in self.horizon_choices):
            raise ValueError("horizon windows must be >= 1")
        if not (MIN_MODEL_CANDIDATES <= len(self.model_names)
                <= MAX_MODEL_CANDIDATES):
            raise ValueError(
                f"need {MIN_MODEL_CANDIDATES}–{MAX_MODEL_CANDIDATES} model "
                "candidates")
        if self.interaction_order != MAX_INTERACTION_ORDER:
            raise ValueError("interaction_order is frozen to pairwise (2)")
        if self.max_seconds <= 0:
            raise ValueError("max_seconds must be > 0")
        if self.sweep_cap < 1:
            raise ValueError("sweep_cap must be >= 1")
        if self.n_jobs < 0:
            raise ValueError("n_jobs must be >= 0 (0/1 = sequential)")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "Bounds":
        raw = raw or {}
        return cls(
            max_candidate_features=int(
                raw.get("max_candidate_features", MAX_CANDIDATE_FEATURES)),
            horizon_choices=tuple(raw.get("horizon_choices",
                                          (3, 5, 8, 10, 12, 15))),
            model_names=tuple(raw.get("model_names",
                                      ("xgboost", "lightgbm", "logistic",
                                       "randomforest", "mlp"))),
            seed=int(raw.get("seed", DEFAULT_SEED)),
            max_seconds=float(raw.get("max_seconds", DEFAULT_MAX_SECONDS)),
            sweep_cap=int(raw.get("sweep_cap", DEFAULT_SWEEP_CAP)),
            n_jobs=int(raw.get("n_jobs", 1)),
        )


@dataclass(frozen=True)
class ModelScope:
    """One model's independent optimization scope.

    The moneyline (binary win probability) and the market/run-engine
    (score/totals/spread distribution) models each carry their own raw
    field pool and feature list — one is NEVER inferred from the other.
    """

    sport: str
    model: str                       # "moneyline" | "market"
    raw_fields: tuple[str, ...]      # raw fields from the durable catalog
    prod_features: tuple[str, ...]   # incumbent served feature list
    target: str                      # column optimized against
    horizon_field: str = ""          # column the horizon candidates shift

    def __post_init__(self) -> None:
        if self.model not in ("moneyline", "market"):
            raise ValueError("model must be 'moneyline' or 'market'")
        if not self.raw_fields:
            raise ValueError(f"{self.sport}/{self.model}: empty raw pool")
        if not self.prod_features:
            raise ValueError(f"{self.sport}/{self.model}: empty incumbent list")


# ---------------------------------------------------------------------------
# §15 training-horizon policies: the BOUNDED, DECLARED candidate set.
# The optimizer may compare only these named policies (never an open
# search over every calendar start / window length); the selected policy
# is locked into the model contract as training_mode + parameters.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingPolicy:
    """One declared training-horizon policy (§15).

    ``mode``: ``expanding`` (from a declared season start) or ``rolling``
    (the last K seasons). ``start_season`` pins the expanding origin;
    ``rolling_seasons`` pins K for rolling. ``decay`` optionally names a
    time-decay weighting for full-history expanding (``"none"`` = no
    decay). A policy is immutable and hashable — it can be locked into
    a production contract verbatim.
    """

    mode: str                        # "expanding" | "rolling"
    start_season: int | None = None  # expanding origin season
    rolling_seasons: int | None = None
    decay: str = "none"              # "none" | "exponential" (reserved)

    def __post_init__(self) -> None:
        if self.mode not in ("expanding", "rolling"):
            raise ValueError(
                f"training policy mode must be 'expanding' or 'rolling', "
                f"got {self.mode!r}")
        if self.mode == "expanding" and self.start_season is None:
            raise ValueError("expanding policy requires start_season")
        if self.mode == "rolling":
            if not self.rolling_seasons or self.rolling_seasons < 1:
                raise ValueError(
                    "rolling policy requires rolling_seasons >= 1")
        if self.decay not in ("none", "exponential"):
            raise ValueError(
                f"training policy decay must be 'none' or 'exponential', "
                f"got {self.decay!r}")

    @property
    def name(self) -> str:
        if self.mode == "expanding":
            base = f"expanding_from_{self.start_season}"
        else:
            base = f"rolling_last_{self.rolling_seasons}_seasons"
        return base if self.decay == "none" else f"{base}_{self.decay}"


def training_policy_candidates(
        earliest_season: int, latest_season: int,
        *, max_policies: int = 10) -> tuple[TrainingPolicy, ...]:
    """The bounded §15 candidate set for one optimization run.

    Declares exactly: expanding from the earliest valid season, expanding
    from a recent-regime season (latest − 4), and rolling windows of the
    2/3/5/8-season forms that fit the data span. Never every calendar
    start — at most ``max_policies`` candidates, deterministic order.
    """
    if latest_season < earliest_season:
        raise ValueError(
            f"latest_season {latest_season} < earliest {earliest_season}")
    span = latest_season - earliest_season + 1
    out: list[TrainingPolicy] = [
        TrainingPolicy("expanding", start_season=earliest_season)]
    if span > 5:
        out.append(TrainingPolicy(
            "expanding", start_season=latest_season - 4))
    for k in (2, 3, 5, 8):
        if k <= span and len(out) < max_policies:
            out.append(TrainingPolicy("rolling", rolling_seasons=k))
    if span >= 10 and len(out) < max_policies:
        out.append(TrainingPolicy(
            "expanding", start_season=earliest_season,
            decay="exponential"))
    return tuple(out)


@dataclass(frozen=True)
class OptimizationConfig:
    """A declarative optimization run configuration."""

    bounds: Bounds
    scopes: tuple[ModelScope, ...] = field(default_factory=tuple)
    coverage_min: float = 0.90            # coverage gate threshold
    logloss_tolerance: float = 0.005      # pooled logloss tolerance vs incumbent
    auc_tolerance: float = 0.005          # sealed AUC tolerance vs incumbent
    ece_tolerance: float = 0.010          # ECE tolerance vs incumbent
    stability_max_delta: float = 0.020    # late-vs-early fold logloss delta

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "OptimizationConfig":
        scopes = tuple(
            ModelScope(
                sport=str(s["sport"]),
                model=str(s["model"]),
                raw_fields=tuple(s["raw_fields"]),
                prod_features=tuple(s["prod_features"]),
                target=str(s["target"]),
                horizon_field=str(s.get("horizon_field", "")),
            )
            for s in raw.get("scopes", [])
        )
        thresholds = raw.get("gates", {}) or {}
        return cls(
            bounds=Bounds.from_mapping(raw.get("bounds")),
            scopes=scopes,
            coverage_min=float(thresholds.get("coverage_min", 0.90)),
            logloss_tolerance=float(
                thresholds.get("logloss_tolerance", 0.005)),
            auc_tolerance=float(thresholds.get("auc_tolerance", 0.005)),
            ece_tolerance=float(thresholds.get("ece_tolerance", 0.010)),
            stability_max_delta=float(
                thresholds.get("stability_max_delta", 0.020)),
        )


def load_optimization_config(raw: Mapping[str, Any] | None = None,
                             env_var: str = "OPTIMIZATION_CONFIG_JSON"
                             ) -> OptimizationConfig:
    """Load the declarative run config from a mapping or a JSON string in
    ``env_var``. Missing config raises — there is no silent default run."""
    if raw is None:
        blob = os.environ.get(env_var, "")
        if not blob:
            raise ValueError(
                f"optimization config required: pass a mapping or set {env_var}")
        import json

        raw = json.loads(blob)
    if "bounds" not in raw and "scopes" not in raw:
        raise ValueError("optimization config must declare bounds/scopes")
    return OptimizationConfig.from_mapping(raw)
