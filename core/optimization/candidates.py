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
from dataclasses import dataclass

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
                     bounds,                     # core.optimization.models.Bounds
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
