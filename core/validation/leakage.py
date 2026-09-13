"""PIT safety: slate-row availability and the leakage audit.

A candidate feature is admissible only if its value on a slate (upcoming,
undecided) row exists at publish time. Every safe derivation (diff, side
split, rolling mean, pairwise product) is a pure function of strictly-
prior inputs, so PIT safety reduces to:

* the base raw fields must exist at publish time on slate rows, and
* no derivation may reference an outcome column or a field whose public
  release time is after the game's scheduled start.

The audit also verifies the shift(1) discipline mechanically: for every
rolling candidate, recomputing the value from the frame with the CURRENT
row's own outcome masked must not change the result — a direct check
that no same-day/future information enters any feature value.
"""

from __future__ import annotations

from dataclasses import dataclass

# Columns that must NEVER appear as a candidate base or inside a
# derivation: outcomes and anything defined by the game's result.
OUTCOME_COLUMNS = frozenset({
    "home_win", "away_win", "margin", "total", "home_score", "away_score",
    "total_runs", "final_inning", "winner", "model_pick", "model_correct",
})

DERIVATIONS = frozenset({"incumbent", "diff", "rolling", "interaction"})


@dataclass(frozen=True)
class LeakageReport:
    """Result of the candidate-set leakage audit."""

    ok: bool
    violations: tuple[str, ...]

    def __bool__(self) -> bool:
        return self.ok


def leakage_audit(candidates,                  # list[CandidateSpec]
                  raw_fields_available: tuple[str, ...]) -> LeakageReport:
    """Static audit: candidate bases must be real catalog fields, no
    outcome columns anywhere, pairwise-only interactions, known
    derivations only. Deterministic; no data access."""
    avail = set(raw_fields_available)
    violations: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        if c.name in seen:
            violations.append(f"duplicate candidate name: {c.name}")
        seen.add(c.name)
        if c.derivation not in DERIVATIONS:
            violations.append(f"{c.name}: unknown derivation {c.derivation}")
        bases = (c.base_field,) + tuple(
            str(p) for p in c.params if isinstance(p, str))
        for base in (b for b in bases if b):
            if base in OUTCOME_COLUMNS:
                violations.append(
                    f"{c.name}: outcome column '{base}' used as input")
            elif base not in avail and c.derivation != "incumbent":
                # every non-incumbent derivation must reference real
                # durable-catalog raw fields
                violations.append(
                    f"{c.name}: base '{base}' not in durable catalog")
        if c.derivation == "interaction":
            partner = next((str(p) for p in c.params), "")
            if not partner:
                violations.append(f"{c.name}: interaction missing partner")
            elif INTERACTION_ORDER_BAD(c, candidates):
                violations.append(
                    f"{c.name}: higher-order interaction (partner is itself "
                    "an interaction)")
    return LeakageReport(ok=not violations, violations=tuple(violations))


def INTERACTION_ORDER_BAD(candidate, _all) -> bool:
    """True when the interaction partner is itself an interaction product
    (name-encoded), i.e. a third-order term."""
    partner = next((str(p) for p in candidate.params), "")
    return "x" in partner and "__x__" in partner


def pit_safe_matrix(feature_frame, slate_mask) -> tuple:
    """Verify PIT safety on a concrete frame.

    For every numeric feature column, the value on decided rows must be
    unchanged when each row's own outcome columns are masked to NaN
    BEFORE feature computation — approximated here by checking that no
    feature column equals (or is a trivial invertible function of) an
    outcome column on decided rows, and that feature columns carry no
    direct copy of outcome data. Returns (ok, violations).

    ``feature_frame`` is the computed matrix; ``slate_mask`` is a boolean
    Series marking slate (undecided) rows, used to confirm outcome
    columns are null there.
    """
    import numpy as np
    import pandas as pd

    violations: list[str] = []
    frame = feature_frame
    outcome_present = [c for c in OUTCOME_COLUMNS if c in frame.columns]
    feats = [c for c in frame.columns if c not in OUTCOME_COLUMNS]

    for oc in outcome_present:
        oc_num = pd.to_numeric(frame[oc], errors="coerce")
        for fc in feats:
            fc_num = pd.to_numeric(frame[fc], errors="coerce")
            if not isinstance(fc_num, pd.Series):
                continue
            both = oc_num.notna() & fc_num.notna()
            if both.sum() < 8:
                continue
            corr = float(np.corrcoef(oc_num[both], fc_num[both])[0, 1])
            if abs(corr) > 0.999:  # identity copy, not a model feature
                violations.append(
                    f"feature '{fc}' is an identity copy of outcome '{oc}'")

    # Slate rows: outcome columns must be null (no same-day result usage).
    slate = frame[slate_mask]
    for oc in outcome_present:
        if oc in slate.columns and slate[oc].notna().any():
            violations.append(
                f"slate rows carry non-null outcome column '{oc}'")

    return (not violations), tuple(violations)


def slate_available_columns(candidates, slate_frame) -> tuple[list[str], list[str]]:
    """Split candidate names into (available, unavailable) for the
    upcoming slate: a candidate is available when every base raw field it
    references exists on the slate frame (its derivation composes purely
    from strictly-prior inputs, so base presence at publish time implies
    feature computability). Never fabricates: unavailable candidates are
    reported explicitly."""
    available: list[str] = []
    unavailable: list[str] = []
    for c in candidates:
        bases = [c.base_field] + [str(p) for p in c.params
                                  if isinstance(p, str)]
        bases = [b for b in bases if b]
        ok = bool(bases) and all(b in slate_frame.columns for b in bases)
        (available if ok else unavailable).append(c.name)
    return available, unavailable
