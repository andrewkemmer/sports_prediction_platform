"""§7.1 per-field availability metadata layer (B-001-RESIDUAL, r6).

Pins:

* every feature in every sport's frozen contracts carries a declared
  availability class (closed over the contracts — enforced at registry
  validation AND here as an independent pin);
* every declared class is one of the §7.1 classes and every declared
  lead is a conservative, documented provenance bound;
* the stamp is ``scheduled_start − lead`` and passes the tested §7.1
  strict validator (``available_at < prediction_cutoff``) for both
  buffers (60/90 min) and any realistic start;
* the gate accepts the UNSTAMPED serving frame, never mutates it, and
  fails closed on unresolvable starts / undeclared features;
* all four study configs activate ``enforce`` and the runners consume
  the study mode key (the 7.6 activation).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from core.features import (
    AVAILABILITY_CLASSES,
    AvailabilityGateResult,
    FIELD_CLASSES,
    FIELD_LEAD_OVERRIDES,
    available_at_for,
    class_for,
    lead_hours_for,
    per_field_gate,
    require_classes_for,
    stamp_frame,
    stamped_feature_records,
)
from core.validation.future_availability import normalize_enforcement_mode
from core.validation.schema import ValidationError

RUNNERS = {
    "mlb": "sports/mlb/run_production.py",
    "nfl": "sports/nfl/run_production.py",
    "nhl": "sports/nhl/run_production.py",
    "nba": "sports/nba/run_production.py",
}
STUDY_YAMLS = {
    "mlb": "sports/mlb/config/study.yaml",
    "nfl": "sports/nfl/config/study.yaml",
    "nhl": "sports/nhl/config/study.yaml",
    "nba": "sports/nba/config/study.yaml",
}
START_COLS = {"mlb": "start_time_utc", "nfl": "gameday",
              "nhl": "game_date", "nba": "game_date"}


def _contract_features(sport: str) -> tuple[str, ...]:
    import importlib
    mod = importlib.import_module(f"sports.{sport}.features.registry")
    return tuple(dict.fromkeys(
        (*mod.MONEYLINE_FEATURE_COLS, *mod.MARKET_FEATURE_COLS)))


def _frame(sport: str, rows: int = 2) -> pd.DataFrame:
    start = datetime(2026, 9, 12, 23, 5, tzinfo=timezone.utc)
    return pd.DataFrame({
        "game_id": [f"{sport}-{i}" for i in range(rows)],
        START_COLS[sport]: [start] * rows,
    })


# ---------------------------------------------------------------------------
# closed coverage over the frozen contracts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sport", ["mlb", "nfl", "nhl", "nba"])
def test_every_contract_field_declared(sport):
    feats = _contract_features(sport)
    missing = [f for f in feats if f"{sport}:{f}" not in FIELD_CLASSES]
    assert not missing, f"{sport}: undeclared contract fields {missing}"


@pytest.mark.parametrize("sport", ["mlb", "nfl", "nhl", "nba"])
def test_no_orphan_declarations(sport):
    """Every sport-prefixed declaration names a REAL contract field — no
    speculative entries (the metadata table is closed over contracts,
    both directions)."""
    owned = {k.split(":", 1)[1] for k in FIELD_CLASSES
             if k.startswith(f"{sport}:")}
    feats = set(_contract_features(sport))
    orphans = owned - feats
    assert not orphans, f"{sport}: availability entries for non-contract fields {sorted(orphans)}"


def test_classes_are_the_declared_set():
    assert set(AVAILABILITY_CLASSES) == {
        "schedule", "ratings", "rolling", "precomputed"}


def test_leads_are_conservative_and_documented():
    """The uniform 24 h lead is the declared provenance bound: guaranteed
    availability from the prior completed-game midnight. Any future
    per-field override must be explicit (FIELD_LEAD_OVERRIDES)."""
    assert FIELD_LEAD_OVERRIDES == {}
    for cls, hours in AVAILABILITY_CLASSES.items():
        assert hours == 24, cls


def test_unknown_feature_fails_closed():
    with pytest.raises(ValidationError, match="no availability class"):
        class_for("mlb:not_a_real_feature")
    with pytest.raises(ValidationError, match="no availability class"):
        available_at_for(
            datetime(2026, 9, 12, tzinfo=timezone.utc), "nfl:zzz")


def test_require_classes_for_rejects_drift():
    with pytest.raises(ValidationError, match="without an availability class"):
        require_classes_for("nfl", ["elo_diff", "brand_new_thing"])


# ---------------------------------------------------------------------------
# stamps + §7.1 strict gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sport", ["mlb", "nfl", "nhl", "nba"])
@pytest.mark.parametrize("buffer_minutes", [60, 90])
def test_gate_passes_for_realistic_starts(sport, buffer_minutes):
    feats = _contract_features(sport)
    report = per_field_gate(
        _frame(sport), sport=sport, start_col=START_COLS[sport],
        features=feats, buffer_minutes=buffer_minutes,
        label=f"{sport} gate")
    assert isinstance(report, AvailabilityGateResult)
    assert report.ok, report.failing_fields()
    assert report.n_fields == len(feats)
    assert report.n_rows_per_field == 2
    assert report.n_missing == 0 and report.n_violations == 0


def test_stamp_is_start_minus_declared_lead():
    start = datetime(2026, 9, 12, 23, 5, tzinfo=timezone.utc)
    lead = lead_hours_for("mlb:elo_diff")
    assert available_at_for(start, "mlb:elo_diff") == start - timedelta(hours=lead)
    recs = stamped_feature_records(
        pd.DataFrame({"game_id": ["g"], "start": [start]}),
        start_col="start", features=("mlb:elo_diff",))
    assert recs[0]["available_at"] == start - timedelta(hours=lead)
    assert recs[0]["scheduled_start_utc"] == start


def test_gate_never_mutates_the_served_frame():
    f = _frame("mlb")
    before = tuple(f.columns)
    per_field_gate(f, sport="mlb", start_col=START_COLS["mlb"],
                   features=("elo_diff",), buffer_minutes=60,
                   label="mlb")
    assert tuple(f.columns) == before
    stamped = stamp_frame(f, start_col=START_COLS["mlb"],
                          features=("mlb:elo_diff",))
    assert f is not stamped and tuple(f.columns) == before


def test_gate_fails_closed_on_unresolvable_start():
    f = _frame("nfl")
    f.loc[0, START_COLS["nfl"]] = None
    with pytest.raises(ValidationError, match="CLOSED"):
        per_field_gate(f, sport="nfl", start_col=START_COLS["nfl"],
                       features=("elo_diff",), buffer_minutes=90,
                       label="nfl gate")


def test_gate_fails_closed_on_undeclared_feature():
    with pytest.raises(ValidationError, match="no availability class"):
        per_field_gate(_frame("nba"), sport="nba",
                       start_col=START_COLS["nba"],
                       features=("not_declared",), buffer_minutes=90,
                       label="nba gate")


def test_gate_requires_required_columns():
    with pytest.raises(ValidationError, match="lacks"):
        per_field_gate(pd.DataFrame({"x": [1]}), sport="mlb",
                       start_col="start", features=("elo_diff",),
                       buffer_minutes=60, label="mlb gate")


# ---------------------------------------------------------------------------
# 7.6 activation: enforce mode live in config and consumed by runners
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sport", ["mlb", "nfl", "nhl", "nba"])
def test_study_yaml_activates_enforce(sport):
    text = open(STUDY_YAMLS[sport], encoding="utf-8").read()
    line = next(l for l in text.splitlines()
                if l.startswith("availability_enforcement:"))
    assert normalize_enforcement_mode(
        line.split(":", 1)[1].split("#", 1)[0].strip(), sport) == "enforce"


@pytest.mark.parametrize("sport", ["mlb", "nfl", "nhl", "nba"])
def test_runner_consumes_study_mode_key(sport):
    src = open(RUNNERS[sport], encoding="utf-8").read()
    assert "study.availability_enforcement" in src, sport
    assert '== "enforce"' in src, sport
    assert "per_field_gate(" in src, sport
