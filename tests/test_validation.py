"""Tests for core.validation."""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core.validation import (
    ValidationError,
    assert_deterministic,
    require_columns,
    stable_json,
    validate_metric,
    validate_probability,
    validate_probability_pair,
)
from core.validation.future_availability import (
    BUFFER_MINUTES,
    SPORT_RESOLUTION,
    SPORT_START_COL,
    FutureAvailabilityReport,
    SlateWindowReport,
    normalize_enforcement_mode,
    prediction_cutoff,
    validate_available_at,
    validate_settlement_record,
    validate_slate_window,
    window_anchor,
)


def test_validate_probability_strict():
    assert validate_probability(0.5) == 0.5
    with pytest.raises(ValidationError):
        validate_probability(0.0)
    with pytest.raises(ValidationError):
        validate_probability(1.0)
    assert validate_probability(0.0, strict=False) == 0.0
    assert validate_probability(1.0, strict=False) == 1.0
    with pytest.raises(ValidationError):
        validate_probability(1.5, strict=False)
    with pytest.raises(ValidationError):
        validate_probability(float("nan"))
    with pytest.raises(ValidationError):
        validate_probability("x")


def test_validate_probability_pair():
    assert validate_probability_pair(0.6, 0.4) == (0.6, 0.4)
    with pytest.raises(ValidationError, match="expected 1.0"):
        validate_probability_pair(0.6, 0.3)


def test_validate_metric():
    assert validate_metric(0.25, "brier") == 0.25
    with pytest.raises(ValidationError):
        validate_metric(float("inf"), "brier")
    with pytest.raises(ValidationError):
        validate_metric(None, "brier")


def test_require_columns():
    import pandas as pd
    df = pd.DataFrame({"a": [1], "b": [2]})
    require_columns(df, ["a"])
    with pytest.raises(ValidationError, match="missing required columns"):
        require_columns(df, ["a", "c"], context="slate")


def test_stable_json_sorted_and_deterministic():
    s1 = stable_json({"b": 1, "a": [1, 2]})
    s2 = stable_json({"a": [1, 2], "b": 1})
    assert s1 == s2
    assert '"a"' in s1 and s1.index('"a"') < s1.index('"b"')


def test_stable_json_rejects_nan():
    with pytest.raises(ValueError):
        stable_json({"x": float("nan")})


def test_stable_json_handles_paths_dataclasses_tuples():
    @dataclass
    class D:
        x: int
        p: Path

    p = Path("/tmp")
    s = stable_json(D(x=1, p=p))
    # Platform-neutral: the path is embedded in its JSON-escaped string form
    # (POSIX ``/tmp``; Windows ``\\tmp``). Compare against the same encoder's
    # rendering rather than hard-coding a POSIX literal.
    assert stable_json(str(p))[1:-1] in s


def test_sha256_and_determinism():
    from core.validation import sha256_bytes, sha256_text
    assert sha256_text("abc").startswith("ba7816bf")
    assert sha256_bytes(b"abc") == sha256_text("abc")

    def fn():
        return {"b": 1, "a": [1.5, 2]}

    digest = assert_deterministic(fn)
    assert isinstance(digest, str) and len(digest) == 64


def test_assert_deterministic_detects_nondeterminism():
    import itertools
    counter = itertools.count()

    def fn():
        next(counter)
        return {"n": next(counter)}

    with pytest.raises(ValidationError, match="nondeterministic"):
        assert_deterministic(fn)


def test_round_float():
    from core.validation import round_float
    assert round_float(0.123456789, 4) == 0.1235
    with pytest.raises(ValidationError):
        round_float(float("nan"))


# ---------------------------------------------------------------------------
# §7.1 point-in-time validators (B-001, Phase 7.5d WS4 — partial remediation;
# per-field available_at metadata layer = B-001-RESIDUAL, Phase 7.6).
# ---------------------------------------------------------------------------

_UTC = timezone.utc
_START = datetime(2026, 9, 7, 23, 5, tzinfo=_UTC)  # 7:05 PM ET game


def _row(avail, start=_START, gid="g1"):
    return {"game_id": gid, "scheduled_start_utc": start,
            "available_at": avail}


def test_s71_buffer_pins_match_spec():
    """MLB/NHL 60 min, NFL/NBA 90 min — hard-pinned per §7.1."""
    assert BUFFER_MINUTES == {"mlb": 60, "nhl": 60, "nfl": 90, "nba": 90}


def test_s71_cutoff_math_per_sport():
    assert prediction_cutoff(_START, 60) == _START - timedelta(minutes=60)
    assert prediction_cutoff(_START, 90) == _START - timedelta(minutes=90)


def test_s71_strict_inequality_and_equality_fails():
    cutoff = prediction_cutoff(_START, 60)
    ok = validate_available_at(
        [_row(cutoff - timedelta(minutes=1))], buffer_minutes=60,
        label="strict")
    assert ok.ok and ok.n_violations == 0
    # == cutoff FAILS (§7.1: strict < only).
    eq = validate_available_at([_row(cutoff)], buffer_minutes=60,
                               label="equality")
    assert not eq.ok and eq.n_violations == 1
    # 1 second after cutoff also fails.
    late = validate_available_at(
        [_row(cutoff + timedelta(seconds=1))], buffer_minutes=60,
        label="late")
    assert not late.ok


def test_s71_fail_closed_on_missing_and_unparseable():
    rep = validate_available_at(
        [_row(None), _row("not-a-timestamp", gid="g2")],
        buffer_minutes=60, label="failclosed")
    assert not rep.ok
    assert rep.n_missing == 2 and rep.n_violations == 0
    assert any("g2" in m for m in rep.missing)
    assert any("fails CLOSED" in m for m in rep.missing)


def test_s71_rejects_non_spec_buffers():
    with pytest.raises(ValidationError, match="must be 60"):
        validate_available_at([_row(None)], buffer_minutes=0, label="zero")
    with pytest.raises(ValidationError, match="per §7.1"):
        validate_available_at([_row(None)], buffer_minutes=75, label="odd")


def test_s71_slate_window_uniform_semantics():
    """Secondary gate: the future-start check ALWAYS runs; resolution only
    selects the comparison granularity (instant vs date)."""
    import pandas as pd
    frame = pd.DataFrame({
        "game_id": ["a", "b"],
        "start_time_utc": ["2026-09-07T16:05:00+00:00",
                           "2026-09-08T01:05:00+00:00"],
    })
    # Anchor = the instant the serving window OPENS (its first date) —
    # deterministic and window-derived, never wall clock.
    anchor = window_anchor("20260907")
    assert anchor == datetime(2026, 9, 7, tzinfo=_UTC)
    rep = validate_slate_window(frame, anchor_utc=anchor,
                                start_col="start_time_utc",
                                resolution="instant", label="t")
    assert isinstance(rep, SlateWindowReport) and rep.ok
    # A PRE-WINDOW start (the day before the window opens) violates.
    bad = validate_slate_window(
        frame.assign(start_time_utc=["2026-09-06T23:05:00+00:00",
                                     "2026-09-08T01:05:00+00:00"]),
        anchor_utc=anchor, start_col="start_time_utc",
        resolution="instant", label="t")
    assert not bad.ok and bad.n_future_violations == 1


def test_s71_slate_window_date_resolution():
    """Date-granularity sports compare on calendar date at the window edge
    (same-or-later than the window's first date passes)."""
    import pandas as pd
    frame = pd.DataFrame({
        "game_id": ["a", "b"],
        "gameday": ["2026-09-07", "2026-09-08"],
    })
    anchor = window_anchor("20260907")
    assert anchor == datetime(2026, 9, 7, tzinfo=_UTC)
    rep = validate_slate_window(frame, anchor_utc=anchor,
                                start_col="gameday",
                                resolution="date", label="t")
    assert rep.ok
    stale = validate_slate_window(
        frame.assign(gameday=["2026-09-06", "2026-09-08"]),
        anchor_utc=anchor, start_col="gameday",
        resolution="date", label="t")
    assert not stale.ok and any("predates the serving window" in v
                                for v in stale.violations)
    with pytest.raises(ValidationError, match="resolution"):
        validate_slate_window(frame, anchor_utc=anchor, start_col="gameday",
                              resolution="bogus", label="t")


def test_s71_window_anchor_is_deterministic():
    """Anchor derives from the serving window only — never wall clock; live
    and replay produce identical anchors."""
    assert window_anchor("20260907") == datetime(2026, 9, 7, tzinfo=_UTC)
    assert window_anchor("20260908") == datetime(2026, 9, 8, tzinfo=_UTC)
    assert window_anchor("20260907") == window_anchor("20260907")


@pytest.mark.parametrize("sport", ["mlb", "nfl", "nhl", "nba"])
def test_s71_slate_gate_catches_pre_window_and_decided_in_every_sport(sport):
    """B-001 WS4 (uniform secondary gate): all four sports catch BOTH a
    pre-window row and a decided row, at the start column/resolution each
    sport's data supports."""
    import pandas as pd
    start_col = SPORT_START_COL[sport]
    resolution = SPORT_RESOLUTION[sport]
    anchor = window_anchor("20260907")
    in_window = ("2026-09-07T23:05:00+00:00" if resolution == "instant"
                 else "2026-09-07")
    pre_window = ("2026-09-06T23:05:00+00:00" if resolution == "instant"
                  else "2026-09-06")
    clean = pd.DataFrame({"game_id": ["g1"], start_col: [in_window],
                          "home_win": [None]})
    assert validate_slate_window(clean, anchor_utc=anchor,
                                 start_col=start_col,
                                 resolution=resolution, label=sport).ok
    pre = pd.DataFrame({"game_id": ["g1"], start_col: [pre_window],
                        "home_win": [None]})
    rep_pre = validate_slate_window(pre, anchor_utc=anchor,
                                    start_col=start_col,
                                    resolution=resolution, label=sport)
    assert not rep_pre.ok and rep_pre.n_future_violations == 1
    decided = pd.DataFrame({"game_id": ["g1"], start_col: [in_window],
                            "home_win": [1.0]})
    rep_dec = validate_slate_window(decided, anchor_utc=anchor,
                                    start_col=start_col,
                                    resolution=resolution, label=sport)
    assert not rep_dec.ok and rep_dec.n_decided_rows == 1


def test_s71_settlement_subgate_semantics():
    class Cfg:
        tie_allowed = False
        push_allowed = True

    class CfgNoPush:
        tie_allowed = False
        push_allowed = False

    # r10 (§20 completion): each market kind settles under its OWN §20
    # vocabulary — a spread/total landing exactly on the line is that
    # MARKET's push, never a moneyline outcome. Caught on real 2019
    # nflverse data (2019_02_DAL_WAS fair-spread push) by the live smoke.
    assert validate_settlement_record(
        {"outcome": "push", "scope": "full_game"}, market_kind="spread",
        settlement_cfg=Cfg()).ok
    assert validate_settlement_record(
        {"outcome": "home_cover", "scope": "full_game"},
        market_kind="spread", settlement_cfg=Cfg()).ok
    assert validate_settlement_record(
        {"outcome": "away_cover", "scope": "full_game"},
        market_kind="spread", settlement_cfg=Cfg()).ok
    assert validate_settlement_record(
        {"outcome": "over", "scope": "full_game"}, market_kind="total",
        settlement_cfg=Cfg()).ok
    assert validate_settlement_record(
        {"outcome": "under", "scope": "full_game"}, market_kind="total",
        settlement_cfg=Cfg()).ok
    assert validate_settlement_record(
        {"outcome": "push", "scope": "full_game"}, market_kind="total",
        settlement_cfg=Cfg()).ok
    # cross-kind outcomes are INVALID: a moneyline 'home' is not a total
    # outcome; an 'over' is not a moneyline outcome.
    for bad_kind, bad_outcome in (("total", "home"),
                                  ("moneyline", "over"),
                                  ("moneyline", "home_cover"),
                                  ("spread", "over")):
        bad = validate_settlement_record(
            {"outcome": bad_outcome, "scope": "full_game"},
            market_kind=bad_kind, settlement_cfg=Cfg())
        assert not bad.ok and "not valid for market kind" in bad.violations[0]

    assert validate_settlement_record(
        {"outcome": "home", "scope": "full_game"}, market_kind="moneyline",
        settlement_cfg=Cfg()).ok
    assert validate_settlement_record(
        {"outcome": "push", "scope": "full_game"}, market_kind="moneyline",
        settlement_cfg=Cfg()).ok
    bad = validate_settlement_record(
        {"outcome": "tie", "scope": "full_game"}, market_kind="moneyline",
        settlement_cfg=Cfg())
    assert not bad.ok and "tie_allowed=False" in bad.violations[0]
    nopush = validate_settlement_record(
        {"outcome": "push"}, market_kind="moneyline",
        settlement_cfg=CfgNoPush())
    assert not nopush.ok

    class CfgTieOk:
        tie_allowed = True
        push_allowed = False

    # r10: NFL-style three-way moneyline — a tie outcome is valid exactly
    # when the kind's config declares tie_allowed=True.
    assert validate_settlement_record(
        {"outcome": "tie", "scope": "full_game"}, market_kind="moneyline",
        settlement_cfg=CfgTieOk()).ok
    scope_bad = validate_settlement_record(
        {"outcome": "home", "scope": "regulation"},
        market_kind="moneyline", settlement_cfg=Cfg())
    assert not scope_bad.ok and "regulation" in scope_bad.violations[0]


def test_s71_enforcement_mode_normalization():
    assert normalize_enforcement_mode("report_only", "x") == "report_only"
    assert normalize_enforcement_mode("enforce", "x") == "enforce"
    assert normalize_enforcement_mode(None, "x") == "report_only"
    # unknown modes still rejected — but 'enforce' is now the ACTIVATED
    # r6 mode (B-001-RESIDUAL metadata layer), no longer reserved.
    with pytest.raises(ValidationError):
        normalize_enforcement_mode("fail_open", "x")


def test_s71_report_fields_present():
    rep = validate_available_at([_row(_START - timedelta(hours=2))],
                                buffer_minutes=60, label="fields")
    assert isinstance(rep, FutureAvailabilityReport)
    assert rep.n_rows == 1 and rep.label == "fields"
