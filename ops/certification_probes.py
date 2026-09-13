"""§27 certification source probes — LIVE, bounded, one-shot each.

Deliberately NOT part of tests/ (network is banned there). Run manually:

    PYTHONUTF8=1 PYTHONPATH=. .venv-pinned/Scripts/python ops/certification_probes.py [sport ...]

Each probe routes through the sport's PRODUCTION transport seam
(``pull_statcast`` / ``load_schedule`` + ``compose_nfl_start_utc`` /
NHL ``pull_schedule`` / NBA ``pull_schedule``), so the probes certify
the real retry/backoff/throttle/cache code paths, not a stand-up
client. Every probe is bounded (one day / one season / today), saves
the RAW payload under ``ops/probes/<date>/`` as §5 evidence, asserts
the source's required columns are present, and prints a PASS/FAIL
line. A failure never blocks the other probes; the exit code reports
the tally.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

OPS = Path(__file__).resolve().parent
REPO_ROOT = OPS.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EVIDENCE = OPS / "probes" / date.today().isoformat()

RESULTS: list[tuple[str, str, str]] = []


def record(sport: str, probe: str, ok: bool, note: str = "") -> None:
    RESULTS.append((sport, probe, ("PASS" if ok else "FAIL") + f" {note}"))
    print(f"[{sport}] {probe}: {'PASS' if ok else 'FAIL'} {note}", flush=True)


def _save_raw(sport: str, name: str, payload) -> Path:
    d = EVIDENCE / sport
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    if isinstance(payload, pd.DataFrame):
        payload.to_parquet(p.with_suffix(".parquet"), index=False)
    else:
        p.write_text(payload if isinstance(payload, str)
                     else json.dumps(payload)[:2_000_000],
                     encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# MLB — production pull_statcast, one day into a probe-local cache
# ---------------------------------------------------------------------------

def probe_mlb() -> None:
    try:
        from sports.mlb.ingestion import (REQUIRED_STATCAST_COLS,
                                          cache_bounds, load_pitches,
                                          pull_statcast)

        day = date.today() - timedelta(days=3)
        out = OPS / "_probe_cache" / "mlb_pitches.parquet"
        pull_statcast(day, day, out_path=out, full_repull=False, resume=False)
        df = load_pitches(out)
        _save_raw("mlb", f"statcast_{day}", df)
        missing = [c for c in REQUIRED_STATCAST_COLS if c not in df.columns]
        b = cache_bounds(out)
        ok = (not missing) and len(df) > 0 and b[0] is not None
        record("mlb", f"pull_statcast({day})", ok,
               f"rows={len(df)} missing={missing[:4] or 'none'} "
               f"bounds={b[0]}..{b[1]}")
    except Exception as exc:  # noqa: BLE001 — certification report, not a test
        record("mlb", "pull_statcast", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# NFL — production load_schedule (nflreadpy) + the r9 instant composition
# ---------------------------------------------------------------------------

def probe_nfl() -> None:
    try:
        from sports.nfl.ingestion import (REQUIRED_COLS, SOURCE_SCHEDULES,
                                          compose_nfl_start_utc,
                                          load_schedule)

        season = date.today().year if date.today().month >= 9 else \
            date.today().year - 1
        out = OPS / "_probe_cache" / "nfl_schedules"
        df = load_schedule([season], out, full_repull=False)
        instants = compose_nfl_start_utc(df)
        _save_raw("nfl", f"schedules_{season}", df)
        composed = int(instants.notna().sum())
        missing = [c for c in REQUIRED_COLS["schedules"]
                   if c not in df.columns]
        # Past weeks must compose fully; far-future weeks may legitimately
        # lack a gametime yet.
        ok = (not missing) and composed > 0 \
            and "start_time_utc" in df.columns
        record("nfl", f"load_schedule({season})", ok,
               f"rows={len(df)} instants={composed} missing={missing or 'none'}")
    except Exception as exc:  # noqa: BLE001
        record("nfl", "load_schedule", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# NHL — production pull_schedule (api-web.nhle.com), a 3-day window
# ---------------------------------------------------------------------------

def probe_nhl() -> None:
    try:
        from sports.nhl.ingestion import (REQUIRED_COLS, load_schedule_cache,
                                          pull_schedule)

        # Early-October opening week (September is offseason; the
        # production pull correctly types an empty window as a genuine
        # gap — the probe must target a window with games).
        start = date(date.today().year, 10, 7)
        end = start + timedelta(days=3)
        out = OPS / "_probe_cache" / "nhl_schedule.parquet"
        pull_schedule(start, end, out_path=out, full_repull=False,
                      resume=False)
        df = load_schedule_cache(out)
        _save_raw("nhl", f"schedule_{start}_{end}", df)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        ok = (not missing) and len(df) > 0
        record("nhl", f"pull_schedule({start}..{end})", ok,
               f"rows={len(df)} missing={missing[:4] or 'none'}")
    except Exception as exc:  # noqa: BLE001
        record("nhl", "pull_schedule", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# NBA — production pull_schedule (stats.nba.com), current season window
# ---------------------------------------------------------------------------

def probe_nba() -> None:
    try:
        from sports.nba.ingestion import (REQUIRED_COLS, cached_seasons,
                                          load_schedule_cache, pull_schedule)

        season = date.today().year if date.today().month >= 10 else \
            date.today().year - 1
        out = OPS / "_probe_cache" / "nba_schedule.parquet"
        # The pull is season-granular: window inside the current season.
        pull_schedule(f"{season}-10-01", f"{season + 1}-04-30",
                      out_path=out, full_repull=False, resume=False)
        df = load_schedule_cache(out)
        _save_raw("nba", f"schedule_{season}", df)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        seasons = sorted(cached_seasons(out))
        ok = (not missing) and len(df) > 0
        record("nba", f"pull_schedule({season}-{season + 1})", ok,
               f"rows={len(df)} seasons={seasons} missing={missing[:4] or 'none'}")
    except Exception as exc:  # noqa: BLE001
        record("nba", "pull_schedule", False, f"{type(exc).__name__}: {exc}")


def main() -> int:
    wanted = set(a.lower() for a in sys.argv[1:]) or {"mlb", "nfl", "nhl",
                                                      "nba"}
    probes = {"mlb": probe_mlb, "nfl": probe_nfl, "nhl": probe_nhl,
              "nba": probe_nba}
    order = [s for s in sorted(probes) if s in wanted]
    for i, sport in enumerate(order):
        probes[sport]()
        if i < len(order) - 1:
            time.sleep(2)  # pacing between probes
    fails = [r for r in RESULTS if r[2].startswith("FAIL")]
    print(f"\n=== CERTIFICATION SOURCE PROBES: "
          f"{len(RESULTS) - len(fails)}/{len(RESULTS)} PASS ===")
    print(f"raw evidence: {EVIDENCE}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
