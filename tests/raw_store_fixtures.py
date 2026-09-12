"""Materialize the committed bounded raw fixtures into the gitignored store.

Phase 7.5e-A rebaseline. The original Phase 7.5 evidence capture ran against a
frozen store built in an external sandbox; those bytes never entered the
repository and are not reproducible from repository-controlled inputs (the
capture window and source versions were not recorded). Rather than force the
unreproducible literals, the permanent evidence tests were re-based onto the
committed, bounded datasets under ``tests/fixtures/raw_store/`` — one compact
Parquet per sport, generated deterministically by the per-sport support
modules' ``evidence_frame()`` builders.

This module only READS those committed fixtures and materializes the
gitignored working store that the production loaders expect, so a fresh
checkout (or CI) can run the store-backed tests with no network and no
external state. Two guarantees:

* **Idempotent / never clobbers** — a target file is written only when it is
  absent. A real store produced by a production pull is never truncated,
  overwritten, or partially replaced.
* **Read-only fixtures** — the committed Parquet files are inputs; nothing
  here regenerates, rewrites, or exports them (``evidence_frame()`` exists
  solely so the fixtures stay reproducible from reviewed code, and a
  provenance test asserts the two still agree).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Committed bounded raw fixtures (one compact Parquet per sport).
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "raw_store"

#: Gitignored working store the production loaders read (relative, matching
#: the paths the store-backed tests and adapters use).
STORE_DIR = Path("store")


def _write_once(path: Path, frame: pd.DataFrame) -> Path:
    """Write ``frame`` to ``path`` only when ``path`` does not already exist."""
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    return path


def fixture_frame(sport: str) -> pd.DataFrame:
    """Read one committed raw fixture (the frozen evidence input)."""
    return pd.read_parquet(FIXTURES_DIR / f"{sport}.parquet")


def ensure_raw_store(store_dir: Path | None = None) -> Path:
    """Materialize the committed raw fixtures into the gitignored store.

    Returns the store root. Safe to call repeatedly; existing files win.
    """
    root = Path(store_dir) if store_dir is not None else STORE_DIR

    _write_once(root / "mlb" / "raw" / "pitches.parquet",
                fixture_frame("mlb"))
    _write_once(root / "nba" / "raw" / "schedule.parquet",
                fixture_frame("nba"))
    _write_once(root / "nhl" / "raw" / "schedule.parquet",
                fixture_frame("nhl"))

    nfl = fixture_frame("nfl")
    for kind in ("schedules", "pbp"):
        for season, group in nfl[nfl["__kind"] == kind].groupby("season"):
            _write_once(root / "nfl" / "raw" / f"{kind}_{int(season)}.parquet",
                        group.drop(columns=["__kind"]))
    return root
