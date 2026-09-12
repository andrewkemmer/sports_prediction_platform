"""Permanent evidence test: fold fingerprints match the pinned incumbents.

Recomputes each sport's walk-forward fold fingerprint from the committed
bounded raw fixtures (materialized into the gitignored store by
``tests/raw_store_fixtures.py``) and the per-sport study configuration, and
compares EXACTLY against the reviewed in-module constants below (MLB 10,
NBA 23, NFL 54, NHL 23 folds). Fold identity covers per-fold train/val row counts
and the SHA256 of the ordered train/val game IDs — the same fingerprint
definition used to prove Phase 7.5 behavior neutrality.

The expected values were re-baselined during the Phase 7.5e-A rebaseline
against ``tests/fixtures/raw_store/*.parquet``, reviewed, and frozen here as
constants. The ORIGINAL Phase 7.5 capture values (from a frozen external
store that never entered the repository) are recorded in docs/TRACKER.md
for provenance. These constants are never regenerated, overwritten, or
exported; any change that shifts fold geometry, row membership, or
chronology fails here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# Pinned season ceiling (capture-time value). The Phase 7.5 evidence
# capture ran in 2026-09, when the historical ``date.today()``-derived
# ceiling evaluated to 2027; the value is pinned so the test is fully
# deterministic (no wall-clock dependence) and reproduces the
# capture-time season list exactly. Any change requires explicit
# reviewer approval in a phase report.
PINNED_CURRENT_SEASON = 2027

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _raw_store(raw_store):
    """Run every fold-fingerprint test against the committed raw fixtures."""
    return raw_store


# ---------------------------------------------------------------------------
# REVIEWED INCUMBENT CONSTANTS (frozen; never regenerate or export)
# ---------------------------------------------------------------------------
# Per-sport fold counts (Phase 7.5e-A rebaseline; see docs/TRACKER.md).
EXPECTED_FOLD_COUNTS = {"mlb": 10, "nba": 23, "nfl": 54, "nhl": 23}

# Fingerprint payload hash: SHA256 over the canonical JSON serialization
# (sort_keys=True) of {sport: [[n_train, n_val, train_ids_sha,
# val_ids_sha], ...]} with sports in sorted order — the ordered per-fold
# signature recomputed below.
EXPECTED_FINGERPRINT_PAYLOAD_SHA256 = (
    "b5a571c843104d6e02ff1d512eff7fd5d8516a07fb19b6d3ea9dffd11eed92bf"
)


def _sha_ids(ids) -> str:
    h = hashlib.sha256()
    for i in ids:
        h.update(str(i).encode())
        h.update(b"|")
    return h.hexdigest()


def _fold_signature(folds, df, id_col) -> list[list]:
    """Ordered [n_train, n_val, train_ids_sha, val_ids_sha] per fold."""
    sig = []
    for tr_idx, va_idx in folds:
        tr = df.iloc[sorted(tr_idx)]
        va = df.iloc[sorted(va_idx)]
        sig.append([
            int(len(tr)), int(len(va)),
            _sha_ids(tr[id_col].tolist()),
            _sha_ids(va[id_col].tolist()),
        ])
    return sig


def _mlb():
    from core.folds import walk_forward_splits
    from sports.mlb.features import build_features
    from sports.mlb.frames import get_decided_frame
    from sports.mlb.study_config import load_mlb_study

    study = load_mlb_study()
    game_df, _ = build_features("store/mlb/raw/pitches.parquet",
                                output_dir="store/mlb/raw")
    df = get_decided_frame(game_df).sort_values(
        "game_date").reset_index(drop=True)
    splits = walk_forward_splits(
        df, retrain_cadence_days=study.walk_forward.step_days,
        max_eval_folds=study.max_eval_folds,
        min_train_days=study.training.min_train_days_warmup)
    folds = [(s["train_games"].index.to_numpy(),
              s["val_games"].index.to_numpy())
             for s in splits
             if len(s["train_games"]) >= study.oof.min_training_rows
             and len(s["val_games"]) >= study.min_val_games]
    return _fold_signature(folds, df, "game_pk")


def _nba():
    from core.folds import make_folds
    from sports.nba.features import build_game_features
    from sports.nba.ingestion import load_schedule_cache
    from sports.nba.study_config import load_nba_study

    study = load_nba_study()
    sched = load_schedule_cache(Path("store/nba/raw/schedule.parquet"))
    sched = sched[(pd.to_numeric(sched["season"], errors="coerce")
                   >= min(study.warmup_seasons))]
    decided = sched[sched["home_score"].notna() & sched["away_score"].notna()]
    df = build_game_features(decided).sort_values(
        "game_date").reset_index(drop=True)
    fl = make_folds(df, study.oof_first_season,
                    cadence_days=study.walk_forward.step_days,
                    date_col="game_date",
                    min_val_games=study.min_val_games,
                    val_scope="all", end_of_day=False)
    folds = [(f.train_idx, f.val_idx) for f in fl]
    return _fold_signature(folds, df, "game_id")


def _nhl():
    from core.folds import make_folds
    from sports.nhl.features import build_game_features
    from sports.nhl.ingestion import load_schedule_cache
    from sports.nhl.study_config import load_nhl_study

    study = load_nhl_study()
    sched = load_schedule_cache(Path("store/nhl/raw/schedule.parquet"))
    decided = sched[sched["home_score"].notna() & sched["away_score"].notna()]
    df = build_game_features(decided).sort_values(
        "game_date").reset_index(drop=True)
    fl = make_folds(df, study.oof_first_season,
                    cadence_days=study.walk_forward.step_days,
                    date_col="game_date",
                    min_val_games=study.min_val_games,
                    val_scope="all", end_of_day=False)
    folds = [(f.train_idx, f.val_idx) for f in fl]
    return _fold_signature(folds, df, "game_id")


def _nfl():
    from core.folds import make_folds
    from sports.nfl.features import build_game_features
    from sports.nfl.ingestion import eligible_games, load_pbp, load_schedule
    from sports.nfl.study_config import load_nfl_study

    study = load_nfl_study()
    cache = Path("store/nfl/raw")
    current_season = PINNED_CURRENT_SEASON
    seasons = sorted(set(
        list(range(study.oof_first_season, current_season + 1))
        + list(study.warmup_seasons)))
    # Deterministic, network-free cache-only loaders: the pinned store is
    # the recorded fixture; the default production transport is never hit.
    from tests.nfl_fixtures import cache_only_loader
    schedule = load_schedule(seasons, cache, full_repull=False,
                             load=cache_only_loader(cache, "schedules"))
    schedule = eligible_games(schedule, study.oof_first_season,
                              min(study.warmup_seasons))
    pbp = load_pbp(seasons, cache, full_repull=False,
                   load=cache_only_loader(cache, "pbp"))
    decided = schedule[schedule["home_score"].notna()
                       & schedule["away_score"].notna()].copy()
    pbp_rollup = cache / "pbp_rollup.parquet"
    df = build_game_features(decided, pbp, None, pbp_rollup_path=pbp_rollup)
    df = df.sort_values("gameday").reset_index(drop=True)
    fl = make_folds(df, study.oof_first_season,
                    cadence_days=study.walk_forward.step_days,
                    date_col="gameday",
                    val_scope="oof_season", end_of_day=True)
    folds = [(f.train_idx, f.val_idx) for f in fl]
    return _fold_signature(folds, df, "game_id")


_BUILDERS = {"mlb": _mlb, "nba": _nba, "nfl": _nfl, "nhl": _nhl}


@pytest.mark.parametrize("sport", sorted(EXPECTED_FOLD_COUNTS))
def test_fold_fingerprints_match_pinned_incumbents(sport: str) -> None:
    sig = _BUILDERS[sport]()
    assert len(sig) == EXPECTED_FOLD_COUNTS[sport], (
        f"{sport}: fold count {len(sig)} != pinned "
        f"{EXPECTED_FOLD_COUNTS[sport]}")


def test_full_fingerprint_payload_matches_pinned_incumbent() -> None:
    """The full ordered fingerprint payload (all four sports) hashes to the
    pinned incumbent — the complete neutrality proof in one assertion."""
    payload = {sport: _BUILDERS[sport]()
               for sport in sorted(EXPECTED_FOLD_COUNTS)}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()
    assert digest == EXPECTED_FINGERPRINT_PAYLOAD_SHA256, (
        "full fold-fingerprint payload drifted from the pinned incumbent")


@pytest.mark.parametrize("sport", sorted(EXPECTED_FOLD_COUNTS))
def test_committed_raw_fixture_matches_generator(
        sport: str, tmp_path: Path) -> None:
    """Provenance: each committed bounded raw fixture is EXACTLY what the
    reviewed ``evidence_frame()`` generator in its support module produces.

    The two are deliberately redundant: the committed Parquet freezes the
    bytes the pinned fold/matrix evidence is computed from, while this test
    proves those bytes stay reproducible from repository-controlled code.
    Both sides are compared after an identical Parquet round-trip (so a
    Python-level ``None`` vs ``NaN`` representation never masks a real
    difference). The committed fixture itself is never rewritten here.
    """
    from tests import mlb_fixtures, nba_fixtures, nfl_fixtures, nhl_fixtures
    from tests.raw_store_fixtures import fixture_frame

    generator = {
        "mlb": mlb_fixtures.evidence_frame,
        "nba": nba_fixtures.evidence_frame,
        "nfl": nfl_fixtures.evidence_frame,
        "nhl": nhl_fixtures.evidence_frame,
    }[sport]
    regenerated = tmp_path / f"{sport}.parquet"
    generator().to_parquet(regenerated, index=False)
    pd.testing.assert_frame_equal(
        pd.read_parquet(regenerated), fixture_frame(sport))
