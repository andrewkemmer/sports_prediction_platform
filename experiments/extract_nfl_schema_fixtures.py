"""One-off extractor: pull NFL artifact headers/keys from the read-only
reference repo into the immutable schema fixture.

Usage (manual, when the reference repo is mounted):
    python3 experiments/extract_nfl_schema_fixtures.py <ref_nfl_data_delivery> \
        [output.json]

Re-running is idempotent against unchanged artifact schemas; the fixture
is otherwise treated as immutable and reviewed by diff.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def csv_columns(path: Path) -> list[str]:
    return path.open(encoding="utf-8").readline().rstrip("\n").split(",")


def json_keys(path: Path) -> list[str]:
    return list(json.loads(path.read_text(encoding="utf-8")).keys())


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        Path("tests/fixtures/nfl_artifact_schemas.json"))

    artifacts: dict = {}
    # Dated families (real 20260907 artifacts in the reference sink).
    dated = {
        "moneyline_json": "nfl_moneyline_v1_20260907.json",
        "calibration_json": "nfl_calibration_20260907.json",
        "predictions_history": "nfl_predictions_history_20260907.csv",
        "power_rankings": "nfl_power_rankings_20260907.csv",
        "markets": "nfl_run_engine_markets_20260907.csv",
        "markets_meta": "nfl_run_engine_markets_20260907.meta.json",
        "markets_monitor": "nfl_run_engine_monitor_20260907.json",
        "qb_matchup": "nfl_qb_matchup_20260907.json",
        "feature_json": "nfl_feature_v1_20260907.json",
        "model_monitor": "nfl_model_monitor_20260907.json",
        "shap_game": "nfl_shap_game_2026_01_ARI_LAC.csv",
    }
    for fam, name in dated.items():
        p = src / name
        if not p.is_file():
            print(f"  MISSING {name}")
            continue
        entry: dict = {"file": name}
        if p.suffix == ".csv":
            entry["columns"] = csv_columns(p)
        else:
            entry["keys"] = json_keys(p)
        artifacts[fam] = entry

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": ("Read-only reference repository sports_prediction_model "
                    "(andrewkemmer-sports_prediction_model-5dce619), "
                    "nfl-backend/data_delivery/ production artifacts dated "
                    "2026-09-07. Extracted verbatim from real artifact "
                    "headers/keys; NOT remembered schemas."),
        "extracted_from_commit": "5dce619",
        "artifacts": artifacts,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} with {len(artifacts)} artifact families")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
