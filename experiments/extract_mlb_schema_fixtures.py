"""One-off extractor: pull artifact headers/keys from the read-only
reference repo into the immutable schema fixture.

Usage (manual, when the reference repo is mounted):
    python3 experiments/extract_mlb_schema_fixtures.py <ref_data_delivery> \
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
    print(f"CSV headers / JSON keys available under {src}:")
    for p in sorted(src.iterdir()):
        if p.name.endswith(".csv"):
            print(f"  {p.name}: {len(csv_columns(p))} cols")
        elif p.name.endswith(".json") and "meta" not in p.name:
            try:
                print(f"  {p.name}: {len(json_keys(p))} keys")
            except Exception as exc:  # noqa: BLE001
                print(f"  {p.name}: ERROR {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
