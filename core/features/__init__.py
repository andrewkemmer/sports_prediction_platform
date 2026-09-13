"""Features package (Phase r1 §2 restructure).

* ``candidate_frame`` — durable DuckDB/Parquet store paths and bounded
                       partition IO (former ``core/storage.py``). The
                       wide point-in-time candidate frame is assembled
                       per sport on top of these primitives.
* ``availability``    — §7.1 per-field availability metadata
                       (B-001-RESIDUAL, Phase 7.6): availability
                       classes with declared conservative leads, row
                       stamping, and the per-field enforcement gate.
"""

from core.features.availability import (  # noqa: F401
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
from core.features.candidate_frame import (  # noqa: F401
    StorageError,
    duckdb_path,
    parquet_dir,
    read_duckdb_table,
    read_parquet_range,
    store_root,
    write_duckdb_table,
    write_parquet_partition,
)

__all__ = [
    # availability (B-001-RESIDUAL)
    "AVAILABILITY_CLASSES", "AvailabilityGateResult", "FIELD_CLASSES",
    "FIELD_LEAD_OVERRIDES", "available_at_for", "class_for",
    "lead_hours_for", "per_field_gate", "require_classes_for",
    "stamp_frame", "stamped_feature_records",
    # candidate_frame storage primitives
    "StorageError", "duckdb_path", "parquet_dir", "read_duckdb_table",
    "read_parquet_range", "store_root", "write_duckdb_table",
    "write_parquet_partition",
]
