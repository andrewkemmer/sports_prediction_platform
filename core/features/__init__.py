"""Features package (Phase r1 §2 restructure).

* ``candidate_frame`` — durable DuckDB/Parquet store paths and bounded
                       partition IO (former ``core/storage.py``). The
                       wide point-in-time candidate frame is assembled
                       per sport on top of these primitives.
"""

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
    "StorageError", "duckdb_path", "parquet_dir", "read_duckdb_table",
    "read_parquet_range", "store_root", "write_duckdb_table",
    "write_parquet_partition",
]
