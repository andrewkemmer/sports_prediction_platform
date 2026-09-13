"""DuckDB/Parquet storage utilities.

The durable store (never subject to frontend-artifact retention) holds raw
historical payloads, feature/training frames, and OOF tables. Layout:

    <root>/sports/<sport>/store/
        duckdb/    one .duckdb file per dataset (queryable history)
        parquet/   one directory per dataset, dated part files

Writes are idempotent per partition key; reads are deterministic. Nothing
here is exposed to the Streamlit frontend except through loaders.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from core.config.platform import normalize_sport_key


class StorageError(RuntimeError):
    """Raised when a storage operation fails or violates the layout."""


def store_root(root: Path, sport: str) -> Path:
    """The durable store root for a sport: ``<root>/sports/<sport>/store``."""
    return Path(root) / "sports" / normalize_sport_key(sport) / "store"


def duckdb_path(root: Path, sport: str, dataset: str) -> Path:
    """The DuckDB file for one dataset (e.g. ``games``, ``features``)."""
    if not dataset.replace("_", "").isalnum():
        raise StorageError(f"invalid dataset name: {dataset!r}")
    return store_root(root, sport) / "duckdb" / f"{dataset}.duckdb"


def parquet_dir(root: Path, sport: str, dataset: str) -> Path:
    """The Parquet partition directory for one dataset."""
    if not dataset.replace("_", "").isalnum():
        raise StorageError(f"invalid dataset name: {dataset!r}")
    return store_root(root, sport) / "parquet" / dataset


def write_parquet_partition(
    df: pd.DataFrame,
    root: Path,
    sport: str,
    dataset: str,
    partition_date: str,
    *,
    overwrite: bool = True,
) -> Path:
    """Write one dated Parquet partition (``dataset=<date>.parquet``).

    Idempotent: without ``overwrite`` an existing partition is returned
    unchanged (never appended-to or duplicated).
    """
    if df is None:
        raise StorageError("cannot write a None frame")
    from core.study import is_valid_compact_date
    if not is_valid_compact_date(partition_date):
        raise StorageError(f"invalid partition date: {partition_date!r}")
    out_dir = parquet_dir(root, sport, dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{partition_date}.parquet"
    if out.exists() and not overwrite:
        return out
    df.to_parquet(out, index=False)
    return out


def read_parquet_range(
    root: Path,
    sport: str,
    dataset: str,
    date_start: str | None = None,
    date_end: str | None = None,
) -> pd.DataFrame:
    """Read Parquet partitions in a compact-date range (inclusive).

    Missing partitions are skipped (partial history is expected during
    backfills). Returns an empty frame when nothing exists.
    """
    d = parquet_dir(root, sport, dataset)
    if not d.is_dir():
        return pd.DataFrame()
    parts = sorted(d.glob("*.parquet"))
    frames = []
    for p in parts:
        stamp = p.stem
        if date_start is not None and stamp < date_start:
            continue
        if date_end is not None and stamp > date_end:
            continue
        frames.append(pd.read_parquet(p))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def write_duckdb_table(
    df: pd.DataFrame,
    root: Path,
    sport: str,
    dataset: str,
    table: str = "games",
    *,
    mode: str = "replace",
) -> Path:
    """Write a DataFrame into the dataset's DuckDB file.

    ``mode`` is ``replace`` (idempotent rebuild) or ``append``. The file and
    schema directory are created on demand.
    """
    if mode not in ("replace", "append"):
        raise StorageError(f"unknown mode: {mode!r}")
    path = duckdb_path(root, sport, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    try:
        con.register("df_view", df)
        if mode == "replace":
            con.execute(
                f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM df_view")
        else:
            con.execute(
                f"INSERT INTO {table} SELECT * FROM df_view")
    finally:
        con.close()
    return path


def read_duckdb_table(
    root: Path,
    sport: str,
    dataset: str,
    table: str = "games",
) -> pd.DataFrame:
    """Read a DuckDB table; empty frame when the file/table doesn't exist."""
    path = duckdb_path(root, sport, dataset)
    if not path.is_file():
        return pd.DataFrame()
    con = duckdb.connect(str(path), read_only=True)
    try:
        tables = con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'").fetchall()
        if (table,) not in tables:
            return pd.DataFrame()
        return con.execute(f"SELECT * FROM {table}").df()
    finally:
        con.close()
