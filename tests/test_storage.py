"""Tests for core.storage."""

from pathlib import Path

import pandas as pd
import pytest

from core.features import (
    StorageError,
    duckdb_path,
    parquet_dir,
    read_duckdb_table,
    read_parquet_range,
    store_root,
    write_duckdb_table,
    write_parquet_partition,
)


def test_layout_paths(tmp_path):
    assert store_root(tmp_path, "mlb") == (
        tmp_path / "sports" / "mlb" / "store")
    assert duckdb_path(tmp_path, "mlb", "games") == (
        store_root(tmp_path, "mlb") / "duckdb" / "games.duckdb")
    assert parquet_dir(tmp_path, "mlb", "games") == (
        store_root(tmp_path, "mlb") / "parquet" / "games")


def test_invalid_dataset_name_rejected(tmp_path):
    with pytest.raises(StorageError):
        duckdb_path(tmp_path, "mlb", "../evil")
    with pytest.raises(StorageError):
        parquet_dir(tmp_path, "mlb", "a/b")


def _frame():
    return pd.DataFrame({"game_id": ["g1", "g2"], "home_win": [1.0, 0.0]})


def test_parquet_partition_roundtrip(tmp_path):
    write_parquet_partition(_frame(), tmp_path, "mlb", "games", "20260907")
    out = read_parquet_range(tmp_path, "mlb", "games")
    assert len(out) == 2
    # range filters
    assert read_parquet_range(tmp_path, "mlb", "games",
                              date_start="20260908").empty
    assert len(read_parquet_range(tmp_path, "mlb", "games",
                                  date_start="20260901",
                                  date_end="20260907")) == 2
    # missing dataset -> empty frame
    assert read_parquet_range(tmp_path, "mlb", "missing").empty


def test_parquet_partition_idempotent_no_overwrite(tmp_path):
    p1 = write_parquet_partition(_frame(), tmp_path, "mlb", "games",
                                 "20260907")
    df2 = pd.DataFrame({"game_id": ["g9"]})
    p2 = write_parquet_partition(df2, tmp_path, "mlb", "games", "20260907",
                                 overwrite=False)
    assert p1 == p2
    assert len(read_parquet_range(tmp_path, "mlb", "games")) == 2  # unchanged


def test_parquet_invalid_partition_date(tmp_path):
    with pytest.raises(StorageError, match="invalid partition date"):
        write_parquet_partition(_frame(), tmp_path, "mlb", "games",
                                "2026-09-07")


def test_parquet_none_frame_rejected(tmp_path):
    with pytest.raises(StorageError, match="None frame"):
        write_parquet_partition(None, tmp_path, "mlb", "games", "20260907")


def test_duckdb_roundtrip(tmp_path):
    write_duckdb_table(_frame(), tmp_path, "mlb", "games")
    out = read_duckdb_table(tmp_path, "mlb", "games")
    assert len(out) == 2
    # replace mode is idempotent
    write_duckdb_table(_frame(), tmp_path, "mlb", "games", mode="replace")
    assert len(read_duckdb_table(tmp_path, "mlb", "games")) == 2
    # append mode grows
    write_duckdb_table(_frame(), tmp_path, "mlb", "games", mode="append")
    assert len(read_duckdb_table(tmp_path, "mlb", "games")) == 4


def test_duckdb_missing_file_empty(tmp_path):
    assert read_duckdb_table(tmp_path, "mlb", "nope").empty


def test_duckdb_bad_mode(tmp_path):
    with pytest.raises(StorageError, match="unknown mode"):
        write_duckdb_table(_frame(), tmp_path, "mlb", "games", mode="upsert")
