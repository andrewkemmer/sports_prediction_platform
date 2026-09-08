"""Tests for core.validation."""

import math
from dataclasses import dataclass
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

    s = stable_json(D(x=1, p=Path("/tmp")))
    assert "/tmp" in s


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
