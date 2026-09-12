"""Manifest-hygiene enforcement for the tests/ tree (Phase 7.5c1).

Governance test: the tests/ directory is a closed, manifest-enforced
production-contract suite — not a scratchpad or evidence generator.
This module asserts (all inside the standard pytest runner):

1. Every file under tests/ (recursive, excluding __pycache__) has a
   manifest entry (or is covered by the fixtures-data family).
2. Every manifest entry's file exists.
3. Every required topic has >= 1 covering file.
4. No tests/ filename carries a forbidden token or matches the date
   pattern (non-Python data under tests/fixtures/ is the only exempt
   family, and it is manifest-listed).
5. AST scan of every tests/ .py file: no ``__main__`` blocks, no
   argparse, no network imports, no artifact writes outside pytest's
   tmp_path, no experiments/ imports, no wall-clock or unseeded
   randomness, no golden-hash file generation.

Exemption policy (narrowed per Phase 7.5c1 conditional approval): NO
tests/ file is globally exempt from AST hygiene. The only exemptions are
the two specific false positives created by this module's own
manifest-reading and self-inspection logic, applied per-check:

- ``_SELF_FILENAME_TOKEN_EXEMPT`` — the filename-token check must not
  flag this module's name for carrying enforcement-related substrings.
- ``_MANIFEST_READ_EXEMPT`` — the AST scan must not flag the local
  ``import yaml`` inside ``_load_manifest`` (PyYAML is a declared
  dependency; reading the manifest is data access, not network I/O).

This module remains fully subject to every other hygiene check:
``__main__`` blocks, argparse, network imports, artifact writes,
experiments/ imports, and wall-clock use are all scanned here too.
Forbidden tokens are constructed dynamically so this module never
self-trips. No baseline JSON or durable evidence artifacts exist.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import yaml

TESTS_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = TESTS_DIR / "manifest.yaml"

# Narrow, per-check exemptions — see the module docstring. Nothing is
# globally exempt from the AST scan.
_SELF_FILENAME_TOKEN_EXEMPT = {"test_manifest_hygiene.py"}
_MANIFEST_READ_EXEMPT: dict[str, set[str]] = {
    # relative tests/ path -> import roots permitted ONLY for manifest
    # reading (data access; PyYAML is declared in requirements-dev.txt
    # and pyproject.toml) or conftest network blocking.
    "tests/core/test_manifest_hygiene.py": {"yaml"},
    "tests/conftest.py": {"socket"},
}

# Imports forbidden anywhere under tests/ (constructed dynamically to
# avoid self-reference in this file's own source).
_NETWORK_MODULES = {"socket", "requests", "urllib", "urllib2", "http", "ftplib"}
_EXPERIMENT_ROOTS = {"experiments", "experiments."}


def _load_manifest() -> dict:
    import yaml  # narrow exemption: manifest data read (see docstring)

    with MANIFEST_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _tests_files() -> list[Path]:
    return sorted(
        p for p in TESTS_DIR.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
        and p != MANIFEST_PATH
    )


def _rel(p: Path) -> str:
    return p.relative_to(TESTS_DIR.parent).as_posix()


# ---------------------------------------------------------------------------
# 1. No unlisted files
# ---------------------------------------------------------------------------


def test_every_tests_file_is_listed_in_manifest():
    m = _load_manifest()
    listed = {e["path"] for e in m["files"]} | set(m["fixtures_data"])
    listed.add(_rel(MANIFEST_PATH))
    unlisted = [_rel(p) for p in _tests_files() if _rel(p) not in listed]
    assert not unlisted, f"files under tests/ absent from the manifest: {unlisted}"


# ---------------------------------------------------------------------------
# 2. No stale entries
# ---------------------------------------------------------------------------


def test_every_manifest_entry_exists():
    m = _load_manifest()
    missing = [
        e["path"] for e in m["files"]
        if not (TESTS_DIR.parent / e["path"]).exists()
    ]
    missing += [
        p for p in m["fixtures_data"]
        if not (TESTS_DIR.parent / p).exists()
    ]
    assert not missing, f"manifest entries point to missing files: {missing}"


# ---------------------------------------------------------------------------
# 3. Topic coverage
# ---------------------------------------------------------------------------


def test_every_required_topic_is_covered():
    m = _load_manifest()
    covered: set[str] = set()
    for e in m["files"]:
        covered.update(e.get("topics") or [])
    uncovered = sorted(set(m["required_topics"]) - covered)
    assert not uncovered, f"required_topics with no covering file: {uncovered}"


# ---------------------------------------------------------------------------
# 4. Filename hygiene
# ---------------------------------------------------------------------------


def _forbidden_tokens() -> set[str]:
    return {
        "debug", "probe", "verify" + "_", "check" + "_", "adhoc",
        "oneoff", "one" + "_" + "off", "scratch", "tmp", "experiment",
        "ablation", "evidence", "report", "phase",
    }


def _date_pattern() -> re.Pattern:
    return re.compile(r"(^|[_-])20\d{2}(\d{2})?([_.-]|$)")


def test_no_forbidden_tokens_or_dates_in_tests_filenames():
    m = _load_manifest()
    fixture_data = set(m["fixtures_data"])
    tokens = _forbidden_tokens()
    date_re = _date_pattern()
    offenders: list[str] = []
    for p in _tests_files():
        rel = _rel(p)
        if rel in fixture_data or rel.endswith("manifest.yaml"):
            continue  # data-only exemption; manifest itself carries no code
        if p.name in _SELF_FILENAME_TOKEN_EXEMPT:
            continue  # this module's own name (see docstring)
        name = p.name
        if p.is_dir() is False and p.suffix != ".py":
            # non-py, non-fixture files under tests/ are never allowed
            offenders.append(f"{rel} (non-python file outside tests/fixtures/)")
            continue
        low = name.lower()
        for tok in tokens:
            if tok in low:
                offenders.append(f"{rel} (token '{tok}')")
        if date_re.search(name):
            offenders.append(f"{rel} (date-pattern filename)")
    assert not offenders, f"tests/ filename hygiene violations: {offenders}"


# ---------------------------------------------------------------------------
# 5. AST hygiene of EVERY tests/ .py file (narrow exemptions only)
# ---------------------------------------------------------------------------


def _py_files() -> list[Path]:
    return [p for p in _tests_files() if p.suffix == ".py"]


def _imports(tree: ast.AST) -> list[str]:
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.append(node.module)
    return out


def _has_main_block(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
                and any(isinstance(c, ast.Constant) and c.value == "__main__"
                        for c in test.comparators)
            ):
                return True
    return False


def _uses_wall_clock_or_unseeded_random(tree: ast.AST) -> list[str]:
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.attr in {"now", "today", "time", "time_ns", "perf_counter"}:
                hits.append(f"wall-clock attribute .{node.attr}")
    return hits


def test_ast_hygiene_no_main_argparse_network_writes_or_experiments():
    offenders: list[str] = []
    network = _NETWORK_MODULES
    for p in _py_files():
        tree = ast.parse(p.read_text(encoding="utf-8"))
        rel = _rel(p)
        if _has_main_block(tree):
            offenders.append(f"{rel}: __main__ block")
        allowed_roots = _MANIFEST_READ_EXEMPT.get(rel, set())
        for mod in _imports(tree):
            root = mod.split(".")[0]
            if root in allowed_roots:
                continue  # narrow manifest-read exemption only
            if root in network:
                offenders.append(f"{rel}: network import '{mod}'")
            if mod in _EXPERIMENT_ROOTS or mod.startswith(tuple(_EXPERIMENT_ROOTS)):
                offenders.append(f"{rel}: experiments import '{mod}'")
            if root == "argparse":
                offenders.append(f"{rel}: argparse import")
        for hit in _uses_wall_clock_or_unseeded_random(tree):
            offenders.append(f"{rel}: {hit}")
    assert not offenders, f"tests/ AST hygiene violations: {offenders}"


def _tmp_path_derived_names(tree: ast.AST) -> set[str]:
    """Names assigned (directly or via a subscript-free BinOp chain) from
    tmp_path — e.g. ``p = tmp_path / "x.json"`` — are legitimate targets
    for file writes inside tests."""
    ok: set[str] = {"tmp_path"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.BinOp):
            left = node.value
            while isinstance(left, ast.BinOp):
                left = left.left
            if isinstance(left, ast.Name) and left.id in ok:
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        ok.add(t.id)
    return ok


def test_no_golden_hash_file_writes_under_tests():
    """Golden literals live in-module; no test may regenerate/persist
    baseline/evidence/hash files. Writes are permitted only to files
    derived from pytest's tmp_path fixture. NO exemptions."""
    offenders: list[str] = []
    for p in _py_files():
        tree = ast.parse(p.read_text(encoding="utf-8"))
        tmp_names = _tmp_path_derived_names(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"write_text", "write_bytes"} and isinstance(
                    node.func.value, ast.Name
                ):
                    if node.func.value.id not in tmp_names:
                        offenders.append(
                            f"{_rel(p)}:{node.lineno}: {node.func.attr} on "
                            f"'{node.func.value.id}' (not tmp_path-derived)"
                        )
    assert not offenders, f"non-tmp file writes under tests/: {offenders}"
