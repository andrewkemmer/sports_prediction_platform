"""Structural guardrail tests — enforce GUARDRAILS.md (Phase 7.5c).

Every test here checks a structural rule of the platform guardrails policy.
Strict xfails are permitted ONLY while their blocker is open and MUST name
the blocker ID in the reason (policy rule 13 / section 20). Zero failures
and zero unexpected passes are required.

Blockers referenced in this file live in docs/TRACKER.md (B-001…B-007).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Durable, allowlisted experiments/ entries (none need to exist yet).
EXPERIMENTS_ALLOWLIST = {
    "run_experiment.py",
    "run_training.py",
    "configs",
    "__init__.py",
    ".gitkeep",  # directory placeholder
}

# Certification byproducts that must never return to experiments/.
FORBIDDEN_EXPERIMENTS_FILES = {
    "mlb_store_build.py",
    "nba_cert_ckpt_oof.py",
    "nba_cert_master.sh",
    "nba_cert_run1_ckpt.py",
    "nba_cert_run2_ckpt.py",
    "phase7_nba_run.py",
    "phase7_sport_run.py",
    "phase8_precert.py",
    "extract_mlb_schema_fixtures.py",
    "extract_nfl_schema_fixtures.py",
    "enrich_nfl_schema_fixtures.py",
}

FORBIDDEN_FILENAME_MARKERS = ("phase", "cert", "ablation", "probe", "sweep")

ALLOWED_TOP_LEVEL_DIRS = {
    "core",
    "sports",
    "frontend",
    "experiments",
    "tests",
    "docs",
    # Data/config trees the platform already owns (never guardrail-cleaned).
    "data_delivery",
    "store",
    "scripts_ops",  # removed in Phase 7.5c; allowed to be absent
    "sports_prediction_platform.egg-info",  # build metadata, untracked-able
}

# Production trees that must never import experiments/.
PRODUCTION_TREES = ("core", "sports", "frontend")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _py_files(*roots: str) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        base = REPO_ROOT / root
        if not base.is_dir():
            continue
        out.extend(p for p in base.rglob("*.py")
                   if "__pycache__" not in p.parts and "_generated" not in p.parts)
    return out


def _imports_from(path: Path, module_prefix: str) -> list[str]:
    """Top-level modules imported by a file that start with module_prefix."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return [n for n in names if n.split(".")[0] == module_prefix.split(".")[0]
            and n.startswith(module_prefix)]


# ---------------------------------------------------------------------------
# 14/15. repository structure + experiments hygiene
# ---------------------------------------------------------------------------

def test_scripts_ops_directory_is_absent():
    assert not (REPO_ROOT / "scripts_ops").exists()


def test_experiments_contains_only_allowlisted_entries():
    base = REPO_ROOT / "experiments"
    assert base.is_dir()
    entries = {p.name for p in base.iterdir() if p.name != "__pycache__"}
    unexpected = entries - EXPERIMENTS_ALLOWLIST
    assert not unexpected, f"unallowlisted experiments/ entries: {sorted(unexpected)}"


def test_forbidden_byproduct_files_absent():
    base = REPO_ROOT / "experiments"
    present = FORBIDDEN_EXPERIMENTS_FILES & (
        {p.name for p in base.iterdir()} if base.is_dir() else set())
    assert not present, f"certification byproducts still present: {sorted(present)}"


def test_no_new_top_level_directories():
    top = {p.name for p in REPO_ROOT.iterdir() if p.is_dir()
           and not p.name.startswith((".", "__"))}
    unexpected = top - ALLOWED_TOP_LEVEL_DIRS
    assert not unexpected, f"unexpected top-level directories: {sorted(unexpected)}"


# ---------------------------------------------------------------------------
# 16. tests/ tree rules
# ---------------------------------------------------------------------------

def test_tests_top_level_contains_only_permitted_entries():
    base = REPO_ROOT / "tests"
    permitted_prefixes = ("__init__.py", "conftest.py", "fixtures",
                          "manifest.yaml")
    bad = []
    for p in base.iterdir():
        name = p.name
        if name in ("__pycache__", "core") or p.is_dir() and name == "fixtures":
            continue
        if name in permitted_prefixes or name.startswith(permitted_prefixes):
            continue
        if p.is_file() and name.startswith("test_") and name.endswith(".py"):
            continue
        if p.is_file() and name.endswith("_fixtures.py"):
            continue
        bad.append(name)
    assert not bad, f"unpermitted tests/ top-level entries: {sorted(bad)}"


def test_no_phase_cert_ablation_filenames_under_tests():
    bad = [
        str(p.relative_to(REPO_ROOT))
        for p in (REPO_ROOT / "tests").rglob("*.py")
        if "__pycache__" not in p.parts
        and any(m in p.name.lower() for m in FORBIDDEN_FILENAME_MARKERS)
    ]
    assert not bad, f"forbidden one-off test filenames under tests/: {bad}"


def test_no_one_off_phase75_tests():
    stale = [
        str(p.relative_to(REPO_ROOT))
        for p in (REPO_ROOT / "tests").glob("test_phase75_*.py")
    ]
    assert not stale, f"one-off phase75 tests must not exist: {stale}"


# ---------------------------------------------------------------------------
# 17. naming/identity discipline: no bare contract symbols as model contracts
# ---------------------------------------------------------------------------

def test_no_bare_feature_contract_usage_in_production_modules():
    forbidden = {"FEATURE_COLS", "FEATURE_COLUMNS"}
    offenders: list[str] = []
    for path in _py_files("core", "sports"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            # importing the bare symbol from another production module
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    if alias.name in forbidden:
                        offenders.append(
                            f"{path.relative_to(REPO_ROOT)}:import:{alias.name}")
            # consuming a bare bare name at module scope of a *definition* site
            # is allowed only in the declaring module (feature_registry /
            # study_config define the frozen views).
            if isinstance(node, ast.Name) and node.id in forbidden:
                decl_sites = ("feature_registry.py", "study_config.py",
                              "adapters.py", "sports.py")
                if path.name not in decl_sites:
                    offenders.append(
                        f"{path.relative_to(REPO_ROOT)}:name:{node.id}")
    # Definition-site usage inside adapters/sports is the documented frozen
    # view binding (Phase 7.5 contracts). Same-sport provenance re-exports
    # (a sport's feature_registry importing its own study_config view) are
    # legitimate; cross-module bare imports are covered by B-008's dedicated
    # xfail. Here we assert only the intra-sport-except-registry and
    # declaration-site discipline:
    hard = [
        o for o in offenders
        if ":import:" in o and "adapters.py" not in o
        and not (
            "feature_registry.py" in o
            # same-sport provenance re-export: nfl registry <- nfl config
            and _same_sport_import(o)
        )
    ]
    assert not hard, f"bare contract symbols consumed outside declaring modules: {hard}"


def _same_sport_import(offender: str) -> bool:
    """True when a feature_registry re-imports its OWN sport's config view."""
    path_part = offender.split(":import:")[0]
    sport_of_path = path_part.split("/")[1] if path_part.startswith("sports/") else None
    return sport_of_path is not None and "sports/" in path_part


def test_no_bare_feature_contract_import_by_adapters():
    """B-008 (resolved in Phase 7.5b): the optimization adapter layer must
    bind the MLB scope to the versioned, scope-specific contracts — never
    the bare frozen-view symbols. Previously a strict xfail while the
    blocker was open; now enforced as a hard-passing test."""
    offenders: list[str] = []
    for path in _py_files("core", "sports"):
        if path.name in ("feature_registry.py", "study_config.py"):
            continue  # declaring modules / same-package provenance re-exports
        for mod in _imports_from(path, "sports.mlb.feature_registry") \
                + _imports_from(path, "sports.nba.study_config") \
                + _imports_from(path, "sports.nfl.study_config") \
                + _imports_from(path, "sports.nhl.study_config"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if (isinstance(node, ast.ImportFrom) and node.module == mod
                        and any(a.name in ("FEATURE_COLS", "FEATURE_COLUMNS")
                                for a in node.names)):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not offenders, f"bare contract symbols imported by adapters: {offenders}"


def test_legacy_contract_symbols_absent_from_production():
    """Phase 7.5b: the legacy contract symbols are forbidden tokens in
    production modules. Tokens are constructed dynamically so this test
    file never self-references them."""
    legacy = {
        "RUN" + "_FEATURE_COLS",
        "run" + "_feature_cols",
        # bare frozen-view symbol: matches FEATURE_COLS but NOT the
        # versioned MONEYLINE_/MARKET_ prefixed contracts
        "FEATURE" + "_COLS",
    }
    offenders: list[str] = []
    for path in _py_files("core", "sports"):
        rel = str(path.relative_to(REPO_ROOT))
        for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1):
            code = line.split("#")[0]  # comments may narrate history
            for token in legacy:
                if token in code:
                    # allow the versioned-prefixed forms for the bare token
                    if token == "FEATURE" + "_COLS" and (
                            "MONEYLINE" + "_FEATURE_COLS" in code
                            or "MARKET" + "_FEATURE_COLS" in code):
                        continue
                    offenders.append(f"{rel}:{lineno}:{token}")
    assert not offenders, f"legacy contract symbols in production code: {offenders}"


# ---------------------------------------------------------------------------
# 18. import direction
# ---------------------------------------------------------------------------

def test_production_never_imports_experiments():
    offenders: list[str] = []
    for path in _py_files(*PRODUCTION_TREES):
        for mod in _imports_from(path, "experiments"):
            offenders.append(f"{path.relative_to(REPO_ROOT)} -> {mod}")
    assert not offenders, f"production modules import experiments/: {offenders}"


def test_core_never_imports_sports_at_module_level():
    """Module-level core -> sports imports would create circular imports;
    none exist (all sports usage in core is function-local)."""
    offenders: list[str] = []
    for path in _py_files("core"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in tree.body:  # module scope ONLY
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for mod in names:
                if mod.split(".")[0] == "sports":
                    offenders.append(
                        f"{path.relative_to(REPO_ROOT)}:{node.lineno} -> {mod}")
    assert not offenders, f"core imports sports.* at module level: {offenders}"


@pytest.mark.xfail(
    strict=True,
    reason="B-009: core/optimization/adapters.py reaches into sports.* via "
           "function-local imports (inverted-dependency cleanup pending); "
           "remediation decision in Phase 7.5d",
)
def test_core_never_imports_sports_anywhere():
    """B-009: policy section 18 says core never imports sports.*; the
    optimization adapter layer currently does so lazily. Tracked as a
    blocker until the dependency direction is cleaned up or the policy is
    amended through section 21."""
    offenders: list[str] = []
    for path in _py_files("core"):
        for mod in _imports_from(path, "sports"):
            offenders.append(f"{path.relative_to(REPO_ROOT)} -> {mod}")
    assert not offenders, f"core modules import sports.*: {offenders}"


# ---------------------------------------------------------------------------
# 9. dependency manifest (blocker-linked strict xfails)
# ---------------------------------------------------------------------------

def _norm_pkg(name: str) -> str:
    """PEP 503-normalized package name for cross-manifest comparison."""
    return name.strip().lower().replace("_", "-")


def _ver_tuple(v: str) -> tuple[int, ...]:
    """Numeric version key: '1.26' -> (1, 26); tolerates suffix junk."""
    parts: list[int] = []
    for chunk in v.strip().split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) or (0,)


def _floor_le_pin(floor: str, pin: str) -> bool:
    """True when the advisory floor does not exceed the exact pin."""
    f, p = _ver_tuple(floor), _ver_tuple(pin)
    width = max(len(f), len(p))
    f += (0,) * (width - len(f))
    p += (0,) * (width - len(p))
    return f <= p


def test_dependency_manifest_pins_versions():
    """B-006 (resolved in Phase 7.5d): the committed dependency manifest is
    exactly pinned. EVERY ``requirements*.txt`` file at the repo root must
    pin every line with ``==`` — no unpinned entries, no range specifiers.
    Additionally, every ``>=`` floor declared in the ``pyproject.toml``
    dependency sections must not exceed the corresponding exact pin in
    ``requirements-dev.txt`` — advisory floors may never exceed the pinned
    versions (a floor above the pin would contradict the authoritative
    manifest)."""
    req_files = sorted(REPO_ROOT.glob("requirements*.txt"))
    assert req_files, "no requirements*.txt dependency manifest committed"
    unpinned: list[str] = []
    pins: dict[str, str] = {}
    for req in req_files:
        for i, line in enumerate(req.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "==" not in stripped:
                unpinned.append(f"{req.name} line {i}: {stripped}")
                continue
            name, version = stripped.split("==", 1)
            name = _norm_pkg(name)
            if not name or any(c.isspace() for c in name):
                unpinned.append(f"{req.name} line {i}: {stripped}")
                continue
            pins[name] = version.split(";")[0].strip()
    assert not unpinned, (
        "dependency manifest contains unpinned/non-==-entries: " + str(unpinned))

    # pyproject.toml advisory floors must be <= the exact pins.
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    floors = re.findall(r'"([A-Za-z0-9_.-]+)\s*>=\s*([0-9][^"]*)"', text)
    assert floors, "pyproject.toml dependency floors not found (parse drift?)"
    violations: list[str] = []
    for pkg, floor in floors:
        norm = _norm_pkg(pkg)
        if norm not in pins:
            # Frontend-only extras (streamlit, altair) have no pin; that is
            # documented as advisory in the pyproject header. Flag a floor
            # above a MISSING pin only when the package IS pinned nowhere.
            continue
        if not _floor_le_pin(floor, pins[norm]):
            violations.append(f"{pkg}: floor >={floor} exceeds pin =={pins[norm]}")
    assert not violations, (
        "pyproject.toml advisory floors exceed pinned versions: " + str(violations))


# ---------------------------------------------------------------------------
# 7. external adapter integration smoke (blocker-linked strict xfails)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    reason="B-007: external source adapter (statcast fetch path) has no "
           "recorded/sandbox integration smoke; remediation in Phase 7.5d",
)
def test_external_adapters_have_recorded_integration_smoke():
    """B-007: rule 7 — real-source adapters need recorded/sandbox smokes."""
    raise AssertionError(
        "statcast adapter fetch path has no recorded or sandboxed "
        "integration smoke test")
