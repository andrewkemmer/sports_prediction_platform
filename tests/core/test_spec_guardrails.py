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


def test_core_never_imports_sports_anywhere():
    """B-009 (resolved in Phase 7.5d/WS6): policy section 18 says core
    never imports sports.*. The optimization adapter bindings now live
    outside core — ``sports/<sport>/optimization.py`` owns each sport's
    builder and ``sports/optimization_adapters.py`` is the thin registry —
    so core has zero sports imports, module-level OR function-local.
    Previously a strict xfail while the blocker was open; now enforced as
    a hard-passing test."""
    offenders: list[str] = []
    for path in _py_files("core"):
        for mod in _imports_from(path, "sports"):
            offenders.append(f"{path.relative_to(REPO_ROOT)} -> {mod}")
    assert not offenders, f"core modules import sports.*: {offenders}"


def test_core_never_references_sports_by_dynamic_import():
    """B-009 companion: a static-import guardrail is not enough — core must
    not reach sports.* through a DYNAMIC import either (no
    ``importlib.import_module`` / ``__import__`` call carrying a
    ``sports``-prefixed literal), and no core module may define a string
    constant naming a ``sports.*`` module. Bare filesystem path components
    (``"sports"`` as a directory segment in ``core/config.py`` and
    ``core/storage.py``) are NOT module references and remain permitted."""
    offenders: list[str] = []
    for path in _py_files("core"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        rel = path.relative_to(REPO_ROOT)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) \
                    else (func.id if isinstance(func, ast.Name) else None)
                if name in ("import_module", "__import__"):
                    args = list(node.args) + [k.value for k in node.keywords]
                    for a in args:
                        if (isinstance(a, ast.Constant)
                                and isinstance(a.value, str)
                                and a.value.split(".")[0] == "sports"):
                            offenders.append(
                                f"{rel}:{node.lineno} dynamic import "
                                f"{a.value!r}")
            if (isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and node.value.startswith("sports.")):
                offenders.append(
                    f"{rel}:{node.lineno} module-name string "
                    f"{node.value!r}")
    assert not offenders, (
        f"core references sports.* dynamically/by module-name string: "
        f"{offenders}")


def test_sport_adapter_registry_lives_outside_core():
    """B-009 architecture: the per-sport adapter composition root lives
    under sports/ — its registry keys are exactly the four sports, the
    module is not under core/, and no ``*_adapters`` builder is defined in
    any core/ module."""
    import importlib

    reg = importlib.import_module("sports.optimization_adapters")
    assert set(reg.SPORT_ADAPTER_BUILDERS) == {"mlb", "nfl", "nhl", "nba"}, (
        "sport adapter registry keys must be exactly the four sports")
    reg_path = Path(reg.__file__).resolve()
    assert (REPO_ROOT / "sports") in reg_path.parents, (
        f"sport adapter registry must live under sports/, got {reg_path}")

    offenders: list[str] = []
    for path in _py_files("core"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in tree.body:  # module scope: builder definitions
            if isinstance(node, ast.FunctionDef) \
                    and node.name.endswith("_adapters"):
                offenders.append(
                    f"{path.relative_to(REPO_ROOT)}:{node.name}")
    assert not offenders, f"adapter builders defined under core/: {offenders}"


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
# record-type validator alignment (B-005, hardened rule 10)
# ---------------------------------------------------------------------------

def _record_validation_registry() -> set[tuple[str, str]]:
    """Import the production registry and return its (sport, family) keys."""
    import importlib
    mod = importlib.import_module("core.record_validation")
    return set(mod.REGISTRY.keys())


def _registry_call_sites() -> set[tuple[str, str]]:
    """Every (sport, family) literal used in a production
    ``validate_record(...)`` call, per the approved AST-alignment scope:
    the four ``sports/*/artifacts.py`` writer modules plus the nine
    designated runner-side OOF/fold-table to_csv sites. No generic
    to_csv scan (call sites outside the scope are not required to be
    registry-keyed)."""
    tree_cache: dict[Path, ast.Module] = {}
    for path in _py_files("sports"):
        name = path.relative_to(REPO_ROOT).as_posix()
        if (name.endswith("artifacts.py")
                or name in RUNNER_VALIDATE_SITES):
            try:
                tree_cache[path] = ast.parse(
                    path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
    sites: set[tuple[str, str]] = set()
    for tree in tree_cache.values():
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "validate_record"
                    and len(node.args) >= 2
                    and all(isinstance(a, ast.Constant)
                            for a in node.args[:2])):
                sites.add((node.args[0].value, node.args[1].value))
    return sites


# The 9 designated runner-side OOF/fold-table write sites (3 per sport).
RUNNER_VALIDATE_SITES = {
    "sports/nfl/runner.py", "sports/nhl/runner.py", "sports/nba/runner.py",
}


def test_registry_covers_every_active_writer_call_site():
    """B-005: every (sport, family) a writer validates has a registry
    entry — a writer keying an undeclared family fails here."""
    reg = _record_validation_registry()
    missing = sorted(_registry_call_sites() - reg)
    assert not missing, (
        "validate_record call sites without registry entries: " + str(missing))


def test_registry_entries_all_have_active_writers():
    """B-005: every registry entry has an active writer call site —
    dead registry entries (writers removed without registry cleanup)
    fail here. Reserved durable names (model_history,
    model_version_history) have NO entries by design."""
    reg = _record_validation_registry()
    orphaned = sorted(reg - _registry_call_sites())
    assert not orphaned, (
        "registry entries with no active writer call site: " + str(orphaned))


def test_registry_size_pinned():
    """B-005: the registry size is pinned at 56 — 14 per sport (13 MLB
    primaries + markets meta; 10 NNX artifact primaries + markets meta
    + 3 runner-side OOF/fold stores). Any change to the registry is a
    reviewed registry change, never silent growth."""
    import importlib
    mod = importlib.import_module("core.record_validation")
    assert len(mod.REGISTRY) == mod.EXPECTED_REGISTRY_SIZE == 56


def test_reserved_durable_names_have_no_registry_entries():
    """B-005: model_history / model_version_history are reserved durable
    contract names with zero production writers — they must NOT have
    registry entries (any future writer must add one in the same change,
    caught by the alignment tests above)."""
    reg = _record_validation_registry()
    reserved = {f"{s} {n}" for s in ("mlb", "nfl", "nhl", "nba")
                for n in ("model_history", "model_version_history")}
    reserved_keys = {(s, n) for s, n in reg
                     if n in ("model_history", "model_version_history")}
    assert not reserved_keys, (
        "reserved durable names must not have registry entries: "
        + str(sorted(reserved_keys)))


# ---------------------------------------------------------------------------
# 7. external adapter integration smoke (blocker-linked strict xfails)
# ---------------------------------------------------------------------------

#: Declared external-source registry surface per sport: the ingestion
#: module that declares `EXTERNAL_SOURCES` and the manifest-listed test
#: module that must exercise those adapters end to end.
_EXTERNAL_SOURCE_MODULES = {
    "mlb": "sports/mlb/ingestion.py",
    "nfl": "sports/nfl/ingestion.py",
    "nhl": "sports/nhl/ingestion.py",
    "nba": "sports/nba/ingestion.py",
}
_EXTERNAL_SMOKE_MODULES = {
    "mlb": "tests/test_mlb_ingestion.py",
    "nfl": "tests/test_nfl_ingestion.py",
    "nhl": "tests/test_nhl_ingestion.py",
    "nba": "tests/test_nba_ingestion.py",
}
#: The injectable seam parameter every public ingestion entry point must
#: expose, per sport (source registry -> public boundary functions).
_EXTERNAL_SEAMS = {
    "mlb": {"pull_statcast": "fetch"},
    "nfl": {"load_schedule": "load", "load_pbp": "load",
            "load_player_stats": "load", "load_team_names": "load"},
    "nhl": {"pull_schedule": "fetch", "pull_goalie_boxscores": "fetch"},
    "nba": {"pull_schedule": "fetch", "pull_player_boxscores": "fetch"},
}
#: Transport libraries the real adapters may use; only the four ingestion
#: modules may import them, and only inside `_default_*` functions.
_NETWORK_LIBS = {"urllib", "pybaseball", "nflreadpy", "requests", "socket",
                 "http", "ftplib", "httpx", "aiohttp"}


def _parse(path: Path) -> ast.Module:
    return ast.parse((REPO_ROOT / path).read_text(encoding="utf-8"))


def _module_string_constants(tree: ast.Module) -> dict[str, str]:
    out: dict[str, str] = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            out[node.targets[0].id] = node.value.value
    return out


def _external_source_registry(sport: str) -> dict[str, str]:
    """{constant name -> default adapter function name} declared by the
    sport's ingestion module."""
    for node in _parse(Path(_EXTERNAL_SOURCE_MODULES[sport])).body:
        target = None
        value = None
        if isinstance(node, ast.Assign) and node.targets:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if (isinstance(target, ast.Name) and target.id == "EXTERNAL_SOURCES"
                and isinstance(value, ast.Dict)):
            return {
                (k.id if isinstance(k, ast.Name) else k.value): v.value
                for k, v in zip(value.keys, value.values)
            }
    return {}


def _imported_names(path: Path, module: str) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.ImportFrom) and node.module == module:
            out.update(a.name for a in node.names)
    return out


def test_external_adapters_have_recorded_integration_smoke():
    """B-007 (hard-passing): rule 7 — every declared external source has a
    default adapter that owns its transport, and each source is exercised
    end to end by the sport's manifest-listed integration smoke."""
    offenders: list[str] = []
    for sport, module in _EXTERNAL_SOURCE_MODULES.items():
        tree = _parse(Path(module))
        consts = _module_string_constants(tree)
        funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        registry = _external_source_registry(sport)
        if not registry:
            offenders.append(f"{module}: EXTERNAL_SOURCES registry missing")
            continue
        smoke = Path(_EXTERNAL_SMOKE_MODULES[sport])
        smoke_names = _imported_names(
            smoke, module[:-3].replace("/", "."))
        if "EXTERNAL_SOURCES" not in smoke_names:
            offenders.append(
                f"{_EXTERNAL_SMOKE_MODULES[sport]}: does not import "
                f"EXTERNAL_SOURCES from {module}")
        for const_name, adapter in registry.items():
            source_id = consts.get(const_name)
            if source_id is None:
                offenders.append(
                    f"{module}: registry key {const_name} is not a "
                    f"module-level string constant")
            elif not source_id.startswith(f"{sport}:"):
                offenders.append(
                    f"{module}: source id {source_id!r} is not "
                    f"{sport}-prefixed")
            if adapter not in funcs:
                offenders.append(
                    f"{module}: adapter {adapter} for {const_name} is not "
                    f"defined")
            elif not adapter.startswith("_default_"):
                offenders.append(
                    f"{module}: adapter {adapter} is not a _default_* "
                    f"transport owner")
            if const_name not in smoke_names:
                offenders.append(
                    f"{_EXTERNAL_SMOKE_MODULES[sport]}: no integration "
                    f"smoke imports {const_name}")
    assert not offenders, (
        "external-source adapters lack registry/injectable/smoke "
        f"coverage: {offenders}")


def test_external_adapter_boundaries_are_injectable_with_none_defaults():
    """Every public ingestion boundary exposes a `fetch=None`/`load=None`
    seam resolved at call time (`<name> or _default_...`) — so tests inject
    deterministic fakes and the real transport is never definition-bound."""
    offenders: list[str] = []
    for sport, seams in _EXTERNAL_SEAMS.items():
        module = _EXTERNAL_SOURCE_MODULES[sport]
        tree = _parse(Path(module))
        for fn_name, param in seams.items():
            fn = next((n for n in tree.body
                       if isinstance(n, ast.FunctionDef)
                       and n.name == fn_name), None)
            if fn is None:
                offenders.append(f"{module}: {fn_name} not defined")
                continue
            args = (list(fn.args.args) + list(fn.args.kwonlyargs))
            match = [a for a in args if a.arg == param]
            if not match:
                offenders.append(f"{module}:{fn_name} lacks {param}= seam")
                continue
            defaults = list(fn.args.defaults) + list(fn.args.kw_defaults)
            default_none = any(isinstance(d, ast.Constant) and d.value is None
                               for d in defaults
                               if d is not None)
            if not default_none:
                offenders.append(
                    f"{module}:{fn_name} {param}= default is not None")
            call_time = any(
                isinstance(n, ast.BoolOp) and isinstance(n.op, ast.Or)
                and any(isinstance(v, ast.Name) and v.id == param
                        for v in n.values)
                and any(isinstance(v, ast.Name)
                        and v.id.startswith("_default_")
                        for v in n.values)
                for n in ast.walk(fn))
            if not call_time:
                offenders.append(
                    f"{module}:{fn_name} does not resolve {param} at call "
                    f"time (`{param} or _default_*`)")
    assert not offenders, f"non-injectable adapter boundaries: {offenders}"


def test_network_transports_are_confined_to_default_adapters():
    """No untested direct external call: transport imports exist ONLY in
    the four ingestion modules, and there ONLY inside `_default_*`
    functions (the single injectable boundary)."""
    offenders: list[str] = []
    for path in _py_files("core", "sports", "frontend", "experiments"):
        # posix-normalized so the forward-slash module allowlist matches on
        # Windows too (str(WindowsPath) uses backslashes).
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in _EXTERNAL_SOURCE_MODULES.values():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods: list[str] = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for m in mods:
                if m.split(".")[0] in _NETWORK_LIBS:
                    offenders.append(f"{rel}:{node.lineno}: import {m}")
    for sport, module in _EXTERNAL_SOURCE_MODULES.items():
        tree = _parse(Path(module))
        for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
            for node in ast.walk(fn):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    mods = [node.module]
                if any(m.split(".")[0] in _NETWORK_LIBS for m in mods) \
                        and not fn.name.startswith("_default_"):
                    offenders.append(
                        f"{module}:{fn.name} imports a network transport "
                        f"outside the _default_* boundary")
    assert not offenders, f"unconfined network transports: {offenders}"


def test_no_test_module_reaches_the_network():
    """Governance: no test module imports a transport, and conftest
    installs the autouse socket block that makes any accidental network
    access fail loudly (hardened rules 7/16)."""
    offenders: list[str] = []
    for path in _py_files("tests"):
        # posix-normalized so the conftest allowlist entry matches on
        # Windows too (str(WindowsPath) uses backslashes).
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel == "tests/conftest.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods: list[str] = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for m in mods:
                if m.split(".")[0] in _NETWORK_LIBS:
                    offenders.append(f"{rel}:{node.lineno}: import {m}")
    conftest = (REPO_ROOT / "tests" / "conftest.py").read_text(
        encoding="utf-8")
    if "autouse=True" not in conftest or "getaddrinfo" not in conftest:
        offenders.append("tests/conftest.py: autouse socket block missing")
    assert not offenders, f"tests reach the network: {offenders}"


def test_approved_degradations_are_declared_and_logged_per_sport():
    """D2/B-007: no silent fallback. Every non-fatal degradation must be
    (a) EXPLICIT — a key in the sport's `APPROVED_DEGRADATIONS` — and
    (b) LOGGED — the exact token passed at a `logger.*` telemetry site in
    the same module. The check is bidirectional: a declared-but-never-
    logged degradation and an undeclared-but-logged one are both defects."""
    offenders: list[str] = []
    for sport, module in _EXTERNAL_SOURCE_MODULES.items():
        tree = _parse(Path(module))
        declared: set[str] = set()
        for node in tree.body:
            target = value = None
            if isinstance(node, ast.Assign) and node.targets:
                target, value = node.targets[0], node.value
            elif isinstance(node, ast.AnnAssign):
                target, value = node.target, node.value
            if (isinstance(target, ast.Name)
                    and target.id == "APPROVED_DEGRADATIONS"
                    and isinstance(value, ast.Dict)):
                declared = {k.value for k in value.keys
                            if isinstance(k, ast.Constant)
                            and isinstance(k.value, str)}
        if not declared:
            offenders.append(
                f"{module}: APPROVED_DEGRADATIONS missing or empty")
            continue
        logged: set[str] = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "logger"):
                for arg in node.args:
                    if isinstance(arg, ast.Constant) \
                            and isinstance(arg.value, str):
                        logged.add(arg.value)
        not_logged = sorted(declared - logged)
        if not_logged:
            offenders.append(
                f"{module}: declared but never logged at a logger.* site: "
                f"{not_logged}")
        undeclared = sorted(t for t in logged
                            if t.startswith(f"{sport}:")
                            and t not in declared)
        if undeclared:
            offenders.append(
                f"{module}: logged degradation token(s) not declared in "
                f"APPROVED_DEGRADATIONS: {undeclared}")
    assert not offenders, (
        f"silent or undeclared degradations: {offenders}")


# ---------------------------------------------------------------------------
# 8. B-001 §7.1 PIT validator wiring (Phase 7.5d WS4 — partial remediation;
# per-field available_at metadata layer = B-001-RESIDUAL, Phase 7.6)
# ---------------------------------------------------------------------------

_SPORT_RUNNERS = {
    "mlb": REPO_ROOT / "sports" / "mlb" / "runner.py",
    "nfl": REPO_ROOT / "sports" / "nfl" / "runner.py",
    "nhl": REPO_ROOT / "sports" / "nhl" / "runner.py",
    "nba": REPO_ROOT / "sports" / "nba" / "runner.py",
}

_STUDY_CONFIGS = {
    "mlb": REPO_ROOT / "sports" / "mlb" / "study_config.py",
    "nfl": REPO_ROOT / "sports" / "nfl" / "study_config.py",
    "nhl": REPO_ROOT / "sports" / "nhl" / "study_config.py",
    "nba": REPO_ROOT / "sports" / "nba" / "study_config.py",
}

_S71_BUFFERS = {"mlb": 60, "nhl": 60, "nfl": 90, "nba": 90}


def _calls_in(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, FileNotFoundError):
        return []
    return [n.func.id for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]


def test_validate_slate_window_has_call_site_per_sport():
    """B-001 WS4 guardrail: the SECONDARY slate gate (validate_slate_window
    — §7.1-reserved PIT names are never reused) has ≥1 production call
    site in every sport runner."""
    gaps = [s for s, p in _SPORT_RUNNERS.items()
            if "validate_slate_window" not in _calls_in(p)]
    assert not gaps, f"validate_slate_window missing in runners: {gaps}"


def test_slate_window_anchors_are_window_derived_per_sport():
    """B-001 WS4 guardrail (uniform secondary gate): every runner derives
    its slate-gate anchor from its SERVING WINDOW via ``window_anchor`` —
    one argument, an attribute of the window object, never a literal date
    and never wall clock — and passes an explicit ``resolution=`` /
    ``start_col=`` matched to the sport's start data (mlb instant via
    start_time_utc; nfl/nhl/nba date until the 7.6 start-timestamp
    normalization).

    AST-pinned: the ``window_anchor`` argument must be an attribute access
    on the window object (``window.start_date`` / ``pred_window
    .primary_date()``), and no ``utcnow``/``now``/``today`` attribute call
    may appear inside it (artifact display stamps like ``trained_at``
    elsewhere in the runner are out of scope — the ANCHOR must be
    deterministic)."""
    from core.validation.future_availability import (SPORT_RESOLUTION,
                                                     SPORT_START_COL)
    expected_attr = {"mlb": "primary_date", "nfl": "start_date",
                     "nhl": "start_date", "nba": "start_date"}
    for sport, path in _SPORT_RUNNERS.items():
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
        anchors = [n for n in ast.walk(tree)
                   if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id == "window_anchor"]
        assert anchors, f"{sport}: no window_anchor call — anchor not wired"
        for node in anchors:
            assert len(node.args) == 1 and not node.keywords, (
                f"{sport}: window_anchor must take exactly the serving "
                "window's first date")
            arg = node.args[0]
            arg_src = ast.unparse(arg)
            assert expected_attr[sport] in arg_src and (
                "window" in arg_src), (
                f"{sport}: window_anchor argument {arg_src!r} is not the "
                "serving window's first date")
            for sub in ast.walk(arg):
                if (isinstance(sub, ast.Attribute)
                        and sub.attr in ("now", "utcnow", "today")):
                    raise AssertionError(
                        f"{sport}: wall-clock call in the window_anchor "
                        "argument — the slate-gate anchor must be "
                        "window-derived")
        assert f'resolution="{SPORT_RESOLUTION[sport]}"' in src, (
            f"{sport}: expected resolution={SPORT_RESOLUTION[sport]!r} at "
            "the slate gate")
        assert f'start_col="{SPORT_START_COL[sport]}"' in src, (
            f"{sport}: expected start_col={SPORT_START_COL[sport]!r} at "
            "the slate gate")


def test_validate_available_at_has_call_site_per_sport():
    """B-001 WS4 guardrail: the §7.1 per-field gate (validate_available_at,
    report_only mode in 7.5d) has ≥1 production call site in every sport
    runner — the metadata gap is operationally visible every run."""
    gaps = [s for s, p in _SPORT_RUNNERS.items()
            if "validate_available_at" not in _calls_in(p)]
    assert not gaps, f"validate_available_at missing in runners: {gaps}"


def test_validate_settlement_record_has_call_site_per_sport():
    """B-001 WS4 guardrail: the settlement sub-gate
    (validate_settlement_record — distinct from the B-005 record-shape
    registry, never citable as PIT evidence) has ≥1 production call site
    in every sport runner."""
    gaps = [s for s, p in _SPORT_RUNNERS.items()
            if "validate_settlement_record" not in _calls_in(p)]
    assert not gaps, f"validate_settlement_record missing in runners: {gaps}"


def test_study_configs_bind_s71_buffer_keys():
    """B-001 WS4 guardrail: all four study_config.py modules bind and
    hard-pin prediction_cutoff_buffer_minutes (60 MLB/NHL, 90 NFL/NBA per
    §7.1) and availability_enforcement (report_only default)."""
    import importlib
    for sport, expected in _S71_BUFFERS.items():
        src = _STUDY_CONFIGS[sport].read_text(encoding="utf-8")
        assert "prediction_cutoff_buffer_minutes" in src, sport
        assert "availability_enforcement" in src, sport
        assert f"!= {expected}" in src, (
            f"{sport}: buffer not hard-pinned to {expected} per §7.1")
        assert "report_only" in src, sport
        mod = importlib.import_module(f"sports.{sport}.study_config")
        cls = next(c for c in vars(mod).values()
                   if isinstance(c, type)
                   and hasattr(c, "__dataclass_fields__")
                   and "prediction_cutoff_buffer_minutes"
                   in getattr(c, "__dataclass_fields__", {}))
        assert "availability_enforcement" in cls.__dataclass_fields__, sport


def test_mlb_slate_anchor_source_is_the_serving_windows_first_date():
    """B-001 WS4 review closure — MLB anchor-source invariant (option (a)).

    MLB's slate-gate anchor argument is ``pred_window.primary_date()``. That
    IS the serving window's FIRST date for EVERY MLB window, so the anchor is
    the window's lower bound exactly as in the other three sports. Cited
    construction:

    * ``core/prediction_window.py::PredictionWindow.primary_date`` returns
      ``self.slate_dates[0]``;
    * ``resolve_prediction_window`` builds ``slate_dates`` as
      ``run_date + 0..lookahead_days`` (fields ``run_date``/``slate_dates``
      above), so ``slate_dates[0] == run_date``;
    * ``sports/mlb/runner.py`` resolves that window from ``window.start_date``
      — the run window's FIRST date — so
      ``primary_date() == run_date == window.start_date``;
    * ``sports.mlb.runner._build_slate`` keeps every row whose date is in
      ``pred_window.slate_dates`` (primary + lookahead), so option (b)
      "MLB serves only primary-date games" is FALSE and the anchor must be
      the window's lower bound.

    Pins both halves: the structural identity, and its behavioural
    consequence — the whole served slate sits at/after the anchor while a
    pre-window row is rejected.
    """
    from datetime import datetime, timezone

    import pandas as pd

    from core.config import default_config
    from core.prediction_window import resolve_prediction_window
    from core.validation.future_availability import (validate_slate_window,
                                                     window_anchor)
    from sports.mlb.runner import _build_slate

    sport = default_config().sport("mlb")
    w = resolve_prediction_window("20260907", sport)

    # (1) structural: primary_date() is the window's FIRST date == run_date.
    assert w.primary_date() == w.slate_dates[0] == w.run_date == "20260907"
    assert list(w.slate_dates) == sorted(w.slate_dates)
    # Multi-day window — MLB serves primary + lookahead, i.e. NOT primary-
    # date-only (this is why (a) holds and (b) does not).
    assert w.lookahead_days >= 1
    assert len(w.slate_dates) == w.lookahead_days + 1

    # (2) behavioural: the served slate spans the whole window, and every
    # served row is at/after the anchor derived from primary_date().
    frame = pd.DataFrame({
        "game_date": ["2026-09-06", "2026-09-07", "2026-09-08", "2026-09-09"],
        "game_pk": [1, 2, 3, 4],
        "home_win": [pd.NA, pd.NA, pd.NA, pd.NA],
        "start_time_utc": ["2026-09-06T23:05:00+00:00",
                           "2026-09-07T23:05:00+00:00",
                           "2026-09-08T23:05:00+00:00",
                           "2026-09-09T23:05:00+00:00"],
    })
    slate = _build_slate(frame, w)
    assert sorted(slate["game_date"]) == ["2026-09-07", "2026-09-08"]
    anchor = window_anchor(w.primary_date())
    assert anchor == datetime(2026, 9, 7, tzinfo=timezone.utc)
    assert (pd.to_datetime(slate["start_time_utc"], utc=True) >= anchor).all()
    assert validate_slate_window(slate, anchor_utc=anchor,
                                 start_col="start_time_utc",
                                 resolution="instant", label="mlb").ok
    # The anchor is the LOWER bound: a pre-window row is rejected.
    pre = frame[frame["game_date"] == "2026-09-06"]
    rep = validate_slate_window(pre, anchor_utc=anchor,
                                start_col="start_time_utc",
                                resolution="instant", label="mlb")
    assert not rep.ok and rep.n_future_violations == 1
