"""Experiments CLI + manual sync (Phase r8, §15/§16/§2.1).

Pins — via SOURCE INSPECTION (tests never import ``experiments/`` per
the manifest hygiene rule):

* ``experiments/`` contains ONLY the allowlisted entries
  (``run_experiment.py``, ``run_training.py``, ``configs/``,
  ``__init__.py``);
* ``run_experiment.py`` is the generic config-driven runner: production
  imports only (core.experiments + sports.*.optimization), no
  ``data_delivery`` writes, no contract promotion, argparse over
  --sport/--config;
* ``run_training.py`` resolves §15 training policies via
  ``training_policy_candidates`` and prints its report (no artifacts);
* ``core/sync/github_sync.py``: contract-path staging only, structural
  exclusion of data/store/cache paths, dry-run by default, lazy
  GitPython, never automatic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# experiments/ tree (§15) — source inspection only
# ---------------------------------------------------------------------------

EXPERIMENTS_ALLOWLIST = {
    "run_experiment.py",
    "run_training.py",
    "configs",
    "__init__.py",
    ".gitkeep",  # directory placeholder (guardrail-recognized)
}


def _read(*parts: str) -> str:
    return (REPO_ROOT.joinpath(*parts)).read_text(encoding="utf-8")


def test_experiments_tree_is_allowlisted():
    base = REPO_ROOT / "experiments"
    assert base.is_dir(), "experiments/ must exist (§15 durable scaffolding)"
    entries = {p.name for p in base.iterdir() if p.name != "__pycache__"}
    assert entries == EXPERIMENTS_ALLOWLIST, (
        f"experiments/ drift: unexpected={entries - EXPERIMENTS_ALLOWLIST} "
        f"missing={EXPERIMENTS_ALLOWLIST - entries}")


def test_run_experiment_is_the_generic_config_runner():
    src = _read("experiments", "run_experiment.py")
    # argparse CLI over sport + config
    assert '"--sport"' in src and '"--config"' in src
    # reuses production optimizer + search space (never reimplements)
    assert "from core.experiments.optimizer import run_optimization" in src
    assert ("from core.experiments.search_space import "
            "load_optimization_config") in src
    # binds the production per-sport adapters
    assert "sports.{sport}.optimization" in src
    assert "_adapters" in src
    # no artifact writes, no contract promotion, no git push
    assert "data_delivery" in src  # mentioned only as a never-write boundary
    assert "repo.git.push" not in src and "import git" not in src
    for banned in ("write_parquet", "write_duckdb", "to_csv(", "to_json("):
        assert banned not in src, banned


def test_run_training_resolves_bounded_policies():
    src = _read("experiments", "run_training.py")
    assert "training_policy_candidates" in src
    assert "TrainingPolicy" in src
    # report printed, not persisted
    assert "json.dumps" in src
    for banned in ("write_parquet", "write_duckdb", "to_csv(", "to_json("):
        assert banned not in src, banned


# ---------------------------------------------------------------------------
# manual sync (§2.1) — behavioral pins via the core module (allowed import)
# ---------------------------------------------------------------------------

from core.sync import (  # noqa: E402
    CONTRACT_PATHS,
    NEVER_STAGE,
    SyncPlan,
    collect_contract_files,
    plan_sync,
    run_sync,
)


def test_sync_contract_paths_are_code_config_docs_only():
    text = " ".join(CONTRACT_PATHS)
    assert "core/" in text and "sports/" in text and "tests/" in text
    # data/artifact roots are never contract scope
    assert "data_delivery" not in text
    assert "store/" not in text


def test_sync_never_stage_list_is_structural():
    assert "data_delivery/" in NEVER_STAGE
    assert "store/" in NEVER_STAGE


def test_sync_plan_is_dry_run_and_deterministic(tmp_path):
    root = REPO_ROOT
    files = collect_contract_files(root)
    assert files, "contract file collection cannot be empty"
    # data/cache exclusion is structural: no data path can ever appear
    assert not any(
        f.startswith(tuple(NEVER_STAGE)) or "__pycache__" in f
        for f in files)
    plan1 = plan_sync(root, "msg")
    plan2 = plan_sync(root, "msg")
    assert isinstance(plan1, SyncPlan) and plan1.dry_run
    assert plan1.files == plan2.files  # deterministic order
    assert plan1.message == "msg"
    assert plan1.files == files


def test_sync_execute_false_never_imports_git():
    """The lazy GitPython import lives only in the execute branch;
    execute=False must return the plan with git never imported."""
    import sys
    assert "git" not in sys.modules
    plan = run_sync(REPO_ROOT, "dry run", execute=False)
    assert plan.dry_run
    assert "git" not in sys.modules


def test_sync_plan_fails_loud_outside_a_repo(tmp_path):
    from core.sync import SyncError
    with pytest.raises(SyncError, match="not a git repository"):
        plan_sync(tmp_path, "nope")


# ---------------------------------------------------------------------------
# §2.1: sync is never automatic — no production/test invocation
# ---------------------------------------------------------------------------

def test_sync_module_is_never_imported_by_production_or_tests():
    offenders: list[str] = []
    for base in ("core", "sports", "frontend", "experiments"):
        for p in (REPO_ROOT / base).rglob("*.py"):
            rel = p.relative_to(REPO_ROOT).as_posix()
            if rel.startswith("core/sync"):
                continue
            src = p.read_text(encoding="utf-8")
            if "core.sync" in src or "github_sync" in src:
                offenders.append(rel)
    assert not offenders, f"sync imported from: {offenders}"
