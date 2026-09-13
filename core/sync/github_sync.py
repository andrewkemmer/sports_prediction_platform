"""Manual, contract-only GitHub sync (§2.1).

Scope boundaries (hard):
* NEVER runs automatically — no production runner, test, or schedule
  invokes this module; it is executed by a human, explicitly.
* NEVER touches raw data, durable stores, or frontend artifacts
  (``data_delivery/``, ``store/``, caches) — those live outside Git.
* NEVER touches experiment output (experiments write nothing anyway).
* Pushes CODE + CONFIG + CONTRACTS + TESTS + DOCS only.

Delivery rules (spec §23): only allowlisted frontend artifact families
would ever be delivered by the pipeline; this module does not deliver
artifacts at all — it is the developer-facing repository sync.
GitPython is imported lazily (not required by the pinned runtime).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: Top-level paths that are part of the repository contract and may be
#: staged by a sync. Everything outside this set is ignored.
CONTRACT_PATHS: tuple[str, ...] = (
    "core/",
    "sports/",
    "frontend/",
    "experiments/run_experiment.py",
    "experiments/run_training.py",
    "experiments/__init__.py",
    "experiments/configs/",
    "tests/",
    "docs/",
    "GUARDRAILS.md",
    "README.md",
    "pyproject.toml",
    ".github/",
)

#: Paths never staged, even if nested under a contract path.
NEVER_STAGE: tuple[str, ...] = (
    "data_delivery/",
    "store/",
    ".freebuff/",
    "__pycache__/",
    ".pytest_cache/",
)


class SyncError(RuntimeError):
    """Raised when a sync cannot be planned or executed safely."""


@dataclass(frozen=True)
class SyncPlan:
    """The dry-run plan (what a sync WOULD stage, commit, push)."""

    branch: str
    files: tuple[str, ...] = field(default_factory=tuple)
    message: str = ""
    dry_run: bool = True


def _is_never_staged(rel: str) -> bool:
    return any(rel.startswith(p) or f"/{p}" in f"/{rel}"
               for p in NEVER_STAGE)


def collect_contract_files(repo_root: Path) -> tuple[str, ...]:
    """Every existing, contract-scope file under ``repo_root`` (forward-
    slash relative paths, deterministic order). Data/cache paths are
    excluded structurally — a file under ``data_delivery/`` can never
    appear regardless of where it sits."""
    out: set[str] = set()
    for top in CONTRACT_PATHS:
        p = repo_root / top
        if not p.exists():
            continue
        if p.is_file():
            rel = p.relative_to(repo_root).as_posix()
            if not _is_never_staged(rel):
                out.add(rel)
            continue
        for f in p.rglob("*"):
            if not f.is_file():
                continue
            rel = f.relative_to(repo_root).as_posix()
            if _is_never_staged(rel):
                continue
            out.add(rel)
    return tuple(sorted(out))


def plan_sync(repo_root: Path, message: str, *,
              branch: str = "main") -> SyncPlan:
    """Build the deterministic sync plan (dry-run; no Git calls)."""
    repo_root = Path(repo_root).resolve()
    if not (repo_root / ".git").exists():
        raise SyncError(f"{repo_root} is not a git repository")
    files = collect_contract_files(repo_root)
    if not files:
        raise SyncError("sync plan is empty — nothing contract-scope found")
    return SyncPlan(branch=branch, files=files, message=message,
                    dry_run=True)


def run_sync(repo_root: Path, message: str, *, branch: str = "main",
             remote: str = "origin", execute: bool = False) -> SyncPlan:
    """Stage the contract files, commit, and push — EXPLICITLY only.

    ``execute=False`` (the default) returns the dry-run plan and
    performs no Git operations. ``execute=True`` requires GitPython;
    it fails loud when unavailable. This function is never called by
    production code or tests (guardrail: sync is manual).
    """
    plan = plan_sync(repo_root, message, branch=branch)
    if not execute:
        return plan
    try:
        import git  # lazy: GitPython is not a pinned runtime dependency
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise SyncError(
            "GitPython is required for execute=True (pip install GitPython)"
        ) from exc
    repo = git.Repo(str(Path(repo_root).resolve()))
    repo.git.checkout(branch)
    repo.git.add(*plan.files)
    repo.index.commit(plan.message)
    repo.git.push(remote, branch)
    return SyncPlan(branch=branch, files=plan.files, message=message,
                    dry_run=False)


def _main(argv: list[str] | None = None) -> int:
    """Manual CLI: ``python -m core.sync.github_sync`` (§2.1).

    Dry-run is the default and prints the plan only; ``--execute`` is
    the explicit, human-initiated staging + commit + push.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="core.sync.github_sync",
        description="Manual, contract-only GitHub sync (spec §2.1). "
                    "Dry-run by default; never touches data, stores, "
                    "caches, or frontend artifacts.",
    )
    parser.add_argument("--message", "-m", required=True,
                        help="commit message for the sync")
    parser.add_argument("--branch", default="main",
                        help="branch to sync (default: main)")
    parser.add_argument("--remote", default="origin",
                        help="remote to push to (default: origin)")
    parser.add_argument("--dry-run", action="store_true",
                        help="explicit dry-run (the default already; "
                             "accepted for readability)")
    parser.add_argument("--execute", action="store_true",
                        help="actually stage/commit/push (default: "
                             "dry-run plan only)")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    try:
        plan = run_sync(repo_root, args.message, branch=args.branch,
                        remote=args.remote, execute=args.execute)
    except SyncError as exc:
        print(f"sync FAILED: {exc}")
        return 1
    mode = "EXECUTE" if not plan.dry_run else "DRY-RUN"
    print(f"[{mode}] branch={plan.branch} files={len(plan.files)}")
    for rel in plan.files:
        print(f"  {rel}")
    if plan.dry_run:
        print("dry-run only — rerun with --execute to stage/commit/push")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
