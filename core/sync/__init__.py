"""Sync package (§2.1): manual, contract-only GitHub sync.

Implemented in Phase r8 (``core/sync/github_sync.py``). Never runs
automatically, never touches raw data / stores / frontend artifacts /
experiment output, and is never imported by production runners or
tests. See the module docstring for the full boundary list.

Re-exports are lazy (PEP 562) so ``python -m core.sync.github_sync``
does not shadow its submodule via this package's eager import.
"""

_EXPORTS = {
    "CONTRACT_PATHS",
    "NEVER_STAGE",
    "SyncError",
    "SyncPlan",
    "collect_contract_files",
    "plan_sync",
    "run_sync",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        from core.sync import github_sync as _m

        return getattr(_m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
