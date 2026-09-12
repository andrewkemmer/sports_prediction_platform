"""Thin composition root for the per-sport optimization adapters.

Registry only — no sport logic lives here. Each sport owns its builder in
``sports/<sport>/optimization.py``; this module binds them to their
registry keys and exposes ``all_adapters()`` for callers that want every
sport's adapters at once. A sport without a real store contributes
nothing — its scopes report DEFERRED in the runner (never fabricated,
never silently skipped).

Lives under ``sports/`` (not ``core/``) so the dependency direction stays
sports -> core: policy section 18 forbids core from importing ``sports.*``
at module level or function-local.
"""

from __future__ import annotations

import logging

from sports.mlb.optimization import mlb_adapters
from sports.nba.optimization import nba_adapters
from sports.nfl.optimization import nfl_adapters
from sports.nhl.optimization import nhl_adapters

logger = logging.getLogger("core.optimization")

#: The four sport builders, keyed by sport. Composition root only —
#: adding a sport means adding its ``sports/<sport>/optimization.py``
#: builder and one entry here.
SPORT_ADAPTER_BUILDERS = {
    "mlb": mlb_adapters,
    "nfl": nfl_adapters,
    "nhl": nhl_adapters,
    "nba": nba_adapters,
}


def all_adapters() -> dict[str, dict]:
    """Every sport's adapters for sports with a real store; sports
    without one contribute nothing (their scopes report DEFERRED)."""
    out: dict[str, dict] = {}
    for builder in SPORT_ADAPTER_BUILDERS.values():
        try:
            out.update(builder(None))
        except Exception as exc:  # noqa: BLE001 — a sport's store problems
            logger.warning("adapter binding failed for %s: %s",
                           builder.__name__, exc)
    return out
