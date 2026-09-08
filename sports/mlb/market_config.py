"""MLB market constants (reference-exact).

Mirrors the read-only reference repo's ``config.AGREEMENT_FILTER_DELTA``:
the fixed threshold at which a run-engine-derived win probability is
flagged as conflicting with the moneyline ensemble. The 0.08/0.10 shares
inside ``agreement_stats`` are the standard secondary read.
"""

from __future__ import annotations

AGREEMENT_FILTER_DELTA: float = 0.08

# k-edge drift discipline (reference run_engine_k_edge module).
K_EDGE_REF: float = 1.53          # challenger C2 k on the reference OOF
K_EDGE_BAND: float = 0.2          # drift alert band: fitted k outside ref±band
