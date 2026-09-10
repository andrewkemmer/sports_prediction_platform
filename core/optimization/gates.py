"""Promotion gates and the recommendation path.

A candidate feature set is RECOMMENDED only when every gate passes
relative to the incumbent production set, evaluated on pooled OOF +
sealed (holdout) geometry. ANY failing gate rejects the candidate
automatically; rejected candidates appear in the log only. No promotion
ever occurs inside the harness — ``recommend`` returns a verdict, it
never mutates production state (features, study.yaml, artifacts).
"""

from __future__ import annotations

from dataclasses import dataclass, field

GATE_NAMES = (
    "pooled_logloss", "sealed_logloss", "sealed_auc", "ece",
    "stability", "coverage", "leakage", "determinism", "slate_availability",
)


@dataclass(frozen=True)
class GateResult:
    """Verdict for one candidate against the incumbent."""

    candidate: str
    recommended: bool
    gates: dict[str, bool] = field(default_factory=dict)
    failures: tuple[str, ...] = ()
    metrics: dict = field(default_factory=dict)
    deltas: dict = field(default_factory=dict)

    def log_line(self) -> str:
        status = "RECOMMEND" if self.recommended else "REJECT"
        p = self.metrics.get("pooled", {})
        if "rmse" in p:
            body = (f"rmse={p.get('rmse')} mae={p.get('mae')} "
                    f"bias={p.get('bias')}")
        else:
            body = (f"logloss={p.get('logloss')} auc={p.get('auc')} "
                    f"brier={p.get('brier')} ece={p.get('ece')}")
        return (f"[{status}] {self.candidate} {body} "
                + ("" if not self.failures else
                   f"failed={','.join(self.failures)}"))


def evaluate_gates(candidate_name: str,
                   cand: dict,      # {"pooled": {...}, "sealed": {...},
                                    #  "fold_logloss": [...], "coverage": float,
                                    #  "leakage_ok": bool, "deterministic": bool,
                                    #  "slate_available": bool}
                   incumbent: dict,  # same shape (metrics used)
                   *, coverage_min: float = 0.90,
                   logloss_tolerance: float = 0.005,
                   auc_tolerance: float = 0.005,
                   ece_tolerance: float = 0.010,
                   stability_max_delta: float = 0.020) -> GateResult:
    """Evaluate every promotion gate. Tolerances are vs the incumbent.

    All metric inputs must be OOF/sealed rows — the caller's runner
    contract guarantees this; the gate only compares numbers.
    """
    gates: dict[str, bool] = {}
    failures: list[str] = []

    cp, ip = cand["pooled"], incumbent["pooled"]
    cs, is_ = cand.get("sealed") or {}, (incumbent.get("sealed") or {})
    regression = "rmse" in cp  # market scope: rmse/mae metrics, no AUC/ECE

    # 1) pooled objective not worse than incumbent by tolerance
    if regression:
        g = cp["rmse"] <= ip["rmse"] + logloss_tolerance * 2
        gates["pooled_logloss"] = g  # same gate slot, RMSE objective
        if not g:
            failures.append(
                f"pooled RMSE {cp['rmse']:.5f} > incumbent "
                f"{ip['rmse']:.5f} + tol {logloss_tolerance * 2}")
    else:
        g = cp["logloss"] <= ip["logloss"] + logloss_tolerance
        gates["pooled_logloss"] = g
        if not g:
            failures.append(
                f"pooled logloss {cp['logloss']:.5f} > incumbent "
                f"{ip['logloss']:.5f} + tol {logloss_tolerance}")

    # 2) sealed objective not materially worse
    if cs and is_:
        if regression:
            g = cs["rmse"] <= is_["rmse"] + logloss_tolerance * 4
            gates["sealed_logloss"] = g
            if not g:
                failures.append(
                    f"sealed RMSE {cs['rmse']:.5f} > incumbent "
                    f"{is_['rmse']:.5f} (+4x tol)")
        else:
            g = cs["logloss"] <= is_["logloss"] + logloss_tolerance * 2
            gates["sealed_logloss"] = g
            if not g:
                failures.append(
                    f"sealed logloss {cs['logloss']:.5f} > incumbent "
                    f"{is_['logloss']:.5f} (+2x tol)")
    else:
        gates["sealed_logloss"] = True

    # 3) sealed AUC not materially worse (binary scope only)
    if regression:
        gates["sealed_auc"] = True
    elif cs and is_ and cs.get("auc") is not None and is_.get("auc") is not None:
        g = cs["auc"] >= is_["auc"] - auc_tolerance
        gates["sealed_auc"] = g
        if not g:
            failures.append(
                f"sealed AUC {cs['auc']:.5f} < incumbent {is_['auc']:.5f} "
                f"− tol {auc_tolerance}")
    else:
        gates["sealed_auc"] = True

    # 4) ECE/calibration acceptable vs incumbent (binary scope only)
    if regression:
        gates["ece"] = True
    else:
        g = cp["ece"] <= ip["ece"] + ece_tolerance
        gates["ece"] = g
        if not g:
            failures.append(
                f"ECE {cp['ece']:.5f} > incumbent {ip['ece']:.5f} + tol "
                f"{ece_tolerance}")

    # 5) no unstable period behavior (late-vs-early fold degradation)
    stability = cand.get("stability") or {"delta": None}
    delta = stability.get("delta")
    g = delta is None or delta <= stability_max_delta
    gates["stability"] = g
    if not g:
        failures.append(
            f"late-period degradation {delta:.5f} > {stability_max_delta}")

    # 6) coverage / non-null policy — measured over the ADDED candidate
    # features by the runner ("feature coverage >= threshold, unavailable
    # fields explicit"); the incumbent set itself is not re-judged.
    cov_raw = cand.get("coverage", 0.0)
    cov = float(cov_raw if not isinstance(cov_raw, dict)
                else cov_raw.get("value", 0.0))
    g = cov >= coverage_min
    gates["coverage"] = g
    if not g:
        failures.append(f"coverage {cov:.3f} < {coverage_min}")

    # 7) leakage audit passed
    g = bool(cand.get("leakage_ok"))
    gates["leakage"] = g
    if not g:
        failures.append("leakage audit failed")

    # 8) deterministic rerun
    g = bool(cand.get("deterministic"))
    gates["determinism"] = g
    if not g:
        failures.append("deterministic rerun mismatch")

    # 9) available for the upcoming slate (no outcome/future data needed)
    g = bool(cand.get("slate_available"))
    gates["slate_availability"] = g
    if not g:
        failures.append("not available for upcoming slate")

    deltas = {
        "pooled_logloss": (cp["logloss"] - ip["logloss"])
        if not regression else (cp["rmse"] - ip["rmse"]),
        "pooled_brier": cp["brier"] - ip["brier"] if "brier" in cp else 0.0,
        "pooled_ece": cp["ece"] - ip["ece"] if "ece" in cp else 0.0,
    }
    if not regression and (cp.get("auc") is not None
                           and ip.get("auc") is not None):
        deltas["pooled_auc"] = cp["auc"] - ip["auc"]
    if cs and is_:
        deltas["sealed_logloss"] = (cs["logloss"] - is_["logloss"]) \
            if not regression else (cs["rmse"] - is_["rmse"])
        if not regression and (cs.get("auc") is not None
                               and is_.get("auc") is not None):
            deltas["sealed_auc"] = cs["auc"] - is_["auc"]

    return GateResult(
        candidate=candidate_name,
        recommended=not failures,
        gates=gates,
        failures=tuple(failures),
        metrics={"pooled": cp, "sealed": cs or {}},
        deltas=deltas,
    )


def recommend(results: list[GateResult]) -> GateResult | None:
    """Pick the single recommended winner (if any): among recommended
    candidates, the lowest pooled primary objective (logloss for binary
    scopes, RMSE for regression/market scopes); AUC breaks binary ties.
    Returns None when nothing passes — NEVER a forced promotion."""
    passing = [r for r in results if r.recommended]
    if not passing:
        return None

    def key(r: GateResult):
        p = r.metrics["pooled"]
        if "rmse" in p:
            return (p["rmse"], 0.0)
        return (p["logloss"], -(p.get("auc") or 0.0))

    return min(passing, key=key)
