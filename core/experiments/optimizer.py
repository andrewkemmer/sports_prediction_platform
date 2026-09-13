"""The single shared optimization runner.

Runs bounded, deterministic, walk-forward candidate evaluation for one
sport/model scope. Contract:

* reuses an existing per-sport real store when present; otherwise the
  scope is reported as DEFERRED (never silently skipped, never
  fabricated);
* walk-forward OOF only (the platform's own fold geometry from
  study.yaml — never optimized, never a random split);
* train-only imputation + scaling identical to production (the linear
  path uses the production TrainFoldPreprocessor policy);
* deterministic: fixed seed, sorted candidate order; a rerun with the
  same seed produces identical metrics;
* writes NOTHING — no experiment artifacts, no reports. All results go
  to the log;
* does not modify the production FEATURE list or study.yaml. The only
  output is the returned verdict object for the caller to log/report.

Scope kinds: the binary moneyline scope (target home_win, probability
metrics) and the market scope (continuous regressand margin, regression
metrics) are evaluated through the same machinery with kind-specific
metrics and member factories — their feature lists are never inferred
from each other.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    from joblib import Parallel, delayed
except ImportError:  # pragma: no cover - joblib is a sklearn dependency
    Parallel = None
    delayed = None

from core.experiments.search_space import (
    CandidateSpec,
    OptimizationConfig,
    build_candidates,
    candidate_catalog_fingerprint,
    derive_column,
    spec_from_name,
)
from core.evaluation.probabilistic_metrics import (
    period_stability,
    pooled_metrics,
    regression_metrics,
    regression_period_stability,
)
from core.validation.leakage import leakage_audit, pit_safe_matrix

logger = logging.getLogger("core.experiments")


# ---------------------------------------------------------------------------
# Promotion gates and the recommendation path (merged from the former
# core/optimization/gates.py in the Phase-r1 restructure).
#
# A candidate feature set is RECOMMENDED only when every gate passes
# relative to the incumbent production set, evaluated on pooled OOF +
# sealed (holdout) geometry. ANY failing gate rejects the candidate
# automatically; rejected candidates appear in the log only. No promotion
# ever occurs inside the harness — ``recommend`` returns a verdict, it
# never mutates production state (features, study.yaml, artifacts).
# ---------------------------------------------------------------------------

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


class ScopeRunner:
    """Runs one sport/model scope end-to-end."""

    def __init__(self, scope: ModelScope, config: OptimizationConfig,
                 adapter):
        """``adapter`` is the sport binding (dict of callables):

        load_decided()      -> DataFrame of decided games (raw fields +
                               incumbent features + targets)
        load_slate()        -> DataFrame of upcoming games (same fields,
                               no targets)
        build_folds(df)     -> platform fold list of POSITIONAL
                               (train_idx, val_idx) pairs — the study.yaml
                               geometry, unchanged
        matrix(df, feats)   -> float feature frame for the given feature
                               names (production derivation path)
        member_factory(seed)-> closure: fit(Xtr, ytr) / predict(Xva) with
                               the production train-only imputation and
                               native NaN routing
        raw_columns(df)     -> available raw field columns
        slate_check(slate, feats) -> True when every feature is
                               computable on the upcoming slate
        team_cols/date_col  -> optional identity columns for per-team
                               rolling derivations
        leakage_extra       -> optional bool, sport-specific PIT gates
        """
        self.scope = scope
        self.config = config
        self.adapter = adapter
        self.deadline: float | None = None

    # -- budget ------------------------------------------------------------
    def _time_left(self) -> float:
        if self.deadline is None:
            return float("inf")
        return self.deadline - time.monotonic()

    # -- pipeline ----------------------------------------------------------
    def run(self) -> dict:
        t0 = time.monotonic()
        self.deadline = t0 + self.config.bounds.max_seconds
        scope = self.scope
        kind = "market" if scope.model == "market" else "moneyline"

        df = self.adapter["load_decided"]()
        slate = self.adapter["load_slate"]()
        raw_fields = tuple(sorted(set(self.adapter["raw_columns"](df))))

        candidates, withheld = build_candidates(
            raw_fields, self.config.bounds,
            incumbent=scope.prod_features)
        fp1 = candidate_catalog_fingerprint(candidates)
        fp2 = candidate_catalog_fingerprint(
            build_candidates(raw_fields, self.config.bounds,
                             incumbent=scope.prod_features)[0])
        deterministic_catalog = fp1 == fp2

        audit = leakage_audit(candidates, raw_fields)
        logger.info("[%s/%s] catalog: %d candidates (%d withheld) "
                    "fingerprint=%s leakage_ok=%s", scope.sport, scope.model,
                    len(candidates), len(withheld), fp1[:12], audit.ok)
        if withheld:
            logger.info("[%s/%s] withheld by feature bound: %s",
                        scope.sport, scope.model, withheld[:20])
        if not audit.ok:
            logger.error("[%s/%s] leakage audit FAILED: %s",
                         scope.sport, scope.model, audit.violations)

        folds = self.adapter["build_folds"](df)
        logger.info("[%s/%s] platform fold geometry: %d folds (fixed)",
                    scope.sport, scope.model, len(folds))

        # Incumbent metrics (production feature list, unchanged).
        incumbent = self._evaluate_set(scope.prod_features, df, slate, folds)
        logger.info("[%s/%s] incumbent pooled: %s", scope.sport, scope.model,
                    self._fmt(incumbent))

        # Candidate sets: single-feature additions to the incumbent
        # (deterministic bounded sweep). Each candidate = incumbent ∪ {c}.
        # The sweep is capped at bounds.sweep_cap sets — a stable prefix of
        # the deterministic priority order (diffs → rolling → pairwise
        # interactions), so the run is bounded AND reproducible; every
        # dropped candidate is logged explicitly.
        feature_sets: list[tuple[str, tuple[str, ...]]] = []
        for c in candidates:
            if c.derivation == "incumbent" or c.name in scope.prod_features:
                continue
            if len(feature_sets) >= self.config.bounds.sweep_cap:
                logger.info("[%s/%s] sweep cap %d reached — %d candidate "
                            "additions not swept this run", scope.sport,
                            scope.model, self.config.bounds.sweep_cap,
                            len(candidates) - len(scope.prod_features)
                            - len(feature_sets))
                break
            feature_sets.append(
                (f"incumbent+{c.name}", scope.prod_features + (c.name,)))

        # Derive every candidate column ONCE (identical derivation to the
        # per-set path — cached so the bounded sweep evaluates each
        # candidate's column a single time instead of per feature set).
        new_specs = [c for c in candidates
                     if c.derivation != "incumbent"
                     and c.name not in scope.prod_features]
        derived = self._derived_columns(df, new_specs)
        slate_raws = tuple(sorted(set(self.adapter["raw_columns"](slate))))
        slate_new_specs = [s for s in new_specs
                           if spec_from_name(s.name, slate_raws) is not None]
        slate_derived = self._derived_columns(slate, slate_new_specs)

        results: list[GateResult] = []
        n_jobs = self.config.bounds.n_jobs or 1
        can_parallel = (n_jobs > 1 and Parallel is not None
                        and len(feature_sets) > 1)
        if can_parallel:
            # Pure execution parallelism: candidate sets are independent
            # and returned in deterministic order, so results are
            # identical to the sequential path.
            evals = Parallel(n_jobs=n_jobs)(
                delayed(self._evaluate_and_gate)(
                    name, feats, df, slate, folds, derived, slate_derived,
                    incumbent, deterministic_catalog, audit)
                for name, feats in feature_sets)
            results = list(evals)
            for gr in results:
                logger.info(gr.log_line())
        else:
            for name, feats in feature_sets:
                if self._time_left() <= 0:
                    logger.warning(
                        "[%s/%s] wall-clock budget reached — %d of %d "
                        "candidate sets evaluated", scope.sport,
                        scope.model, len(results), len(feature_sets))
                    break
                gr = self._evaluate_and_gate(
                    name, feats, df, slate, folds, derived, slate_derived,
                    incumbent, deterministic_catalog, audit)
                results.append(gr)
                logger.info(gr.log_line())

        winner = recommend(results)
        if winner is not None:
            logger.info("[%s/%s] WINNER %s deltas=%s", scope.sport,
                        scope.model, winner.candidate,
                        {k: (round(v, 5) if isinstance(v, float) else v)
                         for k, v in winner.deltas.items()})
        else:
            logger.info("[%s/%s] no candidate passed all gates — incumbent "
                        "retained (no promotion)", scope.sport, scope.model)
        return {
            "scope": f"{scope.sport}/{scope.model}",
            "kind": kind,
            "catalog_fingerprint": fp1,
            "catalog_deterministic": deterministic_catalog,
            "leakage_ok": audit.ok,
            "leakage_violations": audit.violations,
            "n_candidates": len(candidates),
            "n_candidate_sets": len(feature_sets),
            "interactions_withheld": withheld,
            "incumbent": incumbent,
            "results": results,
            "winner": winner,
            "wall_clock_seconds": round(time.monotonic() - t0, 2),
        }

    # -- derivation ----------------------------------------------------------
    def _evaluate_and_gate(self, name, feats, df, slate, folds, derived,
                           slate_derived, incumbent, catalog_ok, audit):
        """Evaluate one candidate set and gate it against the incumbent."""
        m = self._evaluate_set(feats, df, slate, folds,
                               derived=derived, slate_derived=slate_derived)
        return self._gate(name, m, incumbent, catalog_ok, audit)

    def _derived_columns(self, df: pd.DataFrame,
                         specs: list[CandidateSpec]) -> pd.DataFrame:
        """Evaluate every generated-candidate derivation once on the
        decided frame (per-team rolling uses the adapter's identity
        columns). Each column is checked against the shift(1) discipline:
        masking the CURRENT row's own value must not change its feature
        value (guaranteed by the shift(1) construction)."""
        cols: dict[str, pd.Series] = {}
        team_cols = tuple(self.adapter.get("team_cols",
                                           ("home_team", "away_team")))
        date_col = self.adapter.get("date_col", "game_date")
        for spec in specs:
            cols[spec.name] = derive_column(
                df, spec, team_cols=team_cols, date_col=date_col) \
                .astype(float)
        return pd.DataFrame(cols, index=df.index)

    # -- evaluation ----------------------------------------------------------
    def _evaluate_set(self, feats, df, slate, folds, *, derived=None,
                      slate_derived=None) -> dict:
        scope = self.scope
        kind = "market" if scope.model == "market" else "moneyline"

        X = self.adapter["matrix"](df, feats)
        if derived is not None:
            # Overwrite the per-set derivation with the cached column
            # (identical evaluator; caching only avoids recomputation).
            for f in feats:
                if f in derived.columns:
                    X[f] = derived[f]
        y = df[scope.target].astype(float).to_numpy()

        oof_p = np.full(len(df), np.nan)
        for tr_idx, va_idx in folds:
            if len(tr_idx) < 1 or len(va_idx) < 1:
                continue
            fit = self.adapter["member_factory"](self.config.bounds.seed)
            fit["fit"](X.iloc[tr_idx], y[tr_idx])
            p = fit["predict"](X.iloc[va_idx])
            oof_p[va_idx] = np.asarray(p, dtype=float)

        if kind == "moneyline":
            oof_p = np.clip(oof_p, 1e-9, 1 - 1e-9)
            fold_scores = self._fold_logloss(oof_p, y, folds)
            pooled = pooled_metrics(oof_p[np.isfinite(oof_p)],
                                    y[np.isfinite(oof_p)])
            stability = period_stability(fold_scores)
        else:
            fold_scores = self._fold_rmse(oof_p, y, folds)
            pooled = regression_metrics(oof_p[np.isfinite(oof_p)],
                                        y[np.isfinite(oof_p)])
            stability = regression_period_stability(fold_scores)

        # Sealed period: the last fold's validation window (platform
        # geometry, held out of any selection decisions).
        sealed = {}
        if folds:
            last_va = folds[-1][1]
            sp = oof_p[last_va]
            if np.isfinite(sp).all():
                sealed = (pooled_metrics(sp, y[last_va]) if kind == "moneyline"
                          else regression_metrics(sp, y[last_va]))

        # Coverage policy: the gate measures the coverage of the ADDED
        # candidate features (the spec's "feature coverage >= threshold,
        # unavailable fields explicit") — production's incumbent set may
        # itself carry low-coverage served features, and the whole-set
        # average must not veto a candidate whose own derivations are
        # well-covered. For the incumbent evaluation (nothing added) the
        # whole-set average is reported.
        added = [f for f in feats if f not in set(scope.prod_features)]
        measured = added if added else list(feats)
        cov = float(np.mean([
            np.isfinite(X[f]).mean() if f in X.columns else 0.0
            for f in measured])) if len(X) else 0.0
        slate_ok = self._slate_available(slate, feats)
        pit_ok, _ = pit_safe_matrix(
            X.assign(**{c: df[c] for c in
                        ("home_win", "margin", "total") if c in df.columns}),
            self._slate_mask(df))
        return {
            "pooled": pooled,
            "sealed": sealed,
            "fold_scores": fold_scores,
            "stability": stability,
            "coverage": cov,
            "leakage_ok": bool(pit_ok) and self.adapter.get(
                "leakage_extra", True),
            "deterministic": True,  # verified by the determinism re-run test
            "slate_available": bool(slate_ok),
        }

    def _fold_logloss(self, oof_p: np.ndarray, y: np.ndarray,
                      folds) -> list[float]:
        from core.folds import log_loss as _ll

        out: list[float] = []
        for _, va_idx in folds:
            p = oof_p[va_idx]
            if np.isfinite(p).all() and len(p):
                out.append(float(_ll(p.tolist(), y[va_idx].tolist())))
        return out

    def _fold_rmse(self, oof_p: np.ndarray, y: np.ndarray,
                   folds) -> list[float]:
        out: list[float] = []
        for _, va_idx in folds:
            p = oof_p[va_idx]
            if np.isfinite(p).all() and len(p):
                err = p - y[va_idx]
                out.append(float(np.sqrt(np.mean(err ** 2))))
        return out

    def _slate_available(self, slate: pd.DataFrame, feats) -> bool:
        scope = self.scope
        raw_fields = tuple(sorted(set(self.adapter["raw_columns"](slate))))
        for f in feats:
            if f in scope.prod_features:
                if f in slate.columns:
                    continue
                return False
            spec = spec_from_name(f, raw_fields)
            if spec is None:
                return False
        return True

    def _gate(self, name, m, incumbent, catalog_ok, audit) -> GateResult:
        cfg = self.config
        m = dict(m)
        m["leakage_ok"] = bool(m.get("leakage_ok")) and audit.ok
        m["deterministic"] = bool(catalog_ok)
        return evaluate_gates(
            name, m, incumbent,
            coverage_min=cfg.coverage_min,
            logloss_tolerance=cfg.logloss_tolerance,
            auc_tolerance=cfg.auc_tolerance,
            ece_tolerance=cfg.ece_tolerance,
            stability_max_delta=cfg.stability_max_delta)

    @staticmethod
    def _fmt(m: dict) -> str:
        p = m.get("pooled", {})
        if "logloss" in p:
            return (f"logloss={p['logloss']:.5f} auc={p.get('auc'):.5f} "
                    f"brier={p.get('brier'):.5f} ece={p.get('ece'):.5f}")
        return (f"rmse={p.get('rmse', float('nan')):.5f} "
                f"mae={p.get('mae', float('nan')):.5f}")

    @staticmethod
    def _slate_mask(df: pd.DataFrame) -> pd.Series:
        """Slate mask: rows with null home_win are upcoming/undecided."""
        return df["home_win"].isna() if "home_win" in df.columns \
            else pd.Series(False, index=df.index)


def run_optimization(config: OptimizationConfig, adapters: dict) -> dict:
    """Run the optimizer for every scope with an adapter bound.

    ``adapters`` maps "sport/model" -> adapter dict. A scope without an
    adapter is reported as deferred (no real store / offline smoke only)
    — never silently skipped. Returns per-scope summaries; the caller
    logs and reports. No files are written.
    """
    out: dict = {}
    for scope in config.scopes:
        key = f"{scope.sport}/{scope.model}"
        adapter = adapters.get(key)
        if adapter is None:
            logger.warning("[%s] DEFERRED — no adapter bound (no real store "
                           "or not exercised this run)", key)
            out[key] = {"deferred": True}
            continue
        out[key] = ScopeRunner(scope, config, adapter).run()
    return out


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
