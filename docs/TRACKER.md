# TRACKER.md — Phase Ledger & Blocker Table

Status: seeded in Phase 7.5c. Referenced from `GUARDRAILS.md`
(Implementation Governance). Every blocker cited in phase reports and
guardrail-test strict xfails must exist here by ID.

---

## Phase ledger

| Phase | Scope | Status |
|---|---|---|
| 0 | Platform scaffold: contracts, config, dates, storage, retention, markets, validation, warmup, folds, OOF rules, settlement semantics | COMPLETE |
| 1 | MLB sport vertical: ingestion, features, training, artifacts, runner | COMPLETE |
| 2 | Streamlit frontend: dashboard pages, artifact loaders, schemas | COMPLETE |
| 3 | NFL sport vertical | COMPLETE |
| 4 | NHL sport vertical | COMPLETE |
| 5 | NBA sport vertical (distribution model, run engine) | COMPLETE |
| 6 | Deterministic optimization harness (candidates, PIT, adapters, runner, scope pools) | COMPLETE |
| 7 | Certification: per-sport production runs, artifact schemas, frontend smoke, sealed gates | COMPLETE |
| 7.5 | Phase 7.5 remediation (behavior-neutral): fold/OOF consolidation into `core/folds.py`, versioned per-model contracts | COMPLETE (commit `86cff77`, tag `phase-7.5-complete`) |
| 7.5b | MLB adapter wiring + scope pool baseline (commit `70bec2e`, tag `phase-7.5-baseline`) — see history correction note below | NOT EXECUTED as a distinct phase: no 7.5b report or commit exists; `86cff77` is the Phase 7.5-complete commit (pre-7.5b state). `70bec2e` / `phase-7.5-baseline` is the MLB adapter *wiring* commit from the earlier authorized pre-step, not a 7.5b phase delivery |
| 7.5c | Guardrails: policy (`GUARDRAILS.md`), enforcement (`tests/core/test_spec_guardrails.py`), audit, cleanup, governance | COMPLETE (this phase) |
| 7.5b | Canonical per-sport contract ownership: registry-owned versioned contracts, study_config import+bind, legacy symbol removal, adapter rewiring (B-008 resolved) | EXECUTED (pending commit) |
| 7.5d | Blocker remediation: B-001…B-007, B-009 (see blocker table below; B-008 closed in 7.5b) | PENDING |
| 7.6a | Repository structure alignment (per approved Phase 7.6 plan; not started) | PENDING |
| 7.6b | Structure alignment verification + follow-through | PENDING |
| 7.5c1 | Tests lockdown: permanent-test governance, closed manifest (`tests/manifest.yaml`), hygiene enforcement (`tests/core/test_manifest_hygiene.py`), duplicate consolidation (test_markets_tie_push → test_markets; test_retention_production → test_retention), CI workflow, GUARDRAILS §16 amendment | EXECUTED (pending commit) |

## Byproduct ledger

| Byproduct | Created in | Disposition |
|---|---|---|
| `scripts_ops/phase75_*` evidence scripts + manifest | 7.5 (remediation, `86cff77`) | Scripts deleted (7.5c); manifest superseded in 7.5b by the permanent in-module evidence constants (`tests/core/test_fold_fingerprints.py`, `tests/core/test_matrix_hashes.py`) and deleted with `docs/audits/` |
| `scripts_ops/verify_{store,features}_equiv.py` | Phase 8 pre-cert | Deleted (7.5c) |
| `experiments/mlb_store_build.py`, `nba_cert_*`, `phase7_*.py`, `phase8_precert.py` | Phase 7/8 certification | Deleted (7.5c) |
| `experiments/extract_*_schema_fixtures.py`, `enrich_nfl_schema_fixtures.py` | schema-fixture extraction | Deleted (7.5c); fixture DATA lives permanently in `tests/fixtures/` |
| `sports/nba/data_delivery_run2/` | NBA certification run 2 | Deleted (7.5c) |
| `experiments/{run_experiment.py,run_training.py,configs/}` | — | Not yet required; only allowlisted entries permitted (guardrail-enforced) |

## Blocker table

Every blocker: id, source (audit finding), file:line or area, defect class,
suggested fix, assigned phase. Strict xfails in
`tests/core/test_spec_guardrails.py` link to these IDs.

### B-001 — PIT validator: build AND wire (corrected scope)
- **Source:** Task 1 audit 1/1b (`git grep 'validate_point_in_time|validate_sport_cutoff|validate_settlement_record|future_availability'`) — zero matches repo-wide. `core/validation/future_availability.py` does not exist.
- **Area:** prediction/artifact/settlement paths in `sports/*/`, `core/`.
- **Defect class:** hardened rule 1 — validators must exist AND be invoked on every applicable path.
- **Corrected scope:** **Build the PIT validator per spec §7.1 AND wire it into every applicable production path.** This is build+wire, not wire-only: no validator module currently exists to wire.
- **Assigned:** Phase 7.5d as build+wire.

### B-002 — retention family extraction misses stamped-SHAP and `.meta.json` families
- **Source:** Task 1 audit 2 — `core/retention.py::_family_name` (lines 175–189) returns the full stem for `nba_shap_game_20270105_0022600001.csv` (date-stamp-plus-id suffix) and `nba_run_engine_markets_20270105.meta.json` (stem retains `_20270105.meta`); both then fail the allowlist and become permanently protected.
- **Area:** `core/retention.py`, `core/config.py::DEFAULT_ALLOWLISTED_FAMILIES` (non-prefixed vs sport-prefixed family duality).
- **Defect class:** hardened rule 3 — retention must cover every dated artifact family, sport-prefixed included; 20-day window.
- **Suggested fix:** fix family extraction to strip trailing stamp/id segments and suffixes (`.meta`), unify the family allowlist (prefix-agnostic), set `frontend_days=20`, and add family-coverage tests against real emitted filenames for all four sports.
- **Assigned:** Phase 7.5d.

### B-003 — retention window is 10 days, policy hardened to 20
- **Source:** `core/config.py:31` `FRONTEND_ARTIFACT_RETENTION_DAYS = 10`.
- **Defect class:** hardened rule 3 — 20-day retention window.
- **Suggested fix:** change to 20 and update all affected retention tests.
- **Assigned:** Phase 7.5d.

### B-004 — `frontend_days`/allowlist consumers not verified per-path
- **Source:** Task 1 audit 2 cross-check: retention is invoked in all four runners (good), but no test asserts every emitted family is deletable — the coverage gap that hides B-002.
- **Defect class:** rule 2/4 — config fields need live, tested consumers.
- **Suggested fix:** add a runner-emission ↔ allowlist parity test per sport.
- **Assigned:** Phase 7.5d.

### B-005 — record-type validator alignment before write not evidenced
- **Source:** Task 1 audit 5/6 follow-up: artifact schema checks exist post-write (`tests/test_mlb_artifact_schemas.py`, `artifacts.py::_check_columns`), but no pre-write record-type validator gate is invoked on all write paths.
- **Defect class:** hardened rule 10.
- **Suggested fix:** invoke the record-type validator inside each artifact writer before the file hits the sink; test with a deliberately malformed record.
- **Assigned:** Phase 7.5d.

### B-006 — dependency manifest declares lower bounds only, no pinned/lock manifest (RESOLVED in Phase 7.5d)
- **Source:** Task 1 audit 3 — `pyproject.toml` and `requirements-dev.txt` use `>=` floors; no lock file committed.
- **Defect class:** hardened rule 9 — committed (pinned) dependency manifest.
- **Suggested fix:** commit a pinned/locked manifest (exact-pin requirements or lock file) and a CI-consistent install path.
- **Resolution (Phase 7.5d, Workstream 1):** `requirements-dev.txt` converted to exact `==` pins for all 14 dev/CI dependencies (pytest, pytest-cov, pandas, numpy, requests, duckdb, pyarrow, pybaseball, scikit-learn, lightgbm, xgboost, scipy, pyyaml, ruff), resolved from the certification environment. The CI install path (`pip install -r requirements-dev.txt` in `.github/workflows/tests.yml`) is unchanged and now consumes the pinned manifest. No `requirements.txt` exists (none needed — CI uses only requirements-dev.txt). The strict xfail was removed; `test_dependency_manifest_pins_versions` is a hard-passing test. `pyproject.toml` optional-dependency floors remain (they are advisory convenience groups, not the manifest enforced by rule 9).
- **Enforced by:** `tests/core/test_spec_guardrails.py::test_dependency_manifest_pins_versions` (hard test; scans every `requirements*.txt` at the repo root — any future manifest file is covered automatically).
- **Status:** CLOSED (Phase 7.5d, Workstream 1).
- **WS1 addendum (pre-commit review):** `pyproject.toml` dependency sections documented as advisory convenience floors (header note: authoritative manifest is `requirements-dev.txt`; CI installs only from it; sections are non-operative for governance). Pin test generalized from a hard-coded package list to a structural `==` check over all `requirements*.txt`. Clean-venv (Python 3.10.12) resolution check performed: all 14 pins downloadable from PyPI on a bare interpreter (wheel audit, no production sinks touched); full wheel materialization truncated only by sandbox disk space, not by any pin resolution failure.

### B-007 — external-source adapter (statcast fetch path) lacks recorded integration smoke
- **Source:** Task 1 audit 8 — `sports/mlb/ingestion.py:107-109` lazily imports the statcast fetcher; no recorded/sandbox smoke exercises the fetch path end to end.
- **Defect class:** hardened rule 7.
- **Suggested fix:** add a sandboxed integration smoke (recorded response or sandbox endpoint) proving the fetch path executes; never touching production sinks.
- **Enforced by:** `tests/core/test_spec_guardrails.py::test_external_adapters_have_recorded_integration_smoke` (strict xfail, B-007).
- **Assigned:** Phase 7.5d.

### B-008 — MLB scope bound bare `FEATURE_COLS`/`RUN_FEATURE_COLS`; MLB versioned contracts were dead declarations (RESOLVED in Phase 7.5b)
- **Original source (working tree == `86cff77`):** `core/optimization/adapters.py:359-361` imported the unversioned `FEATURE_COLS`/`RUN_FEATURE_COLS` from `sports.mlb.feature_registry` and bound both MLB scopes to them. The MLB versioned contracts had zero consumers outside the declaring module, and `sports/mlb/study_config.py` carried neither binding.
- **Masked verification:** Phase 7.5 Gate 5's adapter-binding verification was **incomplete/masked** — it verified the NFL/NBA/NHL scope bindings and MLB contract *content/identity* but did not check that MLB's adapter path bypasses the versioned contract names entirely.
- **Resolution (Phase 7.5b):** registries own explicit, independent versioned contracts; legacy `FEATURE_COLS`/`RUN_FEATURE_COLS` deleted repo-wide; MLB adapter rewired to the versioned contracts; MLBStudy now binds all four contract fields. Legacy tokens are guardrail-forbidden; the strict xfail is removed and the check is a hard-passing test.
- **Enforced by:** `tests/core/test_spec_guardrails.py::test_no_bare_feature_contract_import_by_adapters` (hard test) + `test_legacy_contract_symbols_absent_from_production` + `tests/test_scope_contracts.py` registry-source/bindings assertions.
- **Status:** CLOSED (Phase 7.5b).

### B-009 — `core/optimization/adapters.py` reaches into `sports.*` via function-local imports
- **Source:** `core/optimization/adapters.py` lines 152–154, 212–215, 283–285 (lazy `sports.*` imports; no module-level circularity, but inverted dependency direction).
- **Defect class:** GUARDRAILS §18 — core never imports `sports.*`.
- **Suggested fix:** either invert the dependency (sport-side adapter registration) or amend §18 through the section-21 procedure — with the decision recorded in the tracker.
- **Enforced by:** `tests/core/test_spec_guardrails.py::test_core_never_imports_sports_anywhere` (strict xfail, B-009); the module-level variant is enforced as a hard-passing test.
- **Assigned:** Phase 7.5d.

## Audit verdicts that produced no blockers

- **Probability conservation (audit 4):** PASS — `core/markets.py::normalize_push_probs/normalize_tie_probs` validate non-negativity and exact mass conservation; `ProbTriple` validates sum==1; NBA/NHL/NFL derivations split tie mass explicitly and document the convention (rule 5 satisfied).
- **Per-game grouping (audit 5):** PASS — event-frame expansion groups strictly per `game_id` in all three non-MLB sport feature builders and runners; MLB market derivation is per-game (run-engine per game pair).
- **Discarded outputs (audit 6):** PASS — market records and monitor outputs are persisted to the sink on every path found; no compute-then-discard sites.
- **Dead config fields (audit 7):** PASS — `regular_season_only`, `train_positions` absent; `rolling`/`expanding` occurrences are live walk-forward/expanding-train semantics and rolling-derivation candidates, not dead config.
