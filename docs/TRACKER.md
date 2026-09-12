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

### B-001 — PIT validator: build AND wire (corrected scope) — PARTIALLY REMEDIATED (Phase 7.5d, Workstream 4)
- **Source:** Task 1 audit 1/1b (`git grep 'validate_point_in_time|validate_sport_cutoff|validate_settlement_record|future_availability'`) — zero matches repo-wide. `core/validation/future_availability.py` does not exist.
- **Area:** prediction/artifact/settlement paths in `sports/*/`, `core/`.
- **Defect class:** hardened rule 1 — validators must exist AND be invoked on every applicable path.
- **Corrected scope:** **Build the PIT validator per spec §7.1 AND wire it into every applicable production path.** This is build+wire, not wire-only: no validator module currently exists to wire.
- **Assigned:** Phase 7.5d as build+wire.
- **Partial remediation (Phase 7.5d, Workstream 4 — CONDITIONALLY APPROVED as PARTIAL):**
  - `core/validation.py` converted to the package `core/validation/` (`__init__.py` re-exports the complete existing 11-symbol public surface verbatim; former module body moved to `core/validation/_legacy_module.py`; old module file removed). Import compatibility proven by the full battery with zero call-site edits.
  - `core/validation/future_availability.py` implements the §7.1-exact per-field gate `validate_available_at`: strict `available_at < prediction_cutoff` (**== FAILS**); `prediction_cutoff = scheduled game start − buffer`; missing/unparseable metadata **FAILS CLOSED** (counted in `n_missing` with representative game IDs; zero fabricated timestamps); buffers restricted to {60, 90}.
  - Cutoff buffers come from the new `prediction_cutoff_buffer_minutes` key in each sport's `study.yaml` **`retention`-adjacent top-level section** (top-level keys, after `training`/`market`): MLB 60, NHL 60, NFL 90, NBA 90 — hard-pinned in all four `study_config.py` loaders (any other value raises; wording pins "must be 60/90 per §7.1").
  - W3 mode key `availability_enforcement: report_only | enforce` in all four `study.yaml` + `study_config.py` (7.5d default `report_only` everywhere; `enforce` rejected as reserved for the Phase 7.6 metadata-layer activation). MANDATORY TELEMETRY: every production run emits/logs the FutureAvailabilityReport (label, n_rows, n_missing, n_violations, representative missing IDs) in BOTH modes via the four runner call sites.
  - SECONDARY gate named `validate_slate_window` (every served start at/after the serving window's lower-bound anchor + no decided served rows); §7.1-reserved PIT names are not reused. Settlement sub-gate `validate_settlement_record` is a distinct semantic check (outcome vs tie/push allowance, scope vs market kind) — NOT part of the B-005 record-shape registry and NOT citable as PIT evidence.
  - Wiring (AST-verified ≥1 call site per sport in `tests/core/test_spec_guardrails.py`): `validate_available_at` — mlb/runner.py (slate), nfl/runner.py (slate), nhl/runner.py (slate), nba/runner.py (slate); `validate_slate_window` — all four runners (slate); `validate_settlement_record` — mlb/runner.py (decided OOF moneyline rows), nfl/runner.py (`_build_oof_market_rows` fair market rows), nhl/runner.py (OOF moneyline rows), nba/runner.py (OOF moneyline rows).
  - Tests: `tests/test_validation.py` (§7.1 suite: strict inequality incl. == cutoff FAILS, per-sport buffer math, fail-closed, non-spec buffer rejection, settlement semantics, enforcement-mode normalization); `tests/core/test_spec_guardrails.py` (per-sport call-site guardrails ×3 + study-config buffer binding guardrail).
- **COVERAGE GAP — B-001-RESIDUAL (OPEN, Phase 7.6):** no per-field `available_at` metadata exists anywhere in production today (raw-field catalogs `sports/*/catalog.py`, ingestion records, and feature metadata carry no availability timestamps; the runner gates run over rows whose `available_at` column is absent, so `n_missing` reports the gap every run by design). Per-field enforcement is therefore NOT yet real: the 7.5d delivery ships the validator, wiring, buffers, mode key, and telemetry, but B-001 stays OPEN until the human-approved Phase 7.6 metadata layer populates per-field `available_at` and the enforcement mode is activated. Phase-end certification must enumerate this residual explicitly.
- **WS4 anchor ruling (Phase 7.5d review closure, anchor corrected):** the require_future_start per-sport asymmetry proposal was REJECTED; the secondary gate now runs the future-start check UNIFORMLY in all four sports with a DETERMINISTIC, serving-window-derived LOWER-BOUND anchor (`window_anchor`, never wall clock), compared at the resolution each sport's start data supports (MLB instant via `start_time_utc`; NFL/NHL/NBA date-granularity), identical semantics live and replay. The anchor is the instant the served window OPENS (`00:00Z` of its first date): MLB passes `pred_window.primary_date()`, NFL/NHL/NBA pass `window.start_date` (the run window the slate rows are filtered to). A served start must be `>=` that anchor, which catches pre-window rows in every sport and, with the `home_win` decided check, decided rows too. An earlier revision anchored on the window's LAST date and rejected legitimate in-window rows (NBA fixture `2015-11-19` vs anchor `2015-11-20`); that revision is superseded. Per-sport guardrails: `tests/core/test_spec_guardrails.py::test_slate_window_anchors_are_window_derived_per_sport` (AST-pins the single-argument window-derived anchor per sport, the explicit per-sport `resolution=`/`start_col=`, and zero wall-clock calls in the anchor expression) and `tests/test_validation.py::test_s71_slate_gate_catches_pre_window_and_decided_in_every_sport` (pre-window AND decided rows caught in all four sports).
- **Status:** PARTIALLY REMEDIATED (Phase 7.5d, Workstream 4) — validator + secondary + settlement + buffers + guardrails in 7.5d; per-field `available_at` metadata layer = **B-001-RESIDUAL**, Phase 7.6.
- **Phase 7.6 start-timestamp normalization (TRACKER line, per WS4 review ruling):** instant-granularity game starts for the NFL/NHL/NBA slates (NFL `gameday`, NHL `game_date`, NBA `game_date` → true UTC start instants), at which point all four sports tighten the secondary gate to `resolution="instant"` and the §7.1 per-field gate compares at instant resolution uniformly. Assigned Phase 7.6 (7.6c); OPEN.
- **Phase 7.6 legacy-module split (TRACKER line, per WS4 review ruling):** `core/validation/_legacy_module.py` is an approved TEMPORARY import bridge only. During the Phase 7.6 restructure its body is split into the §2 validation submodules (leakage/schema/determinism/coverage/future_availability) and `_legacy_module.py` is deleted — it must not become permanent. Assigned Phase 7.6; OPEN.

### B-002 — retention family extraction misses stamped-SHAP and `.meta.json` families (RESOLVED in Phase 7.5d)
- **Source:** Task 1 audit 2 — `core/retention.py::_family_name` (lines 175–189) returns the full stem for `nba_shap_game_20270105_0022600001.csv` (date-stamp-plus-id suffix) and `nba_run_engine_markets_20270105.meta.json` (stem retains `_20270105.meta`); both then fail the allowlist and become permanently protected.
- **Area:** `core/retention.py`, `core/config.py::DEFAULT_ALLOWLISTED_FAMILIES` (non-prefixed vs sport-prefixed family duality).
- **Defect class:** hardened rule 3 — retention must cover every dated artifact family, sport-prefixed included; 20-day window.
- **Suggested fix:** fix family extraction to strip trailing stamp/id segments and suffixes (`.meta`), unify the family allowlist (prefix-agnostic), set `frontend_days=20`, and add family-coverage tests against real emitted filenames for all four sports.
- **Assigned:** Phase 7.5d.
- **Resolution (Phase 7.5d, Workstream 2):** `_family_name` now strips composite `.meta` suffixes and trailing 10-digit run-engine id segments (in addition to 8-digit stamps and team-vs `@` segments); `_artifact_stamp` strips `.meta` before stamp search (the `.meta.json` stem had hidden the date entirely). `DEFAULT_ALLOWLISTED_FAMILIES` expanded to the exact emitted dated families for all four sports (B-004 ruling C1); the stale alias families `nfl_moneyline_json` / `nfl_feature_json` / `nfl_markets` / `nfl_markets_monitor` removed (zero emitters; sink scan found zero historical files). Tests: `test_shap_family_name_extraction` (regression cases), `test_stamped_shap_and_meta_families_deletable` (actual deletability), `test_emitted_dated_families_are_all_allowlisted`, `test_allowlisted_families_are_emitted_or_contracts`, `test_stale_alias_families_removed`.
- **Enforced by:** `tests/test_retention.py` (parity tests, both directions).
- **Status:** CLOSED (Phase 7.5d, Workstream 2).

### B-003 — retention window is 10 days, policy hardened to 20 (RESOLVED in Phase 7.5d)
- **Source:** `core/config.py:31` `FRONTEND_ARTIFACT_RETENTION_DAYS = 10`.
- **Defect class:** hardened rule 3 — 20-day retention window.
- **Suggested fix:** change to 20 and update all affected retention tests.
- **Assigned:** Phase 7.5d.
- **Resolution (Phase 7.5d, Workstream 2):** window changed 10 → 20 per spec §23 (proposal STOP approved pre-implementation). `FRONTEND_ARTIFACT_RETENTION_DAYS = 20`; all four `study.yaml` `retention.frontend_days: 20`; all four sport study-config validators accept 20 only; docstrings/contract descriptions updated. Boundary operator (age == window retained / age > window deleted): `core/retention.py::plan_retention` line `if stamp >= cutoff: continue`. `test_ten_day_boundary_exact` renamed `test_twenty_day_boundary_exact` (zero old-name references; B-003-pending comment removed); `test_retention_cutoff` now pins cutoff 20260907−20 = 20260818. Repo-wide stale-literal sweep: all 12 live "10-day" literals updated; TRACKER historical quotes retained as audit record; GUARDRAILS.md and frontend/ had zero live 10-day literals (no §-amendment needed).
- **Enforced by:** `tests/test_retention.py::test_twenty_day_boundary_exact`, `test_retention_cutoff`; `tests/test_config.py::test_load_config_defaults_when_nothing_given`; `tests/test_mlb_training.py::test_study_yaml_pins_the_agreed_defaults`.
- **Status:** CLOSED (Phase 7.5d, Workstream 2).
- **Future hardening (recorded):** sport study validators should cross-check `core.config.FRONTEND_ARTIFACT_RETENTION_DAYS` instead of hard-coding the window literal.

### B-004 — `frontend_days`/allowlist consumers not verified per-path (RESOLVED in Phase 7.5d)
- **Source:** Task 1 audit 2 cross-check: retention is invoked in all four runners (good), but no test asserts every emitted family is deletable — the coverage gap that hides B-002.
- **Defect class:** rule 2/4 — config fields need live, tested consumers.
- **Suggested fix:** add a runner-emission ↔ allowlist parity test per sport.
- **Assigned:** Phase 7.5d.
- **Resolution (Phase 7.5d, Workstream 2):** audit found 23 emitted dated families not deletable (MLB `run_engine_markets`/`run_engine_monitor`; all `nfl_moneyline_v1`/`nfl_feature_v1`/`nfl_run_engine_*` true names; the entire NHL and NBA sport-prefixed sets — all frontend-consumed). STOP reported per standing condition; ruling **C1** approved: `DEFAULT_ALLOWLISTED_FAMILIES` expanded to the exact emitted families (+16 NHL/NBA/MLB entries) and the 4 stale alias entries removed (zero emitters, zero sink files — verified before removal). Bidirectional parity tests added: every emitted dated family must be allowlisted, and every allowlist entry must be an emitted family or a durable-contract name.
- **Enforced by:** `tests/test_retention.py::test_emitted_dated_families_are_all_allowlisted` + `test_allowlisted_families_are_emitted_or_contracts` + `test_stale_alias_families_removed`.
- **Status:** CLOSED (Phase 7.5d, Workstream 2).
- **Future hardening (recorded, targeted Phase 7.6):** auto-derive the retention allowlist from each sport's `ArtifactContract` frontend families (ruling C2), eliminating the hard-coded list.

### B-005 — record-type validator alignment before write not evidenced (RESOLVED in Phase 7.5d)
- **Source:** Task 1 audit 5/6 follow-up: artifact schema checks exist post-write (`tests/test_mlb_artifact_schemas.py`, `artifacts.py::_check_columns`), but no pre-write record-type validator gate is invoked on all write paths.
- **Defect class:** hardened rule 10.
- **Suggested fix:** invoke the record-type validator inside each artifact writer before the file hits the sink; test with a deliberately malformed record.
- **Assigned:** Phase 7.5d.
- **Resolution (Phase 7.5d, Workstream 3):** `core/record_validation.py` adds a per-record-type validation registry — 56 pinned entries (14 per sport: MLB 13 primaries + `run_engine_markets.meta`; each of NFL/NHL/NBA 10 artifact primaries + `.meta` companion + 3 runner-side OOF/fold stores). Keys are the EXACT emitted family names (legacy delivery-allowlist aliases `markets`/`markets_monitor` and the non-emitting prefixed `feature_drift`/`feature_coverage` keys for NFL/NHL/NBA are documented in the module docstring; those sports ship drift/coverage as fields inside `<sport>_model_monitor.json`). Every active writer invokes `validate_record(sport, family, record)` post-normalization and immediately pre-serialization: primary and `.meta.json` companions are validated independently; `RecordValidationError` raises before any bytes reach the sink (zero partial files). Semantics: required fields/keys, NaN/inf rejected (JSON-strict), `None` only in contract-declared nullable slots, declared probability fields plus the `p_*` column convention range-checked in closed [0, 1], probability-mass groups checked read-only (never renormalized), optional absent grid groups skipped, `games[]` card probability fields checked per card for `moneyline_v1`. Exclusions (by design): raw ingestion caches (`sports/mlb/ingestion.py` parquet writes — raw source payloads) and `core/storage.py` generic writers (zero production callers). Reserved durable names `model_history`/`model_version_history` have NO registry entries (zero production writers); any future writer must add one in the same change, enforced bidirectionally.
- **Enforced by:** `tests/core/test_spec_guardrails.py::test_registry_covers_every_active_writer_call_site` + `test_registry_entries_all_have_active_writers` (bidirectional AST alignment over the four `sports/*/artifacts.py` writer modules + the 9 designated runner-side OOF/fold sites), `test_registry_size_pinned` (56), `test_reserved_durable_names_have_no_registry_entries`; `tests/test_contracts.py` parametrizes valid + missing-field records over ALL 56 entries plus NaN/inf, nullability, probability-range, mass-group read-only, and game-card checks; writer-level zero-byte-on-failure proofs in `tests/test_mlb_artifacts.py`, `tests/test_nfl_runner.py`, `tests/test_nhl_runner.py`, `tests/test_nba_runner.py`.
- **Impact class:** INTENDED-IMPACT (approved): valid data is a no-op (all pre-existing artifact tests pass byte-identically); invalid records now fail the run loudly instead of being written.
- **Future hardening (C2, targeted 7.6):** auto-derive the retention allowlist from the ArtifactContract (eliminating the registry/allowlist/fixture triple-maintenance); sport study validators cross-check `core.config.FRONTEND_ARTIFACT_RETENTION_DAYS` instead of hard-coding the window literal. Reconcile the legacy delivery-allowlist aliases — `markets`/`markets_monitor` (MLB, resolve to the emitted `run_engine_markets`/`run_engine_monitor` files via the frontend globs) and the non-emitting prefixed `feature_drift`/`feature_coverage` keys for NFL/NHL/NBA (that data ships as fields inside `<sport>_model_monitor.json`) — against the emitted-family registry so the allowlist, the registry, and the schema fixtures share one source of truth.
- **Status:** CLOSED (Phase 7.5d, Workstream 3).

### B-006 — dependency manifest declares lower bounds only, no pinned/lock manifest (RESOLVED in Phase 7.5d)
- **Source:** Task 1 audit 3 — `pyproject.toml` and `requirements-dev.txt` use `>=` floors; no lock file committed.
- **Defect class:** hardened rule 9 — committed (pinned) dependency manifest.
- **Suggested fix:** commit a pinned/locked manifest (exact-pin requirements or lock file) and a CI-consistent install path.
- **Resolution (Phase 7.5d, Workstream 1):** `requirements-dev.txt` converted to exact `==` pins for all 14 dev/CI dependencies (pytest, pytest-cov, pandas, numpy, requests, duckdb, pyarrow, pybaseball, scikit-learn, lightgbm, xgboost, scipy, pyyaml, ruff), resolved from the certification environment. The CI install path (`pip install -r requirements-dev.txt` in `.github/workflows/tests.yml`) is unchanged and now consumes the pinned manifest. No `requirements.txt` exists (none needed — CI uses only requirements-dev.txt). The strict xfail was removed; `test_dependency_manifest_pins_versions` is a hard-passing test. `pyproject.toml` optional-dependency floors remain (they are advisory convenience groups, not the manifest enforced by rule 9).
- **Enforced by:** `tests/core/test_spec_guardrails.py::test_dependency_manifest_pins_versions` (hard test; scans every `requirements*.txt` at the repo root — any future manifest file is covered automatically).
- **Status:** CLOSED (Phase 7.5d, Workstream 1).
- **WS1 addendum (pre-commit review):** `pyproject.toml` dependency sections documented as advisory convenience floors (header note: authoritative manifest is `requirements-dev.txt`; CI installs only from it; sections are non-operative for governance). Pin test generalized from a hard-coded package list to a structural `==` check over all `requirements*.txt`. Clean-venv (Python 3.10.12) resolution check performed: all 14 pins downloadable from PyPI on a bare interpreter (wheel audit, no production sinks touched); full wheel materialization truncated only by sandbox disk space, not by any pin resolution failure.
- **WS1 process deviation (recorded post-hoc):** commit `4c7a01c` was made without a formally presented and approved validation report preceding it (the executing agent treated the earlier conditional WS1 approval as sufficient). The substantive WS1 content — exact `==` pins for all 14 dependencies, generalized `requirements*.txt` scan, pyproject floor≤pin guard, advisory-floor documentation, clean-venv (3.10.12) installability proof — was verified by battery before the commit and re-verified post-hoc. Future workstreams commit only after their report is explicitly approved.

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
