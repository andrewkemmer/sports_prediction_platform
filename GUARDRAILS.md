# GUARDRAILS.md — Platform Guardrails Policy

Status: ACTIVE (Phase 7.5c)
Authority: this file is the binding policy for all development phases.
Enforcement: `tests/core/test_spec_guardrails.py` (structural/AST checks).
Violation ledger: `docs/TRACKER.md` (blockers by ID, assigned by phase).

---

## 1. Purpose

This policy defines the non-negotiable structural and behavioral guardrails
for the sports prediction platform. It exists so that correctness properties
proven in certification (leakage freedom, settlement honesty, artifact
integrity) remain true as the platform evolves, and so that every exception
is visible, tracked, and scheduled for remediation rather than silently
absorbed.

## 2. Scope

These guardrails apply to all production code (`core/`, `sports/`,
`frontend/`), all tests (`tests/`), all experiment scaffolding
(`experiments/`), and all repository configuration. They bind every
development phase and every contributor, human or automated.

## 3. Definitions

- **Production code** — modules imported by the four sport runners, the
  optimization harness, or the dashboard: `core/`, `sports/`, `frontend/`.
- **Byproduct** — a file produced by a one-time certification or
  certification-support run (scripts, generated stores, verification
  outputs). Byproducts never live in production trees.
- **Blocker** — a recorded violation of a hardened rule, assigned a unique
  ID (B-001…), a defect class, and a remediation phase. Recorded in
  `docs/TRACKER.md`.
- **Guardrail test** — a test in `tests/core/test_spec_guardrails.py` that
  enforces a structural rule of this policy. A guardrail test may carry a
  strict xfail ONLY when linked to a blocker ID.

## 4. Rule hierarchy

Hardened rules (Section 6) outrank convenience, velocity, and historical
convention. Where a rule and an existing implementation conflict, the
conflict is a blocker — never a reinterpretation of the rule.

## 5. How to read the hardened rules

Each rule is a testable invariant. "Must" is mandatory; "should" is
default-and-derogable only with a tracked blocker. Every rule maps to at
least one guardrail test or to a documented certification step.

## 6. The 13 hardened rules

1. **Mandatory validators on every applicable path.** Point-in-time,
   sport-cutoff, and settlement-record validators must have real production
   call sites on every path that produces predictions, artifacts, or
   settlements — definitions without invocations are violations.
2. **No silent no-op config branches.** Every configuration field must have
   a live consumer. A field read by nothing, or a branch that exists only
   to accept and ignore a value, is a violation.
3. **Retention covers every sport-prefixed artifact family, 20-day.** The
   rolling retention policy must match every dated artifact family the
   runners emit — including sport-prefixed names — with a 20-day window;
   durable contracts remain exempt.
4. **Config fields need live consumers.** (Reinforcement of 2 at the study
   level: every study.yaml key must be bound into a dataclass field that a
   production module reads.)
5. **Probability simplex conservation.** Home/away(/tie/push) probabilities
   must sum to 1 with no silently dropped mass; renormalization is explicit
   and validated at the market boundary.
6. **No compute-then-discard.** Computed outputs (market records, monitor
   metrics, distributions) must be either persisted or explicitly declared
   as diagnostic; computing then dropping is a violation.
7. **Real-source adapters need recorded integration smokes.** Any adapter
   that fetches from an external source must have a recorded or sandboxed
   integration smoke test proving the fetch path executes end to end.
8. **Settlement/push/tie semantics from market_rules.yaml.** Settlement
   scope, push, and tie semantics come from the sport's market rules
   configuration — one market type per record, never hard-coded per sport.
9. **Committed dependency manifest.** All runtime and dev dependencies are
   declared in committed manifests (`pyproject.toml`,
   `requirements-dev.txt`); no undeclared imports in production code.
10. **Record-type validator alignment before write.** The record-type
    validator for each artifact family runs against the frame or object
    before it is written to the sink.
11. **Batch transforms tested with ≥2 distinct games.** Any transform that
    processes multiple games in one call is tested with at least two
    distinct games so grouping, ordering, and cross-game leakage are
    exercised.
12. **Approval requires inspectable artifacts, never narrative alone.** A
    phase is approved only on artifacts (fingerprint manifests, matrix
    hashes, test outputs, run logs) that a reviewer can independently
    re-derive — not on narrative claims.
13. **Violations never hidden.** Guardrail tests are never weakened,
    deleted, or skipped to make a phase pass. A failing guardrail is either
    fixed or recorded as a blocker with a remediation phase — nothing else.

## 7. Mandatory validator coverage

Validators that guard point-in-time correctness, sport slate cutoffs, and
settlement-record shape must be defined AND invoked on every applicable
production path (features, training, artifact generation, settlement).
The guardrail test verifies at minimum that each mandatory validator has at
least one production call site; coverage-per-path is verified in
certification.

## 8. Configuration integrity

Configuration is honest or it is not configuration. Fields that exist in
study/config objects must be consumed; branches that normalize-and-ignore
are removed or wired. Dead config fields are blockers under rule 2/4.

## 9. Artifact retention

Retention is policy-driven, allowlisted, dry-run by default, and executed
only after successful generation. The window is 20 days (hardened rule 3;
the historical 10-day default is a tracked deviation). Family extraction
must match every emitted filename family, sport-prefixed families included.

## 10. Probability and settlement honesty

Probabilities conserve mass explicitly (rule 5). Push and tie are
first-class masses with display buckets; they are never silently dropped or
forced to zero. Settlement reads its scope (regulation vs full-game),
tie-allowed, and push-allowed from the sport's market rules (rule 8), one
market type per record.

## 11. Compute discipline

Outputs of expensive computation are persisted or explicitly labeled.
"Compute-then-discard" in production paths is a blocker (rule 6).

## 12. External adapters

Any adapter fetching from an external source (statcast-like APIs, schedule
scrapes) carries a recorded or sandboxed integration smoke (rule 7). The
smoke proves the fetch path executes; it never mutates production sinks.

## 13. Dependencies

`pyproject.toml` and `requirements-dev.txt` are the committed dependency
manifests (rule 9). Production imports are restricted to the standard
library plus declared dependencies. Undeclared imports fail review.

## 14. Repository structure

Top-level directories are exactly: `core/`, `sports/`, `frontend/`,
`experiments/`, `tests/`, `docs/`, plus data/config directories the
platform already owns (`data_delivery/`, `store/`, and existing packaging
metadata). No new top-level directories without a phase plan that names
them. One-off certification scripts never live in `scripts_ops/` or
similar ad-hoc locations: `scripts_ops/` is removed by Phase 7.5c and is
absent thereafter.

## 15. Experiments directory

`experiments/` holds only the durable, allowlisted scaffolding:
`run_experiment.py`, `run_training.py`, `configs/`, `__init__.py` (entries
need not exist yet, but nothing else may). Certification byproducts are
deleted at the end of the phase that created them (see `docs/TRACKER.md`
for the byproduct ledger). Production code never imports `experiments/`;
`experiments/` may import production code.

## 16. Tests directory

`tests/` is flat and permanent: only `__init__.py`, `conftest.py`,
per-sport fixture modules (`*_fixtures.py`), `fixtures/`, and `test_*.py`
files at the top level. Filenames carrying phase, certification, ablation,
probe, or sweep markers (`*phase*`, `*cert*`, `*ablation*`, `*probe*`,
`*sweep*`) are forbidden — one-off tests do not belong in the permanent
suite. A `tests/core/` subpackage is reserved for structural guardrail
tests.

## 17. Naming and identity discipline

Feature contracts are scope-specific and versioned: no production module
may consume a bare, unversioned `FEATURE_COLS`/`FEATURE_COLUMNS` symbol as
its model contract (see Phase 7.5 contracts). The guardrail test enforces
this at the module level.

## 18. Import direction

Production trees (`core/`, `sports/`, `frontend/`) never import
`experiments/`. `experiments/` may import production modules. `core/` never
imports `sports.*` (no circular dependencies).

## 19. Evidence over narrative

Every phase report must link to artifacts a reviewer can re-derive:
command invocations, exit codes, file hashes, test counts. Narrative
claims without inspectable artifacts do not constitute approval evidence
(rule 12).

## 20. Violation handling

A guardrail failure is never hidden (rule 13). The options are exactly:
(a) fix the violation in-phase when in scope; (b) record a blocker with a
unique ID in `docs/TRACKER.md`, assigned to a named remediation phase,
with defect class and suggested fix. Strict xfails in guardrail tests are
permitted only while their blocker is open and must name the blocker ID.

## 21. Policy change procedure

Changing a hardened rule, adding a rule, or closing a blocker requires a
phase plan that names the change, the evidence that motivates it, and an
updated guardrail test. Silent edits to this file are void.

---

## Implementation Governance

- **One phase at a time.** Each phase executes exactly its approved scope;
  out-of-scope fixes are blockers, never drive-by edits.
- **Evidence over narrative.** Approval is granted on inspectable artifacts
  (rule 12), never on summaries alone.
- **Never advance past a known violation.** A phase with an open, unassigned
  violation cannot be declared complete; violations are recorded in
  `docs/TRACKER.md` and carried forward until closed.
- **Tracker.** The phase ledger and blocker table live in
  `docs/TRACKER.md`; every blocker cited in reports and strict xfails must
  exist there by ID.
- **Vendor neutrality.** This policy names no vendor, model, or service
  brand; capabilities are described functionally.
