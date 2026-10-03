# 04: Evaluate Score rubrics

**What to build:** Let clients evaluate ordered Score rubrics and receive an interpretable expected level, confidence, legend, and level probabilities. Support valid one-level rubrics and the model configuration's isolated-level behavior, including requests that also contain Choice questions.

**Blocked by:** 02: Evaluate Choice asynchronously.

**Status:** ready-for-agent

**Implementation:** complete

- [x] Named Score questions use the shared public HTTP, preparation, async evaluation, and upstream assembly path. The numeric score is the probability-weighted expected zero-based level, including fractional values, rather than the winning integer level; confidence follows the pinned assembler's semantics.
- [x] Accept ordered nonempty rubric arrays with text, object, or array descriptions and instructions wherever the pinned HTTP contract permits them. Preserve every rubric description in `legend`; legend and probability keys are string representations of zero-based rubric positions. Invalid rubric shapes, option-capacity violations, and complete-prompt context overflow receive field-oriented 422 errors.
- [x] Preserve matching model layout, option neutralization, per-type calibration applied exactly once, and configured isolated Score-level behavior. Every remote evaluation row has one final answer slot, even when a Score question requires several rows.
- [x] Handle a schema-valid one-level rubric explicitly with correct score, confidence, legend, and probability semantics, rather than relying on a backend two-option assertion. Document the chosen handling against the pinned assembler; preparation-only results report zero backend work, while any actual remote evaluations count their attributable work.
- [x] Score requests, including mixtures with Choice, return complete answer maps or an error. Usage includes all attributable isolated-level rows and repeated State processing, and concurrent callers cannot share preparation state, results, or counters.
- [x] Public HTTP checks use controlled distributions to verify fractional expected scores, confidence, descriptions, isolated-level row behavior, valid one-level rubrics, structured fields, invalid inputs, accounting, and independence under added or reordered questions. Justify numerical tolerances using pinned assembler rounding.

## Comments

Implemented on 2026-10-03 through the shared public FastAPI and controlled
llama.cpp HTTP boundary. Multi-level rubrics retain Decider 1.8.1's configured
isolated or non-isolated layout, type calibration, and upstream assembly. A
one-level rubric is answered during preparation with level 0 probability 1,
expected score 0, confidence 1, and zero backend usage, matching the pinned
assembler formulas without entering its two-level prompt assertion. Tests use
the assembler's four-decimal probability/confidence output and two-decimal
expected-score output as the numerical boundary.
