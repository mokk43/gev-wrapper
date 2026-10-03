# 04: Evaluate Score rubrics

**What to build:** Let clients evaluate ordered Score rubrics and receive an interpretable expected level, confidence, legend, and level probabilities. Support valid one-level rubrics and the model configuration's isolated-level behavior, including requests that also contain Choice questions.

**Blocked by:** 02: Evaluate Choice asynchronously.

**Status:** ready-for-agent

- [ ] Named Score questions use the shared public HTTP, preparation, async evaluation, and upstream assembly path. The numeric score is the probability-weighted expected zero-based level, including fractional values, rather than the winning integer level; confidence follows the pinned assembler's semantics.
- [ ] Accept ordered nonempty rubric arrays with text, object, or array descriptions and instructions wherever the pinned HTTP contract permits them. Preserve every rubric description in `legend`; legend and probability keys are string representations of zero-based rubric positions. Invalid rubric shapes, option-capacity violations, and complete-prompt context overflow receive field-oriented 422 errors.
- [ ] Preserve matching model layout, option neutralization, per-type calibration applied exactly once, and configured isolated Score-level behavior. Every remote evaluation row has one final answer slot, even when a Score question requires several rows.
- [ ] Handle a schema-valid one-level rubric explicitly with correct score, confidence, legend, and probability semantics, rather than relying on a backend two-option assertion. Document the chosen handling against the pinned assembler; preparation-only results report zero backend work, while any actual remote evaluations count their attributable work.
- [ ] Score requests, including mixtures with Choice, return complete answer maps or an error. Usage includes all attributable isolated-level rows and repeated State processing, and concurrent callers cannot share preparation state, results, or counters.
- [ ] Public HTTP checks use controlled distributions to verify fractional expected scores, confidence, descriptions, isolated-level row behavior, valid one-level rubrics, structured fields, invalid inputs, accounting, and independence under added or reordered questions. Justify numerical tolerances using pinned assembler rounding.
