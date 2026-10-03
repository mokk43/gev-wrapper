# 03: Evaluate Noul alongside Choice

**What to build:** Let clients ask named yes/no questions as Noul probabilities, alone or alongside Choice questions against shared State. Preserve the HTTP contract's optional descriptions and instructions without turning Noul into a Boolean answer.

**Blocked by:** 02: Evaluate Choice asynchronously.

**Status:** ready-for-agent

**Implementation:** complete

- [x] Named Noul questions pass through the shared public HTTP, preparation, async evaluation, and answer-assembly path. Each answer preserves its submitted name and contains `type` and numeric `noul` within zero to one.
- [x] Accept omitted or null Noul criteria and an object containing optional `true` and `false` descriptions. Preserve instructions and structured descriptions wherever the pinned HTTP contract permits them; reject invalid field shapes with field-oriented 422 errors before inference.
- [x] Preserve pinned Noul prompt construction, option neutralization, and calibration, applying its configured temperature exactly once. Controlled option distributions produce the expected truth probability rather than a thresholded Boolean.
- [x] Mixed Choice/Noul requests return all requested answers or an error, identify the actual configured Decider model, and account for every attributable backend row. Preparation state, probabilities, and usage remain isolated across callers.
- [x] Public HTTP checks cover optional/null/structured fields, invalid inputs, calibrated values, name fidelity, mixed requests, and numerical tolerances justified by the pinned assembler. Adding or reordering independent questions leaves an existing question's answer unchanged.

## Comments

Implemented on 2026-10-03 through the shared public FastAPI and controlled
llama.cpp HTTP boundary. Empty Noul fields use Decider 1.8.1's pinned generic
Noul prompt because the captured HTTP contract permits both instructions and
criteria to be omitted or null. Mock-backed checks cover per-type calibration,
mixed-row accounting, structured prompt values, and concurrent caller isolation.
