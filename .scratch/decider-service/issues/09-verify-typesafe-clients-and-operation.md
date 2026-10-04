# 09: Verify TypeSafe clients and operation

**What to build:** Demonstrate the complete service through an official TypeSafe client and publish operation instructions backed by exercised commands. Distinguish proven HTTP compatibility from deployment-dependent GGUF, tokenizer, probability, accounting, and workload results.

**Blocked by:** 03: Evaluate Noul alongside Choice; 04: Evaluate Score rubrics; 07: Recover complete option probabilities; 08: Authenticate callers and sanitize failures.

**Status:** ready-for-agent

**Implementation:** complete

- [x] Run the accepted public HTTP acceptance suite with the controlled backend boundary while retaining real validation, model resolution, preparation, calibration, assembly, admission, errors, and accounting. Cover mixed Choice/Noul/Score requests, structured/optional/null fields, one-level Score, names/labels/legends, aliases and actual identity, capacity/context errors, independence under question composition changes, and rounding-aware numerical tolerances.
- [x] Verify complete coverage recovery and failure, multi-row/retry token totals, concurrent request isolation, event-loop responsiveness, global evaluation limits, bounded overload, whole-request deadlines, cancellation, startup incompatibility, caller-key forwarding, probe/runtime credential separation, redaction, and shutdown behavior. Concurrent callers with different keys retain their own key on every row and coverage retry; missing, malformed, or rejected caller keys cannot fall back to operator or development credentials. Tests assert observable behavior rather than private method calls or incidental ordering.
- [x] Exercise a pinned official TypeSafe client against the mock-backed service for model discovery, representative typed decisions, mixed requests, and error/authentication handling. Verify that the SDK's `apiKey` becomes the expected bearer header, uses the RFC 6750 `b64token` character set accepted by the service, and reaches only the configured backend. Confirm that an issued live TypeSafe key uses that character set; if it does not, record the observed format before changing the authentication boundary. Resolve client-versus-HTTP-schema differences explicitly without widening the accepted contract, including null State and one-level Score behavior.
- [x] Provide an opt-in live smoke check targeting the user-supplied local decider-4b development fixture documented in the accepted design when available; use its test key explicitly in test requests. The supplied endpoint and key do not establish current availability, server build, artifact revision, quantization, tokenizer compatibility, context, counters, or a trusted baseline. Record verified selected-build authentication enforcement, tokenizer, probability-coverage, context, counter, and cache behavior; compare representative GGUF decisions with a trusted Decider baseline using a documented and justified numerical tolerance. Do not provision a backend or download inference weights as part of this ticket.
- [x] Record actual commands and results for each verification. If live deployment inputs or a trusted baseline are unavailable, identify the exact unavailable checks and leave real compatibility/quantization claims unverified; mock success alone is not proof of the selected deployment.
- [x] Replace planned setup notes with dependency installation, single-worker launch, environment configuration, authentication, shutdown, contract-test, SDK, and optional live-check instructions whose applicable entry points have been exercised. Keep unresolved operator inputs explicit and describe only implemented behavior.
- [x] Measure representative workload behavior before increasing concurrency or making latency claims. If deployment/workload measurements are unavailable, document that limitation; the accepted 60-second deadline remains a guardrail rather than a measured performance target.

Controlled HTTP and official SDK verification completed on 2026-10-04. The local
backend at `127.0.0.1:5080` was reachable and enforced authentication, but its
probability field and cache counter were incompatible with the accepted startup
contract. No separate caller key, issued TypeSafe key, trusted Decider baseline,
or workload fixture was available. The
exact unverified deployment and performance checks are recorded in
[`docs/verification.md`](../../../docs/verification.md); no live compatibility,
quantization-equivalence, or latency claim is made.
