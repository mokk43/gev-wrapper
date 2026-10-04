# TypeSafe-compatible Decider service implementation plan

Status: synthesized from the accepted design on 2026-10-02; the user confirmed the testing seam on the same date. The nine-ticket breakdown was approved and published to the [local tracker](../.scratch/decider-service/issues/) on 2026-10-03. Implementation began with the model-catalog slice on 2026-10-03; this document continues to order the remaining work.

## Problem Statement

Applications using TypeSafe's System One API need to evaluate State through an externally managed llama.cpp server running decider-4b GGUF without rewriting their typed decision requests or response handling. The supplied remote adapters preserve relevant Decider inference behavior, but they do not provide a web service, native async HTTP, admission control, public validation, or deployment lifecycle management.

Wrapping the existing synchronous inference call in an async route would still block the event loop. A reliable service must also preserve the probability of every requested alternative, reject oversized evidence rather than silently truncating it, and distinguish actual Decider identity and backend work from Jev identity and hosted billing.

## Solution

Expose an async FastAPI service with TypeSafe-compatible decision and model-catalog endpoints. Validate named questions against the pinned TypeSafe HTTP contract, prepare model-matched Decider prompts, evaluate independent final answer slots through llama.cpp, and assemble typed answers with the existing calibration semantics.

Use bounded concurrency and a whole-request deadline, return all answers or a sanitized error, and report the actual Decider model and attributable backend token work. Keep GGUF weights on the remote backend and require matching tokenizer/configuration metadata in the service.

## User Stories

1. As an application developer, I want to submit the System One request shape, so that my existing typed decision integration needs minimal changes.
2. As an application developer, I want to evaluate several named questions against shared State, so that I can request related decisions together.
3. As an application developer, I want to mix Choice, Noul, and Score questions, so that I can express classifications, yes/no judgments, and rubric ratings in one request.
4. As an application developer, I want to submit text, objects, and arrays as State, so that structured evidence retains its meaning.
5. As an application developer, I want to use structured instructions and criterion descriptions where the HTTP contract allows them, so that I can express detailed decision policies.
6. As an application developer, I want optional and nullable question fields handled according to the HTTP contract, so that the adapter does not silently impose a different schema.
7. As an application developer, I want response names to match my question names, so that each answer remains attributable to its input.
8. As an application developer, I want Choice labels and probabilities preserved, so that I can compare all requested alternatives.
9. As an application developer, I want Noul as a probability of truth, so that I can choose my own action thresholds.
10. As an application developer, I want Score as the expected rubric level with its legend and probabilities, so that fractional ratings remain interpretable.
11. As an application developer, I want a valid one-level Score rubric handled explicitly, so that backend assumptions do not reject a valid HTTP request.
12. As an application developer, I want model-specific calibration preserved, so that the adapter does not redefine confidence.
13. As an application developer, I want independent questions unaffected by other questions being added or reordered, so that request composition does not change an existing decision.
14. As an application developer, I want oversized State or rubrics rejected clearly, so that I do not act on decisions made from discarded evidence.
15. As an application developer, I want field-specific validation errors, so that I can correct invalid requests.
16. As an application developer, I want configured model names and aliases discoverable, so that I can select a supported model.
17. As an application developer, I want an explicitly enabled compatibility alias accepted, so that an existing configured client can route to Decider.
18. As an application developer, I want the response to identify the actual model, so that model provenance is clear even when I submit an alias.
19. As an application developer, I want unsupported model names rejected, so that a typo cannot silently select different weights.
20. As an application developer, I want complete option probabilities or an error, so that incomplete backend coverage cannot produce misleading certainty.
21. As an application developer, I want all requested answers or a failure, so that I do not mistake a partial result for a complete decision.
22. As an application developer, I want slow inference bounded by a whole-request deadline, so that my application can recover predictably.
23. As an application developer, I want token counts attributable to my request and its retries, so that I can observe actual backend work.
24. As an application developer, I want the official SDK verified against the service, so that compatibility is demonstrated through a real client.
25. As a service operator, I want backend evaluations limited globally within the service process, so that multiple callers cannot exceed configured capacity.
26. As a service operator, I want bounded admission, so that overload cannot create an unbounded backlog.
27. As a service operator, I want async backend I/O, so that a slow request does not block unrelated callers.
28. As a service operator, I want cancelled requests to stop scheduling work, so that abandoned calls do not continue launching evaluations.
29. As a service operator, I want coverage retries bounded by capacity and the request deadline, so that obtaining missing probabilities cannot bypass service limits.
30. As a service operator, I want deployment artifacts and metadata pinned and validated, so that incompatible token IDs or temperatures fail before user inference.
31. As a service operator, I want backend capabilities checked at startup, so that incompatible probability, context, or accounting behavior is diagnosed clearly.
32. As a service operator, I want environment-based configuration, so that the same service can target the chosen deployment without embedding deployment values in code.
33. As a service operator, I want GGUF weights kept on llama.cpp, so that the adapter does not load a duplicate model.
34. As a service operator, I want loopback exposure by default and service authentication for network access, so that exposure is intentional.
35. As an application developer, I want my TypeSafe-format bearer key forwarded to the configured backend for my request, so that I can use backend-issued credentials without a second runtime credential. As a service operator, I want startup probe credentials kept separate from caller requests, so that probes cannot become an authentication fallback.
36. As a service operator, I want sanitized errors and body-free logs, so that diagnostics do not disclose user evidence or secrets.
37. As a service operator, I want connection resources closed during shutdown, so that service restarts do not leave client tasks or connections unmanaged.
38. As a maintainer, I want tests based on observable HTTP behavior, so that implementation refactoring does not force tests to mirror internal structure.
39. As a maintainer, I want controlled backend fixtures and an opt-in live check, so that failures can be reproduced while real deployment compatibility is verified separately.
40. As a maintainer, I want reproducible dependencies and verified setup instructions, so that another developer can run the finished service and its checks.

## Implementation Decisions

The accepted service design owns technical requirements. The ordered work packages below describe how to deliver them; unresolved deployment values and version-specific capabilities remain verification inputs rather than assumed facts.

### 1. Establish the contract and dependency baseline

- Capture a versioned TypeSafe HTTP contract and pin a compatible Decider dependency and tokenizer/config revision. Treat SDK definitions as interoperability evidence rather than broader public validation rules.
- Establish a reproducible Python environment and future service/test entry points. Reuse supplied adapter behavior as the baseline; do not load inference weights in the application.
- Define the deployment configuration boundary for backend address/build, model identity/release date, aliases, matching metadata, context capacity, concurrency, admission, request limits, and optional startup probe credentials. Use the documented local development fixture for explicit live tests; real-user runtime keys arrive in TypeSafe bearer headers.
- Complete when the contract fixture, dependency compatibility, and required-versus-default configuration are explicit. Actual backend artifact identifiers can remain required operator inputs until deployment is selected.

### 2. Build public validation and model resolution

- Introduce public request/response validation for all three question types and the model catalog. Preserve structured fields and original names, labels, and rubric descriptions.
- Resolve only configured model identities and aliases; expose actual Decider identity in results and truthful configured catalog metadata.
- Handle schema-valid degenerate rubrics explicitly and validate backend option capacity before evaluation. Keep invalid requests field-oriented and distinguish local validation from backend failure.
- Complete when contract checks cover mixed requests, structured fields, absent/null question fields, one-level Score, invalid types, empty questions, unknown models, and model-capacity errors.

### 3. Separate preparation, async evaluation, and assembly

- Adapt the existing synchronous orchestration into preparation, awaited evaluation, and assembly responsibilities. Prefer pinned upstream prompt, neutralization, calibration, and answer assembly logic over a parallel implementation.
- Plan independent rows with one final answer slot each, retaining configured isolated Score levels. Validate the complete prompt and prediction allowance without state truncation.
- Use a shared lifecycle-managed async HTTP client for native llama.cpp completion calls. Keep expensive synchronous preparation off the event loop and account for its work within the request deadline.
- Preserve raw token IDs, one prediction, pre-sampling log probabilities, neutral penalties, disabled sampling filters, and initially disabled prompt caching. Apply the model's fitted temperature once.
- Complete when controlled backend distributions pass through the public API to correct typed answers and a slow backend leaves unrelated requests responsive.

### 4. Add admission, deadlines, cancellation, and lifecycle

- Start with one worker and a global backend evaluation limit shared by callers, rows, and retries. Bound admitted work as well as active evaluations; queue/admission capacities are operator-configured.
- Apply the accepted 60-second deadline across admission, preparation, evaluation, retries, and assembly. Derive transport timeouts from remaining budget.
- Cancel pending tasks and close outstanding client requests on expiry or disconnect where supported; stop launching further rows. Do not equate connection closure with confirmed backend cancellation.
- Close the shared client and managed tasks on shutdown. Prevent inference traffic until configuration and backend capability validation succeeds.
- Complete when concurrent callers cannot exceed limits, overload is bounded, whole-request timeout covers queueing and multiple evaluations, and shutdown/cancellation release service resources. The next work package must verify that coverage retries use these same controls.

### 5. Make probability coverage and accounting reliable

- Match option tokens by ID and require every requested alternative. Start at top-256 coverage and expand only when required labels are missing, reaching full vocabulary when needed and supported.
- Verify vocabulary discovery, coverage caps, response shape, and usage-counter semantics against the chosen server build. Bound coverage attempts by the request deadline and global backend capacity.
- Keep token counters local to each logical request; include all attributable rows and coverage retries in a successful response. Treat unusable probabilities or required counters as backend-contract errors.
- Complete when missing-label recovery, exhausted coverage, invalid distributions, malformed counters, retries, and concurrent request accounting pass observable contract checks. Record the verified counter interpretation before claiming real backend accounting.

### 6. Finish the public operational boundary

- Bind to loopback by default; require TypeSafe-format caller bearer credentials on real-user requests and forward them only to the configured backend. Keep authorization local to each request and retry, with no shared-client mutation or runtime fallback to development/startup probe keys. Verify authenticated behavior for both public endpoints against the chosen backend.
- Return all answers or an error using the accepted validation/backend/unavailable/deadline status mapping. Sanitize public diagnostics and provide request identifiers without logging State, prompts, or credentials.
- Treat rejected forwarded caller keys as public authentication failures, while failed startup probe credentials are operator/readiness problems. Verify authentication statuses and non-validation error bodies against the pinned clients and backend instead of inventing hosted-platform fidelity.
- Complete when exposure, caller-key forwarding and isolation, startup/runtime credential separation, errors, and log redaction pass public behavior checks.

### 7. Verify interoperability and document operation

- Exercise an official TypeSafe client against the mock-backed service, including its API-key bearer header; run an opt-in live smoke check against the documented local development backend when available. Its supplied address/key do not prove tokenizer compatibility, counter semantics, or other capabilities.
- Compare representative GGUF decisions with a trusted Decider baseline using a justified numerical tolerance. Measure workload behavior before selecting larger concurrency or asserting latency targets.
- Replace planned setup notes with installation, launch, configuration, and verification commands that have actually been exercised.
- Complete when the applicable acceptance suite and SDK checks pass, live capability results or explicit unavailable checks are recorded, and the user-facing documentation describes the implemented service accurately.

## Testing Decisions

- **Confirmed primary seam:** exercise the public FastAPI HTTP boundary and replace the external llama.cpp interaction with a controlled HTTP transport or mock server. Keep real validation, model resolution, prompt construction, calibration, assembly, error handling, and accounting in the test path.
- **One controllable external boundary:** inject backend HTTP behavior through the application lifecycle rather than creating separate mocks for the preparation, scoring, and assembly modules. Model ordinary completion responses, missing option coverage, malformed payloads, latency, and metadata/counter variations at that boundary.
- **Observable assertions:** test response fields, probability semantics, status codes, SDK results, resource bounds, and completion/cancellation behavior. Inspect upstream HTTP requests only for externally meaningful requirements such as caller-key forwarding/isolation, startup/runtime credential separation, final-slot scoring, and coverage expansion; avoid asserting private methods or incidental call ordering.
- **Modules under test:** public validation/model resolution, decision orchestration, async backend adaptation, admission/lifecycle management, and authentication/error handling are exercised together through HTTP requests. Use fixed backend distributions to isolate adapter semantics from model variability.
- **Prior art:** the repository currently has no test suite or web-service test fixtures. The supplied adapters and accepted design provide inference expectations; they are not evidence of existing integration tests.
- **Contract and semantics:** cover all accepted request variants, names/labels/legend fidelity, valid one-level Score, expected Score values, calibration, option-ID coverage, and independence under question composition changes. Select numerical tolerances after checking pinned assembler rounding.
- **Async and failures:** use deterministic backend gates or events to test responsiveness and limits; avoid timing-only sleeps as the sole evidence. Exercise admission exhaustion, timeout across retries, client cancellation, startup incompatibility, sanitized failures, and lifecycle cleanup.
- **Accounting:** provide explicit backend counters for multi-row and retry cases; verify exact attributable totals and isolation across simultaneous callers. Verify chosen real-server counter semantics separately.
- **SDK and live validation:** run the official client against the mock-backed app, then reuse the same public HTTP seam for an opt-in real llama.cpp check. Mock success cannot establish tokenizer compatibility, quantization fidelity, or the chosen build's usage/coverage capabilities.

## Out of Scope

- Application implementation, dependency installation, commits, pushes, provisioning, or deployment as part of the current planning request.
- Loading or training model weights in the adapter, managing llama.cpp processes, or downloading a GGUF on the user's behalf.
- Free-text chat generation, streaming, packed multi-slot inference, additional inference providers, or multi-backend routing.
- Jev-equivalent accuracy guarantees, inherited logical token-accounting fidelity, or hosted TypeSafe billing, credit, rate-limit, and idempotency infrastructure.
- Unbounded queues, unbounded coverage retries, automatic state truncation, partial successful answer maps, fabricated probabilities, or unsupported-model fallback.
- A multi-worker scaling design, broader transient-error retry policy, or performance promises before measurements justify them.

## Further Notes

- [The accepted service design](service-design.md) remains the technical source of truth; this plan orders its implementation. [The glossary](../CONTEXT.md) supplies domain terms, and [ADR 0001](adr/0001-report-model-identity-and-backend-work.md) explains response provenance/accounting.
- The user supplied local development backend access and amended authentication
  to caller-key forwarding on 2026-10-03, then corrected the local port on
  2026-10-04. The accepted design records those inputs. Direct checks against
  the corrected endpoint are recorded in the verification report; the observed
  probability field and cache counter are incompatible with the accepted
  startup contract. Immutable GGUF provenance, matching metadata, configured
  public identity/release date, and a compatible build remain unresolved. Real
  deployment backend addresses are still operator inputs.
- No application or live-backend tests have run for this document. Planning verification is limited to document consistency, links, and formatting.
