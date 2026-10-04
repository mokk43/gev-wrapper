# Accepted service design

Status: accepted on 2026-10-02; local development inputs and caller-key
forwarding amended by the user on 2026-10-03; local backend port corrected by
the user on 2026-10-04. Implementation began with the model-catalog slice on
2026-10-03.

## Purpose and boundary

Provide an async Python FastAPI service between TypeSafe clients and one externally managed llama.cpp server running decider-4b GGUF. Preserve the TypeSafe HTTP request and response shapes while retaining Decider's prompt construction, option probabilities, and fitted calibration. Protocol compatibility does not promise Jev-equivalent predictions or calibration after quantization.

The accepted scope includes `POST /v1/systemone`, `GET /v1/models`, request validation, backend adaptation, service authentication, bounded admission, and error mapping. It excludes loading GGUF weights in the service, provisioning llama.cpp, free-text generation, streaming, and a claim to reproduce the hosted TypeSafe platform's billing or rate-limit system.

## Contract authority

[TypeSafe's HTTP OpenAPI](https://api.typesafe.ai/openapi.json), inspected on 2026-10-02, is authoritative for public payload shapes. The [official SDK types](https://github.com/typesafe-ai/typesafe-sdk-js/blob/main/src/types.ts) are an interoperability reference. Some SDK types are broader than HTTP OpenAPI, including nullable state; the service follows HTTP OpenAPI rather than silently widening the contract.

External schemas and package source can change. At implementation time, capture and pin the contract revision and compatible Decider dependency version. Resolve differences explicitly rather than assuming that the current branch still matches this design.

TypeSafe's HTTP contract, rechecked on 2026-10-03, supplies the API key through `Authorization: Bearer <API_KEY>` for both public endpoints. Preserve this header-based authentication rather than adding a credential field to the decision JSON body. In real-user operation, the supplied key authenticates the configured llama.cpp backend through the service, according to the credential policy below.

### Requests

`POST /v1/systemone` takes a JSON object with required `model`, `state`, and nonempty `questions` fields.

- `model` selects a configured model name or compatibility alias.
- `state` is text, a JSON object, or an array. HTTP OpenAPI does not permit null state.
- `questions` maps caller-chosen names to typed definitions. Support mixed Choice, Noul, and Score questions in the same request.
- Instructions may be text, objects, arrays, null, or omitted, as permitted by each question schema.
- Choice criteria are an object mapping labels to descriptions; descriptions may be text, objects, arrays, or null. An array of labels is an upstream Decider extension, not part of the accepted public contract.
- Noul criteria may be omitted, null, or an object with optional `true` and `false` descriptions.
- Score criteria are an ordered, nonempty array of text, object, or array descriptions. The inspected HTTP schema permits one level, while some SDK validation requires two. Handle a valid one-level rubric explicitly rather than letting a backend two-option assertion define the public contract.

Validate backend option capacity and configured request limits before inference. Inputs within the HTTP schema but beyond supported model capacity must receive a field-specific validation error. Preserve question names and criterion labels in the answer mapping.

### Successful responses

Return a JSON object with exactly the required contract structure: `model`, `answers`, and `usage`. Answer names match the submitted question names, and each answer's `type` matches its question.

- Choice: `type`, `choice`, `confidence`, and `probabilities` keyed by the original labels.
- Noul: `type` and numeric `noul` in the range zero to one.
- Score: `type`, `score`, `confidence`, `legend`, and `probabilities`. Legend and probability keys are string representations of zero-based rubric positions; legend values preserve the requested descriptions.
- Usage: integer `input_tokens` and `output_tokens`, using the accounting policy below.

Use upstream Decider's answer assembly and confidence semantics rather than introducing a new definition. In particular, Score represents the expected level, not the winning integer level. Require finite probabilities within zero to one and distributions that sum to approximately one; account for the pinned assembler's rounding when defining numerical tolerances.

`GET /v1/models` returns TypeSafe's `{models: [...]}` shape, describing configured names and accepted aliases. Metadata includes the contract's name, description, and release date fields; supply an actual configured release date rather than inventing one. Clearly identify aliases as routing to Decider.

## Model identity and artifacts

Use a single configured backend. Accept only explicitly configured names and aliases, allowing `jev-latest` when an operator deliberately enables it for existing clients. An alias does not identify the backend weights. Return the actual configured Decider model identity and reject unknown requested models with a validation error.

The local metadata directory must match the GGUF checkpoint: tokenizer vocabulary, option label token IDs, prompt layout, and Decider calibration configuration. Pin the model revision, quantization, tokenizer/config revision, and compatible Decider package version. Validate configuration and tokenizer compatibility before accepting inference traffic. For metadata explicitly declaring the native Qwen3.5 pretokenization regex, honor that regex and raw-text BPE behavior rather than the loader's older Qwen2 regex and NFC normalization. Compare decoding and encoding across the entire local vocabulary. A larger backend vocabulary is compatible only when every trailing backend ID detokenizes to empty with special-token rendering enabled; reject a smaller backend vocabulary or any visible extra entry. Probability coverage and token-ID bounds use the backend vocabulary size, including those verified empty entries. Do not substitute stock Qwen metadata or default calibration silently.

The supplied wrapper constructs its name from the configuration's version field. That name alone may not uniquely identify decider-4b or its artifact revision; implementation must expose an explicit, truthful model identity.

## Async inference flow

1. Authenticate and validate the complete request, resolve its model name, and apply admission limits.
2. Render state and questions using the pinned Decider implementation. Preserve criterion ordering, model-specific prompt layout, option neutralization, and isolated Score-level behavior from matching configuration.
3. Plan independent rows with `independent=True`. Every remote row must contain exactly one answer slot at the final prompt position. An isolated Score question may require one row per rubric level.
4. Check the entire rendered prompt, including question and rubric tokens plus the one-token prediction allowance, against the deployed backend's available context capacity. Reject excess rather than truncating state or allowing backend context shifting to discard evidence.
5. Submit token IDs through a shared, lifecycle-managed `httpx.AsyncClient` to llama.cpp's native `/completion` endpoint, with bounded concurrent backend evaluations.
6. Read the distribution at the final answer slot, match option tokens by ID, and apply the configured Decider temperature exactly once. Assemble the typed answers using the existing semantics.
7. Validate the complete response and return it only if every question succeeded.

The supplied engine is a reference for the payload and scoring math; its blocking `requests.Session` and sequential item loop do not satisfy native async I/O. The inherited `Decider.system_one` method is also synchronous. Adapt the preparation, awaited evaluation, and assembly boundary explicitly; replacing the engine with an async method without adapting its callers will not work.

Reuse upstream prompt and assembly logic where practical and pin any private interfaces used. Offload sufficiently expensive synchronous tokenization or preparation so it cannot monopolize the event loop. Keep request results and usage accounting local to each request instead of deriving them from shared mutable counters.

## Probability extraction and coverage

Preserve the existing raw-token `/completion` approach: one prediction, `temperature=-1`, pre-sampling probabilities, neutral penalties, disabled sampling filters, and non-streaming responses. Set `min_keep` equal to the requested `n_probs` coverage so the selected llama.cpp sampler must retain at least that many candidates; validate the returned coverage rather than assuming the request was honored. Start with prompt caching disabled, matching the supplied engine; changing caching later requires equivalence checks.

The expected native llama.cpp response contains
`completion_probabilities[0].top_logprobs`, with token IDs and log
probabilities. Ignore generated text as a source of decision answers. Relative
log probabilities can stand in for logits because the shared full-vocabulary
normalization term cancels when softmax is applied over the requested options.

Start coverage at 256 tokens. If required option tokens are absent, double the requested coverage for each retry and clamp the final attempt to the readiness-verified maximum. Use full-vocabulary coverage only when that maximum equals the verified vocabulary size. Every response must contain exactly the requested number of unique, in-vocabulary token probabilities at the single final slot. Log probabilities must be finite and nonpositive; returned probability mass may not exceed one by more than `1e-6`, and full-vocabulary mass must sum to one within the same absolute tolerance. Retry under the same admission limits and whole-request deadline. Never assign fabricated probability mass to absent options or return a distribution normalized over only the options that happened to appear.

The maximum coverage and vocabulary discovery method must be verified against the chosen llama.cpp build. The supplied engine assumes vocabulary size is available at `/v1/models` as `data[0].meta.n_vocab`; treat that as an unverified deployment capability. If required coverage cannot be obtained, fail the request with a backend error. Full-vocabulary responses can be large and increase latency.

## Token accounting

Report actual llama.cpp input and output token work across all rows and coverage retries attributable to the successful request. Repeated state processing counts repeatedly. Map verified backend counters into `usage.input_tokens` and `usage.output_tokens`; do not assume a field named `tokens_evaluated` has the intended accounting semantics without checking the pinned server version and caching behavior.

Use one documented interpretation of backend counters and ensure accounting
remains isolated across concurrent requests. A malformed or missing usage
counter is a backend-contract error, not a reason to invent zero usage. For the
selected native llama.cpp response, `tokens_cached` is the prompt length held
in the slot after evaluation, not the number of tokens reused. With prompt
caching disabled, require `timings.cache_n == 0`,
`timings.prompt_n == tokens_evaluated == tokens_cached`, and
`timings.predicted_n == tokens_predicted == 1`. Report `tokens_evaluated` and
`tokens_predicted` as request usage. Preparation-only results require no
backend token work; normal remote inference requests generate one scoring token
per evaluation attempt.

This intentionally differs from inherited Decider accounting, which reports logical shared input tokens and zero output tokens. See [ADR 0001](adr/0001-report-model-identity-and-backend-work.md).

## Limits, deadlines, and failures

Start with one service process/worker. Configure a global backend evaluation limit and bounded admission; share that limit across callers and independent rows. Multiple worker processes would each enforce a separate limit, so increasing worker count requires a coordinated capacity strategy.

Use a 60-second whole-request deadline covering admission, preparation, inference, coverage retries, and assembly. Per-call transport timeouts must fit within the remaining budget. Stop scheduling additional work after the deadline or caller cancellation; cancel pending tasks and close outstanding client requests where supported. A closed HTTP connection does not prove backend inference stopped.

Return all answers or an error; do not send partial successful answer maps. Error policy:

- 422: invalid public input, unsupported configured model, model-capacity violation, or oversized prompt. Use TypeSafe's field-oriented `detail` validation shape.
- 502: malformed backend response, unusable probability coverage, or invalid/missing required backend counters.
- 503: unavailable backend or exhausted admission capacity.
- 504: whole-request deadline expired.

Keep public backend errors sanitized; retain a request identifier and concise operational diagnostics. For decision requests, default timing logs to disabled and enable them through an environment or `.env` boolean setting. When enabled, emit monotonic wall-clock timings in milliseconds for admission, preparation, backend-slot waiting, every backend call and coverage attempt, backend-response validation, aggregate backend evaluation, answer assembly, and the complete request. Correlate timings with the public response's request identifier, report stage outcomes, and include only numeric workload metadata and validated optional backend prompt/prediction timings. Missing or malformed caller bearer credentials and backend rejection of the forwarded caller key are public authentication failures, distinct from unavailable transport or backend-contract failures. Rejection of an operator-supplied startup probe credential is instead an operator/readiness problem. Final non-validation error body details and authentication statuses must be checked against the pinned client and chosen backend behavior during implementation; do not report caller-key rejection as backend unavailability.

The coverage retries above are accepted. A broader transient-error retry policy is not established; avoid adding retries without accounting for the deadline and duplicate backend work.

## Configuration and deployment inputs

Use environment-based configuration. The user supplied the following local
development fixture on 2026-10-03 and corrected its port on 2026-10-04:

- Backend base URL: `http://127.0.0.1:5080`.
- Native completion endpoint: `http://127.0.0.1:5080/completion`, serving decider-4b.
- Local test API key: `llama5080`. Explicit local test clients send it as `Authorization: Bearer llama5080`; bounded startup probes may use it through operator configuration.

These are user-provided test inputs. Direct observations from 2026-10-04 are
recorded in the [verification record](verification.md); they do not establish
complete deployment compatibility. Keep the test key in
explicit local configuration and test invocations, not a hardcoded production
credential or runtime fallback. The remaining required deployment information
includes immutable GGUF provenance, matching metadata, public model identity
and release date, aliases, and a traceable backend build. A real deployment
still needs its own configured backend address.

Also configure backend concurrency, admission capacity, request limits, and an operator startup probe credential when backend authentication requires one. Their numerical values and variable names are implementation/deployment choices, not settled measurements. Expected traffic, typical question counts, and a measured latency target have not been supplied; the 60-second deadline is a guardrail, not a performance claim.

Bind to loopback by default. Real-user requests to both public endpoints require the caller's TypeSafe-format bearer key. Forward that key only to the configured llama.cpp backend, with credentials local to the request across every row and coverage retry. Do not compare it with a separate static service token, replace it with an operator runtime key, or fall back to the local test/probe key when it is missing or rejected. Keep per-request authorization out of shared client defaults and mutable global state so concurrent callers cannot exchange credentials. Omit request bodies from logs, redact credentials from diagnostics, and never place keys in State, prompts, or decision JSON.

The backend validates caller keys; verify its authentication behavior for both inference and the model catalog using supported, bounded checks without assuming an authentication endpoint. Startup checks have no caller request, so they may use a separately configured operator probe credential for metadata and synthetic readiness checks. That credential is confined to startup and never authenticates a real user's inference request in normal operation. The loopback-only manual identity-bypass mode may explicitly reuse it as a caller credential so a single local backend key can exercise the wrapper. A backend that cannot establish the required authenticated behavior must fail readiness rather than silently accepting arbitrary caller keys.

At startup, validate the model metadata, configured alias mapping, backend readiness and credential-validation capabilities, tokenizer compatibility, probability response shape, usage counters, and context capacity. Missing or incompatible requirements must produce actionable diagnostics and prevent serving inference until resolved. Scope probes to metadata and a bounded synthetic fixture; real user evidence is unnecessary for startup checks. An explicit loopback-only manual-development override may skip deployment-manifest hashes and backend identity, build, path, quantization, context, and slot comparisons. Health, authentication, tokenizer, probability, and counter probes still run. The override does not skip local tokenizer/config loading, runtime response validation, authentication, or request limits, and it provides no artifact-provenance evidence.

## Acceptance checks for future implementation

These are required checks, not tests already executed.

- Contract: valid mixed requests and structured descriptions yield the required answer shapes; names, labels, rubric descriptions, and configured identity survive conversion. Exercise nullable/omitted fields according to HTTP OpenAPI, one-level Score, invalid discriminators, empty questions, unknown models, and backend option limits.
- Semantics: controlled backend distributions produce the expected Choice, Noul, Score, and confidence values through the pinned assembler. Verify per-type calibration once, option-ID matching, isolated levels, rounding tolerance, and independence when questions are reordered or added.
- Backend: missing labels trigger bounded coverage retries; exhausted coverage, malformed distributions, wrong token IDs, incompatible tokenizers, missing counters, and context overflow fail explicitly. Compare representative GGUF fixtures with a trusted Decider inference baseline using a documented numerical tolerance.
- Async behavior: a slow backend does not block unrelated requests; concurrent work respects the global limit and bounded admission. Exercise cancellation and a deadline that includes queueing and multiple coverage attempts.
- Accounting: multiple rows, repeated state, retries, caching behavior, and concurrent callers produce attributable counters rather than shared totals.
- Operations: TypeSafe bearer credentials are required for real-user requests and forwarded to the configured backend, including coverage retries. Different concurrent caller keys remain isolated; startup probe credentials cannot substitute for missing or rejected caller keys. Verify catalog and inference authentication, caller-key rejection, unavailable-backend and shutdown behavior, and that failures/logs never expose State, prompts, or secrets.
- SDK interoperability: run an official client against a mock-backed service, then an opt-in live smoke check against the configured llama.cpp build. Record actual commands and results; mock checks alone do not prove real GGUF/server compatibility.

Future implementation should include a reproducible dependency manifest and setup/run instructions based on actual entry points. Keep performance claims pending until measurements exist.

## Evidence

- [Supplied wrapper](../decider_wrapper.py): model/config initialization and synchronous inherited inference seam.
- [Supplied engine](../remote_llamacpp_engine.py): raw token payloads, final-slot restrictions, probability parsing, and calibrated scoring.
- [Upstream Decider inference](https://github.com/Mapika/decider/blob/main/decider/infer.py): row preparation, synchronous orchestration, answer assembly, and logical token accounting.
- [Upstream prompt construction](https://github.com/Mapika/decider/blob/main/decider/prompt.py): option token labels, layouts, and state truncation that the service must avoid.
- [llama.cpp server contract](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md): native completion and probability controls.
- [decider-4b model card](https://huggingface.co/Mapika/decider-4b): model-specific calibration and metadata requirements.
