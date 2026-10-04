# Service configuration

All service configuration uses `DECIDER_*` environment variables. Required
operator inputs have no defaults because the running llama.cpp artifact and
capacity have not been verified. Secret values must be supplied at runtime,
not committed.

## Required inputs

| Variable | Meaning |
| --- | --- |
| `DECIDER_BACKEND_URL` | Base URL of the external llama.cpp server. |
| `DECIDER_BACKEND_BUILD` | Exact `build_info` expected from llama.cpp `GET /props`. |
| `DECIDER_BACKEND_MODEL_ID` | Exact loaded-model `id` expected from llama.cpp `GET /v1/models`. Configure a stable backend alias instead of accepting an incidental file name. |
| `DECIDER_BACKEND_MODEL_PATH` | Exact model path expected from llama.cpp `GET /props`. |
| `DECIDER_GGUF_REVISION` | Immutable revision or digest of the served GGUF artifact. Floating values such as `main` and `latest` are rejected. |
| `DECIDER_GGUF_QUANTIZATION` | Quantization of the served GGUF artifact. |
| `DECIDER_METADATA_DIRECTORY` | Existing local directory containing matching tokenizer and Decider configuration metadata. |
| `DECIDER_METADATA_REVISION` | Immutable revision of the local metadata. Floating values are rejected. |
| `DECIDER_MODEL_NAME` | Truthful public identity of the configured Decider model. |
| `DECIDER_MODEL_DESCRIPTION` | Description returned by the catalog. |
| `DECIDER_MODEL_RELEASE_DATE` | Actual model release date in `YYYY-MM-DD` form. Future dates are rejected. |
| `DECIDER_CONTEXT_CAPACITY` | Verified backend context capacity. |
| `DECIDER_BACKEND_SLOTS` | Maximum concurrent llama.cpp completion attempts in this service process. Configure it from verified backend capacity. |
| `DECIDER_ADMISSION_CAPACITY` | Maximum concurrent decision requests admitted in this service process. Exhaustion returns 503. It must be at least the backend slot count. |
| `DECIDER_MAX_REQUEST_BYTES` | Maximum accepted `/v1/systemone` request-body size. Declared and chunked oversized bodies are rejected before full application buffering. |
| `DECIDER_MAX_QUESTIONS` | Maximum questions per `/v1/systemone` request. |
| `DECIDER_MAX_OPTIONS` | Maximum alternatives per question, from 2 through Decider's limit of 255. |
| `DECIDER_MAXIMUM_PROBABILITY_COVERAGE` | Largest supported llama.cpp `n_probs` value. Startup probes this exact bound. Set it to the vocabulary size only when the selected build supports full-vocabulary probability output. |
| `DECIDER_OPERATOR_PROBE_API_KEY` | Backend credential used only for bounded startup checks. A public request presenting this value is rejected before backend work, and it never becomes a runtime caller fallback. |

## Defaults and optional inputs

| Variable | Default | Meaning |
| --- | --- | --- |
| `DECIDER_MODEL_ALIASES` | `[]` | JSON array of deliberately enabled compatibility aliases. No alias, including `jev-latest`, is implicit. |
| `DECIDER_BIND_HOST` | `127.0.0.1` | Listen address. Set a non-loopback address only for intentional network exposure. Every service start still requires the backend to enforce bearer credentials on model discovery and inference before traffic is accepted. |
| `DECIDER_BIND_PORT` | `8000` | Listen port. |
| `DECIDER_REQUEST_DEADLINE_SECONDS` | `60` | Whole-request deadline in seconds across admission, offloaded preparation, backend-slot waiting, evaluation, and assembly. Each backend transport timeout is limited to the remaining request budget. |
| `DECIDER_INITIAL_PROBABILITY_COVERAGE` | `256` | Compatibility input accepted only as `256`; each backend row always starts recovery at 256. Missing required option token IDs trigger the bounded schedule below. |
| `DECIDER_STARTUP_PROBE_ATTEMPTS` | `3` | Maximum readiness attempts against `GET /health`. |
| `DECIDER_STARTUP_PROBE_TIMEOUT_SECONDS` | `30` | Whole deadline for all startup compatibility checks. |
| `DECIDER_STARTUP_TOKENIZER_PROBE_CHUNK_SIZE` | `4096` | Token IDs per `/detokenize` request while comparing the complete local and backend vocabularies. Accepted range: 1 through 8192. |

## Deployment manifest

`DECIDER_METADATA_DIRECTORY` must contain `deployment-manifest.json`. The
manifest pins the local files and binds them to the configured deployment:

```json
{
  "metadata_revision": "<immutable tokenizer/config revision>",
  "gguf_revision": "<immutable GGUF revision or digest>",
  "gguf_quantization": "Q4_K_M",
  "model_name": "decider-4b-q4-k-m",
  "decider_config_version": "<decider_config.json version>",
  "decider_dependency_version": "1.8.1",
  "prompt_layout": "plain",
  "files": {
    "decider_config.json": "<sha256>",
    "tokenizer.json": "<sha256>",
    "tokenizer_config.json": "<sha256>"
  }
}
```

`files` must list every regular file below the metadata directory except the
manifest itself, using relative paths and lowercase SHA-256 digests. Startup
rejects missing, extra, or changed files. `decider_config.json` must explicitly
set `version`, `temperature`, `neutralize_none`, and `isolated_levels`; upstream
defaults are not accepted as deployment calibration. The manifest records the
resolved `plain` or `chat` prompt layout, including when older official metadata
expresses `plain` by omitting a layout field.

## Startup compatibility gate

Application lifespan completes only after all checks pass. The gate:

- loads and hashes the local tokenizer/configuration manifest and verifies the
  installed `decider-ai` version;
- requires `GET /health` readiness, then compares `/v1/models` and `/props`
  model identity, path, build, quantization, vocabulary, training/effective
  context, and slot capacity with configuration;
- compares every backend token ID with the local tokenizer through bounded
  `/detokenize` chunks, then re-tokenizes that complete vocabulary-derived
  corpus through `/tokenize` to check tokenization rules and all option-label
  IDs;
- proves that invalid bearer credentials are rejected by both `/v1/models` and
  `/completion`, while the operator probe credential is accepted;
- sends one synthetic two-option Decider row with prompt caching disabled and
  the configured maximum probability coverage, sets `min_keep` to that same
  bound, then validates final-slot probability shape, required option IDs, and
  counters. Runtime requests likewise set `min_keep` to their requested
  coverage.

For disabled prompt caching, the supported counter interpretation is
`tokens_evaluated == submitted raw prompt token count`,
`tokens_predicted == 1`, and `tokens_cached == 0` for the startup fixture.
Runtime responses also reject missing or nonzero `tokens_cached`; usage sums
the request-local input/output counters across rows. A
selected build that does not expose the documented `/v1/models`, `/props`,
`/detokenize`, `/tokenize`, and `/completion` shapes is unsupported and fails
startup.

## Runtime probability coverage recovery

Each backend row starts at 256. If an otherwise valid response omits a required
option token ID, the service doubles coverage for the next attempt and clamps
the final attempt to the readiness-verified
`DECIDER_MAXIMUM_PROBABILITY_COVERAGE`. For example, a verified maximum of 600
produces `256`, `512`, then `600`. The final attempt is full-vocabulary only
when that verified maximum equals the verified backend vocabulary size.

Every attempt requests matching `n_probs` and `min_keep` values. The response
must contain exactly one final-slot `probs` element and exactly the requested
number of `top_logprobs` entries. Each entry must have a unique integer token ID
within the verified vocabulary and a finite, nonpositive numeric log
probability. Per the accepted service design, returned probability mass may not
exceed 1 by more than `1e-6`; full-vocabulary coverage must sum to 1 within the
same absolute tolerance. Only a valid response that lacks at least one required
option token ID is retried; malformed coverage, token IDs, probabilities, or
counters fail with a sanitized 502 response.

Retries remain inside the admitted request. Each attempt acquires capacity from
the same global backend-slot limit and uses the same whole-request deadline.
Deadline or cancellation stops further attempts. On success,
`usage.input_tokens` and `usage.output_tokens` sum the verified backend counters
from every attempt, including attempts whose valid coverage omitted required
option IDs. These runtime guarantees are covered by controlled fixtures; no
live backend result is recorded in this repository.

The documented local backend and key are test inputs, not defaults. Both public
operations require one syntactically valid TypeSafe bearer credential. Model
catalog requests forward it only to backend `GET /v1/models`; ordinary decision
requests forward it on each backend row and coverage retry. Preparation-only
decisions validate it through the backend catalog. Per-request headers do not
mutate the shared client. Missing, malformed, rejected, and operator-probe
credentials never fall back to the development or probe key.

Backend 401 and 403 responses are returned as sanitized public authentication
failures with the same status. The pinned client maps those statuses to its
authentication and permission error types and reads textual `detail` fields.
The pinned OpenAPI documents bearer security but no non-validation error schema,
so the service makes no hosted-platform body-fidelity claim. Public validation
failures contain only `loc`, `msg`, and `type`; backend-contract, unavailable,
admission, and deadline failures use fixed bodies. Every public response carries
`x-typesafe-request-id`. Operational failure logs contain that identifier,
method, path, status, and a fixed category, never request bodies, State, prompts,
credentials, upstream bodies, or exception text.

Creating the application establishes one shared, lifecycle-managed backend HTTP
client. Backend evaluations and admitted requests are bounded per process by
the configured capacities. Controlled fixtures prove gate behavior; they do not
prove that any actual selected deployment is compatible. Each real service
start performs the checks against its configured backend, and live results must
be recorded during final operational verification.

Run one service worker. Each additional worker would create independent
admission and backend-slot limits; multi-worker operation requires coordinated
capacity management that this service does not implement. Deadline expiry,
caller cancellation or disconnect, and shutdown cancel pending asyncio work and
release per-process capacity. Cancelling an outstanding HTTP request closes the
client-side operation but does not prove that llama.cpp stopped inference.
Python cannot stop preparation or assembly already executing in a worker thread;
the request stops awaiting that work, but it continues to occupy one of the
`DECIDER_ADMISSION_CAPACITY`-bounded offload slots until completion. Shutdown
awaits all tracked offloads before closing the service lifecycle.
