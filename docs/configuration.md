# Service configuration

All service configuration uses `DECIDER_*` environment variables. Required
operator inputs have no defaults because the running llama.cpp artifact and
capacity have not been verified. Secret values must be supplied at runtime,
not committed.

## Required inputs

| Variable | Meaning |
| --- | --- |
| `DECIDER_BACKEND_URL` | Base URL of the external llama.cpp server. |
| `DECIDER_BACKEND_BUILD` | Exact deployed llama.cpp build identifier. |
| `DECIDER_GGUF_REVISION` | Immutable revision or digest of the served GGUF artifact. Floating values such as `main` and `latest` are rejected. |
| `DECIDER_GGUF_QUANTIZATION` | Quantization of the served GGUF artifact. |
| `DECIDER_METADATA_DIRECTORY` | Existing local directory containing matching tokenizer and Decider configuration metadata. |
| `DECIDER_METADATA_REVISION` | Immutable revision of the local metadata. Floating values are rejected. |
| `DECIDER_MODEL_NAME` | Truthful public identity of the configured Decider model. |
| `DECIDER_MODEL_DESCRIPTION` | Description returned by the catalog. |
| `DECIDER_MODEL_RELEASE_DATE` | Actual model release date in `YYYY-MM-DD` form. Future dates are rejected. |
| `DECIDER_CONTEXT_CAPACITY` | Verified backend context capacity. |
| `DECIDER_BACKEND_SLOTS` | Verified parallel evaluation capacity. |
| `DECIDER_ADMISSION_CAPACITY` | Maximum admitted requests. It must be at least the backend slot count. |
| `DECIDER_MAX_REQUEST_BYTES` | Maximum accepted `/v1/systemone` request-body size. |
| `DECIDER_MAX_QUESTIONS` | Maximum questions per `/v1/systemone` request. |
| `DECIDER_MAX_OPTIONS` | Maximum alternatives per question, from 2 through Decider's limit of 255. |

## Defaults and optional inputs

| Variable | Default | Meaning |
| --- | --- | --- |
| `DECIDER_MODEL_ALIASES` | `[]` | JSON array of deliberately enabled compatibility aliases. No alias, including `jev-latest`, is implicit. |
| `DECIDER_BIND_HOST` | `127.0.0.1` | Listen address. This catalog slice rejects non-loopback addresses until authentication and exposure checks are implemented. |
| `DECIDER_BIND_PORT` | `8000` | Listen port. |
| `DECIDER_REQUEST_DEADLINE_SECONDS` | `60` | Whole-request deadline reserved for the later deadline/admission slice; not yet enforced. |
| `DECIDER_INITIAL_PROBABILITY_COVERAGE` | `256` | llama.cpp `n_probs` used by Choice evaluation. Missing required labels currently fail with 502; recovery is a later slice. |
| `DECIDER_OPERATOR_PROBE_API_KEY` | unset | Optional startup-readiness credential. This slice does not send it or use it as a caller fallback. |

The documented local backend and key are test inputs, not defaults. Choice
requests forward their caller's bearer credential on each backend row without
mutating the shared client. Final authentication behavior for the catalog and
backend credential rejection remains part of the operational-boundary slice;
the service stays loopback-only. Creating the application establishes one
shared, lifecycle-managed backend HTTP client. No readiness request is made and
backend availability, counter semantics, and artifact compatibility remain
unverified.
