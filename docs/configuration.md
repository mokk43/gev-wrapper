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
| `DECIDER_MAX_REQUEST_BYTES` | Maximum accepted request size for later decision endpoints. |
| `DECIDER_MAX_QUESTIONS` | Maximum questions per later decision request. |
| `DECIDER_MAX_OPTIONS` | Maximum alternatives per question, from 2 through Decider's limit of 255. |

## Defaults and optional inputs

| Variable | Default | Meaning |
| --- | --- | --- |
| `DECIDER_MODEL_ALIASES` | `[]` | JSON array of deliberately enabled compatibility aliases. No alias, including `jev-latest`, is implicit. |
| `DECIDER_BIND_HOST` | `127.0.0.1` | Listen address. This catalog slice rejects non-loopback addresses until authentication and exposure checks are implemented. |
| `DECIDER_BIND_PORT` | `8000` | Listen port. |
| `DECIDER_REQUEST_DEADLINE_SECONDS` | `60` | Whole-request deadline for later decision requests. |
| `DECIDER_INITIAL_PROBABILITY_COVERAGE` | `256` | Initial llama.cpp probability coverage for later inference. |
| `DECIDER_OPERATOR_PROBE_API_KEY` | unset | Optional startup-readiness credential. This slice does not send it or use it as a caller fallback. |

The documented local backend and key are test inputs, not defaults. Caller
authentication and forwarding are implemented by issue 08; until then the
catalog remains loopback-only. Creating the application establishes one shared,
lifecycle-managed backend HTTP client, but this slice deliberately performs no
readiness request and makes no claim that the backend is available.
