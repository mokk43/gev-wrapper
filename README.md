# TypeSafe-compatible Decider service

An async FastAPI service that accepts TypeSafe's `/v1/systemone` requests,
evaluates them using a remotely served decider-4b GGUF model in llama.cpp, and
returns TypeSafe-shaped decisions.

## Status

The design was accepted on 2026-10-02. The service now provides the configured
`GET /v1/models` catalog and async `POST /v1/systemone` evaluation for Choice,
Noul, and Score questions, including mixed requests. Decision requests use
pinned Decider preparation, per-type calibration, upstream assembly, and
llama.cpp's native `/completion` endpoint through one shared async client.
Per-process admission and backend evaluation limits are enforced from the
configured capacities. One whole-request deadline covers admitted preparation,
backend-slot waiting, evaluation, and assembly; expiry, caller disconnect, and
shutdown cancel pending request tasks and release service capacity. Offloaded
preparation and assembly remain separately bounded until their worker threads
finish. Startup blocks traffic until pinned local metadata and the selected
llama.cpp deployment pass bounded artifact, build, tokenizer, context,
probability, counter, and authentication checks. Missing option probabilities
now trigger bounded coverage recovery. Both public endpoints require caller
bearer credentials and validate them through the configured backend without
installing credentials on the shared client. Public failures carry request IDs
and use sanitized bodies and operational logs. Loopback remains the default;
an explicit network bind is accepted only behind the same mandatory startup
authentication gate. An explicit loopback-only manual mode may skip deployment
identity comparisons while retaining functional remote probes and runtime
response validation. Manual startup and a mixed decision request passed against
the supplied local backend after native Qwen3.5 tokenizer and vocabulary-padding
compatibility fixes. Deployment identity and provenance remain unverified, so
the service is not production-ready.

## Install and operate

Install the locked environment:

```shell
uv sync --python 3.12.5 --locked
```

For verified operation, create the pinned `deployment-manifest.json` and export
the complete environment described in the
[configuration reference](docs/configuration.md). Validate the environment,
then start exactly one service worker:

```shell
uv run decider-service --validate-config
uv run decider-service
```

`--validate-config` checks environment parsing only. Starting the application
runs the bounded backend compatibility gate and fails startup with a sanitized,
actionable diagnostic when the deployment is unavailable or incompatible. The
loopback-only `DECIDER_SKIP_DEPLOYMENT_IDENTITY_VALIDATION=true` override skips
artifact identity comparisons for manual development while retaining functional
remote probes. The entry point intentionally provides no worker-count option.
Do not place it behind a process manager that starts multiple workers; every
process would own independent admission and backend-slot limits.

The service is pinned to CPython 3.12.5 and `decider-ai==1.8.1`. It loads only
matching tokenizer/configuration metadata. Model weights remain on the external
llama.cpp backend.

The [operation and verification guide](docs/verification.md) is the canonical
runbook for authentication checks, graceful shutdown, contract tests, the
official SDK check, the optional live check, exercised results, and exact
unavailable deployment evidence.

Submit Choice, Noul, and Score questions with the caller's backend credential:

```shell
curl http://127.0.0.1:8000/v1/systemone \
  -H 'Authorization: Bearer <API_KEY>' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "decider-4b-q4-k-m",
    "state": "The production service is down.",
    "questions": {
      "priority": {
        "type": "choice",
        "instructions": "Choose the response priority.",
        "criteria": {"routine": "Can wait", "urgent": "Act now"}
      },
      "is_outage": {
        "type": "noul",
        "instructions": "Is the production service unavailable?"
      },
      "urgency": {
        "type": "score",
        "instructions": "How urgently should we respond?",
        "criteria": ["Can wait", "Needs attention", "Act now"]
      }
    }
  }'
```

The response maps native `tokens_evaluated` and `tokens_predicted` counters to
`usage.input_tokens` and `usage.output_tokens`. Normal startup verifies their
uncached one-row semantics against the selected build; manual identity-bypass
startup retains the counter probe without the build comparison. Mock-backed
tests establish request-local summing. When a valid response omits a required
option token ID,
the row retries from 256 with doubled probability coverage, clamping the final
attempt to the readiness-verified maximum. Retries use the same backend slot
limit and whole-request deadline. Successful usage includes the counters from
every coverage attempt.

## Project documents

- [Service design](docs/service-design.md): accepted requirements, inference flow, deployment inputs, and implementation acceptance checks.
- [Implementation plan](docs/implementation-plan.md): ordered work packages, user stories, testing seam, and completion criteria.
- [Domain glossary](CONTEXT.md): the meaning of State, Choice, Noul, Score, Decider, and Jev.
- [Response identity and accounting decision](docs/adr/0001-report-model-identity-and-backend-work.md): why compatible responses expose the actual model and backend work.
- [Service configuration](docs/configuration.md): required operator inputs, safe defaults, and alias configuration.
- [Verification and operation](docs/verification.md): exercised commands, SDK and contract checks, live-check procedure, and unavailable evidence.
- [Contract and dependency baseline](docs/dependency-baseline.md): pinned revisions, inspected upstream seams, and explicit adaptations.
- [Captured TypeSafe contract](contracts/README.md): source, hashes, and interoperability client pin.
- [Agent instructions](AGENTS.md): project scope, document ownership, and working conventions.

## Reference code

- [decider_wrapper.py](decider_wrapper.py) initializes a remote-backed subclass of upstream `Decider`, preserving model configuration, prompt layout, and calibration settings.
- [remote_llamacpp_engine.py](remote_llamacpp_engine.py) is the original synchronous payload/scoring reference. The running service adapts that boundary to async HTTP and request-local accounting.

GGUF weights belong to the external llama.cpp deployment. The service needs matching tokenizer and Decider configuration metadata locally; it does not load model weights.

## Development and authentication inputs

The user supplied a local decider-4b test backend; its address and test
credential are recorded in the [service configuration requirements](docs/service-design.md#configuration-and-deployment-inputs).
The 2026-10-04 direct checks and their limits are recorded in the
[verification record](docs/verification.md); the backend was reachable and its
native probability and uncached-work counters were verified directly, but no
selected deployment has passed the complete startup gate. Normal service starts
verify the configured deployment before accepting traffic; manual
identity-bypass starts retain only the documented functional probes. Decision
requests require TypeSafe-format bearer credentials, forwarded to the configured
backend per request and never installed on the shared client. Model-catalog requests
validate the caller credential through the backend catalog before returning
the configured public metadata. Preparation-only decisions use the same catalog
validation because they perform no inference call. Backend 401 and 403
responses become sanitized public authentication failures with the same status,
which the pinned client maps to its authentication and permission error types.
Every public response includes `x-typesafe-request-id`; failure logs retain that
identifier, status, and a fixed category without request bodies, State, prompts,
credentials, upstream bodies, or exception text. Startup probe credentials are
rejected at the public boundary and cannot serve as runtime authentication
fallback, except as an explicit caller credential in the loopback-only manual
identity-bypass mode.

## References

- [TypeSafe HTTP OpenAPI](https://api.typesafe.ai/openapi.json)
- [Official TypeSafe SDK types](https://github.com/typesafe-ai/typesafe-sdk-js/blob/main/src/types.ts)
- [Decider inference source](https://github.com/Mapika/decider/blob/main/decider/infer.py)
- [decider-4b model card](https://huggingface.co/Mapika/decider-4b)
- [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
