# TypeSafe-compatible Decider service

An async FastAPI service being built to accept TypeSafe's `/v1/systemone` requests, evaluate them using a remotely served decider-4b GGUF model in llama.cpp, and return TypeSafe-shaped decisions.

## Status

The design was accepted on 2026-10-02. The service now provides the configured
`GET /v1/models` catalog and async `POST /v1/systemone` evaluation for Choice and
Noul questions, including mixed requests. Decision requests use pinned Decider
preparation, per-type calibration, upstream assembly, and llama.cpp's native
`/completion` endpoint through one shared async client. Per-process admission and
backend evaluation limits are enforced from the configured capacities. Score,
coverage recovery, whole-request deadlines, disconnect/cancellation handling,
readiness checks, and the final catalog authentication boundary remain separate
implementation slices. The service remains restricted to loopback and is not
production-ready.

## Catalog setup

Install the locked catalog environment:

```shell
uv sync --python 3.12.5 --locked
```

Set every required variable in [the configuration reference](docs/configuration.md), then validate and run:

```shell
uv run decider-service --validate-config
uv run decider-service
```

Check the public wire response from another shell. Supply an API key explicitly
when checking a future authenticated slice:

```shell
uv run decider-service-check --base-url http://127.0.0.1:8000
```

The service is pinned to CPython 3.12.5 and `decider-ai==1.8.1`. It loads only
matching tokenizer/configuration metadata. Model weights remain on the external
llama.cpp backend.

Submit Choice and Noul questions with the caller's backend credential:

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
      }
    }
  }'
```

The response maps native `tokens_evaluated` and `tokens_predicted` counters to
`usage.input_tokens` and `usage.output_tokens`. Their real accounting semantics
remain unverified until the selected llama.cpp build is checked; mock-backed
tests establish request-local summing only.

## Project documents

- [Service design](docs/service-design.md): accepted requirements, inference flow, deployment inputs, and implementation acceptance checks.
- [Implementation plan](docs/implementation-plan.md): ordered work packages, user stories, testing seam, and completion criteria.
- [Domain glossary](CONTEXT.md): the meaning of State, Choice, Noul, Score, Decider, and Jev.
- [Response identity and accounting decision](docs/adr/0001-report-model-identity-and-backend-work.md): why compatible responses expose the actual model and backend work.
- [Service configuration](docs/configuration.md): required operator inputs, safe defaults, and alias configuration.
- [Contract and dependency baseline](docs/dependency-baseline.md): pinned revisions, inspected upstream seams, and explicit adaptations.
- [Captured TypeSafe contract](contracts/README.md): source, hashes, and interoperability client pin.
- [Agent instructions](AGENTS.md): project scope, document ownership, and working conventions.

## Reference code

- [decider_wrapper.py](decider_wrapper.py) initializes a remote-backed subclass of upstream `Decider`, preserving model configuration, prompt layout, and calibration settings.
- [remote_llamacpp_engine.py](remote_llamacpp_engine.py) is the original synchronous payload/scoring reference. The running service adapts that boundary to async HTTP and request-local accounting.

GGUF weights belong to the external llama.cpp deployment. The planned service needs matching tokenizer and Decider configuration metadata locally; it does not load model weights.

## Development and authentication inputs

The user supplied a local decider-4b test backend; its address and test credential are recorded in the [service configuration requirements](docs/service-design.md#configuration-and-deployment-inputs). Availability and capabilities have not been verified. Decision requests require TypeSafe-format bearer credentials, forwarded to the configured backend per request and never installed on the shared client. Backend 401 and 403 responses become sanitized public authentication failures with the same status. Verification against the selected backend and pinned TypeSafe client, plus authentication for the public model catalog, remains part of the operational-boundary slice. Startup probe credentials remain separate and cannot serve as runtime authentication fallback.

## References

- [TypeSafe HTTP OpenAPI](https://api.typesafe.ai/openapi.json)
- [Official TypeSafe SDK types](https://github.com/typesafe-ai/typesafe-sdk-js/blob/main/src/types.ts)
- [Decider inference source](https://github.com/Mapika/decider/blob/main/decider/infer.py)
- [decider-4b model card](https://huggingface.co/Mapika/decider-4b)
- [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
