# TypeSafe-compatible Decider service

A planned async FastAPI service that accepts TypeSafe's `/v1/systemone` requests, evaluates them using a remotely served decider-4b GGUF model in llama.cpp, and returns TypeSafe-shaped decisions.

## Status

The design was accepted on 2026-10-02. Nine implementation tickets were approved and published to the [local tracker](.scratch/decider-service/issues/) on 2026-10-03. Implementation is deferred. The two supplied Python files are reference adapters, not a running web service; there is no supported installation or launch command yet.

## Project documents

- [Service design](docs/service-design.md): accepted requirements, inference flow, deployment inputs, and implementation acceptance checks.
- [Implementation plan](docs/implementation-plan.md): ordered work packages, user stories, testing seam, and completion criteria.
- [Domain glossary](CONTEXT.md): the meaning of State, Choice, Noul, Score, Decider, and Jev.
- [Response identity and accounting decision](docs/adr/0001-report-model-identity-and-backend-work.md): why compatible responses expose the actual model and backend work.
- [Agent instructions](AGENTS.md): project scope, document ownership, and working conventions.

## Reference code

- [decider_wrapper.py](decider_wrapper.py) initializes a remote-backed subclass of upstream `Decider`, preserving model configuration, prompt layout, and calibration settings.
- [remote_llamacpp_engine.py](remote_llamacpp_engine.py) evaluates final answer slots through llama.cpp's native `/completion` endpoint. Its synchronous transport must be adapted for the accepted async design.

GGUF weights belong to the external llama.cpp deployment. The planned service needs matching tokenizer and Decider configuration metadata locally; it does not load model weights.

## Development and authentication inputs

The user supplied a local decider-4b test backend; its address and test credential are recorded in the [service configuration requirements](docs/service-design.md#configuration-and-deployment-inputs). Availability and capabilities have not been verified. Real users will supply TypeSafe-format bearer credentials that the service forwards to the configured backend, isolated per request. Startup probe credentials remain separate and cannot serve as runtime authentication fallback.

## References

- [TypeSafe HTTP OpenAPI](https://api.typesafe.ai/openapi.json)
- [Official TypeSafe SDK types](https://github.com/typesafe-ai/typesafe-sdk-js/blob/main/src/types.ts)
- [Decider inference source](https://github.com/Mapika/decider/blob/main/decider/infer.py)
- [decider-4b model card](https://huggingface.co/Mapika/decider-4b)
- [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
