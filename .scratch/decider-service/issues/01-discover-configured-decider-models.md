# 01: Discover configured Decider models

**What to build:** Let clients discover the configured Decider model and deliberately enabled compatibility aliases through a runnable TypeSafe-compatible model catalog. Establish the reproducible contract, dependency, and configuration baseline used by subsequent decision requests.

**Blocked by:** None (can start immediately).

**Status:** needs-triage

**Implementation:** complete

- [x] Capture and version the authoritative TypeSafe HTTP OpenAPI contract and pin a compatible Decider dependency, tokenizer/config revision, and interoperability client version. Inspect the pinned Decider preparation and assembly interfaces before relying on them; resolve differences from the accepted design explicitly rather than silently widening the contract to match SDK or upstream extensions.
- [x] Provide a reproducible Python dependency environment and exercised entry points for the catalog and public HTTP checks. Keep inference weights on the external llama.cpp deployment; the service uses matching local tokenizer/configuration metadata.
- [x] Define environment-based configuration for backend address/build, GGUF revision and quantization, metadata directory/revision, public model identity and release date, aliases, context and evaluation capacity, bounded admission, and request limits. Runtime backend credentials come from each caller's TypeSafe `Authorization: Bearer <API_KEY>` header; optionally configure an operator key solely for startup readiness probes, never as a runtime fallback. Distinguish required operator inputs from defaults without inventing unresolved deployment values or embedding production credentials. Explicit local test requests may use the development fixture documented in the accepted design.
- [x] `GET /v1/models` returns the pinned TypeSafe `{models: [...]}` shape with configured names, descriptions, and actual release dates. Only explicitly configured aliases are listed, and their descriptions identify routing to Decider; `jev-latest` is never enabled implicitly.
- [x] Validate local configuration and alias mappings. Keep the default bind address on loopback; the catalog does not establish backend readiness or authorize network exposure before the operational checks are complete.
- [x] Exercise the catalog through public HTTP requests, including configured identities, enabled and absent aliases, missing required catalog metadata, and invalid configuration. Establish one controllable backend HTTP boundary through the application lifecycle for subsequent slices instead of separate mocks for preparation, scoring, and assembly.

## Comments

Implemented on 2026-10-03. Verification uses the pinned CPython 3.12.5 runtime.
`decider-ai==1.8.1` is locked as the inspected preparation/assembly dependency;
the catalog does not load weights or import inference code.
