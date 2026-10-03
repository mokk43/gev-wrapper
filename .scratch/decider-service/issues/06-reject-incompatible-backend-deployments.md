# 06: Reject incompatible backend deployments

**What to build:** Prevent inference traffic until the configured model artifacts and external llama.cpp deployment satisfy the service's requirements. Give operators actionable diagnostics for incompatible tokenizer metadata, probability behavior, token accounting, or context capacity.

**Blocked by:** 02: Evaluate Choice asynchronously.

**Status:** ready-for-agent

**Implementation:** complete

- [x] Startup validates required environment configuration, explicit alias mappings, the configured Decider identity, and pinned GGUF/model, quantization, tokenizer/config, Decider dependency, and backend build inputs. Match tokenizer vocabulary, option token IDs, prompt layout, neutralization, and calibration to the deployed checkpoint; never silently substitute stock metadata or calibration defaults.
- [x] Validate backend readiness, effective available context capacity, probability response shape and token-ID compatibility, required usage counters, and caller credential-validation capabilities for both public model discovery and inference before accepting inference traffic. Verify authentication enforcement and the supported validation route against the chosen build rather than assuming a new authentication endpoint exists. Probe only metadata and a bounded synthetic fixture; startup checks never need real user evidence.
- [x] When backend authentication requires a startup credential, use an explicitly configured operator probe key solely for bounded readiness checks. Probe-key rejection is an operator/readiness failure and prevents readiness; it is not a public caller authentication failure. Runtime requests must always use their caller's bearer key, with no probe-key or development-key fallback.
- [x] Establish the selected build's supported vocabulary discovery method and probability-coverage cap, including whether full-vocabulary coverage is supported. Treat the reference adapter's assumed metadata shape as unverified until checked; unsupported required capabilities fail explicitly.
- [x] Define and record one interpretation of backend input/output counters, accounting for disabled prompt caching and repeated evaluation work. Reject missing, malformed, or incompatible counters; field names alone do not establish accounting semantics.
- [x] Missing configuration, unavailable backend, or incompatible capabilities produce actionable sanitized startup diagnostics and prevent inference until resolved. Probes have bounded attempts, payloads, and time rather than indefinite retries.
- [x] Controlled backend fixtures exercise both successful startup and failures for tokenizer/option-ID mismatch, malformed probability payloads, missing counters, unsupported coverage, unavailable backend, invalid context capacity, unavailable credential-validation capabilities, and rejected probe credentials. Verify that a successful probe cannot authorize runtime requests or supply their keys. Keep fixture-verified checks distinct from actual selected-deployment verification; record unverified live capabilities for the final operational verification rather than claiming mocks prove real compatibility.

## Comments

Implemented on 2026-10-03 through application lifespan and the controlled
llama.cpp HTTP boundary. Startup now verifies a hashed local deployment
manifest, explicit Decider configuration, the installed dependency, backend
health/build/model/path/quantization/vocabulary/context/slots, every tokenizer
token ID in bounded `/detokenize` chunks (including every option-label ID), and
tokenization rules over the resulting complete vocabulary-derived corpus,
invalid-key rejection on model discovery and inference, maximum probability
coverage, uncached counters, and one bounded synthetic Decider row. The operator
probe key is used only by startup; successful readiness does not install it on
the shared client or provide a runtime fallback. Controlled fixtures establish
gate behavior only. No result from the user-supplied live backend is recorded;
that remains an explicit input to final operational verification.
