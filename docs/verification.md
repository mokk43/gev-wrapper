# Verification and operation

This record distinguishes controlled contract evidence from evidence about an
actual llama.cpp deployment. Results below were collected on 2026-10-04.

## Operator procedure

Install the pinned Python environment:

```shell
uv sync --python 3.12.5 --locked
```

Install the pinned official TypeSafe JavaScript client with Node.js 20 or newer:

```shell
npm ci
```

`package-lock.json` pins `@typesafe-ai/sdk@0.6.0` to the npm registry artifact
and integrity recorded in the [captured contract](../contracts/README.md).

Create the pinned metadata manifest and export every required variable using
the [complete environment example](configuration.md#complete-environment-example).
Configuration values must describe the already-running llama.cpp deployment;
the service does not provision it or download weights. Validate parsing, then
start one worker:

```shell
uv run decider-service --validate-config
uv run decider-service
```

Successful process startup means the configured metadata and selected backend
passed the bounded compatibility gate for build identity, authentication,
tokenizer agreement, context, probability coverage, token counters, and
disabled caching. It does not establish prediction equivalence with another
Decider artifact.

From another shell, use a caller credential, not the operator probe credential,
to check authentication and the catalog:

```shell
uv run decider-service-check \
  --base-url http://127.0.0.1:8000 \
  --api-key '<CALLER_API_KEY>'
```

Both public operations require `Authorization: Bearer <CALLER_API_KEY>`. The
credential is forwarded only to the configured backend for that request.
Missing or malformed credentials receive 401; a backend-rejected credential
receives the backend's 401 or 403 status with a sanitized body. There is no
probe-key or development-key fallback.

Stop the foreground process with `Ctrl-C`, or send it `SIGTERM`. Shutdown stops
new admission, cancels tracked requests, waits for preparation and assembly
threads already running, and closes the shared HTTP client. Closing the client
side of an in-flight request does not prove the backend stopped inference.

The launch, public catalog check, and process-signal shutdown commands above
were not run against a selected deployment on 2026-10-04 because the backend
and required operator inputs were unavailable. The controlled suite exercises
the same application startup, authenticated public catalog, request
cancellation, and shutdown behavior without claiming live compatibility.

## Controlled verification

The locked installation command completed:

```text
$ uv sync --python 3.12.5 --locked
Resolved 76 packages in 15ms
Audited 56 packages in 8ms
$ npm ci
added 1 package in 329ms
```

The installed console entry points and environment parsing were also exercised.
The validation command used a complete temporary controlled fixture, not the
unavailable local deployment:

```text
$ uv run decider-service --help
usage: decider-service [-h] [--validate-config]
$ uv run decider-service-check --help
usage: decider-service-check [-h] --base-url BASE_URL [--api-key API_KEY]
$ .venv/bin/decider-service --validate-config
configuration valid
```

The catalog-check console script was exercised against the same controlled
real-TCP service used by interoperability verification:

```text
$ .venv/bin/decider-service-check --base-url '<CONTROLLED_URL>' \
    --api-key 'caller-key'
decider-4b-q4-k-m  2025-07-04  Decider 4B served by the configured llama.cpp backend.
jev-latest  2025-07-04  Compatibility alias routing to Decider model 'decider-4b-q4-k-m'.
```

Run static checks and the complete public contract suite with:

```shell
uv run ruff check .
uv run mypy
uv run pytest -q
```

All three completed successfully in the locked environment:

```text
$ uv run ruff check .
All checks passed!
$ uv run mypy
mypy: No issues found
$ uv run pytest -q
157 passed
```

The controlled suite uses an in-process service and deterministic backend
fixtures. It covers catalog and decision payloads, mixed Choice/Noul/Score
requests, structured and optional values, null State rejection, one-level
Score, aliases and resolved identity, context and capacity failures,
composition independence, probability coverage recovery, retry accounting,
concurrent caller isolation, event-loop responsiveness, global limits, bounded
overload, deadlines, cancellation, startup incompatibility, key forwarding,
probe/runtime credential separation, redaction, and shutdown. These results do
not prove compatibility with an actual GGUF or llama.cpp build.

Run the pinned official TypeSafe JavaScript client checks separately:

```shell
npm test
```

The pinned client check completed successfully:

```text
$ npm test
TypeSafe SDK 0.6.0 interoperability check passed: models, typed decisions,
mixed request, bearer auth, and 401/403/422 errors
```

The SDK checks use the public HTTP seam and the repository's controlled backend.
They verify the configured base URL, model discovery, typed and mixed decisions,
bearer-header construction, authentication errors, and the documented SDK versus
HTTP-schema differences. They do not contact TypeSafe's hosted service.

## Optional live check

First configure the service against the user-managed local fixture at
`http://127.0.0.1:8080` and start the service on `http://127.0.0.1:8000`. Then
run the public smoke check with a backend-issued caller key explicitly:

```shell
TYPESAFE_API_KEY='<CALLER_API_KEY>' \
  uv run python scripts/verify_live_service.py \
  --base-url http://127.0.0.1:8000 \
  --actual-model '<CANONICAL_DECIDER_MODEL>'
```

`TYPESAFE_API_KEY` may be replaced by `--api-key`; one form is required. The
canonical configured identity must be supplied through `--actual-model` or
`DECIDER_LIVE_ACTUAL_MODEL`; an optional `--model` may select a catalogued alias.
The script rejects keys outside the RFC 6750 `b64token` grammar before connecting.
It checks missing and malformed authentication, authenticated model discovery,
and one representative request containing Choice, Noul, and Score questions.
It validates the public response shape, labels, ranges, probability sums, and
positive request-local usage counters. It also checks that aliases resolve to
the explicitly supplied canonical identity. Supply a known rejected
backend key with `--rejected-api-key` or
`DECIDER_REJECTED_CALLER_API_KEY` to verify selected-backend 401/403 handling;
without one, the script reports that check as unavailable. It does not start
llama.cpp, create metadata, provision credentials, or download artifacts.

The public caller key must differ from `DECIDER_OPERATOR_PROBE_API_KEY`; the
service rejects its probe key at the public boundary by design. The only
supplied local key, `llama5080`, is used explicitly by the direct-backend check
below. No separate caller key was supplied, so it cannot currently drive this
public smoke procedure when it is also the startup probe credential.

This is a smoke check, not a benchmark or a baseline comparison. Its 65-second
transport timeout sits outside the service's default 60-second whole-request
deadline so the service, rather than the client, can report deadline expiry.
If either deadline is reconfigured, the operator must keep the transport timeout
above the service deadline. The service's 60 seconds remains a guardrail, not a
measured performance target.

## Actual live result and unavailable evidence

The supplied backend address was checked directly before attempting to launch a
fully configured service:

```text
$ curl --silent --show-error --connect-timeout 2 --max-time 3 \
    http://127.0.0.1:8080/v1/models \
    -H 'Authorization: Bearer llama5080'
curl: (7) Failed to connect to 127.0.0.1 port 8080 after 0 ms: Couldn't connect to server
```

Therefore no public live smoke run was possible on 2026-10-04. The following
checks remain unavailable, not failed compatibility claims:

- Selected-build authentication enforcement on `/v1/models` and `/completion`,
  including acceptance of `llama5080` and rejection of an invalid key.
- Exact llama.cpp build, loaded model identity and path, GGUF revision and
  quantization, matching metadata revision, vocabulary, effective context, and
  parallel-slot capacity.
- Complete tokenizer agreement, option-label token IDs, supported maximum
  `n_probs`/`min_keep` coverage, returned probability mass, counter semantics,
  and disabled-cache behavior on that deployment.
- The character format of an issued live TypeSafe key. `llama5080` is a supplied
  local backend test key, not evidence about TypeSafe's issuer.
- A separate backend-issued runtime caller key. The service cannot use its
  configured startup probe credential as a public caller fallback.
- Numerical comparison with a trusted Decider baseline. No trusted baseline,
  fixed comparison corpus, matching artifact identity, or justified tolerance
  was supplied, so GGUF prediction and quantization equivalence are unverified.
- Representative workload measurements for latency, queueing, concurrency, and
  throughput. No workload fixture or reachable deployment was supplied; the
  60-second request deadline must not be reported as observed latency.

Controlled fixtures establish HTTP behavior only. They do not close any of the
live-deployment gaps above.
