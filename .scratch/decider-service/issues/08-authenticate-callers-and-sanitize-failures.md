# 08: Authenticate callers and sanitize failures

**What to build:** Expose the service intentionally, authenticate callers using their TypeSafe bearer key forwarded to the configured llama.cpp backend, and return useful errors without disclosing State or credentials. Keep operator readiness credentials confined to startup probes.

**Blocked by:** 05: Bound admitted requests and backend work.

**Status:** ready-for-agent

- [ ] Bind to loopback by default. Require caller `Authorization: Bearer <API_KEY>` credentials on both public endpoints; missing or malformed headers fail public authentication before backend work. Network exposure requires verified backend enforcement of caller credentials for model discovery as well as inference, using a validation path supported by the chosen build.
- [ ] Forward each caller's bearer key only to the configured llama.cpp backend, retaining it locally across rows and coverage retries without mutating shared client authentication or default headers. Never include keys in prompts, request bodies, logs, or public diagnostics. Never replace a missing or rejected caller key with an operator probe key or the documented development key; the latter is only for explicit local test requests.
- [ ] Return field-oriented TypeSafe 422 validation errors for invalid public input, unknown models, option-capacity violations, and oversized prompts; sanitized 502 for backend-contract failures; 503 for unavailable backend or exhausted admission; and 504 for the whole-request deadline. A request returns every answer or an error, never a partial successful map.
- [ ] Backend rejection of the forwarded caller key maps to a public authentication failure consistent with the pinned TypeSafe clients. Operator startup probe-key rejection remains a readiness failure; transport/backend unavailability remains 503. Verify exact authentication statuses and non-validation error bodies against the pinned contract and client behavior rather than guessing hosted-platform fidelity.
- [ ] Retain request identifiers and concise operational diagnostics while omitting request bodies, State, prompts, and credentials from logs. Sanitize backend error bodies, exceptions, and startup diagnostics so sensitive upstream content cannot escape.
- [ ] Public HTTP and captured-log checks cover loopback/network configuration, valid, missing, malformed, and backend-rejected caller credentials on both endpoints, validation/backend/unavailable/overload/deadline errors, and request isolation. Concurrent requests with different keys and coverage retries always forward the originating caller's key; a configured probe key cannot satisfy runtime authentication. Inject recognizable evidence and secret markers in controlled failures and confirm none appear in public errors or operational logs.

## Comments

Partial implementation on 2026-10-03 now maps backend 401 and 403 responses to
sanitized public authentication failures with the same status. Exact behavior
still requires verification against the selected backend and pinned TypeSafe
client. Catalog authentication, request identifiers, captured-log checks, and
the remaining operational error boundary are still open.
