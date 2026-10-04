# 07: Recover complete option probabilities

**What to build:** Recover missing option probabilities by increasing backend coverage while retaining the request's capacity and deadline bounds. Return a complete typed decision or an error, and include every attributable coverage attempt in successful request usage.

**Blocked by:** 05: Bound admitted requests and backend work; 06: Reject incompatible backend deployments.

**Status:** ready-for-agent

**Implementation:** complete

- [x] Start each row with top-256 probability coverage and match required alternatives by token ID. Expand coverage only when required option tokens are absent, following a documented schedule and maximum supported by the checked backend build; use full-vocabulary coverage when needed and supported.
- [x] Every retry uses the existing global evaluation limit, admission controls, and remaining whole-request deadline. Expiry or cancellation stops further attempts; large full-vocabulary responses remain bounded by the verified coverage limits. Do not introduce a broader transient-error retry policy.
- [x] Complete coverage yields probabilities calibrated exactly once with the configured per-type temperature and assembled through the shared Decider path. Never invent mass for absent options or normalize only the alternatives that happened to appear. Exhausted or unsupported coverage returns sanitized 502, and deadline expiry returns 504, with no partial answer map.
- [x] Reject malformed final-slot responses, wrong token IDs, nonfinite or invalid distributions, and missing/malformed required counters as backend-contract errors. Apply the documented numerical tolerance and verified counter interpretation consistently.
- [x] Successful `usage` sums actual backend input/output work across all attributable rows and coverage retries, including repeated State processing and each scoring prediction. Usage and intermediate results stay request-local across concurrent callers; preparation-only results incur no backend work.
- [x] Public HTTP checks cover recovery at increasing coverage levels, supported full-vocabulary recovery, exhausted coverage, malformed responses/counters, exact retry-inclusive totals, and isolation across callers. Deterministic gates demonstrate that retries respect the global bound and that the deadline includes waiting plus multiple attempts.

## Comments

Implemented on 2026-10-04 through the public FastAPI boundary. Each row starts
at 256, doubles only after a valid response omits a required option token ID,
and clamps its final attempt to the readiness-verified maximum. Every attempt
reuses the request's global backend slots, caller credential, and absolute
deadline; successful usage includes all valid incomplete attempts.

Runtime parsing now requires exact final-slot coverage, unique in-vocabulary
token IDs, valid uncached counters, finite nonpositive log probabilities, and
probability mass within the accepted `1e-6` tolerance. Controlled tests cover
full-vocabulary recovery and mass validation, exhaustion, malformed attempts,
retry totals, concurrent caller isolation, the global evaluation bound, and
deadline cancellation. They do not establish live backend compatibility.

Final verification: `127 passed`; mypy reported no issues in 15 source files;
Ruff reported all checks passed. Independent Standards and Spec reviews found
no remaining findings.
