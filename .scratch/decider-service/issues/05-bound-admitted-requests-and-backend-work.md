# 05: Bound admitted requests and backend work

**What to build:** Keep the service responsive and its backend workload bounded when callers overlap, overload the service, disconnect, or time out. Each admitted request has one complete deadline, and abandoned requests stop launching evaluations.

**Blocked by:** 02: Evaluate Choice asynchronously.

**Status:** ready-for-agent

- [ ] Start with one service process/worker and an operator-configured global backend evaluation limit shared by all callers and independent rows. Every evaluation attempt uses the same capacity controls, including coverage retries when they are added; document that additional workers would require coordinated capacity management.
- [ ] Bound admitted work as well as active evaluations. Admission/queue capacities are explicit operator configuration, and exhausted admission returns a sanitized 503 without creating an unbounded backlog.
- [ ] Enforce the accepted 60-second whole-request deadline across admission, synchronous preparation/offloaded work, evaluation, retries, and assembly. Derive transport timeouts from the remaining budget rather than granting a fresh full timeout to each row or attempt; expiry returns 504 with no partial successful answers.
- [ ] On deadline expiry or caller cancellation/disconnect where supported, stop scheduling further rows and attempts, cancel pending tasks, and close outstanding client requests. Release service capacity and account for the limits of cancelling offloaded preparation; a closed connection must not be reported as proof that remote backend inference stopped.
- [ ] Service shutdown manages pending tasks and closes the shared client without leaking admission/evaluation capacity or leaving unmanaged client requests.
- [ ] Public HTTP checks use deterministic backend gates/events to verify the global bound across simultaneous callers and multiple rows, bounded overload, deadlines spanning queueing/preparation/evaluation/assembly, cancellation, and shutdown cleanup. Verify released capacity can serve subsequent callers; timing-only sleeps are insufficient evidence.

## Comments

Partial implementation on 2026-10-03 now enforces per-process backend slots
and non-waiting request admission, returning sanitized 503 on exhaustion.
Deterministic public checks cover multiple rows, overlapping callers, and
capacity release. Deadlines, disconnect/cancellation behavior, retry controls,
and shutdown cleanup remain open, so this ticket is not complete.
