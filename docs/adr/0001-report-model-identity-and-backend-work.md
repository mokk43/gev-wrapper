---
status: accepted
---

# Report the actual model and backend work

TypeSafe clients may send Jev aliases, while this service runs an independently trained Decider checkpoint and repeats state evaluation across remote rows. We accept explicitly configured aliases for client compatibility, but return the actual Decider identity and backend input/output token work, including coverage retries, rather than echoing a Jev identity or preserving upstream Decider's logical shared-input accounting and zero output count. This retains the public response structure while making model provenance and inference cost observable; consumers must tolerate resolved model names and avoid comparing these usage counts directly with hosted TypeSafe billing.

The behavioral requirements live in [the service design](../service-design.md); this record preserves the reason for the deliberate compatibility tradeoff.
