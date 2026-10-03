# Working in this project

## Current scope

This project is in the documentation stage. The service design is accepted; implementation is deferred until the user explicitly requests it. Documentation approval does not authorize application code, dependency installation, backend provisioning, or deployment.

## Read when relevant

- Before changing the service contract, inference flow, configuration, or tests, read [the accepted service design](docs/service-design.md).
- When planning implementation or preparing a tracker issue, read [the implementation plan](docs/implementation-plan.md); it orders the accepted requirements rather than replacing them.
- When naming domain concepts, read [the glossary](CONTEXT.md). Keep it terminology-only.
- Before changing response model identity or token accounting, read [ADR 0001](docs/adr/0001-report-model-identity-and-backend-work.md).
- When changing project status or onboarding information, update [README.md](README.md).

The service design owns technical requirements. ADRs own decision rationale. Add each rule in its authoritative location; merge or replace existing coverage instead of creating a parallel rule.

## Behavior and changes

- Verify claims against source and observed behavior. Challenge weak assumptions and state uncertainty and material tradeoffs plainly.
- Keep changes surgical and within the active request. Note unrelated issues briefly; address them only if they block the work.
- Read source or current primary documentation before relying on APIs, signatures, flags, or behavior. Distinguish planned behavior from implemented and verified behavior.
- Resolve routine problems independently. For ambiguous or expensive work, ask one focused batch of questions with recommendations before building; ask about decisions, and investigate discoverable facts yourself.
- Batch independent operations. Delegate genuinely independent subtasks when coordinating the overall work; give each editing agent explicit ownership and preserve others' changes.
- Name artifacts and abstractions by responsibility or intention.
- Commit or push only when the user explicitly requests it.

## Communication and verification

- Lead with the result. Use concise, scannable chat; omit pleasantries, flattery, preambles, tool narration, decorative tables, and em dashes. Use normal prose in authored documents and code, and full explanations when nuance matters.
- Support claims of completion or correctness with a command, output, or file. Quote only the decisive error line.
- End with the single next action, or state that nothing is pending.
- Verify changes at the appropriate scope. Documentation changes need consistency, link, and whitespace checks; service implementation needs the acceptance checks in the design. Report unavailable verification explicitly.

## Agent skills

### Issue tracker

Issues live as local markdown files under `.scratch/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the default five triage labels. See `docs/agents/triage-labels.md`.

### Domain docs

Use the single-context layout. See `docs/agents/domain.md`.
