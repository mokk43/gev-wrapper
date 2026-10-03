# Contract and dependency baseline

## Runtime pins

CPython 3.12.5 is the pinned runtime. Direct dependencies and all
transitive versions are locked in `uv.lock`.

Decider inference interfaces and the runtime dependency are pinned to
`decider-ai==1.8.1`. The wheel has SHA-256
`ac9414054b29a44d34eae1b057844cf8f997e41cf2e8ee952320fb12ba35bb55`
and was published from upstream revision
`50d0be0d7cb43d2066965ce5fa7f3fe4e489a60f`.

The compatible official metadata baseline is
`Mapika/decider-4b-GGUF@b79f09d9ba7837f1b744295ea267b55d08e958ec`,
which contains the tokenizer and `decider_config.json` for model source
`Mapika/decider-4b@eb5fbdfc9448473ec25e399882912863afbdb70e`.
Operators must still configure the exact metadata revision, GGUF revision, and
quantization that match their running backend. The pin above does not assert
that the user-supplied local backend serves that artifact.

The official TypeSafe JavaScript interoperability baseline is documented next
to the captured contract in [`contracts/README.md`](../contracts/README.md).

## Inspected Decider seam

The pinned Decider revision exposes this inspected adapter seam:

1. `Decider._system_one_items(...)` prepares rendered questions, the answer
   index, and prompt items. Independent mode produces one final answer slot per
   item.
2. `Decider._system_one_probs(...)` is the synchronous evaluation step over
   prepared `ids`, `slots`, and `nopts`.
3. Per-type temperatures come from `decider.temperature.for_items(...)` and are
   applied once with `slot_temperatures(...)` and `scaled_softmax(...)`.
4. `decider.systemone.assemble(...)` supplies answer and confidence semantics.

These preparation interfaces are private, which makes the exact package and
source revision pin part of the compatibility boundary.

## Observed upstream differences

The [accepted service design](service-design.md) owns the corresponding
requirements. Inspection found these differences in the pinned upstream code:

- Upstream silently treats missing or unreadable `decider_config.json` as an
  empty configuration; the design requires incompatible metadata to fail
  startup.
- Upstream prompt construction truncates State to a token limit; the design
  rejects complete prompts that exceed capacity.
- Upstream accepts Choice arrays, Score maps, a `bool` type, and omitted Choice
  discriminators, while the pinned HTTP contract is narrower.
- Upstream rejects a one-level Score rubric before assembly. The HTTP contract
  permits one, so the service handles it as a preparation-only answer: level 0
  has probability 1, expected score 0, and confidence 1, with zero backend
  work. These values match the pinned assembler's normalization, expected-level,
  and single-level confidence formulas without invoking its two-level prompt
  assertion.
- Upstream assembly emits extension fields such as `x_p_max`, `certainty`,
  `level_fit`, and `fit_mass`; those fields are absent from the HTTP contract.
- Upstream logical token accounting and shared mutable engine counters do not
  represent actual remote backend work described by the design's accounting
  policy.
