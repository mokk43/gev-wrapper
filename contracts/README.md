# Pinned TypeSafe HTTP contract

`typesafe-openapi-0.2.0.json` is the authoritative TypeSafe HTTP OpenAPI
document retrieved from <https://api.typesafe.ai/openapi.json> on 2026-10-03.
The source URL is mutable and returned no `ETag` or `Last-Modified` header, so
`info.version` alone is not an immutable identifier.

- Source OpenAPI version: `3.1.0`
- Source API version: `0.2.0`
- SHA-256 of the exact 14,158-byte response:
  `a191f8a7df6bd6fedced8120dd0fd106f88575d1d1c8360d08900a6c7c0360d5`
- SHA-256 after formatting with `jq .`, as committed here:
  `72452d6951dbaadd1030af76434917ef103e470bf0cd6ac035b02b111bfd4d24`

The pinned `GET /v1/models` response is `{models: [...]}`. Every model entry
requires string `name`, `description`, and `release_date` fields. The contract
describes release dates as `YYYY-MM-DD` but does not enforce a JSON Schema
format. Both public operations declare HTTP bearer security.

The interoperability client baseline is `@typesafe-ai/sdk@0.6.0`, source
revision `66880ccded6cb642dc1809620c2b108c33730214`, Node.js 20 or newer. Its
`models.list()` method unwraps the HTTP object and returns `ModelCard[]`. The
npm package integrity is
`sha512-IddX+Q0XM+VagOUZFeP7wZjaO4SHMdvnh2zEBdrZZnXedWI3BNK1lKhMx3ayrkFWvVLbVcUHJy6AVZlY+e6Jaw==`.
