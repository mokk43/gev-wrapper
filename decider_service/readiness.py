from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, cast

import httpx

from decider_service.config import INVALID_PROBE_CREDENTIAL, Settings
from decider_service.decision import (
    BackendContractError,
    BackendCounterSemanticsError,
    BackendProbabilityCoverageError,
    BackendProbabilityDataError,
    BackendTokenIdentityError,
    DecisionRuntime,
    PublicInputError,
    completion_payload,
    parse_backend_row,
)


class StartupValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class BackendCapabilities:
    vocabulary_size: int
    context_capacity: int
    maximum_probability_coverage: int
    full_vocabulary_probability_coverage: bool


_LLAMA_FTYPE_ALIASES = {
    "ALL F32": "F32",
    "MXFP4 MOE": "MXFP4_MOE",
    "Q2_K - MEDIUM": "Q2_K",
    "Q2_K - SMALL": "Q2_K_S",
    "Q3_K - SMALL": "Q3_K_S",
    "Q3_K - MEDIUM": "Q3_K_M",
    "Q3_K - LARGE": "Q3_K_L",
    "Q4_K - SMALL": "Q4_K_S",
    "Q4_K - MEDIUM": "Q4_K_M",
    "Q5_K - SMALL": "Q5_K_S",
    "Q5_K - MEDIUM": "Q5_K_M",
    "TQ1_0 - 1.69 BPW TERNARY": "TQ1_0",
    "TQ2_0 - 2.06 BPW TERNARY": "TQ2_0",
    "IQ2_XXS - 2.0625 BPW": "IQ2_XXS",
    "IQ2_XS - 2.3125 BPW": "IQ2_XS",
    "IQ2_S - 2.5 BPW": "IQ2_S",
    "IQ2_M - 2.7 BPW": "IQ2_M",
    "IQ3_XS - 3.3 BPW": "IQ3_XS",
    "IQ3_XXS - 3.0625 BPW": "IQ3_XXS",
    "IQ1_S - 1.5625 BPW": "IQ1_S",
    "IQ1_M - 1.75 BPW": "IQ1_M",
    "IQ4_NL - 4.5 BPW": "IQ4_NL",
    "IQ4_XS - 4.25 BPW": "IQ4_XS",
    "IQ3_S - 3.4375 BPW": "IQ3_S",
    "IQ3_S MIX - 3.66 BPW": "IQ3_M",
}


def _json_object(response: httpx.Response, route: str) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        raise StartupValidationError(
            f"backend {route} returned malformed JSON during startup"
        ) from None
    if not isinstance(body, dict):
        raise StartupValidationError(
            f"backend {route} returned an incompatible JSON shape during startup"
        )
    return cast(dict[str, Any], body)


async def _request(
    client: httpx.AsyncClient,
    method: str,
    route: str,
    *,
    deadline: float,
    bearer_token: str | None = None,
    json: object | None = None,
) -> httpx.Response:
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise TimeoutError
    headers = (
        {"Authorization": f"Bearer {bearer_token}"}
        if bearer_token is not None
        else None
    )
    try:
        return await client.request(
            method,
            route,
            headers=headers,
            json=json,
            timeout=remaining,
        )
    except httpx.RequestError:
        raise StartupValidationError(
            f"backend {route} was unavailable during startup"
        ) from None


def _require_success(response: httpx.Response, route: str) -> None:
    if response.status_code in {401, 403}:
        raise StartupValidationError(
            f"operator probe credential was rejected by backend {route}"
        )
    if not 200 <= response.status_code < 300:
        raise StartupValidationError(
            f"backend {route} returned HTTP {response.status_code} during startup"
        )


async def _verify_health(
    client: httpx.AsyncClient,
    settings: Settings,
    deadline: float,
) -> None:
    last_status: int | None = None
    for _attempt in range(settings.startup_probe_attempts):
        try:
            response = await _request(client, "GET", "/health", deadline=deadline)
        except StartupValidationError:
            continue
        last_status = response.status_code
        if response.status_code == 200:
            body = _json_object(response, "/health")
            if body.get("status") != "ok":
                raise StartupValidationError(
                    "backend /health did not report ready status"
                )
            return
        await asyncio.sleep(0)
    status = f"; last HTTP status was {last_status}" if last_status is not None else ""
    raise StartupValidationError(
        "backend /health did not become ready after "
        f"{settings.startup_probe_attempts} bounded attempts{status}"
    )


async def _verify_authentication(
    client: httpx.AsyncClient,
    route: str,
    *,
    method: str,
    deadline: float,
    json: object | None = None,
) -> None:
    response = await _request(
        client,
        method,
        route,
        deadline=deadline,
        bearer_token=INVALID_PROBE_CREDENTIAL,
        json=json,
    )
    if response.status_code not in {401, 403}:
        raise StartupValidationError(
            f"backend {route} does not reject invalid bearer credentials"
        )


def _positive_integer(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise StartupValidationError(description)
    return value


def _canonical_quantization(value: str) -> str:
    normalized = " ".join(value.strip().upper().split())
    return _LLAMA_FTYPE_ALIASES.get(normalized, normalized)


async def validate_backend_deployment(
    settings: Settings,
    runtime: DecisionRuntime,
    client: httpx.AsyncClient,
    *,
    validate_identity: bool = True,
) -> BackendCapabilities:
    deadline = (
        asyncio.get_running_loop().time() + settings.startup_probe_timeout_seconds
    )
    probe_key = settings.operator_probe_api_key.get_secret_value()
    try:
        async with asyncio.timeout_at(deadline):
            await _verify_health(client, settings, deadline)
            await _verify_authentication(
                client,
                "/v1/models",
                method="GET",
                deadline=deadline,
            )
            models_response = await _request(
                client,
                "GET",
                "/v1/models",
                deadline=deadline,
                bearer_token=probe_key,
            )
            _require_success(models_response, "/v1/models")
            models_body = _json_object(models_response, "/v1/models")
            data = models_body.get("data")
            if (
                not isinstance(data, list)
                or len(data) != 1
                or not isinstance(data[0], dict)
            ):
                raise StartupValidationError(
                    "backend /v1/models must describe exactly one loaded model"
                )
            model = cast(dict[str, Any], data[0])
            if validate_identity and model.get("id") != settings.backend_model_id:
                raise StartupValidationError(
                    "backend model identity does not match configured backend_model_id"
                )
            metadata = model.get("meta")
            if not isinstance(metadata, dict):
                raise StartupValidationError(
                    "backend /v1/models did not expose required model metadata"
                )
            vocabulary_size = _positive_integer(
                metadata.get("n_vocab"),
                "backend /v1/models returned an invalid vocabulary size",
            )
            context_capacity = settings.context_capacity
            if validate_identity:
                training_context = _positive_integer(
                    metadata.get("n_ctx_train"),
                    "backend /v1/models returned an invalid training context capacity",
                )
                backend_ftype = metadata.get("ftype")
                if not isinstance(backend_ftype, str) or not backend_ftype.strip():
                    raise StartupValidationError(
                        "backend /v1/models did not expose model quantization"
                    )
                configured_quantization = settings.gguf_quantization
                if configured_quantization is None:
                    raise StartupValidationError(
                        "configured GGUF quantization is missing"
                    )
                if _canonical_quantization(backend_ftype) != _canonical_quantization(
                    configured_quantization
                ):
                    raise StartupValidationError(
                        "backend model quantization does not match configured "
                        "gguf_quantization"
                    )

            props_response = await _request(
                client,
                "GET",
                "/props",
                deadline=deadline,
                bearer_token=probe_key,
            )
            _require_success(props_response, "/props")
            props = _json_object(props_response, "/props")
            if validate_identity:
                if props.get("build_info") != settings.backend_build:
                    raise StartupValidationError(
                        "backend build does not match configured backend_build"
                    )
                if props.get("model_path") != settings.backend_model_path:
                    raise StartupValidationError(
                        "backend model path does not match configured "
                        "backend_model_path"
                    )
                generation_settings = props.get("default_generation_settings")
                if not isinstance(generation_settings, dict):
                    raise StartupValidationError(
                        "backend /props omitted default generation settings"
                    )
                context_capacity = _positive_integer(
                    generation_settings.get("n_ctx"),
                    "backend /props returned an invalid effective context capacity",
                )
                total_slots = _positive_integer(
                    props.get("total_slots"),
                    "backend /props returned an invalid slot capacity",
                )
                if min(context_capacity, training_context) < settings.context_capacity:
                    raise StartupValidationError(
                        "backend effective context capacity is smaller than configured "
                        "context_capacity"
                    )
                if total_slots < settings.backend_slots:
                    raise StartupValidationError(
                        "backend slot capacity is smaller than configured backend_slots"
                    )

            try:
                fixture = runtime.readiness_fixture()
            except (BackendContractError, PublicInputError):
                raise StartupValidationError(
                    "configured context or Decider metadata cannot produce the "
                    "bounded startup fixture"
                ) from None
            if fixture.vocabulary_size > vocabulary_size:
                raise StartupValidationError(
                    "local tokenizer vocabulary size does not match the backend model"
                )
            if settings.maximum_probability_coverage > vocabulary_size:
                raise StartupValidationError(
                    "maximum_probability_coverage exceeds the backend vocabulary size"
                )
            for first_token_id in range(
                fixture.vocabulary_size,
                vocabulary_size,
                settings.startup_tokenizer_probe_chunk_size,
            ):
                padding_response = await _request(
                    client,
                    "POST",
                    "/detokenize",
                    deadline=deadline,
                    bearer_token=probe_key,
                    json={
                        "tokens": list(
                            range(
                                first_token_id,
                                min(
                                    first_token_id
                                    + settings.startup_tokenizer_probe_chunk_size,
                                    vocabulary_size,
                                ),
                            )
                        )
                    },
                )
                _require_success(padding_response, "/detokenize")
                padding_body = _json_object(padding_response, "/detokenize")
                # Native /detokenize renders special tokens and concatenates
                # pieces. Any visible extra token therefore makes this nonempty.
                if padding_body.get("content") != "":
                    raise StartupValidationError(
                        "backend trailing token IDs are not empty vocabulary padding"
                    )
            for (
                token_ids,
                expected_content,
                expected_encoded_token_ids,
            ) in runtime.tokenizer_identity_chunks(
                settings.startup_tokenizer_probe_chunk_size
            ):
                token_response = await _request(
                    client,
                    "POST",
                    "/detokenize",
                    deadline=deadline,
                    bearer_token=probe_key,
                    json={"tokens": token_ids},
                )
                _require_success(token_response, "/detokenize")
                token_body = _json_object(token_response, "/detokenize")
                if token_body.get("content") != expected_content:
                    raise StartupValidationError(
                        "backend tokenizer vocabulary does not match local metadata"
                    )
                encoding_response = await _request(
                    client,
                    "POST",
                    "/tokenize",
                    deadline=deadline,
                    bearer_token=probe_key,
                    json={
                        "content": expected_content,
                        "add_special": False,
                        "parse_special": True,
                    },
                )
                _require_success(encoding_response, "/tokenize")
                encoding_body = _json_object(encoding_response, "/tokenize")
                if encoding_body.get("tokens") != list(expected_encoded_token_ids):
                    raise StartupValidationError(
                        "backend tokenizer tokenization rules do not match local "
                        "metadata"
                    )

            rejection_payload = completion_payload(
                fixture.row,
                settings.initial_probability_coverage,
            )
            await _verify_authentication(
                client,
                "/completion",
                method="POST",
                deadline=deadline,
                json=rejection_payload,
            )
            probe_payload = completion_payload(
                fixture.row,
                settings.maximum_probability_coverage,
            )
            completion_response = await _request(
                client,
                "POST",
                "/completion",
                deadline=deadline,
                bearer_token=probe_key,
                json=probe_payload,
            )
            _require_success(completion_response, "/completion")
            completion_body = _json_object(completion_response, "/completion")
            try:
                parse_backend_row(
                    completion_body,
                    fixture.label_token_ids[: fixture.row.option_count],
                    expected_input_tokens=len(fixture.row.token_ids),
                    expected_probability_coverage=(
                        settings.maximum_probability_coverage
                    ),
                    vocabulary_size=vocabulary_size,
                )
            except BackendProbabilityCoverageError:
                raise StartupValidationError(
                    "backend does not support configured maximum probability coverage"
                ) from None
            except BackendProbabilityDataError:
                raise StartupValidationError(
                    "backend /completion returned malformed probability data"
                ) from None
            except BackendTokenIdentityError:
                raise StartupValidationError(
                    "backend /completion returned incompatible token IDs"
                ) from None
            except BackendCounterSemanticsError:
                raise StartupValidationError(
                    "backend counters do not represent uncached prompt and "
                    "generated work"
                ) from None
            except BackendContractError:
                raise StartupValidationError(
                    "backend /completion probability or counter contract is "
                    "incompatible"
                ) from None
    except TimeoutError:
        raise StartupValidationError(
            "backend startup checks exceeded startup_probe_timeout_seconds"
        ) from None

    return BackendCapabilities(
        vocabulary_size=vocabulary_size,
        context_capacity=context_capacity,
        maximum_probability_coverage=settings.maximum_probability_coverage,
        full_vocabulary_probability_coverage=(
            settings.maximum_probability_coverage == vocabulary_size
        ),
    )
