from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError
from transformers import AutoTokenizer

from decider_service.app import create_app
from decider_service.config import INVALID_PROBE_CREDENTIAL
from decider_service.readiness import StartupValidationError, _canonical_quantization
from tests.readiness_support import (
    PROBE_KEY,
    ControlledBackend,
    completion_response,
    configured_settings,
    write_test_manifest,
    write_test_metadata,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def metadata_directory(tmp_path: Path) -> Path:
    return write_test_metadata(tmp_path)


@pytest.mark.parametrize(
    ("descriptor", "canonical"),
    [
        ("Q4_K - Medium", "Q4_K_M"),
        ("IQ3_S mix - 3.66 bpw", "IQ3_M"),
        ("Q8_0 - unknown-detail", "Q8_0 - UNKNOWN-DETAIL"),
        ("IQ3_S MIXED WITH Q8", "IQ3_S MIXED WITH Q8"),
    ],
)
def test_quantization_normalization_uses_exact_llama_ftype_aliases(
    descriptor: str,
    canonical: str,
) -> None:
    assert _canonical_quantization(descriptor) == canonical


@pytest.mark.anyio
async def test_tokenizer_probe_matches_backend_parse_special_semantics(
    metadata_directory: Path,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        metadata_directory,
        local_files_only=True,
    )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|readiness_special|>"]}
    )
    tokenizer.split_special_tokens = True
    tokenizer.save_pretrained(metadata_directory)
    write_test_manifest(metadata_directory)
    backend = ControlledBackend(metadata_directory)
    app = create_app(
        configured_settings(metadata_directory),
        backend_transport=backend.transport(),
    )

    async with app.router.lifespan_context(app):
        assert app.state.backend_capabilities.vocabulary_size == len(tokenizer)


@pytest.mark.anyio
async def test_startup_verifies_backend_before_serving_inference(
    metadata_directory: Path,
) -> None:
    runtime_requests: list[httpx.Request] = []

    def runtime_response(request: httpx.Request) -> httpx.Response:
        runtime_requests.append(request)
        return completion_response({1: 0.87, 2: 0.13})

    backend = ControlledBackend(metadata_directory, runtime_response)
    app = create_app(
        configured_settings(
            metadata_directory,
            startup_tokenizer_probe_chunk_size=128,
        ),
        backend_transport=backend.transport(),
    )

    async with app.router.lifespan_context(app):
        assert app.state.backend_capabilities.vocabulary_size == 703
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/systemone",
                headers={"Authorization": "Bearer caller-key"},
                json={
                    "model": "decider-4b-q4-k-m",
                    "state": "evidence",
                    "questions": {
                        "priority": {
                            "type": "choice",
                            "criteria": {"urgent": None, "routine": None},
                        }
                    },
                },
            )

    assert response.status_code == 200
    assert len(runtime_requests) == 1
    assert runtime_requests[0].headers["Authorization"] == "Bearer caller-key"
    assert PROBE_KEY not in runtime_requests[0].content.decode()
    startup_completions = [
        request
        for request in backend.requests
        if request.url.path == "/completion"
        and request.headers.get("Authorization") == f"Bearer {PROBE_KEY}"
    ]
    assert len(startup_completions) == 1
    assert json.loads(startup_completions[0].content)["cache_prompt"] is False
    tokenizer_probes = [
        json.loads(request.content)["tokens"]
        for request in backend.requests
        if request.url.path == "/detokenize"
    ]
    assert all(len(chunk) <= 128 for chunk in tokenizer_probes)
    assert [token_id for chunk in tokenizer_probes for token_id in chunk] == list(
        range(703)
    )
    assert all(
        json.loads(request.content)["parse_special"] is True
        for request in backend.requests
        if request.url.path == "/tokenize"
    )


@pytest.mark.anyio
async def test_loopback_manual_run_skips_identity_but_keeps_functional_probes(
    metadata_directory: Path,
) -> None:
    (metadata_directory / "deployment-manifest.json").unlink()
    backend = ControlledBackend(metadata_directory)

    settings = configured_settings(
        metadata_directory,
        backend_build=None,
        backend_model_id=None,
        backend_model_path=None,
        gguf_revision=None,
        gguf_quantization=None,
        metadata_revision=None,
        skip_deployment_identity_validation=True,
    )
    app = create_app(
        settings,
        backend_transport=backend.transport(),
    )

    async with app.router.lifespan_context(app):
        capabilities = app.state.backend_capabilities
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            catalog_response = await client.get(
                "/v1/models",
                headers={"Authorization": f"Bearer {PROBE_KEY}"},
            )

    assert capabilities.vocabulary_size == 703
    assert capabilities.context_capacity == settings.context_capacity
    assert (
        capabilities.maximum_probability_coverage
        == settings.maximum_probability_coverage
    )
    assert capabilities.full_vocabulary_probability_coverage is False
    probed_paths = {request.url.path for request in backend.requests}
    assert {
        "/health",
        "/v1/models",
        "/props",
        "/detokenize",
        "/tokenize",
        "/completion",
    } <= probed_paths
    for path in ("/v1/models", "/completion"):
        authorization_values = {
            request.headers.get("Authorization")
            for request in backend.requests
            if request.url.path == path
        }
        assert f"Bearer {INVALID_PROBE_CREDENTIAL}" in authorization_values
        assert f"Bearer {PROBE_KEY}" in authorization_values
    assert catalog_response.status_code == 200


async def assert_startup_fails(
    metadata_directory: Path,
    handler: httpx.MockTransport,
    message: str,
    **settings_overrides: object,
) -> None:
    app = create_app(
        configured_settings(metadata_directory, **settings_overrides),
        backend_transport=handler,
    )
    with pytest.raises(StartupValidationError, match=message):
        async with app.router.lifespan_context(app):
            raise AssertionError("incompatible deployment must not become ready")


@pytest.mark.anyio
@pytest.mark.parametrize("backend_padding", [False, True])
async def test_startup_rejects_backend_tokenizer_vocabulary_mismatch(
    metadata_directory: Path,
    backend_padding: bool,
) -> None:
    backend = (
        VocabularyPaddingBackend(metadata_directory)
        if backend_padding
        else ControlledBackend(metadata_directory)
    )
    unprobed_label_id = backend.tokenizer.convert_tokens_to_ids("K")

    def mismatch(request: httpx.Request) -> httpx.Response:
        if (
            request.url.path == "/detokenize"
            and request.headers.get("Authorization") == f"Bearer {PROBE_KEY}"
            and unprobed_label_id in json.loads(request.content)["tokens"]
        ):
            response = backend(request)
            return httpx.Response(
                200,
                json={"content": response.json()["content"] + " mismatch"},
            )
        return backend(request)

    await assert_startup_fails(
        metadata_directory,
        httpx.MockTransport(mismatch),
        "vocabulary does not match",
    )


class VocabularyPaddingBackend(ControlledBackend):
    def __init__(
        self,
        metadata_directory: Path,
        *,
        padding_content: str = "",
        padding_size: int = 137,
    ) -> None:
        super().__init__(metadata_directory)
        self.padding_content = padding_content
        self.vocabulary_size = len(self.tokenizer) + padding_size

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if (
            request.url.path == "/detokenize"
            and request.headers.get("Authorization") == f"Bearer {PROBE_KEY}"
        ):
            self.requests.append(request)
            tokens = json.loads(request.content)["tokens"]
            content = self.tokenizer.decode(
                [token for token in tokens if token < len(self.tokenizer)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            content += self.padding_content * sum(
                token >= len(self.tokenizer) for token in tokens
            )
            return httpx.Response(200, json={"content": content})
        response = super().__call__(request)
        if (
            request.url.path == "/v1/models"
            and request.headers.get("Authorization") == f"Bearer {PROBE_KEY}"
        ):
            body = response.json()
            body["data"][0]["meta"]["n_vocab"] = self.vocabulary_size
            return httpx.Response(200, json=body)
        return response


@pytest.mark.anyio
@pytest.mark.parametrize("manual_mode", [False, True])
@pytest.mark.parametrize("full_backend_coverage", [False, True])
async def test_startup_accepts_only_verified_empty_backend_vocabulary_padding(
    metadata_directory: Path,
    manual_mode: bool,
    full_backend_coverage: bool,
) -> None:
    backend = VocabularyPaddingBackend(metadata_directory)
    local_vocabulary_size = len(backend.tokenizer)
    maximum_coverage = (
        backend.vocabulary_size if full_backend_coverage else local_vocabulary_size
    )
    app = create_app(
        configured_settings(
            metadata_directory,
            skip_deployment_identity_validation=manual_mode,
            startup_tokenizer_probe_chunk_size=128,
            maximum_probability_coverage=maximum_coverage,
        ),
        backend_transport=backend.transport(),
    )

    async with app.router.lifespan_context(app):
        capabilities = app.state.backend_capabilities

    assert capabilities.vocabulary_size == backend.vocabulary_size
    assert capabilities.maximum_probability_coverage == maximum_coverage
    assert capabilities.full_vocabulary_probability_coverage is full_backend_coverage
    detokenization_chunks = [
        json.loads(request.content)["tokens"]
        for request in backend.requests
        if request.url.path == "/detokenize"
    ]
    assert all(len(chunk) <= 128 for chunk in detokenization_chunks)
    assert sorted(token for chunk in detokenization_chunks for token in chunk) == list(
        range(backend.vocabulary_size)
    )
    padding_chunks = [
        chunk for chunk in detokenization_chunks if min(chunk) >= local_vocabulary_size
    ]
    assert [len(chunk) for chunk in padding_chunks] == [128, 9]
    encoding_probes = [
        request for request in backend.requests if request.url.path == "/tokenize"
    ]
    local_chunks = [
        chunk for chunk in detokenization_chunks if max(chunk) < local_vocabulary_size
    ]
    assert len(encoding_probes) == len(local_chunks)
    assert all(
        json.loads(request.content)["parse_special"] for request in encoding_probes
    )
    probability_probe = next(
        request
        for request in backend.requests
        if request.url.path == "/completion"
        and request.headers.get("Authorization") == f"Bearer {PROBE_KEY}"
    )
    assert json.loads(probability_probe.content)["n_probs"] == maximum_coverage


@pytest.mark.anyio
@pytest.mark.parametrize("manual_mode", [False, True])
@pytest.mark.parametrize("padding_content", ["additional text", "<|control|>", "\x00"])
async def test_startup_rejects_nonempty_backend_vocabulary_extensions(
    metadata_directory: Path,
    manual_mode: bool,
    padding_content: str,
) -> None:
    backend = VocabularyPaddingBackend(
        metadata_directory,
        padding_content=padding_content,
    )
    await assert_startup_fails(
        metadata_directory,
        backend.transport(),
        "trailing token IDs are not empty vocabulary padding",
        skip_deployment_identity_validation=manual_mode,
        startup_tokenizer_probe_chunk_size=128,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("manual_mode", [False, True])
async def test_startup_rejects_backend_vocabulary_smaller_than_local_metadata(
    metadata_directory: Path,
    manual_mode: bool,
) -> None:
    backend = VocabularyPaddingBackend(metadata_directory, padding_size=-1)
    await assert_startup_fails(
        metadata_directory,
        backend.transport(),
        "local tokenizer vocabulary size does not match the backend model",
        skip_deployment_identity_validation=manual_mode,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("backend_padding", [False, True])
async def test_startup_rejects_backend_tokenization_rules_mismatch(
    metadata_directory: Path,
    backend_padding: bool,
) -> None:
    backend = (
        VocabularyPaddingBackend(metadata_directory)
        if backend_padding
        else ControlledBackend(metadata_directory)
    )

    def mismatch(request: httpx.Request) -> httpx.Response:
        if (
            request.url.path == "/tokenize"
            and request.headers.get("Authorization") == f"Bearer {PROBE_KEY}"
        ):
            return httpx.Response(200, json={"tokens": [999]})
        return backend(request)

    await assert_startup_fails(
        metadata_directory,
        httpx.MockTransport(mismatch),
        "tokenization rules do not match",
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("backend_ftype", "configured_quantization"),
    [
        ("Q8_0", "Q4_K_M"),
        ("Q8_0 - unknown-detail", "Q8_0"),
        ("IQ3_S MIXED WITH Q8", "IQ3_M"),
    ],
)
async def test_startup_rejects_backend_quantization_mismatch(
    metadata_directory: Path,
    backend_ftype: str,
    configured_quantization: str,
) -> None:
    manifest_path = metadata_directory / "deployment-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["gguf_quantization"] = configured_quantization
    manifest_path.write_text(json.dumps(manifest))
    backend = ControlledBackend(metadata_directory)

    def mismatch(request: httpx.Request) -> httpx.Response:
        response = backend(request)
        if (
            request.url.path == "/v1/models"
            and request.headers.get("Authorization") == f"Bearer {PROBE_KEY}"
        ):
            body = response.json()
            body["data"][0]["meta"]["ftype"] = backend_ftype
            return httpx.Response(200, json=body)
        return response

    await assert_startup_fails(
        metadata_directory,
        httpx.MockTransport(mismatch),
        "quantization does not match",
        gguf_quantization=configured_quantization,
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            {"completion_probabilities": "invalid"},
            "returned malformed probability data",
        ),
        (
            {
                "completion_probabilities": [
                    {
                        "top_logprobs": [
                            {"id": token_id, "logprob": -float(token_id + 1)}
                            for token_id in range(512)
                        ]
                    }
                ],
                "tokens_cached": 0,
                "tokens_predicted": 1,
                "truncated": False,
            },
            "probability or counter contract is incompatible",
        ),
    ],
)
async def test_startup_rejects_malformed_probability_or_counter_payloads(
    metadata_directory: Path,
    body: dict[str, object],
    message: str,
) -> None:
    backend = ControlledBackend(metadata_directory)

    def malformed(request: httpx.Request) -> httpx.Response:
        if (
            request.url.path == "/completion"
            and request.headers.get("Authorization") == f"Bearer {PROBE_KEY}"
        ):
            return httpx.Response(200, json=body)
        return backend(request)

    await assert_startup_fails(
        metadata_directory,
        httpx.MockTransport(malformed),
        message,
    )


@pytest.mark.anyio
async def test_startup_rejects_unsupported_probability_coverage(
    metadata_directory: Path,
) -> None:
    backend = ControlledBackend(metadata_directory)

    def limited_coverage(request: httpx.Request) -> httpx.Response:
        if (
            request.url.path == "/completion"
            and request.headers.get("Authorization") == f"Bearer {PROBE_KEY}"
        ):
            body = json.loads(request.content)
            body["n_probs"] = 511
            modified = httpx.Request(
                request.method,
                request.url,
                headers=request.headers,
                json=body,
            )
            return backend(modified)
        return backend(request)

    await assert_startup_fails(
        metadata_directory,
        httpx.MockTransport(limited_coverage),
        "does not support configured maximum probability coverage",
    )


@pytest.mark.anyio
async def test_startup_rejects_out_of_vocabulary_probability_token_ids(
    metadata_directory: Path,
) -> None:
    backend = ControlledBackend(metadata_directory)

    def wrong_token_id(request: httpx.Request) -> httpx.Response:
        if (
            request.url.path == "/completion"
            and request.headers.get("Authorization") == f"Bearer {PROBE_KEY}"
        ):
            body = backend(request).json()
            body["completion_probabilities"][0]["top_logprobs"][0]["id"] = 99_999
            return httpx.Response(200, json=body)
        return backend(request)

    await assert_startup_fails(
        metadata_directory,
        httpx.MockTransport(wrong_token_id),
        "returned incompatible token IDs",
    )


@pytest.mark.anyio
async def test_startup_rejects_probability_coverage_without_required_option_id(
    metadata_directory: Path,
) -> None:
    backend = ControlledBackend(metadata_directory)

    def missing_option(request: httpx.Request) -> httpx.Response:
        if (
            request.url.path == "/completion"
            and request.headers.get("Authorization") == f"Bearer {PROBE_KEY}"
        ):
            body = backend(request).json()
            body["completion_probabilities"][0]["top_logprobs"][2]["id"] = 512
            return httpx.Response(200, json=body)
        return backend(request)

    await assert_startup_fails(
        metadata_directory,
        httpx.MockTransport(missing_option),
        "probability or counter contract is incompatible",
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("counter", "value", "message"),
    [
        (
            "tokens_predicted",
            2,
            "probability or counter contract is incompatible",
        ),
        ("tokens_cached", 1, "counters do not represent uncached prompt"),
        ("tokens_evaluated", 1, "counters do not represent uncached prompt"),
    ],
)
async def test_startup_rejects_incompatible_counter_semantics(
    metadata_directory: Path,
    counter: str,
    value: int,
    message: str,
) -> None:
    backend = ControlledBackend(metadata_directory)

    def incompatible_counter(request: httpx.Request) -> httpx.Response:
        if (
            request.url.path == "/completion"
            and request.headers.get("Authorization") == f"Bearer {PROBE_KEY}"
        ):
            body = backend(request).json()
            body[counter] = value
            return httpx.Response(200, json=body)
        return backend(request)

    await assert_startup_fails(
        metadata_directory,
        httpx.MockTransport(incompatible_counter),
        message,
    )


@pytest.mark.anyio
async def test_startup_health_attempts_are_bounded(
    metadata_directory: Path,
) -> None:
    attempts = 0
    backend = ControlledBackend(metadata_directory)

    def unavailable(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        if request.url.path == "/health":
            attempts += 1
            return httpx.Response(503, json={"status": "loading"})
        return backend(request)

    await assert_startup_fails(
        metadata_directory,
        httpx.MockTransport(unavailable),
        "did not become ready after 3 bounded attempts",
    )
    assert attempts == 3


@pytest.mark.anyio
async def test_startup_rejects_transport_failure_after_health_check(
    metadata_directory: Path,
) -> None:
    attempts = 0
    backend = ControlledBackend(metadata_directory)

    def unavailable(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        if request.url.path == "/props":
            attempts += 1
            raise httpx.ConnectError("backend disconnected", request=request)
        return backend(request)

    await assert_startup_fails(
        metadata_directory,
        httpx.MockTransport(unavailable),
        "backend /props was unavailable during startup",
    )
    assert attempts == 1


@pytest.mark.anyio
async def test_startup_probe_has_one_whole_deadline(
    metadata_directory: Path,
) -> None:
    async def blocked(_request: httpx.Request) -> httpx.Response:
        await asyncio.Event().wait()
        raise AssertionError("cancelled startup request must not resume")

    app = create_app(
        configured_settings(
            metadata_directory,
            startup_probe_timeout_seconds=0.01,
        ),
        backend_transport=httpx.MockTransport(blocked),
    )

    with pytest.raises(
        StartupValidationError,
        match="exceeded startup_probe_timeout_seconds",
    ):
        async with app.router.lifespan_context(app):
            raise AssertionError("expired startup must not become ready")


@pytest.mark.anyio
async def test_startup_rejects_insufficient_effective_context_capacity(
    metadata_directory: Path,
) -> None:
    backend = ControlledBackend(metadata_directory)

    def insufficient_context(request: httpx.Request) -> httpx.Response:
        if (
            request.url.path == "/props"
            and request.headers.get("Authorization") == f"Bearer {PROBE_KEY}"
        ):
            return httpx.Response(
                200,
                json={
                    "default_generation_settings": {"n_ctx": 1024},
                    "total_slots": 2,
                    "model_path": "/models/decider-4b-test-Q4_K_M.gguf",
                    "build_info": "llama.cpp-b1234",
                },
            )
        return backend(request)

    await assert_startup_fails(
        metadata_directory,
        httpx.MockTransport(insufficient_context),
        "effective context capacity is smaller",
    )


@pytest.mark.anyio
async def test_startup_rejects_insufficient_backend_slot_capacity(
    metadata_directory: Path,
) -> None:
    backend = ControlledBackend(metadata_directory)

    def insufficient_slots(request: httpx.Request) -> httpx.Response:
        if (
            request.url.path == "/props"
            and request.headers.get("Authorization") == f"Bearer {PROBE_KEY}"
        ):
            body = backend(request).json()
            body["total_slots"] = 1
            return httpx.Response(200, json=body)
        return backend(request)

    await assert_startup_fails(
        metadata_directory,
        httpx.MockTransport(insufficient_slots),
        "slot capacity is smaller than configured backend_slots",
    )


@pytest.mark.anyio
@pytest.mark.parametrize("route", ["/v1/models", "/completion"])
async def test_startup_rejects_backend_routes_without_authentication_enforcement(
    metadata_directory: Path,
    route: str,
) -> None:
    backend = ControlledBackend(metadata_directory)

    def accepts_invalid_key(request: httpx.Request) -> httpx.Response:
        if (
            request.url.path == route
            and request.headers.get("Authorization")
            == f"Bearer {INVALID_PROBE_CREDENTIAL}"
        ):
            return httpx.Response(200, json={})
        return backend(request)

    await assert_startup_fails(
        metadata_directory,
        httpx.MockTransport(accepts_invalid_key),
        f"{route} does not reject invalid bearer credentials",
    )


@pytest.mark.anyio
async def test_rejected_operator_probe_credential_prevents_readiness(
    metadata_directory: Path,
) -> None:
    backend = ControlledBackend(metadata_directory)

    def reject_probe(request: httpx.Request) -> httpx.Response:
        if (
            request.url.path == "/v1/models"
            and request.headers.get("Authorization") == f"Bearer {PROBE_KEY}"
        ):
            return httpx.Response(401, json={"secret": PROBE_KEY})
        return backend(request)

    await assert_startup_fails(
        metadata_directory,
        httpx.MockTransport(reject_probe),
        "operator probe credential was rejected",
    )


@pytest.mark.anyio
async def test_startup_rejects_unpinned_local_metadata_changes(
    metadata_directory: Path,
) -> None:
    (metadata_directory / "tokenizer_config.json").write_text("{}")
    app = create_app(
        configured_settings(metadata_directory),
        backend_transport=ControlledBackend(metadata_directory).transport(),
    )

    with pytest.raises(RuntimeError, match="do not match the deployment manifest"):
        async with app.router.lifespan_context(app):
            raise AssertionError("changed metadata must not become ready")


@pytest.mark.anyio
async def test_startup_rejects_implicit_calibration_defaults(
    metadata_directory: Path,
) -> None:
    config_path = metadata_directory / "decider_config.json"
    config = json.loads(config_path.read_text())
    del config["temperature"]
    config_path.write_text(json.dumps(config))
    write_test_manifest(metadata_directory)
    app = create_app(
        configured_settings(metadata_directory),
        backend_transport=ControlledBackend(metadata_directory).transport(),
    )

    with pytest.raises(RuntimeError, match="explicitly define temperature"):
        async with app.router.lifespan_context(app):
            raise AssertionError("default calibration must not become ready")


def test_configuration_rejects_probability_coverage_above_declared_maximum(
    metadata_directory: Path,
) -> None:
    with pytest.raises(
        ValidationError,
        match="initial_probability_coverage must not exceed",
    ):
        configured_settings(
            metadata_directory,
            initial_probability_coverage=513,
            maximum_probability_coverage=512,
        )


def test_configuration_requires_top_256_initial_probability_coverage(
    metadata_directory: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="initial_probability_coverage must be 256",
    ):
        configured_settings(
            metadata_directory,
            initial_probability_coverage=128,
            maximum_probability_coverage=512,
        )


def test_configuration_rejects_empty_operator_probe_key(
    metadata_directory: Path,
) -> None:
    with pytest.raises(
        ValidationError,
        match="operator_probe_api_key must not be empty",
    ):
        configured_settings(metadata_directory, operator_probe_api_key="")


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"backend_build": "other-build"}, "backend build does not match"),
        ({"backend_model_id": "other-model"}, "model identity does not match"),
        ({"backend_model_path": "/models/other.gguf"}, "model path does not match"),
    ],
)
async def test_startup_rejects_wrong_backend_identity_inputs(
    metadata_directory: Path,
    override: dict[str, object],
    message: str,
) -> None:
    app = create_app(
        configured_settings(metadata_directory, **override),
        backend_transport=ControlledBackend(metadata_directory).transport(),
    )

    with pytest.raises(StartupValidationError, match=message):
        async with app.router.lifespan_context(app):
            raise AssertionError("wrong backend identity must not become ready")


@pytest.mark.anyio
@pytest.mark.parametrize(
    "override",
    [
        {"model_name": "other-decider"},
        {"gguf_revision": "1" * 40},
        {"gguf_quantization": "Q8_0"},
        {"metadata_revision": "2" * 40},
    ],
)
async def test_startup_rejects_local_manifest_identity_mismatch(
    metadata_directory: Path,
    override: dict[str, object],
) -> None:
    app = create_app(
        configured_settings(metadata_directory, **override),
        backend_transport=ControlledBackend(metadata_directory).transport(),
    )

    with pytest.raises(RuntimeError, match="does not match configured"):
        async with app.router.lifespan_context(app):
            raise AssertionError("wrong artifact identity must not become ready")
