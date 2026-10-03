from datetime import date, timedelta
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from decider_service.app import create_app
from decider_service.config import Settings


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def service_client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )


def configured_settings(metadata_directory: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "backend_url": "http://127.0.0.1:8080",
        "backend_build": "llama.cpp-b1234",
        "gguf_revision": "b79f09d9ba7837f1b744295ea267b55d08e958ec",
        "gguf_quantization": "Q4_K_M",
        "metadata_directory": metadata_directory,
        "metadata_revision": "b79f09d9ba7837f1b744295ea267b55d08e958ec",
        "model_name": "decider-4b-q4-k-m",
        "model_description": "Decider 4B served by the configured llama.cpp backend.",
        "model_release_date": date(2025, 7, 4),
        "context_capacity": 32_768,
        "backend_slots": 2,
        "admission_capacity": 8,
        "max_request_bytes": 1_048_576,
        "max_questions": 32,
        "max_options": 255,
    }
    values.update(overrides)
    return Settings.model_validate(values)


@pytest.mark.anyio
async def test_catalog_lists_the_configured_decider_identity(tmp_path: Path) -> None:
    app = create_app(configured_settings(tmp_path))

    async with service_client(app) as client:
        response = await client.get(
            "/v1/models",
            headers={"Authorization": "Bearer local-test-key"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "models": [
            {
                "name": "decider-4b-q4-k-m",
                "description": "Decider 4B served by the configured llama.cpp backend.",
                "release_date": "2025-07-04",
            }
        ]
    }


@pytest.mark.anyio
async def test_catalog_lists_only_explicit_compatibility_aliases(
    tmp_path: Path,
) -> None:
    settings = configured_settings(
        tmp_path,
        model_aliases=("legacy-decider", "jev-latest"),
    )
    app = create_app(settings)

    async with service_client(app) as client:
        response = await client.get("/v1/models")

    assert response.status_code == 200
    assert response.json()["models"] == [
        {
            "name": "decider-4b-q4-k-m",
            "description": "Decider 4B served by the configured llama.cpp backend.",
            "release_date": "2025-07-04",
        },
        {
            "name": "legacy-decider",
            "description": (
                "Compatibility alias routing to Decider model "
                "'decider-4b-q4-k-m'."
            ),
            "release_date": "2025-07-04",
        },
        {
            "name": "jev-latest",
            "description": (
                "Compatibility alias routing to Decider model "
                "'decider-4b-q4-k-m'."
            ),
            "release_date": "2025-07-04",
        },
    ]


def test_configuration_rejects_duplicate_aliases(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="model_aliases must be unique"):
        configured_settings(
            tmp_path,
            model_aliases=("legacy-decider", "legacy-decider"),
        )


def test_configuration_rejects_canonical_name_as_alias(tmp_path: Path) -> None:
    with pytest.raises(
        ValidationError,
        match="model_aliases must not repeat model_name",
    ):
        configured_settings(
            tmp_path,
            model_aliases=("decider-4b-q4-k-m",),
        )


def test_configuration_rejects_non_string_alias(tmp_path: Path) -> None:
    with pytest.raises(
        ValidationError,
        match="model_aliases must contain only strings",
    ):
        configured_settings(tmp_path, model_aliases=(123,))


@pytest.mark.parametrize(
    ("field", "floating_revision"),
    [
        ("gguf_revision", "main"),
        ("metadata_revision", "latest"),
        ("gguf_revision", "develop"),
        ("metadata_revision", "release"),
    ],
)
def test_configuration_rejects_floating_artifact_revisions(
    tmp_path: Path,
    field: str,
    floating_revision: str,
) -> None:
    with pytest.raises(ValidationError, match="must identify an immutable revision"):
        configured_settings(tmp_path, **{field: floating_revision})


def test_catalog_slice_rejects_non_loopback_binding(tmp_path: Path) -> None:
    with pytest.raises(
        ValidationError,
        match="must be loopback until authenticated network exposure is implemented",
    ):
        configured_settings(tmp_path, bind_host="0.0.0.0")


def test_configuration_requires_bounded_admission_for_backend_slots(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValidationError,
        match="admission_capacity must be at least backend_slots",
    ):
        configured_settings(
            tmp_path,
            backend_slots=4,
            admission_capacity=3,
        )


def test_configuration_rejects_future_model_release_date(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="must not be in the future"):
        configured_settings(
            tmp_path,
            model_release_date=date.today() + timedelta(days=1),
        )


@pytest.mark.parametrize("field", ["model_description", "model_release_date"])
def test_configuration_requires_catalog_metadata(
    tmp_path: Path,
    field: str,
) -> None:
    with pytest.raises(ValidationError):
        configured_settings(tmp_path, **{field: None})


def test_configuration_requires_existing_metadata_directory(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidationError, match="Path does not point to a directory"):
        configured_settings(tmp_path / "missing")


@pytest.mark.anyio
async def test_environment_factory_serves_configured_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = {
        "DECIDER_BACKEND_URL": "http://127.0.0.1:8080",
        "DECIDER_BACKEND_BUILD": "llama.cpp-b1234",
        "DECIDER_GGUF_REVISION": "b79f09d9ba7837f1b744295ea267b55d08e958ec",
        "DECIDER_GGUF_QUANTIZATION": "Q4_K_M",
        "DECIDER_METADATA_DIRECTORY": str(tmp_path),
        "DECIDER_METADATA_REVISION": "b79f09d9ba7837f1b744295ea267b55d08e958ec",
        "DECIDER_MODEL_NAME": "decider-4b-q4-k-m",
        "DECIDER_MODEL_DESCRIPTION": "Configured Decider model.",
        "DECIDER_MODEL_RELEASE_DATE": "2025-07-04",
        "DECIDER_MODEL_ALIASES": '["jev-latest"]',
        "DECIDER_CONTEXT_CAPACITY": "32768",
        "DECIDER_BACKEND_SLOTS": "2",
        "DECIDER_ADMISSION_CAPACITY": "8",
        "DECIDER_MAX_REQUEST_BYTES": "1048576",
        "DECIDER_MAX_QUESTIONS": "32",
        "DECIDER_MAX_OPTIONS": "255",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    from decider_service.app import create_app_from_environment

    app = create_app_from_environment()
    async with service_client(app) as client:
        response = await client.get("/v1/models")

    assert [model["name"] for model in response.json()["models"]] == [
        "decider-4b-q4-k-m",
        "jev-latest",
    ]


@pytest.mark.anyio
async def test_public_catalog_check_validates_the_wire_response(
    tmp_path: Path,
) -> None:
    from decider_service.catalog_check import fetch_catalog

    app = create_app(configured_settings(tmp_path))
    catalog = await fetch_catalog(
        "http://test",
        api_key="local-test-key",
        transport=httpx.ASGITransport(app=app),
    )

    assert catalog.models[0].name == "decider-4b-q4-k-m"


@pytest.mark.anyio
async def test_application_lifecycle_owns_the_injected_backend_boundary(
    tmp_path: Path,
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": []})

    app = create_app(
        configured_settings(tmp_path),
        backend_transport=httpx.MockTransport(respond),
    )

    async with app.router.lifespan_context(app):
        backend_client: httpx.AsyncClient = app.state.backend_client
        response = await backend_client.get("/v1/models")

        assert response.status_code == 200
        assert requests[0].url == "http://127.0.0.1:8080/v1/models"
        assert not backend_client.is_closed

    assert backend_client.is_closed
