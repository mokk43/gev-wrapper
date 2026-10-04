from datetime import date, timedelta
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from decider_service.app import create_app
from decider_service.config import Settings
from tests.readiness_support import (
    BACKEND_MODEL_ID,
    BACKEND_MODEL_PATH,
    GGUF_REVISION,
    METADATA_REVISION,
    PROBE_KEY,
    ready_backend_transport,
    write_test_metadata,
)
from tests.readiness_support import (
    configured_settings as base_configured_settings,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def service_client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )


def configured_settings(
    metadata_directory: Path,
    **overrides: object,
) -> Settings:
    if metadata_directory.is_dir():
        write_test_metadata(metadata_directory)
    overrides.setdefault("model_aliases", ())
    return base_configured_settings(metadata_directory, **overrides)


def authenticated_catalog_transport(
    metadata_directory: Path,
) -> httpx.MockTransport:
    return ready_backend_transport(
        metadata_directory,
        lambda _request: httpx.Response(200, json={"data": []}),
    )


@pytest.mark.anyio
async def test_catalog_lists_the_configured_decider_identity(tmp_path: Path) -> None:
    settings = configured_settings(tmp_path)
    app = create_app(
        settings,
        backend_transport=authenticated_catalog_transport(tmp_path),
    )

    async with (
        app.router.lifespan_context(app),
        service_client(app) as client,
    ):
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
    app = create_app(
        settings,
        backend_transport=authenticated_catalog_transport(tmp_path),
    )

    async with (
        app.router.lifespan_context(app),
        service_client(app) as client,
    ):
        response = await client.get(
            "/v1/models",
            headers={"Authorization": "Bearer local-test-key"},
        )

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
                "Compatibility alias routing to Decider model 'decider-4b-q4-k-m'."
            ),
            "release_date": "2025-07-04",
        },
        {
            "name": "jev-latest",
            "description": (
                "Compatibility alias routing to Decider model 'decider-4b-q4-k-m'."
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


def test_binding_defaults_to_loopback(tmp_path: Path) -> None:
    settings = configured_settings(tmp_path)

    assert settings.bind_host.is_loopback
    assert settings.skip_deployment_identity_validation is False


def test_configuration_allows_intentional_network_binding(tmp_path: Path) -> None:
    settings = configured_settings(tmp_path, bind_host="0.0.0.0")
    assert str(settings.bind_host) == "0.0.0.0"


def test_configuration_allows_skipping_deployment_identity_on_loopback(
    tmp_path: Path,
) -> None:
    settings = configured_settings(
        tmp_path,
        backend_build=None,
        backend_model_id=None,
        backend_model_path=None,
        gguf_revision=None,
        gguf_quantization=None,
        metadata_revision=None,
        skip_deployment_identity_validation=True,
    )

    assert settings.skip_deployment_identity_validation is True


def test_configuration_requires_identity_settings_by_default(tmp_path: Path) -> None:
    with pytest.raises(
        ValidationError,
        match="deployment identity settings are required: backend_build",
    ):
        configured_settings(tmp_path, backend_build=None)


def test_configuration_rejects_skipping_deployment_identity_on_network_bind(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValidationError,
        match="skip_deployment_identity_validation requires a loopback bind_host",
    ):
        configured_settings(
            tmp_path,
            bind_host="0.0.0.0",
            skip_deployment_identity_validation=True,
        )


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
    monkeypatch.chdir(tmp_path)
    write_test_metadata(tmp_path)
    environment = {
        "DECIDER_BACKEND_URL": "http://127.0.0.1:8080",
        "DECIDER_BACKEND_BUILD": "llama.cpp-b1234",
        "DECIDER_BACKEND_MODEL_ID": BACKEND_MODEL_ID,
        "DECIDER_BACKEND_MODEL_PATH": BACKEND_MODEL_PATH,
        "DECIDER_GGUF_REVISION": GGUF_REVISION,
        "DECIDER_GGUF_QUANTIZATION": "Q4_K_M",
        "DECIDER_METADATA_DIRECTORY": str(tmp_path),
        "DECIDER_METADATA_REVISION": METADATA_REVISION,
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
        "DECIDER_MAXIMUM_PROBABILITY_COVERAGE": "512",
        "DECIDER_OPERATOR_PROBE_API_KEY": PROBE_KEY,
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    from decider_service.app import create_app_from_environment

    app = create_app_from_environment(
        backend_transport=authenticated_catalog_transport(tmp_path)
    )
    async with (
        app.router.lifespan_context(app),
        service_client(app) as client,
    ):
        response = await client.get(
            "/v1/models",
            headers={"Authorization": "Bearer local-test-key"},
        )

    assert [model["name"] for model in response.json()["models"]] == [
        "decider-4b-q4-k-m",
        "jev-latest",
    ]


@pytest.mark.anyio
async def test_public_catalog_check_validates_the_wire_response(
    tmp_path: Path,
) -> None:
    from decider_service.catalog_check import fetch_catalog

    app = create_app(
        configured_settings(tmp_path),
        backend_transport=authenticated_catalog_transport(tmp_path),
    )
    async with app.router.lifespan_context(app):
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
        backend_transport=ready_backend_transport(tmp_path, respond),
    )

    async with app.router.lifespan_context(app):
        backend_client: httpx.AsyncClient = app.state.backend_client
        response = await backend_client.get("/v1/models")

        assert response.status_code == 200
        assert requests[0].url == "http://127.0.0.1:8080/v1/models"
        assert not backend_client.is_closed

    assert backend_client.is_closed
