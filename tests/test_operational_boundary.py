from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from decider_service.app import create_app
from decider_service.readiness import StartupValidationError
from tests.readiness_support import (
    assert_failure_was_logged,
    configured_settings,
    ready_backend_transport,
    write_test_metadata,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def metadata_directory(tmp_path: Path) -> Path:
    return write_test_metadata(tmp_path)


def service_client(
    app: FastAPI,
    *,
    raise_app_exceptions: bool = True,
) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(
            app=app,
            raise_app_exceptions=raise_app_exceptions,
        ),
        base_url="http://test",
    )


def choice_request_payload() -> dict[str, Any]:
    return {
        "model": "decider-4b-q4-k-m",
        "state": "evidence",
        "questions": {
            "priority": {
                "type": "choice",
                "criteria": {"urgent": None, "routine": None},
            }
        },
    }


def one_level_score_request_payload() -> dict[str, Any]:
    return {
        "model": "decider-4b-q4-k-m",
        "state": "evidence",
        "questions": {
            "priority": {
                "type": "score",
                "criteria": ["only level"],
            }
        },
    }


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("method", "path", "json"),
    [
        ("GET", "/v1/models", None),
        ("POST", "/v1/systemone", choice_request_payload()),
        ("POST", "/v1/systemone", {"secret-state": "must-not-be-read"}),
    ],
)
async def test_public_endpoints_reject_missing_credentials_before_request_work(
    metadata_directory: Path,
    method: str,
    path: str,
    json: object | None,
) -> None:
    app = create_app(configured_settings(metadata_directory))

    async with service_client(app) as client:
        response = await client.request(method, path, json=json)

    assert response.status_code == 401
    assert response.json() == {"detail": "Bearer credential required."}
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.headers["x-typesafe-request-id"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "authorization",
    [
        "Basic caller-key",
        "Bearer",
        "Bearer ",
        "Bearer  caller-key",
        "Bearer caller key",
        "Bearer caller-key,other-key",
    ],
)
@pytest.mark.parametrize(
    ("method", "path", "json"),
    [
        ("GET", "/v1/models", None),
        ("POST", "/v1/systemone", choice_request_payload()),
    ],
)
async def test_public_endpoints_reject_malformed_bearer_credentials(
    metadata_directory: Path,
    authorization: str,
    method: str,
    path: str,
    json: object | None,
) -> None:
    app = create_app(configured_settings(metadata_directory))

    async with service_client(app) as client:
        response = await client.request(
            method,
            path,
            headers={"Authorization": authorization},
            json=json,
        )

    assert response.status_code == 401
    assert response.json() == {"detail": "Bearer credential required."}


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("method", "path", "json"),
    [
        ("GET", "/v1/models", None),
        ("POST", "/v1/systemone", choice_request_payload()),
    ],
)
async def test_operator_probe_credential_cannot_authenticate_public_requests(
    metadata_directory: Path,
    method: str,
    path: str,
    json: object | None,
) -> None:
    app = create_app(configured_settings(metadata_directory))

    async with service_client(app) as client:
        response = await client.request(
            method,
            path,
            headers={"Authorization": "Bearer operator-probe-key"},
            json=json,
        )

    assert response.status_code == 401
    assert response.json() == {"detail": "Bearer credential required."}


@pytest.mark.anyio
async def test_catalog_forwards_only_the_callers_credential_to_the_backend(
    metadata_directory: Path,
) -> None:
    caller_requests: list[httpx.Request] = []

    def backend(request: httpx.Request) -> httpx.Response:
        caller_requests.append(request)
        return httpx.Response(200, json={"data": []})

    app = create_app(
        configured_settings(metadata_directory),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )

    async with (
        app.router.lifespan_context(app),
        service_client(app) as client,
    ):
        response = await client.get(
            "/v1/models",
            headers={"Authorization": "bearer caller-key"},
        )
        assert "Authorization" not in app.state.backend_client.headers

    assert response.status_code == 200
    assert response.json()["models"][0]["name"] == "decider-4b-q4-k-m"
    assert len(caller_requests) == 1
    assert caller_requests[0].url.path == "/v1/models"
    assert caller_requests[0].headers["Authorization"] == "Bearer caller-key"
    assert caller_requests[0].content == b""


@pytest.mark.anyio
@pytest.mark.parametrize("backend_status", [401, 403])
@pytest.mark.parametrize(
    ("path", "json"),
    [
        ("/v1/models", None),
        ("/v1/systemone", one_level_score_request_payload()),
    ],
)
async def test_backend_credential_rejection_is_sanitized_without_probe_fallback(
    metadata_directory: Path,
    caplog: pytest.LogCaptureFixture,
    backend_status: int,
    path: str,
    json: object | None,
) -> None:
    caplog.set_level(logging.INFO, logger="decider_service")
    caller_requests: list[httpx.Request] = []

    def backend(request: httpx.Request) -> httpx.Response:
        caller_requests.append(request)
        return httpx.Response(
            backend_status,
            json={"error": "upstream-secret-marker"},
        )

    app = create_app(
        configured_settings(metadata_directory),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )

    async with (
        app.router.lifespan_context(app),
        service_client(app) as client,
    ):
        response = await client.request(
            "GET" if json is None else "POST",
            path,
            headers={"Authorization": "Bearer rejected-caller-key"},
            json=json,
        )

    assert response.status_code == backend_status
    assert response.json() == {"detail": "Backend rejected caller credentials."}
    assert "upstream-secret-marker" not in response.text
    assert response.headers["x-typesafe-request-id"]
    assert [request.headers["Authorization"] for request in caller_requests] == [
        "Bearer rejected-caller-key"
    ]
    assert all(
        "operator-probe-key" not in request.headers["Authorization"]
        for request in caller_requests
    )
    messages = assert_failure_was_logged(
        response,
        caplog,
        "caller_authentication",
    )
    assert "upstream-secret-marker" not in messages
    assert "rejected-caller-key" not in messages
    assert "operator-probe-key" not in messages


@pytest.mark.anyio
async def test_validation_errors_keep_only_field_oriented_contract_data(
    metadata_directory: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="decider_service")
    app = create_app(configured_settings(metadata_directory))

    async with service_client(app) as client:
        response = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer secret-caller-key"},
            json={
                "model": "decider-4b-q4-k-m",
                "state": "secret-state-marker",
                "questions": "secret-invalid-questions-marker",
            },
        )

    assert response.status_code == 422
    assert all(
        set(error) == {"loc", "msg", "type"} for error in response.json()["detail"]
    )
    messages = assert_failure_was_logged(
        response,
        caplog,
        "request_validation",
    )
    assert "secret-state-marker" not in response.text + messages
    assert "secret-invalid-questions-marker" not in response.text + messages
    assert "secret-caller-key" not in response.text + messages


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("backend_status", "expected_status", "expected_detail"),
    [
        (
            400,
            502,
            "Backend response did not satisfy the catalog contract.",
        ),
        (503, 503, "Decision backend unavailable."),
    ],
)
async def test_catalog_backend_failures_are_sanitized_and_logged(
    metadata_directory: Path,
    caplog: pytest.LogCaptureFixture,
    backend_status: int,
    expected_status: int,
    expected_detail: str,
) -> None:
    caplog.set_level(logging.INFO, logger="decider_service")

    def backend(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            backend_status,
            text="secret-upstream-response-marker",
        )

    app = create_app(
        configured_settings(metadata_directory),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )

    async with (
        app.router.lifespan_context(app),
        service_client(app) as client,
    ):
        response = await client.get(
            "/v1/models",
            headers={"Authorization": "Bearer secret-caller-key"},
        )

    assert response.status_code == expected_status
    assert response.json() == {"detail": expected_detail}
    messages = assert_failure_was_logged(
        response,
        caplog,
        "backend_contract" if expected_status == 502 else "backend_unavailable",
    )
    assert "secret-upstream-response-marker" not in response.text + messages
    assert "secret-caller-key" not in response.text + messages


@pytest.mark.anyio
async def test_unexpected_runtime_exceptions_are_sanitized_and_logged(
    metadata_directory: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="decider_service")

    def backend(_request: httpx.Request) -> httpx.Response:
        raise RuntimeError("secret-upstream-exception-marker")

    app = create_app(
        configured_settings(metadata_directory),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )

    async with (
        app.router.lifespan_context(app),
        service_client(app, raise_app_exceptions=False) as client,
    ):
        response = await client.get(
            "/v1/models",
            headers={"Authorization": "Bearer secret-caller-key"},
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal service error."}
    messages = assert_failure_was_logged(
        response,
        caplog,
        "unexpected_service_error",
    )
    assert "secret-upstream-exception-marker" not in response.text + messages
    assert "secret-caller-key" not in response.text + messages


@pytest.mark.anyio
async def test_startup_transport_exceptions_hide_upstream_diagnostics(
    metadata_directory: Path,
) -> None:
    def unavailable(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        raise httpx.ConnectError(
            "secret-startup-exception-marker",
            request=request,
        )

    app = create_app(
        configured_settings(metadata_directory),
        backend_transport=httpx.MockTransport(unavailable),
    )

    with pytest.raises(StartupValidationError) as captured:
        async with app.router.lifespan_context(app):
            pass

    diagnostic = "".join(traceback.format_exception(captured.value))
    assert "secret-startup-exception-marker" not in diagnostic
