from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from starlette.types import ASGIApp, Message, Receive, Scope, Send

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decider_service.app import create_app  # noqa: E402
from tests.readiness_support import (  # noqa: E402
    ControlledBackend,
    completion_response,
    configured_settings,
    write_test_manifest,
    write_test_metadata,
)

FULL_ALPHABET_KEY = "AZaz09-._~+/=="
REJECTED_KEY = "rejected-caller-key"
BACKEND_ORIGIN = "http://127.0.0.1:8080"


class RecordingApp:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.requests: list[dict[str, Any]] = []

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        body_parts: list[bytes] = []

        async def recording_receive() -> Message:
            message = await receive()
            if message["type"] == "http.request":
                body_parts.append(message.get("body", b""))
            return message

        await self.app(scope, recording_receive, send)
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope["headers"]
        }
        self.requests.append(
            {
                "method": scope["method"],
                "path": scope["path"],
                "headers": headers,
                "body": b"".join(body_parts),
            }
        )


@asynccontextmanager
async def local_service(app: ASGIApp) -> AsyncIterator[str]:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            log_level="critical",
            access_log=False,
            lifespan="on",
        )
    )
    thread = threading.Thread(
        target=server.run,
        kwargs={"sockets": [listener]},
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("local TypeSafe service stopped during startup")
        if time.monotonic() >= deadline:
            raise RuntimeError("local TypeSafe service did not start within 10 seconds")
        await asyncio.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await asyncio.to_thread(thread.join, 10)
        listener.close()
        if thread.is_alive():
            raise RuntimeError("local TypeSafe service did not stop within 10 seconds")


def assert_sdk_requests(
    requests: list[dict[str, Any]],
    *,
    service_origin: str,
) -> None:
    assert requests, "the SDK did not reach the local service"
    expected_host = service_origin.removeprefix("http://")
    assert {request["headers"]["host"] for request in requests} == {expected_host}
    assert {request["path"] for request in requests} <= {
        "/v1/models",
        "/v1/systemone",
    }

    primary_requests = [
        request
        for request in requests
        if request["headers"].get("authorization")
        == f"Bearer {FULL_ALPHABET_KEY}"
    ]
    assert primary_requests, "the full-alphabet API key was not sent as a bearer token"
    sdk_requests = [
        request
        for request in primary_requests
        if request["headers"].get("x-typesafe-sdk") == "typesafe-sdk/0.6.0"
    ]
    assert len(sdk_requests) >= 6, "official SDK request headers were not observed"

    null_state_requests = []
    for request in primary_requests:
        if not request["body"]:
            continue
        body = json.loads(request["body"])
        if "state" in body and body["state"] is None:
            null_state_requests.append(request)
    assert len(null_state_requests) == 1, "the SDK did not send exactly one null State"


def assert_backend_requests(requests: list[httpx.Request]) -> None:
    assert requests, "the service did not reach the controlled backend"
    origins = {
        f"{request.url.scheme}://{request.url.host}:{request.url.port}"
        for request in requests
    }
    assert origins == {
        BACKEND_ORIGIN
    }
    caller_requests = [
        request
        for request in requests
        if request.headers.get("authorization")
        in {
            f"Bearer {FULL_ALPHABET_KEY}",
            f"Bearer {REJECTED_KEY}",
        }
    ]
    assert caller_requests
    assert any(
        request.headers["authorization"] == f"Bearer {FULL_ALPHABET_KEY}"
        for request in caller_requests
    )
    assert any(
        request.headers["authorization"] == f"Bearer {REJECTED_KEY}"
        for request in caller_requests
    )


async def run() -> None:
    script = ROOT / "tests" / "typesafe_sdk_interop.mjs"
    if not (ROOT / "node_modules" / "@typesafe-ai" / "sdk").is_dir():
        raise RuntimeError("official SDK is not installed; run `npm ci` first")

    with tempfile.TemporaryDirectory(prefix="typesafe-sdk-interop-") as temporary:
        metadata_directory = write_test_metadata(Path(temporary))
        (metadata_directory / "decider_config.json").write_text(
            json.dumps(
                {
                    "version": "test",
                    "temperature": 1.0,
                    "neutralize_none": True,
                    "isolated_levels": False,
                }
            )
        )
        write_test_manifest(metadata_directory)

        def runtime_backend(request: httpx.Request) -> httpx.Response:
            if request.headers.get("authorization") == f"Bearer {REJECTED_KEY}":
                return httpx.Response(403, json={"error": "rejected test credential"})
            return completion_response(
                {1: 0.1, 2: 0.9},
                request=request if request.url.path == "/completion" else None,
            )

        backend = ControlledBackend(metadata_directory, runtime_backend)
        service = RecordingApp(
            create_app(
                configured_settings(metadata_directory),
                backend_transport=backend.transport(),
            )
        )
        async with local_service(service) as service_origin:
            completed = await asyncio.to_thread(
                subprocess.run,
                ["node", str(script), service_origin],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

        if completed.returncode != 0:
            raise RuntimeError(
                "official SDK check failed\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        result = json.loads(completed.stdout)
        assert result["sdkVersion"] == "0.6.0"
        assert result["serviceOrigin"] == service_origin
        assert result["typedDecisions"] == ["choice", "noul", "score"]
        assert result["mixedDecision"] is True
        assert result["oneLevelScoreRejectedLocally"] is True
        assert result["oneLevelScoreAcceptedOverHTTP"] is True
        assert result["nullStateRejectedByService"] is True
        assert result["errors"] == [401, 403, 422]
        assert_sdk_requests(service.requests, service_origin=service_origin)
        assert_backend_requests(backend.requests)

    print(
        "TypeSafe SDK 0.6.0 interoperability check passed: "
        "models, typed decisions, mixed request, bearer auth, and 401/403/422 errors"
    )


if __name__ == "__main__":
    asyncio.run(run())
