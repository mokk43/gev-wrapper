from __future__ import annotations

import asyncio
import json
import logging
import math
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
import uvicorn
from uvicorn.config import LOGGING_CONFIG

import decider_service.cli as cli_module
from decider_service.app import create_app
from decider_service.config import Settings
from tests.readiness_support import (
    assert_failure_was_logged,
    completion_response,
    configured_settings,
    ready_backend_transport,
    write_test_metadata,
)


class TimingRecord(logging.LogRecord):
    stage: str
    outcome: str
    duration_ms: float
    status_code: int
    row_index: int
    attempt: int
    n_probs: int
    prompt_tokens: int
    backend_prompt_ms: float
    backend_predicted_ms: float


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def metadata_directory(tmp_path: Path) -> Path:
    return write_test_metadata(tmp_path)


def timing_settings(metadata_directory: Path, **overrides: object) -> Settings:
    return configured_settings(
        metadata_directory,
        timing_logging_enabled=True,
        **overrides,
    )


def choice_payload(*, rows: int = 1) -> dict[str, Any]:
    return {
        "model": "decider-4b-q4-k-m",
        "state": "private-state-marker",
        "questions": {
            f"private-question-marker-{index}": {
                "type": "choice",
                "instructions": "private-instruction-marker",
                "criteria": {
                    "private-option-marker": None,
                    "other": None,
                },
            }
            for index in range(rows)
        },
    }


def timing_records(
    caplog: pytest.LogCaptureFixture,
    response: httpx.Response,
) -> list[TimingRecord]:
    request_id = response.headers["x-typesafe-request-id"]
    records = [
        cast(TimingRecord, record)
        for record in caplog.records
        if record.name == "decider_service"
        and record.getMessage().startswith("request_timing ")
        and getattr(record, "request_id", None) == request_id
    ]
    assert records
    for record in records:
        duration = getattr(record, "duration_ms", None)
        assert isinstance(duration, float)
        assert math.isfinite(duration) and duration >= 0
        assert f"request_id={request_id}" in record.getMessage()
        assert f"stage={record.stage}" in record.getMessage()
        assert f"outcome={record.outcome}" in record.getMessage()
    return records


def assert_safe_logs(caplog: pytest.LogCaptureFixture) -> None:
    messages = "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.name == "decider_service"
    )
    for private_value in (
        "private-state-marker",
        "private-question-marker",
        "private-instruction-marker",
        "private-option-marker",
        "private-caller-key",
        "private-backend-content",
        "private-exception-message",
        "operator-probe-key",
    ):
        assert private_value not in messages


@pytest.mark.anyio
async def test_request_timings_separate_backend_duration_and_backend_metrics(
    metadata_directory: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def backend(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.03)
        body = completion_response({1: 0.2, 2: 0.8}, request=request).json()
        body["content"] = "private-backend-content"
        body["timings"].update(prompt_ms=12.5, predicted_ms=3.25)
        return httpx.Response(200, json=body)

    app = create_app(
        timing_settings(metadata_directory),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client,
    ):
        caplog.set_level(logging.INFO, logger="decider_service")
        caplog.clear()
        response = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer private-caller-key"},
            json=choice_payload(),
        )

    assert response.status_code == 200
    records = timing_records(caplog, response)
    stages = {record.stage: record for record in records}
    assert {
        "request_total",
        "admission",
        "preparation",
        "backend_slot_wait",
        "backend_completion",
        "backend_response_validation",
        "backend_evaluation",
        "answer_assembly",
    } <= stages.keys()
    assert all(record.outcome == "success" for record in records)
    completion = stages["backend_completion"]
    assert completion.duration_ms >= 20
    assert stages["backend_evaluation"].duration_ms >= completion.duration_ms
    assert stages["request_total"].duration_ms >= completion.duration_ms
    assert stages["request_total"].status_code == 200
    assert completion.row_index == 0
    assert completion.attempt == 1
    assert completion.n_probs == 256
    assert completion.prompt_tokens > 0
    validation = stages["backend_response_validation"]
    assert validation.backend_prompt_ms == 12.5
    assert validation.backend_predicted_ms == 3.25
    assert_safe_logs(caplog)


@pytest.mark.anyio
async def test_row_timings_identify_retries_and_backend_slot_wait(
    metadata_directory: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def backend(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.03)
        coverage = json.loads(request.content)["n_probs"]
        return completion_response(
            {1: 0.4, 2: 0.6} if coverage == 512 else {1: 0.9},
            request=request,
            coverage=coverage,
            excluded_token_ids=() if coverage == 512 else (2,),
        )

    app = create_app(
        timing_settings(metadata_directory, backend_slots=1),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client,
    ):
        caplog.set_level(logging.INFO, logger="decider_service")
        caplog.clear()
        response = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer private-caller-key"},
            json=choice_payload(rows=2),
        )

    assert response.status_code == 200
    records = timing_records(caplog, response)
    for stage in ("backend_completion", "backend_response_validation"):
        attempts = [record for record in records if record.stage == stage]
        assert {
            (record.row_index, record.attempt, record.n_probs)
            for record in attempts
        } == {(0, 1, 256), (0, 2, 512), (1, 1, 256), (1, 2, 512)}
        assert len(attempts) == 4
    validation = [
        record for record in records if record.stage == "backend_response_validation"
    ]
    assert all(
        record.outcome == ("retry" if record.attempt == 1 else "success")
        for record in validation
    )
    waits = [record for record in records if record.stage == "backend_slot_wait"]
    assert len(waits) == 4
    assert any(record.duration_ms >= 20 for record in waits)
    assert_safe_logs(caplog)


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["deadline", "connection", "invalid_json"])
async def test_failed_backend_work_still_emits_request_and_stage_timings(
    metadata_directory: Path,
    caplog: pytest.LogCaptureFixture,
    failure: str,
) -> None:
    async def backend(request: httpx.Request) -> httpx.Response:
        if failure == "deadline":
            await asyncio.Event().wait()
        if failure == "connection":
            raise httpx.ConnectError("private-exception-message", request=request)
        return httpx.Response(200, text="private-backend-content")

    app = create_app(
        timing_settings(metadata_directory, request_deadline_seconds=0.05),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client,
    ):
        caplog.set_level(logging.INFO, logger="decider_service")
        caplog.clear()
        response = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer private-caller-key"},
            json=choice_payload(),
        )

    assert response.status_code == {
        "deadline": 504,
        "connection": 503,
        "invalid_json": 502,
    }[failure]
    records = timing_records(caplog, response)
    totals = [record for record in records if record.stage == "request_total"]
    assert len(totals) == 1
    assert totals[0].outcome == "error"
    assert totals[0].status_code == response.status_code
    failed_stage = (
        "backend_response_validation" if failure == "invalid_json"
        else "backend_completion"
    )
    failed = [record for record in records if record.stage == failed_stage]
    assert len(failed) == 1
    assert failed[0].outcome == ("cancelled" if failure == "deadline" else "error")
    assert not any(record.stage == "answer_assembly" for record in records)
    assert_safe_logs(caplog)


@pytest.mark.anyio
@pytest.mark.parametrize("valid_credential", [False, True])
async def test_request_total_is_logged_for_authentication_and_validation_rejection(
    metadata_directory: Path,
    caplog: pytest.LogCaptureFixture,
    valid_credential: bool,
) -> None:
    def backend(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("rejected request must not reach the backend")

    app = create_app(
        timing_settings(metadata_directory),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client,
    ):
        caplog.set_level(logging.INFO, logger="decider_service")
        caplog.clear()
        response = await client.post(
            "/v1/systemone",
            headers=(
                {"Authorization": "Bearer private-caller-key"}
                if valid_credential else {}
            ),
            json={"state": "private-state-marker"},
        )

    assert response.status_code == (422 if valid_credential else 401)
    records = timing_records(caplog, response)
    assert len(records) == 1
    assert records[0].stage == "request_total"
    assert records[0].outcome == "error"
    assert records[0].status_code == response.status_code
    assert_safe_logs(caplog)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("prompt_ms", "predicted_ms"),
    [(-1, float("inf")), (True, "private-backend-content"), (float("nan"), None)],
)
async def test_untrusted_backend_timing_values_are_omitted(
    metadata_directory: Path,
    caplog: pytest.LogCaptureFixture,
    prompt_ms: object,
    predicted_ms: object,
) -> None:
    def backend(request: httpx.Request) -> httpx.Response:
        body = completion_response({1: 0.2, 2: 0.8}, request=request).json()
        body["timings"].update(prompt_ms=prompt_ms, predicted_ms=predicted_ms)
        return httpx.Response(200, text=json.dumps(body))

    app = create_app(
        timing_settings(metadata_directory),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client,
    ):
        caplog.set_level(logging.INFO, logger="decider_service")
        caplog.clear()
        response = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer private-caller-key"},
            json=choice_payload(),
        )

    assert response.status_code == 200
    records = timing_records(caplog, response)
    assert any(record.stage == "backend_response_validation" for record in records)
    assert all(not hasattr(record, "backend_prompt_ms") for record in records)
    assert all(not hasattr(record, "backend_predicted_ms") for record in records)
    assert_safe_logs(caplog)


@pytest.mark.anyio
async def test_prepared_answer_times_backend_authentication_without_completion(
    metadata_directory: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    backend_routes: list[str] = []

    async def backend(request: httpx.Request) -> httpx.Response:
        backend_routes.append(request.url.path)
        await asyncio.sleep(0.03)
        return httpx.Response(200, json={"data": []})

    app = create_app(
        timing_settings(metadata_directory),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client,
    ):
        caplog.set_level(logging.INFO, logger="decider_service")
        caplog.clear()
        response = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer private-caller-key"},
            json={
                "model": "decider-4b-q4-k-m",
                "state": "private-state-marker",
                "questions": {"only": {"type": "score", "criteria": ["only"]}},
            },
        )

    assert response.status_code == 200
    assert backend_routes == ["/v1/models"]
    records = timing_records(caplog, response)
    authentication = [
        record for record in records if record.stage == "caller_authentication"
    ]
    assert len(authentication) == 1
    assert authentication[0].outcome == "success"
    assert authentication[0].duration_ms >= 20
    assert not any(record.stage == "backend_completion" for record in records)
    assert_safe_logs(caplog)


@pytest.mark.anyio
async def test_overlapping_requests_keep_timing_ids_and_rows_separate(
    metadata_directory: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    callers_seen: set[str] = set()
    both_callers_started = asyncio.Event()

    async def backend(request: httpx.Request) -> httpx.Response:
        caller = request.headers["Authorization"]
        callers_seen.add(caller)
        if len(callers_seen) == 2:
            both_callers_started.set()
        await both_callers_started.wait()
        body = completion_response({1: 0.2, 2: 0.8}, request=request).json()
        body["timings"]["prompt_ms"] = 11 if caller.endswith("first") else 22
        return httpx.Response(200, json=body)

    app = create_app(
        timing_settings(metadata_directory, backend_slots=3),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client,
    ):
        caplog.set_level(logging.INFO, logger="decider_service")
        caplog.clear()
        responses = await asyncio.wait_for(
            asyncio.gather(
                client.post(
                    "/v1/systemone",
                    headers={"Authorization": "Bearer private-caller-key-first"},
                    json=choice_payload(rows=1),
                ),
                client.post(
                    "/v1/systemone",
                    headers={"Authorization": "Bearer private-caller-key-second"},
                    json=choice_payload(rows=2),
                ),
            ),
            timeout=2,
        )

    assert responses[0].headers["x-typesafe-request-id"] != (
        responses[1].headers["x-typesafe-request-id"]
    )
    for response, row_count, prompt_ms in zip(responses, (1, 2), (11, 22), strict=True):
        assert response.status_code == 200
        records = timing_records(caplog, response)
        totals = [record for record in records if record.stage == "request_total"]
        assert len(totals) == 1
        for stage in ("backend_completion", "backend_response_validation"):
            rows = [record for record in records if record.stage == stage]
            assert len(rows) == row_count
            assert {record.row_index for record in rows} == set(range(row_count))
        validations = [
            record for record in records
            if record.stage == "backend_response_validation"
        ]
        assert all(record.backend_prompt_ms == prompt_ms for record in validations)
    assert_safe_logs(caplog)


def test_cli_logging_config_emits_service_info_without_mutating_uvicorn_defaults(
    metadata_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    defaults = deepcopy(LOGGING_CONFIG)
    supplied: dict[str, Any] = {}

    def run_server(_app: object, **kwargs: Any) -> None:
        supplied.update(kwargs)

    monkeypatch.setattr(sys, "argv", ["decider-service"])
    monkeypatch.setattr(
        cli_module, "load_settings", lambda: configured_settings(metadata_directory)
    )
    monkeypatch.setattr(uvicorn, "run", run_server)
    cli_module.main()

    assert defaults == LOGGING_CONFIG
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, logging, logging.config, sys; "
            "logging.config.dictConfig(json.load(sys.stdin)); "
            "logging.getLogger('decider_service').info('timing-visibility-marker')",
        ],
        input=json.dumps(supplied["log_config"]),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "timing-visibility-marker" in result.stderr


@pytest.mark.anyio
@pytest.mark.parametrize("explicit_setting", [None, False])
async def test_disabled_timing_preserves_responses_and_operational_failure_logs(
    metadata_directory: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    explicit_setting: bool | None,
) -> None:
    monkeypatch.delenv("DECIDER_TIMING_LOGGING_ENABLED", raising=False)
    overrides: dict[str, object] = {}
    if explicit_setting is not None:
        overrides["timing_logging_enabled"] = explicit_setting
    settings = configured_settings(metadata_directory, **overrides)
    assert settings.timing_logging_enabled is False

    def backend(request: httpx.Request) -> httpx.Response:
        if request.headers["Authorization"].endswith("failed"):
            raise httpx.ConnectError("private-exception-message", request=request)
        return completion_response({1: 0.2, 2: 0.8}, request=request)

    app = create_app(
        settings,
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client,
    ):
        caplog.set_level(logging.INFO, logger="decider_service")
        caplog.clear()
        successful = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer private-caller-key"},
            json=choice_payload(),
        )
        failed = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer private-caller-key-failed"},
            json=choice_payload(),
        )

    assert successful.status_code == 200
    assert successful.headers["x-typesafe-request-id"]
    assert failed.status_code == 503
    assert_failure_was_logged(failed, caplog, "backend_unavailable")
    assert not any(
        record.getMessage().startswith("request_timing ")
        for record in caplog.records
        if record.name == "decider_service"
    )
    assert_safe_logs(caplog)
