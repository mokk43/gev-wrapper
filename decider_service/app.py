from __future__ import annotations

import asyncio
import logging
import re
import uuid
from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager
from datetime import date
from typing import Any, TypeVar, cast

import httpx
from decider.systemone import MAX_LEVELS  # type: ignore[import-untyped]
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from decider_service.config import Settings, load_settings
from decider_service.contracts import SystemOneRequest, SystemOneResponse, Usage
from decider_service.decision import (
    AdmissionCapacityError,
    BackendContractError,
    BackendUnavailableError,
    CallerAuthenticationError,
    DecisionCapacity,
    DecisionResult,
    DecisionRuntime,
    ProbabilityCoverage,
    PublicInputError,
    authenticate_caller,
    evaluate_request,
)
from decider_service.readiness import validate_backend_deployment
from decider_service.timing import measure_stage, request_id_context


class ModelMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    release_date: date


class ModelMetadataList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: list[ModelMetadata]


_REQUEST_TOO_LARGE_DETAIL = [
    {
        "loc": ["body"],
        "msg": "request body exceeds configured byte limit",
        "type": "request_too_large",
    }
]
_Result = TypeVar("_Result")
_PUBLIC_OPERATIONS = {
    ("GET", "/v1/models"),
    ("POST", "/v1/systemone"),
}
_BEARER_CREDENTIAL = re.compile(rb"(?i:Bearer) ([A-Za-z0-9\-._~+/]+={0,})\Z")
_REQUEST_ID_HEADER = "x-typesafe-request-id"
_LOGGER = logging.getLogger("decider_service")


def _scope_state(scope: Scope) -> dict[str, Any]:
    state = scope.setdefault("state", {})
    return cast(dict[str, Any], state)


def _log_scope_failure(
    scope: Scope,
    status_code: int,
    category: str,
    *,
    exception_type: str | None = None,
) -> None:
    state = _scope_state(scope)
    fields = (
        state.get("request_id", "unavailable"),
        scope.get("method", "unavailable"),
        scope.get("path", "unavailable"),
        status_code,
        category,
    )
    if exception_type is None:
        _LOGGER.info(
            "request_failed request_id=%s method=%s path=%s status=%d category=%s",
            *fields,
        )
        return
    _LOGGER.error(
        "request_failed request_id=%s method=%s path=%s status=%d category=%s "
        "exception_type=%s",
        *fields,
        exception_type,
    )


def _public_http_error(
    request: Request,
    status_code: int,
    detail: str | list[dict[str, object]],
    category: str,
) -> HTTPException:
    request.state.failure_category = category
    return HTTPException(status_code=status_code, detail=detail)


def _safe_validation_message(kind: str) -> str:
    messages = {
        "missing": "Field required",
        "extra_forbidden": "Extra inputs are not permitted",
        "dict_type": "Input should be a valid object",
        "list_type": "Input should be a valid array",
        "string_type": "Input should be a valid string",
        "string_too_short": "String does not meet the minimum length",
        "too_short": "Value does not meet the minimum length",
        "union_tag_invalid": "Question type must be choice, noul, or score",
        "union_tag_not_found": "Question type is required",
    }
    return messages.get(kind, "Invalid value")


class RequestIdentifierMiddleware:
    def __init__(self, app: ASGIApp, timing_logging_enabled: bool = False) -> None:
        self._app = app
        self._timing_logging_enabled = timing_logging_enabled

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        request_id = uuid.uuid4().hex
        _scope_state(scope)["request_id"] = request_id

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers[_REQUEST_ID_HEADER] = request_id
            await send(message)

        if not self._timing_logging_enabled or (
            scope.get("method"), scope.get("path")
        ) != ("POST", "/v1/systemone"):
            await self._app(scope, receive, send_with_request_id)
            return

        context_token = request_id_context.set(request_id)
        try:
            with measure_stage("request_total", status_code=500) as timing:

                async def send_with_timing(message: Message) -> None:
                    if message["type"] == "http.response.start":
                        status_code = message["status"]
                        timing.details["status_code"] = status_code
                        if status_code >= 400:
                            timing.outcome = "error"
                    await send_with_request_id(message)

                try:
                    await self._app(scope, receive, send_with_timing)
                except asyncio.CancelledError:
                    timing.details["status_code"] = 499
                    raise
        finally:
            request_id_context.reset(context_token)


class CallerAuthenticationMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        operator_probe_api_key: str | None,
    ) -> None:
        self._app = app
        self._operator_probe_api_key = (
            operator_probe_api_key.encode("utf-8")
            if operator_probe_api_key is not None
            else None
        )

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        operation = (scope.get("method"), scope.get("path"))
        if scope["type"] != "http" or operation not in _PUBLIC_OPERATIONS:
            await self._app(scope, receive, send)
            return
        authorization_values = [
            value
            for name, value in scope.get("headers", [])
            if name.lower() == b"authorization"
        ]
        match = (
            _BEARER_CREDENTIAL.fullmatch(authorization_values[0])
            if len(authorization_values) == 1
            else None
        )
        if match is None or (
            self._operator_probe_api_key is not None
            and match.group(1) == self._operator_probe_api_key
        ):
            _log_scope_failure(scope, 401, "caller_authentication")
            response = JSONResponse(
                status_code=401,
                content={"detail": "Bearer credential required."},
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return
        try:
            bearer_token = match.group(1).decode("ascii")
        except UnicodeDecodeError:
            _log_scope_failure(scope, 401, "caller_authentication")
            response = JSONResponse(
                status_code=401,
                content={"detail": "Bearer credential required."},
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return
        _scope_state(scope)["bearer_token"] = bearer_token
        await self._app(scope, receive, send)


class ActiveRequests:
    def __init__(self) -> None:
        self._work_tasks: set[asyncio.Task[Any]] = set()
        self._request_tasks: set[asyncio.Task[Any]] = set()
        self._idle = asyncio.Event()
        self._idle.set()
        self._closing = False

    async def run(
        self,
        work: Coroutine[Any, Any, _Result],
        request: Request,
    ) -> _Result:
        if self._closing:
            work.close()
            raise RuntimeError("decision service is shutting down")
        request_task = asyncio.current_task()
        if request_task is None:
            work.close()
            raise RuntimeError("decision request has no owning task")
        work_task = asyncio.create_task(work)
        disconnected = asyncio.create_task(self._wait_for_disconnect(request))
        self._work_tasks.add(work_task)
        self._request_tasks.add(request_task)
        self._idle.clear()
        try:
            completed, _pending = await asyncio.wait(
                (work_task, disconnected),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if work_task in completed:
                return await work_task
            work_task.cancel()
            await asyncio.gather(work_task, return_exceptions=True)
            raise asyncio.CancelledError
        except BaseException:
            work_task.cancel()
            await asyncio.gather(work_task, return_exceptions=True)
            raise
        finally:
            disconnected.cancel()
            await asyncio.gather(disconnected, return_exceptions=True)
            self._work_tasks.discard(work_task)
            self._request_tasks.discard(request_task)
            if not self._request_tasks:
                self._idle.set()

    @staticmethod
    async def _wait_for_disconnect(request: Request) -> None:
        while True:
            if (await request.receive())["type"] == "http.disconnect":
                return

    async def close(self) -> None:
        self._closing = True
        work_tasks = tuple(self._work_tasks)
        for work_task in work_tasks:
            work_task.cancel()
        if work_tasks:
            await asyncio.gather(*work_tasks, return_exceptions=True)
        await self._idle.wait()


def _request_too_large_response() -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": _REQUEST_TOO_LARGE_DETAIL},
    )


class RequestBodyLimitMiddleware:
    def __init__(self, app: ASGIApp, maximum_bytes: int) -> None:
        self._app = app
        self._maximum_bytes = maximum_bytes

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != "/v1/systemone"
        ):
            await self._app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError:
                declared_size = -1
            if declared_size > self._maximum_bytes:
                _log_scope_failure(scope, 422, "request_too_large")
                await _request_too_large_response()(scope, receive, send)
                return

        received_bytes = 0

        async def limited_receive() -> Message:
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self._maximum_bytes:
                    _scope_state(scope)["failure_category"] = "request_too_large"
                    raise HTTPException(
                        status_code=422,
                        detail=_REQUEST_TOO_LARGE_DETAIL,
                    )
            return message

        await self._app(scope, limited_receive, send)


def _field_validation_error(
    location: tuple[str | int, ...],
    message: str,
    kind: str,
) -> HTTPException:
    return HTTPException(
        status_code=422,
        detail=[
            {
                "loc": ["body", *location],
                "msg": message,
                "type": kind,
            }
        ],
    )


def create_app(
    settings: Settings,
    *,
    backend_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with httpx.AsyncClient(
            base_url=str(settings.backend_url),
            transport=backend_transport,
        ) as backend_client:
            app.state.backend_client = backend_client
            app.state.decision_capacity = DecisionCapacity(
                backend_slots=settings.backend_slots,
                admission_capacity=settings.admission_capacity,
            )
            app.state.active_requests = ActiveRequests()
            try:
                app.state.decision_runtime = (
                    await app.state.decision_capacity.run_offloaded(
                        lambda: DecisionRuntime(settings)
                    )
                )
                if settings.skip_deployment_identity_validation:
                    _LOGGER.warning(
                        "deployment identity validation skipped for loopback manual run"
                    )
                app.state.backend_capabilities = await validate_backend_deployment(
                    settings,
                    app.state.decision_runtime,
                    backend_client,
                    validate_identity=(
                        not settings.skip_deployment_identity_validation
                    ),
                )
                yield
            finally:
                await app.state.active_requests.close()
                await app.state.decision_capacity.close()

    app = FastAPI(title="TypeSafe-compatible Decider service", lifespan=lifespan)
    app.add_middleware(
        RequestBodyLimitMiddleware,
        maximum_bytes=settings.max_request_bytes,
    )
    app.add_middleware(
        CallerAuthenticationMiddleware,
        operator_probe_api_key=(
            None
            if settings.skip_deployment_identity_validation
            else settings.operator_probe_api_key.get_secret_value()
        ),
    )
    app.add_middleware(
        RequestIdentifierMiddleware,
        timing_logging_enabled=settings.timing_logging_enabled,
    )

    @app.exception_handler(RequestValidationError)
    async def field_oriented_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        errors: list[dict[str, object]] = []
        for error in exc.errors():
            kind = str(error["type"])
            location = tuple(error["loc"])
            if kind in {"union_tag_invalid", "union_tag_not_found"}:
                location = (*location, "type")
            errors.append(
                {
                    "loc": location,
                    "msg": _safe_validation_message(kind),
                    "type": kind,
                }
            )
        _log_scope_failure(request.scope, 422, "request_validation")
        return JSONResponse(
            status_code=422,
            content={"detail": jsonable_encoder(errors)},
        )

    @app.exception_handler(StarletteHTTPException)
    async def operational_http_error(
        request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        category = getattr(
            request.state,
            "failure_category",
            f"http_{exc.status_code}",
        )
        _log_scope_failure(request.scope, exc.status_code, category)
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": jsonable_encoder(exc.detail)},
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def unexpected_service_error(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        _log_scope_failure(
            request.scope,
            500,
            "unexpected_service_error",
            exception_type=type(exc).__name__,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal service error."},
            headers={
                _REQUEST_ID_HEADER: cast(
                    str,
                    request.state.request_id,
                )
            },
        )

    @app.get("/v1/models", response_model=ModelMetadataList)
    async def list_models(http_request: Request) -> ModelMetadataList:
        deadline = asyncio.get_running_loop().time() + settings.request_deadline_seconds
        try:
            async with asyncio.timeout_at(deadline):
                await authenticate_caller(
                    app.state.backend_client,
                    app.state.decision_capacity,
                    cast(str, http_request.state.bearer_token),
                    deadline,
                )
        except TimeoutError as exc:
            raise _public_http_error(
                http_request,
                504,
                "Model catalog request deadline expired.",
                "deadline",
            ) from exc
        except BackendContractError as exc:
            raise _public_http_error(
                http_request,
                502,
                "Backend response did not satisfy the catalog contract.",
                "backend_contract",
            ) from exc
        except BackendUnavailableError as exc:
            raise _public_http_error(
                http_request,
                503,
                "Decision backend unavailable.",
                "backend_unavailable",
            ) from exc
        except CallerAuthenticationError as exc:
            raise _public_http_error(
                http_request,
                exc.status_code,
                "Backend rejected caller credentials.",
                "caller_authentication",
            ) from exc
        release_date = settings.model_release_date
        return ModelMetadataList(
            models=[
                ModelMetadata(
                    name=settings.model_name,
                    description=settings.model_description,
                    release_date=release_date,
                ),
                *(
                    ModelMetadata(
                        name=alias,
                        description=(
                            "Compatibility alias routing to Decider model "
                            f"'{settings.model_name}'."
                        ),
                        release_date=release_date,
                    )
                    for alias in settings.model_aliases
                ),
            ]
        )

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    async def system_one(
        request: SystemOneRequest,
        http_request: Request,
    ) -> SystemOneResponse:
        bearer_token = cast(str, http_request.state.bearer_token)
        if request.model not in (settings.model_name, *settings.model_aliases):
            raise _field_validation_error(
                ("model",),
                "unknown configured model",
                "unknown_model",
            )
        if len(request.questions) > settings.max_questions:
            raise _field_validation_error(
                ("questions",),
                "question count exceeds configured limit",
                "too_many_questions",
            )
        for name, question in request.questions.items():
            if question.type == "choice":
                option_count = len(question.criteria)
                maximum_options = settings.max_options
            elif question.type == "score":
                option_count = len(question.criteria)
                maximum_options = min(settings.max_options, MAX_LEVELS)
            else:
                continue
            if option_count > maximum_options:
                raise _field_validation_error(
                    ("questions", name, "criteria"),
                    "option count exceeds configured capacity",
                    "option_capacity",
                )
        deadline = asyncio.get_running_loop().time() + settings.request_deadline_seconds

        async def evaluate_admitted_request() -> DecisionResult:
            async with app.state.decision_capacity.admit():
                runtime = app.state.decision_runtime
                capabilities = app.state.backend_capabilities
                return await evaluate_request(
                    request,
                    runtime=runtime,
                    client=app.state.backend_client,
                    capacity=app.state.decision_capacity,
                    bearer_token=bearer_token,
                    probability_coverage=ProbabilityCoverage(
                        initial=settings.initial_probability_coverage,
                        maximum=capabilities.maximum_probability_coverage,
                        vocabulary_size=capabilities.vocabulary_size,
                    ),
                    deadline=deadline,
                )

        try:
            async with asyncio.timeout_at(deadline):
                result = await app.state.active_requests.run(
                    evaluate_admitted_request(),
                    http_request,
                )
        except TimeoutError as exc:
            raise _public_http_error(
                http_request,
                504,
                "Decision request deadline expired.",
                "deadline",
            ) from exc
        except AdmissionCapacityError as exc:
            raise _public_http_error(
                http_request,
                503,
                "Decision service is at capacity.",
                "admission_capacity",
            ) from exc
        except PublicInputError as exc:
            raise _field_validation_error(
                exc.location,
                exc.message,
                exc.kind,
            ) from exc
        except BackendContractError as exc:
            raise _public_http_error(
                http_request,
                502,
                "Backend response did not satisfy the inference contract.",
                "backend_contract",
            ) from exc
        except BackendUnavailableError as exc:
            raise _public_http_error(
                http_request,
                503,
                "Decision backend unavailable.",
                "backend_unavailable",
            ) from exc
        except CallerAuthenticationError as exc:
            raise _public_http_error(
                http_request,
                exc.status_code,
                "Backend rejected caller credentials.",
                "caller_authentication",
            ) from exc
        return SystemOneResponse(
            model=settings.model_name,
            answers=result.answers,
            usage=Usage(
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            ),
        )

    return app


def create_app_from_environment(
    *,
    backend_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    return create_app(load_settings(), backend_transport=backend_transport)
