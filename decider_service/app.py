from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager
from datetime import date
from typing import Annotated, Any, TypeVar

import httpx
from decider.systemone import MAX_LEVELS  # type: ignore[import-untyped]
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
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
    PublicInputError,
    evaluate_request,
)
from decider_service.readiness import validate_backend_deployment


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
                await _request_too_large_response()(scope, receive, send)
                return

        received_bytes = 0

        async def limited_receive() -> Message:
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self._maximum_bytes:
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
                app.state.backend_capabilities = await validate_backend_deployment(
                    settings,
                    app.state.decision_runtime,
                    backend_client,
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

    @app.exception_handler(RequestValidationError)
    async def field_oriented_validation_error(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        errors = exc.errors()
        for error in errors:
            if error["type"] in {"union_tag_invalid", "union_tag_not_found"}:
                error["loc"] = (*error["loc"], "type")
        return JSONResponse(
            status_code=422,
            content={"detail": jsonable_encoder(errors)},
        )

    @app.get("/v1/models", response_model=ModelMetadataList)
    async def list_models() -> ModelMetadataList:
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
        authorization: Annotated[str | None, Header()] = None,
    ) -> SystemOneResponse:
        if authorization is None or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Bearer credential required")
        bearer_token = authorization.removeprefix("Bearer ")
        if not bearer_token:
            raise HTTPException(status_code=401, detail="Bearer credential required")
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
                return await evaluate_request(
                    request,
                    runtime=runtime,
                    client=app.state.backend_client,
                    capacity=app.state.decision_capacity,
                    bearer_token=bearer_token,
                    probability_coverage=settings.initial_probability_coverage,
                    deadline=deadline,
                )

        try:
            async with asyncio.timeout_at(deadline):
                result = await app.state.active_requests.run(
                    evaluate_admitted_request(),
                    http_request,
                )
        except TimeoutError as exc:
            raise HTTPException(
                status_code=504,
                detail="Decision request deadline expired.",
            ) from exc
        except AdmissionCapacityError as exc:
            raise HTTPException(
                status_code=503,
                detail="Decision service is at capacity.",
            ) from exc
        except PublicInputError as exc:
            raise _field_validation_error(
                exc.location,
                exc.message,
                exc.kind,
            ) from exc
        except BackendContractError as exc:
            raise HTTPException(
                status_code=502,
                detail="Backend response did not satisfy the inference contract.",
            ) from exc
        except BackendUnavailableError as exc:
            raise HTTPException(
                status_code=503,
                detail="Decision backend unavailable.",
            ) from exc
        except CallerAuthenticationError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail="Backend rejected caller credentials.",
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


def create_app_from_environment() -> FastAPI:
    return create_app(load_settings())
