from __future__ import annotations

import asyncio
import json
import logging
import math
import sys
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from typing import Any, TypeVar, cast

import httpx
import torch
from decider import temperature as decider_temperature  # type: ignore[import-untyped]
from decider.infer import Decider  # type: ignore[import-untyped]
from decider.prompt import (  # type: ignore[import-untyped]
    MAX_OPTIONS,
    chat_template,
    letter_ids,
    resolve_layout,
)
from decider.systemone import (  # type: ignore[import-untyped]
    NOUL_WITHOUT_INSTRUCTIONS,
    assemble,
)

from decider_service.config import Settings
from decider_service.contracts import (
    Answer,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    StructuredValue,
    SystemOneRequest,
)
from decider_service.deployment import (
    load_local_runtime_config,
    validate_local_deployment,
)
from decider_service.timing import measure_stage, request_id_context
from decider_service.tokenizer import load_backend_tokenizer

_LOGGER = logging.getLogger("decider_service")


class PublicInputError(Exception):
    def __init__(
        self,
        location: tuple[str | int, ...],
        message: str,
        kind: str,
    ) -> None:
        super().__init__(message)
        self.location = location
        self.message = message
        self.kind = kind


class BackendContractError(Exception):
    pass


class BackendProbabilityCoverageError(BackendContractError):
    pass


class BackendMissingOptionCoverageError(BackendContractError):
    def __init__(
        self,
        input_tokens: int,
        output_tokens: int,
        missing_token_ids: tuple[int, ...],
    ) -> None:
        super().__init__("backend probability coverage omitted a required option")
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.missing_token_ids = missing_token_ids


class BackendProbabilityDataError(BackendContractError):
    pass


class BackendCounterSemanticsError(BackendContractError):
    pass


class BackendTokenIdentityError(BackendContractError):
    pass


class BackendUnavailableError(Exception):
    pass


class CallerAuthenticationError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__("backend rejected caller credentials")
        self.status_code = status_code


class AdmissionCapacityError(Exception):
    pass


_OffloadResult = TypeVar("_OffloadResult")
_BACKEND_PROBABILITY_TOLERANCE = 1e-6


class DecisionCapacity:
    def __init__(self, *, backend_slots: int, admission_capacity: int) -> None:
        self._backend_slots = asyncio.Semaphore(backend_slots)
        self._admitted: asyncio.Queue[None] = asyncio.Queue(maxsize=admission_capacity)
        self._offload_slots = asyncio.Semaphore(admission_capacity)
        self._offload_tasks: set[asyncio.Task[Any]] = set()
        self._closing = False

    @asynccontextmanager
    async def admit(self) -> AsyncIterator[None]:
        with measure_stage("admission"):
            try:
                self._admitted.put_nowait(None)
            except asyncio.QueueFull as exc:
                raise AdmissionCapacityError from exc
        try:
            yield
        finally:
            self._admitted.get_nowait()

    @asynccontextmanager
    async def backend_slot(self, **timing_details: int | float) -> AsyncIterator[None]:
        with measure_stage("backend_slot_wait", **timing_details):
            await self._backend_slots.acquire()
        try:
            yield
        finally:
            self._backend_slots.release()

    async def run_offloaded(
        self,
        operation: Callable[[], _OffloadResult],
    ) -> _OffloadResult:
        await self._offload_slots.acquire()
        if self._closing:
            self._offload_slots.release()
            raise RuntimeError("decision service is shutting down")
        work_task = asyncio.create_task(asyncio.to_thread(operation))
        self._offload_tasks.add(work_task)

        def release_offload(completed: asyncio.Task[Any]) -> None:
            if not completed.cancelled():
                completed.exception()
            self._offload_tasks.discard(completed)
            self._offload_slots.release()

        work_task.add_done_callback(release_offload)
        return await asyncio.shield(work_task)

    async def close(self) -> None:
        self._closing = True
        tasks = tuple(self._offload_tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


@dataclass(frozen=True)
class PreparedRow:
    token_ids: tuple[int, ...]
    option_count: int


@dataclass(frozen=True)
class PreparedDecision:
    rendered_questions: dict[str, dict[str, Any]]
    answer_index: list[tuple[str, str, int, int]]
    rows: tuple[PreparedRow, ...]
    temperatures: float | list[float]
    label_token_ids: tuple[int, ...]
    score_legends: dict[str, tuple[StructuredValue, ...]]
    prepared_answers: dict[str, Answer]
    answer_order: tuple[str, ...]


@dataclass(frozen=True)
class BackendRow:
    log_probabilities: tuple[float, ...]
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class DecisionResult:
    answers: dict[str, Answer]
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class ProbabilityCoverage:
    initial: int
    maximum: int
    vocabulary_size: int


@dataclass(frozen=True)
class ReadinessFixture:
    vocabulary_size: int
    label_token_ids: tuple[int, ...]
    row: PreparedRow


def _log_backend_validation_failure(
    *,
    row_index: int,
    attempt: int,
    error: BackendContractError,
    missing_option_positions: tuple[int, ...] | None = None,
) -> None:
    log_values = (
        request_id_context.get() or "unavailable",
        row_index,
        attempt,
        type(error).__name__,
    )
    if missing_option_positions is None:
        _LOGGER.error(
            "backend_response_validation_failed request_id=%s row_index=%d "
            "attempt=%d error_type=%s",
            *log_values,
        )
        return
    positions = ",".join(str(position) for position in missing_option_positions)
    _LOGGER.error(
        "backend_response_validation_failed request_id=%s row_index=%d "
        "attempt=%d error_type=%s missing_option_positions=%s",
        *log_values,
        positions,
    )


def _question_name_for_row(
    answer_index: list[tuple[str, str, int, int]],
    row_number: int,
) -> str:
    for question_name, _kind, first_row, row_count in answer_index:
        if first_row <= row_number < first_row + row_count:
            return question_name
    raise BackendContractError(
        "Pinned Decider preparation did not map every row to an answer"
    )


class DecisionRuntime:
    def __init__(self, settings: Settings) -> None:
        metadata_directory = Path(settings.metadata_directory)
        config = (
            load_local_runtime_config(metadata_directory)
            if settings.skip_deployment_identity_validation
            else validate_local_deployment(settings)
        )

        tokenizer = load_backend_tokenizer(metadata_directory)
        decider = Decider.__new__(Decider)
        decider.schema_first = False
        decider.isolated_levels = bool(config.get("isolated_levels", False))
        decider.neutralize_none = bool(config.get("neutralize_none", True))
        decider.m = SimpleNamespace(tok=tokenizer)
        decider.chat = (
            chat_template(tokenizer) if resolve_layout(config) == "chat" else None
        )
        (decider.T, decider.T_by_type), _ = decider_temperature.from_config(
            config,
            None,
            None,
        )

        self._decider = decider
        self._context_capacity = settings.context_capacity
        self._label_token_ids = tuple(int(token) for token in letter_ids(tokenizer))
        self._tokenizer = tokenizer
        self._model_name = settings.model_name
        self._preparation_lock = Lock()

    def readiness_fixture(self) -> ReadinessFixture:
        request = SystemOneRequest.model_validate(
            {
                "model": self._model_name,
                "state": "synthetic startup readiness evidence",
                "questions": {
                    "readiness": {
                        "type": "choice",
                        "criteria": {
                            "compatible": None,
                            "incompatible": None,
                        },
                    }
                },
            }
        )
        prepared = self.prepare(request)
        if len(prepared.rows) != 1:
            raise BackendContractError(
                "Pinned Decider preparation did not produce one readiness row"
            )
        return ReadinessFixture(
            vocabulary_size=len(self._tokenizer),
            label_token_ids=self._label_token_ids,
            row=prepared.rows[0],
        )

    def tokenizer_identity_chunks(
        self,
        chunk_size: int,
    ) -> Iterator[tuple[tuple[int, ...], str, tuple[int, ...]]]:
        for first_token_id in range(0, len(self._tokenizer), chunk_size):
            token_ids = tuple(
                range(
                    first_token_id,
                    min(first_token_id + chunk_size, len(self._tokenizer)),
                )
            )
            content = cast(
                str,
                self._tokenizer.decode(
                    list(token_ids),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
            )
            encoded_token_ids = tuple(
                cast(
                    list[int],
                    self._tokenizer.encode(
                        content,
                        add_special_tokens=False,
                        split_special_tokens=False,
                    ),
                )
            )
            yield token_ids, content, encoded_token_ids

    def prepare(self, request: SystemOneRequest) -> PreparedDecision:
        with self._preparation_lock:
            return self._prepare_request(request)

    def _prepare_request(self, request: SystemOneRequest) -> PreparedDecision:
        score_legends = {
            name: tuple(question.criteria)
            for name, question in request.questions.items()
            if question.type == "score"
        }
        prepared_answers: dict[str, Answer] = {}
        for name, question in request.questions.items():
            if question.type == "choice" and len(question.criteria) == 1:
                choice = next(iter(question.criteria))
                prepared_answers[name] = ChoiceAnswer(
                    type="choice",
                    choice=choice,
                    confidence=1.0,
                    probabilities={choice: 1.0},
                )
            elif question.type == "score" and len(question.criteria) == 1:
                prepared_answers[name] = ScoreAnswer(
                    type="score",
                    score=0.0,
                    confidence=1.0,
                    legend={"0": question.criteria[0]},
                    probabilities={"0": 1.0},
                )
        questions: dict[str, dict[str, Any]] = {
            name: question.model_dump(mode="python")
            for name, question in request.questions.items()
            if name not in prepared_answers
        }
        for question in questions.values():
            if question["type"] == "noul" and question.get("instructions") in (
                None,
                "",
            ):
                question["instructions"] = NOUL_WITHOUT_INSTRUCTIONS
            elif question["type"] == "score" and question.get("instructions") == "":
                # Upstream rejects the contract-valid empty string, so render its
                # JSON literal through the otherwise unchanged prompt path.
                question["instructions"] = json.dumps("")
        if questions:
            try:
                rendered, answer_index, items = self._decider._system_one_items(
                    request.state,
                    questions,
                    independent=True,
                    max_state_tokens=sys.maxsize,
                    layout="state_first",
                    isolated=None,
                )
            except (AssertionError, TypeError, ValueError):
                raise PublicInputError(
                    ("questions",),
                    "question configuration could not be prepared",
                    "value_error",
                ) from None
            if not items:
                raise BackendContractError(
                    "Pinned Decider preparation omitted evaluable questions"
                )
        else:
            rendered, answer_index, items = {}, [], []

        prepared_rows: list[PreparedRow] = []
        for item_number, item in enumerate(items):
            slots = item.get("slots")
            token_ids = item.get("ids")
            option_counts = item.get("nopts")
            if (
                not isinstance(slots, list)
                or not isinstance(token_ids, list)
                or not isinstance(option_counts, list)
                or len(option_counts) != 1
                or not isinstance(option_counts[0], int)
                or slots != [len(token_ids) - 1]
            ):
                raise BackendContractError(
                    "Pinned Decider preparation did not produce one final answer slot"
                )
            if len(token_ids) + 1 > self._context_capacity:
                question_name = _question_name_for_row(answer_index, item_number)
                raise PublicInputError(
                    ("questions", question_name),
                    "complete rendered prompt exceeds configured backend "
                    "context capacity",
                    "context_capacity",
                )
            prepared_rows.append(
                PreparedRow(
                    token_ids=tuple(int(token_id) for token_id in token_ids),
                    option_count=option_counts[0],
                )
            )

        temperatures_by_item = decider_temperature.for_items(
            self._decider.T,
            self._decider.T_by_type,
            items,
        )
        temperatures = cast(
            float | list[float],
            decider_temperature.slot_temperatures(temperatures_by_item, items),
        )
        return PreparedDecision(
            rendered_questions=rendered,
            answer_index=answer_index,
            rows=tuple(prepared_rows),
            temperatures=temperatures,
            label_token_ids=self._label_token_ids,
            score_legends=score_legends,
            prepared_answers=prepared_answers,
            answer_order=tuple(request.questions),
        )


def _required_nonnegative_integer(body: dict[str, Any], field: str) -> int:
    value = body.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BackendContractError("backend returned malformed token counters")
    return value


def parse_backend_row(
    body: object,
    required_token_ids: tuple[int, ...],
    *,
    expected_input_tokens: int | None = None,
    expected_probability_coverage: int | None = None,
    vocabulary_size: int | None = None,
    allow_missing_option_tokens: bool = False,
) -> BackendRow:
    if not isinstance(body, dict):
        raise BackendContractError("backend returned malformed completion data")
    typed_body = cast(dict[str, Any], body)
    probabilities = typed_body.get("completion_probabilities")
    if (
        not isinstance(probabilities, list)
        or len(probabilities) != 1
        or not isinstance(probabilities[0], dict)
    ):
        raise BackendProbabilityDataError(
            "backend returned malformed probability data"
        )
    try:
        top_logprobs = probabilities[0]["top_logprobs"]
    except (KeyError, TypeError) as exc:
        raise BackendProbabilityDataError(
            "backend returned malformed probability data"
        ) from exc
    if not isinstance(top_logprobs, list):
        raise BackendProbabilityDataError("backend returned malformed probability data")
    if (
        expected_probability_coverage is not None
        and len(top_logprobs) != expected_probability_coverage
    ):
        raise BackendProbabilityCoverageError(
            "backend returned incompatible probability coverage"
        )

    by_token_id: dict[int, float] = {}
    for entry in top_logprobs:
        if not isinstance(entry, dict):
            raise BackendProbabilityDataError(
                "backend returned malformed probability data"
            )
        token_id = entry.get("id")
        log_probability = entry.get("logprob")
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise BackendTokenIdentityError("backend returned incompatible token IDs")
        if (
            vocabulary_size is not None and not 0 <= token_id < vocabulary_size
        ) or token_id in by_token_id:
            raise BackendTokenIdentityError("backend returned incompatible token IDs")
        if (
            isinstance(log_probability, bool)
            or not isinstance(log_probability, int | float)
            or not math.isfinite(log_probability)
            or log_probability > 0
        ):
            raise BackendProbabilityDataError(
                "backend returned malformed probability data"
            )
        by_token_id[token_id] = float(log_probability)

    probability_mass = math.fsum(
        math.exp(log_probability) for log_probability in by_token_id.values()
    )
    if probability_mass > 1.0 + _BACKEND_PROBABILITY_TOLERANCE or (
        vocabulary_size is not None
        and len(by_token_id) == vocabulary_size
        and not math.isclose(
            probability_mass,
            1.0,
            rel_tol=0.0,
            abs_tol=_BACKEND_PROBABILITY_TOLERANCE,
        )
    ):
        raise BackendProbabilityDataError(
            "backend returned an invalid probability distribution"
        )

    input_tokens = _required_nonnegative_integer(typed_body, "tokens_evaluated")
    output_tokens = _required_nonnegative_integer(typed_body, "tokens_predicted")
    cached_tokens = _required_nonnegative_integer(typed_body, "tokens_cached")
    timings = typed_body.get("timings")
    if not isinstance(timings, dict):
        raise BackendContractError("backend returned malformed token counters")
    typed_timings = cast(dict[str, Any], timings)
    reused_tokens = _required_nonnegative_integer(typed_timings, "cache_n")
    processed_tokens = _required_nonnegative_integer(typed_timings, "prompt_n")
    predicted_tokens = _required_nonnegative_integer(typed_timings, "predicted_n")
    if output_tokens != 1:
        raise BackendContractError("backend returned an unexpected prediction count")
    if (
        (expected_input_tokens is not None and input_tokens != expected_input_tokens)
        or reused_tokens != 0
        or processed_tokens != input_tokens
        or cached_tokens != input_tokens
        or predicted_tokens != output_tokens
    ):
        raise BackendCounterSemanticsError(
            "backend returned inconsistent uncached prompt counters"
        )
    if typed_body.get("truncated") is True:
        raise BackendContractError("backend truncated a rendered prompt")
    missing_token_ids = tuple(
        token_id for token_id in required_token_ids if token_id not in by_token_id
    )
    if missing_token_ids and (
        not allow_missing_option_tokens
        or all(token_id not in by_token_id for token_id in required_token_ids)
    ):
        raise BackendMissingOptionCoverageError(
            input_tokens,
            output_tokens,
            missing_token_ids,
        )
    required_probabilities = tuple(
        by_token_id.get(token_id, -math.inf) for token_id in required_token_ids
    )
    return BackendRow(required_probabilities, input_tokens, output_tokens)


def _probability_coverage_schedule(
    initial_coverage: int,
    maximum_coverage: int,
) -> Iterator[int]:
    coverage = initial_coverage
    while True:
        yield coverage
        if coverage == maximum_coverage:
            return
        coverage = min(coverage * 2, maximum_coverage)


async def _evaluate_row(
    client: httpx.AsyncClient,
    capacity: DecisionCapacity,
    row: PreparedRow,
    label_token_ids: tuple[int, ...],
    bearer_token: str,
    coverage_policy: ProbabilityCoverage,
    deadline: float,
    row_index: int,
) -> BackendRow:
    input_tokens = 0
    output_tokens = 0
    for attempt, probability_coverage in enumerate(
        _probability_coverage_schedule(
            coverage_policy.initial,
            coverage_policy.maximum,
        ),
        start=1,
    ):
        timing_details = {
            "row_index": row_index,
            "attempt": attempt,
            "n_probs": probability_coverage,
            "prompt_tokens": len(row.token_ids),
        }
        payload = completion_payload(row, probability_coverage)
        response = await _request_backend(
            client,
            capacity,
            "POST",
            "/completion",
            bearer_token=bearer_token,
            deadline=deadline,
            json=payload,
            timing_details=timing_details,
        )
        with measure_stage("backend_response_validation", **timing_details) as timing:
            try:
                body = response.json()
            except ValueError as exc:
                error = BackendContractError("backend returned malformed JSON")
                _log_backend_validation_failure(
                    row_index=row_index,
                    attempt=attempt,
                    error=error,
                )
                raise error from exc
            if isinstance(body, dict) and isinstance(body.get("timings"), dict):
                for name in ("prompt_ms", "predicted_ms"):
                    value = body["timings"].get(name)
                    if (
                        not isinstance(value, bool)
                        and isinstance(value, int | float)
                        and 0 <= value <= sys.float_info.max
                    ):
                        timing.details[f"backend_{name}"] = value
            try:
                result = parse_backend_row(
                    body,
                    label_token_ids[: row.option_count],
                    expected_input_tokens=len(row.token_ids),
                    expected_probability_coverage=probability_coverage,
                    vocabulary_size=coverage_policy.vocabulary_size,
                    allow_missing_option_tokens=(
                        probability_coverage == coverage_policy.maximum
                    ),
                )
            except BackendMissingOptionCoverageError as exc:
                input_tokens += exc.input_tokens
                output_tokens += exc.output_tokens
                if probability_coverage == coverage_policy.maximum:
                    missing_option_positions = tuple(
                        position
                        for position, token_id in enumerate(
                            label_token_ids[: row.option_count],
                            start=1,
                        )
                        if token_id in exc.missing_token_ids
                    )
                    _log_backend_validation_failure(
                        row_index=row_index,
                        attempt=attempt,
                        error=exc,
                        missing_option_positions=missing_option_positions,
                    )
                    raise
                timing.outcome = "retry"
                continue
            except BackendContractError as exc:
                _log_backend_validation_failure(
                    row_index=row_index,
                    attempt=attempt,
                    error=exc,
                )
                raise
            missing_option_positions = tuple(
                position
                for position, log_probability in enumerate(
                    result.log_probabilities,
                    start=1,
                )
                if log_probability == -math.inf
            )
            if missing_option_positions:
                positions = ",".join(
                    str(position) for position in missing_option_positions
                )
                _LOGGER.warning(
                    "backend_missing_option_probability_zeroed request_id=%s "
                    "row_index=%d attempt=%d n_probs=%d "
                    "missing_option_positions=%s",
                    request_id_context.get() or "unavailable",
                    row_index,
                    attempt,
                    probability_coverage,
                    positions,
                )
        return BackendRow(
            result.log_probabilities,
            input_tokens + result.input_tokens,
            output_tokens + result.output_tokens,
        )
    raise AssertionError("probability coverage schedule must not be empty")


async def _request_backend(
    client: httpx.AsyncClient,
    capacity: DecisionCapacity,
    method: str,
    route: str,
    *,
    bearer_token: str,
    deadline: float,
    json: object | None = None,
    timing_details: dict[str, int] | None = None,
) -> httpx.Response:
    try:
        details = timing_details or {}
        async with capacity.backend_slot(**details):
            stage = (
                "backend_completion"
                if route == "/completion"
                else "caller_authentication"
            )
            with measure_stage(stage, **details) as timing:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError
                response = await client.request(
                    method,
                    route,
                    headers={"Authorization": f"Bearer {bearer_token}"},
                    json=json,
                    timeout=remaining,
                )
                timing.details["status_code"] = response.status_code
                if response.status_code in {401, 403}:
                    raise CallerAuthenticationError(response.status_code)
                if response.status_code == 200:
                    return response
                if response.status_code in {408, 429} or response.status_code >= 500:
                    raise BackendUnavailableError("backend rejected the request")
                raise BackendContractError("backend rejected a supported request")
    except httpx.TimeoutException as exc:
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError from exc
        raise BackendUnavailableError("backend request timed out") from exc
    except httpx.RequestError as exc:
        raise BackendUnavailableError("backend request failed") from exc


async def authenticate_caller(
    client: httpx.AsyncClient,
    capacity: DecisionCapacity,
    bearer_token: str,
    deadline: float,
) -> None:
    await _request_backend(
        client,
        capacity,
        "GET",
        "/v1/models",
        bearer_token=bearer_token,
        deadline=deadline,
    )


def completion_payload(
    row: PreparedRow,
    probability_coverage: int,
) -> dict[str, Any]:
    return {
        "prompt": list(row.token_ids),
        "n_predict": 1,
        "temperature": -1.0,
        "n_probs": probability_coverage,
        "min_keep": probability_coverage,
        "post_sampling_probs": False,
        "repeat_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "top_k": 0,
        "top_p": 1.0,
        "min_p": 0.0,
        "typical_p": 1.0,
        "cache_prompt": False,
        "stream": False,
    }


def _calibrate_and_assemble(
    prepared: PreparedDecision,
    rows: list[BackendRow],
) -> dict[str, Answer]:
    logits = torch.full((len(rows), MAX_OPTIONS), float("-inf"))
    for row_number, row in enumerate(rows):
        logits[row_number, : len(row.log_probabilities)] = torch.tensor(
            row.log_probabilities,
            dtype=logits.dtype,
        )
    calibrated = decider_temperature.scaled_softmax(
        logits,
        prepared.temperatures,
    )
    probability_rows = [row.tolist() for row in calibrated]
    assembled = assemble(
        prepared.rendered_questions,
        prepared.answer_index,
        probability_rows,
    )

    answers: dict[str, Answer] = {}
    for name, raw_answer in assembled.items():
        if raw_answer.get("type") == "noul":
            noul = raw_answer.get("noul")
            if (
                isinstance(noul, bool)
                or not isinstance(noul, int | float)
                or not math.isfinite(noul)
                or not 0.0 <= noul <= 1.0
            ):
                raise BackendContractError("assembled Noul probability is invalid")
            answers[name] = NoulAnswer(type="noul", noul=float(noul))
            continue
        probabilities = cast(dict[str, float], raw_answer["probabilities"])
        tolerance = len(probabilities) * 0.00005 + 1e-7
        if (
            not probabilities
            or any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in probabilities.values()
            )
            or abs(sum(probabilities.values()) - 1.0) > tolerance
        ):
            raise BackendContractError("assembled probabilities are invalid")
        confidence = raw_answer["confidence"]
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, int | float)
            or not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
        ):
            raise BackendContractError("assembled confidence is invalid")
        if raw_answer.get("type") == "score":
            score = raw_answer.get("score")
            legend = prepared.score_legends[name]
            if (
                isinstance(score, bool)
                or not isinstance(score, int | float)
                or not math.isfinite(score)
                or not 0.0 <= score <= len(legend) - 1
            ):
                raise BackendContractError("assembled Score value is invalid")
            answers[name] = ScoreAnswer(
                type="score",
                score=float(score),
                confidence=float(confidence),
                legend={str(index): value for index, value in enumerate(legend)},
                probabilities=probabilities,
            )
            continue
        answers[name] = ChoiceAnswer(
            type="choice",
            choice=raw_answer["choice"],
            confidence=float(confidence),
            probabilities=probabilities,
        )
    answers.update(prepared.prepared_answers)
    return {name: answers[name] for name in prepared.answer_order}


async def evaluate_request(
    request: SystemOneRequest,
    *,
    runtime: DecisionRuntime,
    client: httpx.AsyncClient,
    capacity: DecisionCapacity,
    bearer_token: str,
    probability_coverage: ProbabilityCoverage,
    deadline: float,
) -> DecisionResult:
    with measure_stage("preparation", question_count=len(request.questions)) as timing:
        prepared = await capacity.run_offloaded(lambda: runtime.prepare(request))
        timing.details["row_count"] = len(prepared.rows)
    if not prepared.rows:
        await authenticate_caller(
            client,
            capacity,
            bearer_token,
            deadline,
        )
        return DecisionResult(prepared.prepared_answers.copy(), 0, 0)
    with measure_stage("backend_evaluation", row_count=len(prepared.rows)):
        rows = await asyncio.gather(
            *(
                _evaluate_row(
                    client,
                    capacity,
                    row,
                    prepared.label_token_ids,
                    bearer_token,
                    probability_coverage,
                    deadline,
                    row_index,
                )
                for row_index, row in enumerate(prepared.rows)
            )
        )
    with measure_stage("answer_assembly"):
        answers = await capacity.run_offloaded(
            lambda: _calibrate_and_assemble(prepared, rows)
        )
    return DecisionResult(
        answers=answers,
        input_tokens=sum(row.input_tokens for row in rows),
        output_tokens=sum(row.output_tokens for row in rows),
    )
