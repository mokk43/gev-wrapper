from __future__ import annotations

import asyncio
import json
import math
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

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
from decider.systemone import assemble  # type: ignore[import-untyped]
from transformers import AutoTokenizer

from decider_service.config import Settings
from decider_service.contracts import ChoiceAnswer, SystemOneRequest


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


class BackendUnavailableError(Exception):
    pass


class CallerAuthenticationError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__("backend rejected caller credentials")
        self.status_code = status_code


class AdmissionCapacityError(Exception):
    pass


class InferenceCapacity:
    def __init__(self, *, backend_slots: int, admission_capacity: int) -> None:
        self._backend_slots = asyncio.Semaphore(backend_slots)
        self._admitted: asyncio.Queue[None] = asyncio.Queue(
            maxsize=admission_capacity
        )

    @asynccontextmanager
    async def admit(self) -> AsyncIterator[None]:
        try:
            self._admitted.put_nowait(None)
        except asyncio.QueueFull as exc:
            raise AdmissionCapacityError from exc
        try:
            yield
        finally:
            self._admitted.get_nowait()

    @asynccontextmanager
    async def backend_slot(self) -> AsyncIterator[None]:
        async with self._backend_slots:
            yield


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


@dataclass(frozen=True)
class BackendRow:
    log_probabilities: tuple[float, ...]
    input_tokens: int
    output_tokens: int


class DecisionRuntime:
    def __init__(self, settings: Settings) -> None:
        metadata_directory = Path(settings.metadata_directory)
        config_path = metadata_directory / "decider_config.json"
        try:
            config = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Cannot load required Decider metadata: {config_path}"
            ) from exc
        if not isinstance(config, dict):
            raise RuntimeError("decider_config.json must contain a JSON object")

        tokenizer = AutoTokenizer.from_pretrained(
            metadata_directory,
            local_files_only=True,
        )
        decider = Decider.__new__(Decider)
        decider.schema_first = False
        decider.isolated_levels = bool(config.get("isolated_levels", False))
        decider.neutralize_none = bool(config.get("neutralize_none", True))
        decider.m = SimpleNamespace(tok=tokenizer)
        decider.chat = (
            chat_template(tokenizer)
            if resolve_layout(config) == "chat"
            else None
        )
        (decider.T, decider.T_by_type), _ = decider_temperature.from_config(
            config,
            None,
            None,
        )

        self._decider = decider
        self._context_capacity = settings.context_capacity
        self._label_token_ids = tuple(int(token) for token in letter_ids(tokenizer))

    def prepare(self, request: SystemOneRequest) -> PreparedDecision:
        questions: dict[str, dict[str, Any]] = {
            name: question.model_dump(mode="python")
            for name, question in request.questions.items()
        }
        try:
            rendered, answer_index, items = self._decider._system_one_items(
                request.state,
                questions,
                independent=True,
                max_state_tokens=sys.maxsize,
                layout="state_first",
                isolated=None,
            )
        except (AssertionError, TypeError, ValueError) as exc:
            raise PublicInputError(
                ("questions",),
                str(exc),
                "value_error",
            ) from exc

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
                question_name = answer_index[item_number][0]
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
        )


def _required_nonnegative_integer(body: dict[str, Any], field: str) -> int:
    value = body.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BackendContractError("backend returned malformed token counters")
    return value


def _parse_backend_row(
    body: object,
    required_token_ids: tuple[int, ...],
) -> BackendRow:
    if not isinstance(body, dict):
        raise BackendContractError("backend returned malformed completion data")
    typed_body = cast(dict[str, Any], body)
    try:
        top_logprobs = typed_body["probs"][0]["top_logprobs"]
    except (KeyError, IndexError, TypeError) as exc:
        raise BackendContractError(
            "backend returned malformed probability data"
        ) from exc
    if not isinstance(top_logprobs, list):
        raise BackendContractError("backend returned malformed probability data")

    by_token_id: dict[int, float] = {}
    for entry in top_logprobs:
        if not isinstance(entry, dict):
            raise BackendContractError("backend returned malformed probability data")
        token_id = entry.get("id")
        log_probability = entry.get("logprob")
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or isinstance(log_probability, bool)
            or not isinstance(log_probability, int | float)
            or not math.isfinite(log_probability)
            or token_id in by_token_id
        ):
            raise BackendContractError("backend returned malformed probability data")
        by_token_id[token_id] = float(log_probability)

    try:
        probabilities = tuple(by_token_id[token_id] for token_id in required_token_ids)
    except KeyError as exc:
        raise BackendContractError(
            "backend probability coverage omitted a required option"
        ) from exc

    input_tokens = _required_nonnegative_integer(typed_body, "tokens_evaluated")
    output_tokens = _required_nonnegative_integer(typed_body, "tokens_predicted")
    if output_tokens != 1:
        raise BackendContractError("backend returned an unexpected prediction count")
    if typed_body.get("truncated") is True:
        raise BackendContractError("backend truncated a rendered prompt")
    return BackendRow(probabilities, input_tokens, output_tokens)


async def _evaluate_row(
    client: httpx.AsyncClient,
    capacity: InferenceCapacity,
    row: PreparedRow,
    label_token_ids: tuple[int, ...],
    bearer_token: str,
    probability_coverage: int,
) -> BackendRow:
    payload = {
        "prompt": list(row.token_ids),
        "n_predict": 1,
        "temperature": -1.0,
        "n_probs": probability_coverage,
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
    try:
        async with capacity.backend_slot():
            response = await client.post(
                "/completion",
                headers={"Authorization": f"Bearer {bearer_token}"},
                json=payload,
            )
            if response.status_code in {401, 403}:
                raise CallerAuthenticationError(response.status_code)
            response.raise_for_status()
    except httpx.RequestError as exc:
        raise BackendUnavailableError("backend request failed") from exc
    except httpx.HTTPStatusError as exc:
        raise BackendUnavailableError("backend rejected the request") from exc
    try:
        body = response.json()
    except ValueError as exc:
        raise BackendContractError("backend returned malformed JSON") from exc
    return _parse_backend_row(body, label_token_ids[: row.option_count])


def _calibrate_and_assemble(
    prepared: PreparedDecision,
    rows: list[BackendRow],
) -> dict[str, ChoiceAnswer]:
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

    answers: dict[str, ChoiceAnswer] = {}
    for name, raw_answer in assembled.items():
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
        answers[name] = ChoiceAnswer(
            type="choice",
            choice=raw_answer["choice"],
            confidence=float(confidence),
            probabilities=probabilities,
        )
    return answers


async def evaluate_choice_request(
    request: SystemOneRequest,
    *,
    runtime: DecisionRuntime,
    client: httpx.AsyncClient,
    capacity: InferenceCapacity,
    bearer_token: str,
    probability_coverage: int,
) -> tuple[dict[str, ChoiceAnswer], int, int]:
    prepared = await asyncio.to_thread(runtime.prepare, request)
    rows = await asyncio.gather(
        *(
            _evaluate_row(
                client,
                capacity,
                row,
                prepared.label_token_ids,
                bearer_token,
                probability_coverage,
            )
            for row in prepared.rows
        )
    )
    answers = await asyncio.to_thread(_calibrate_and_assemble, prepared, rows)
    return (
        answers,
        sum(row.input_tokens for row in rows),
        sum(row.output_tokens for row in rows),
    )
