from __future__ import annotations

import asyncio
import json
import math
import string
import threading
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from fastapi import FastAPI
from starlette.types import Message, Scope
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

import decider_service.decision as decision_module
from decider_service.app import create_app
from decider_service.config import Settings
from decider_service.decision import DecisionRuntime
from tests.readiness_support import (
    configured_settings as base_configured_settings,
)
from tests.readiness_support import (
    ready_backend_transport,
    write_test_manifest,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def metadata_directory(tmp_path: Path) -> Path:
    labels = list(string.ascii_uppercase) + [
        first + second
        for first in string.ascii_uppercase
        for second in string.ascii_uppercase
    ]
    vocabulary = {"[UNK]": 0, **{label: i + 1 for i, label in enumerate(labels)}}
    next_token_id = len(vocabulary)
    vocabulary.update(
        {
            word: next_token_id + offset
            for offset, word in enumerate(
                (
                    "none",
                    "not",
                    "listed",
                    "here",
                    '["instruction-marker"]',
                    '["true-marker"]',
                    '["false-marker"]',
                )
            )
        }
    )
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
    ).save_pretrained(tmp_path)
    (tmp_path / "decider_config.json").write_text(
        json.dumps(
            {
                "version": "test",
                "temperature": 1.0,
                "neutralize_none": True,
                "isolated_levels": True,
            }
        )
    )
    write_test_manifest(tmp_path)
    return tmp_path


def configured_settings(
    metadata_directory: Path,
    **overrides: object,
) -> Settings:
    write_test_manifest(metadata_directory)
    return base_configured_settings(metadata_directory, **overrides)


def service_client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
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


def multi_row_choice_request_payload() -> dict[str, Any]:
    payload = choice_request_payload()
    payload["questions"]["owner"] = {
        "type": "choice",
        "criteria": {"support": None, "sales": None},
    }
    return payload


def completion_response(
    probabilities: dict[int, float],
    *,
    input_tokens: int = 11,
    output_tokens: int = 1,
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "content": "generated text is ignored",
            "probs": [
                {
                    "top_logprobs": [
                        {"id": token_id, "logprob": math.log(probability)}
                        for token_id, probability in probabilities.items()
                    ]
                }
            ],
            "tokens_evaluated": input_tokens,
            "tokens_predicted": output_tokens,
            "truncated": False,
        },
    )


@pytest.mark.anyio
async def test_choice_is_evaluated_through_the_public_http_boundary(
    metadata_directory: Path,
) -> None:
    backend_requests: list[httpx.Request] = []

    async def backend(request: httpx.Request) -> httpx.Response:
        backend_requests.append(request)
        return completion_response({1: 0.2, 2: 0.8})

    app = create_app(
        configured_settings(metadata_directory),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )

    async with (
        app.router.lifespan_context(app),
        service_client(app) as client,
    ):
        response = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer caller-key"},
            json={
                "model": "jev-latest",
                "state": "The production service is down.",
                "questions": {
                    "priority": {
                        "type": "choice",
                        "instructions": "Choose the response priority.",
                        "criteria": {
                            "routine": "Can wait",
                            "urgent": "Act now",
                        },
                    }
                },
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "model": "decider-4b-q4-k-m",
        "answers": {
            "priority": {
                "type": "choice",
                "choice": "urgent",
                "confidence": 0.6,
                "probabilities": {"routine": 0.2, "urgent": 0.8},
            }
        },
        "usage": {"input_tokens": 11, "output_tokens": 1},
    }
    assert len(backend_requests) == 1
    backend_request = backend_requests[0]
    assert backend_request.url == "http://127.0.0.1:8080/completion"
    assert backend_request.headers["Authorization"] == "Bearer caller-key"
    payload: dict[str, Any] = json.loads(backend_request.content)
    assert payload == {
        "prompt": payload["prompt"],
        "n_predict": 1,
        "temperature": -1.0,
        "n_probs": 256,
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
    assert payload["prompt"]
    assert all(isinstance(token_id, int) for token_id in payload["prompt"])
    assert "caller-key" not in backend_request.content.decode()


@pytest.mark.anyio
async def test_noul_is_evaluated_through_the_public_http_boundary(
    metadata_directory: Path,
) -> None:
    backend_requests: list[httpx.Request] = []

    async def backend(request: httpx.Request) -> httpx.Response:
        backend_requests.append(request)
        return completion_response({1: 0.123456, 2: 0.876544})

    app = create_app(
        configured_settings(metadata_directory),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )

    async with (
        app.router.lifespan_context(app),
        service_client(app) as client,
    ):
        response = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer caller-key"},
            json={
                "model": "jev-latest",
                "state": "The message advertises a product.",
                "questions": {
                    "is_spam": {
                        "type": "noul",
                        "instructions": "Is this message spam?",
                        "criteria": {
                            "true": "Unsolicited advertising",
                            "false": "A legitimate conversation",
                        },
                    }
                },
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "model": "decider-4b-q4-k-m",
        "answers": {"is_spam": {"type": "noul", "noul": 0.8765}},
        "usage": {"input_tokens": 11, "output_tokens": 1},
    }
    assert body["answers"]["is_spam"]["noul"] == pytest.approx(
        0.8765,
        abs=0.00005,
    )
    assert len(backend_requests) == 1


@pytest.mark.anyio
async def test_score_is_evaluated_as_an_expected_level_through_public_http(
    metadata_directory: Path,
) -> None:
    backend_requests: list[httpx.Request] = []
    fit_probabilities = [0.1, 0.3, 0.6]

    def backend(request: httpx.Request) -> httpx.Response:
        backend_requests.append(request)
        fit_probability = fit_probabilities[len(backend_requests) - 1]
        return completion_response(
            {1: 1.0 - fit_probability, 2: fit_probability},
            input_tokens=len(backend_requests) * 10,
        )

    app = create_app(
        configured_settings(metadata_directory),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )

    async with (
        app.router.lifespan_context(app),
        service_client(app) as client,
    ):
        response = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer caller-key"},
            json={
                "model": "jev-latest",
                "state": {"incident": ["checkout", "down"]},
                "questions": {
                    "urgency": {
                        "type": "score",
                        "instructions": ["Rate", {"timeframe": "today"}],
                        "criteria": [
                            "Can wait",
                            {"window": "this week"},
                            ["Act", "today"],
                        ],
                    }
                },
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "model": "decider-4b-q4-k-m",
        "answers": {
            "urgency": {
                "type": "score",
                "score": 1.5,
                "confidence": 0.25,
                "legend": {
                    "0": "Can wait",
                    "1": {"window": "this week"},
                    "2": ["Act", "today"],
                },
                "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
            }
        },
        "usage": {"input_tokens": 60, "output_tokens": 3},
    }
    assert len(backend_requests) == 3
    assert all(
        json.loads(request.content)["n_predict"] == 1 for request in backend_requests
    )


@pytest.mark.anyio
async def test_one_level_score_is_prepared_without_backend_work(
    metadata_directory: Path,
) -> None:
    backend_requests: list[httpx.Request] = []

    def backend(request: httpx.Request) -> httpx.Response:
        backend_requests.append(request)
        raise AssertionError("one-level Score must not require inference")

    app = create_app(
        configured_settings(metadata_directory),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )

    async with (
        app.router.lifespan_context(app),
        service_client(app) as client,
    ):
        response = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer caller-key"},
            json={
                "model": "decider-4b-q4-k-m",
                "state": "evidence",
                "questions": {
                    "classification": {
                        "type": "score",
                        "criteria": [{"only": ["applicable", "level"]}],
                    }
                },
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "model": "decider-4b-q4-k-m",
        "answers": {
            "classification": {
                "type": "score",
                "score": 0.0,
                "confidence": 1.0,
                "legend": {"0": {"only": ["applicable", "level"]}},
                "probabilities": {"0": 1.0},
            }
        },
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
    assert backend_requests == []


@pytest.mark.anyio
async def test_score_accepts_empty_text_instructions(
    metadata_directory: Path,
) -> None:
    def backend(_request: httpx.Request) -> httpx.Response:
        return completion_response({1: 0.4, 2: 0.6})

    app = create_app(
        configured_settings(metadata_directory),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )

    async with (
        app.router.lifespan_context(app),
        service_client(app) as client,
    ):
        response = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer caller-key"},
            json={
                "model": "decider-4b-q4-k-m",
                "state": "evidence",
                "questions": {
                    "severity": {
                        "type": "score",
                        "instructions": "",
                        "criteria": ["low", "high"],
                    }
                },
            },
        )

    assert response.status_code == 200
    assert response.json()["answers"]["severity"] == {
        "type": "score",
        "score": 0.5,
        "confidence": 0.0,
        "legend": {"0": "low", "1": "high"},
        "probabilities": {"0": 0.5, "1": 0.5},
    }


@pytest.mark.anyio
async def test_mixed_choice_and_nonisolated_score_use_per_type_calibration_once(
    metadata_directory: Path,
) -> None:
    (metadata_directory / "decider_config.json").write_text(
        json.dumps(
            {
                "version": "test",
                "temperature": 1.0,
                "temperature_by_type": {"choice": 1.0, "score": 2.0},
                "neutralize_none": True,
                "isolated_levels": False,
            }
        )
    )
    backend_requests: list[httpx.Request] = []

    def backend(request: httpx.Request) -> httpx.Response:
        backend_requests.append(request)
        return completion_response(
            {1: 0.81, 2: 0.09},
            input_tokens=len(backend_requests) * 10,
        )

    app = create_app(
        configured_settings(metadata_directory),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )

    async with (
        app.router.lifespan_context(app),
        service_client(app) as client,
    ):
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
                    },
                    "severity": {
                        "type": "score",
                        "instructions": "Rate severity.",
                        "criteria": ["low", "high"],
                    },
                },
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "model": "decider-4b-q4-k-m",
        "answers": {
            "priority": {
                "type": "choice",
                "choice": "urgent",
                "confidence": 0.8,
                "probabilities": {"urgent": 0.9, "routine": 0.1},
            },
            "severity": {
                "type": "score",
                "score": 0.25,
                "confidence": 0.5,
                "legend": {"0": "low", "1": "high"},
                "probabilities": {"0": 0.75, "1": 0.25},
            },
        },
        "usage": {"input_tokens": 30, "output_tokens": 2},
    }
    assert len(backend_requests) == 2


@pytest.mark.anyio
async def test_multiple_choices_apply_temperature_once_and_sum_row_usage(
    metadata_directory: Path,
) -> None:
    (metadata_directory / "decider_config.json").write_text(
        json.dumps(
            {
                "version": "test",
                "temperature": 2.0,
                "neutralize_none": True,
                "isolated_levels": True,
            }
        )
    )
    backend_requests: list[httpx.Request] = []

    async def backend(request: httpx.Request) -> httpx.Response:
        backend_requests.append(request)
        row_number = len(backend_requests)
        return completion_response(
            {1: 0.81, 2: 0.09},
            input_tokens=row_number * 10,
        )

    app = create_app(
        configured_settings(metadata_directory),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )
    questions = {
        "priority": {
            "type": "choice",
            "instructions": "Choose the response priority.",
            "criteria": {"urgent": None, "routine": None},
        },
        "owner": {
            "type": "choice",
            "instructions": "Choose the owner.",
            "criteria": {"support": None, "sales": None},
        },
    }

    async with (
        app.router.lifespan_context(app),
        service_client(app) as client,
    ):
        response = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer multi-row-key"},
            json={
                "model": "decider-4b-q4-k-m",
                "state": {"message": "Please help", "attempts": [1, 2]},
                "questions": questions,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["usage"] == {"input_tokens": 30, "output_tokens": 2}
    assert body["answers"] == {
        "priority": {
            "type": "choice",
            "choice": "urgent",
            "confidence": 0.5,
            "probabilities": {"urgent": 0.75, "routine": 0.25},
        },
        "owner": {
            "type": "choice",
            "choice": "support",
            "confidence": 0.5,
            "probabilities": {"support": 0.75, "sales": 0.25},
        },
    }
    assert len(backend_requests) == 2
    assert {request.headers["Authorization"] for request in backend_requests} == {
        "Bearer multi-row-key"
    }


@pytest.mark.anyio
async def test_mixed_questions_apply_type_calibration_once_and_sum_usage(
    metadata_directory: Path,
) -> None:
    (metadata_directory / "decider_config.json").write_text(
        json.dumps(
            {
                "version": "test",
                "temperature": 1.0,
                "temperature_by_type": {"choice": 1.0, "noul": 2.0},
                "neutralize_none": True,
                "isolated_levels": True,
            }
        )
    )
    backend_requests: list[httpx.Request] = []

    def backend(request: httpx.Request) -> httpx.Response:
        backend_requests.append(request)
        return completion_response(
            {1: 0.09, 2: 0.81},
            input_tokens=len(backend_requests) * 10,
        )

    app = create_app(
        configured_settings(metadata_directory),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )

    async with (
        app.router.lifespan_context(app),
        service_client(app) as client,
    ):
        response = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer mixed-key"},
            json={
                "model": "jev-latest",
                "state": "evidence",
                "questions": {
                    "priority": {
                        "type": "choice",
                        "criteria": {"routine": None, "urgent": None},
                    },
                    "is_spam": {
                        "type": "noul",
                        "instructions": "Is this spam?",
                    },
                },
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "model": "decider-4b-q4-k-m",
        "answers": {
            "priority": {
                "type": "choice",
                "choice": "urgent",
                "confidence": 0.8,
                "probabilities": {"routine": 0.1, "urgent": 0.9},
            },
            "is_spam": {"type": "noul", "noul": 0.75},
        },
        "usage": {"input_tokens": 30, "output_tokens": 2},
    }
    assert len(backend_requests) == 2
    assert {request.headers["Authorization"] for request in backend_requests} == {
        "Bearer mixed-key"
    }


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("payload", "location"),
    [
        (
            {
                "model": "decider-4b-q4-k-m",
                "state": None,
                "questions": {
                    "priority": {
                        "type": "choice",
                        "criteria": {"urgent": None, "routine": None},
                    }
                },
            },
            "state",
        ),
        (
            {
                "model": "decider-4b-q4-k-m",
                "state": "evidence",
                "questions": {
                    "priority": {
                        "type": "choice",
                        "criteria": ["urgent", "routine"],
                    }
                },
            },
            "criteria",
        ),
        (
            {
                "model": "decider-4b-q4-k-m",
                "state": "evidence",
                "questions": {},
            },
            "questions",
        ),
        (
            {
                "model": "decider-4b-q4-k-m",
                "state": "evidence",
                "questions": {
                    "priority": {
                        "type": "unknown",
                        "criteria": {"urgent": None, "routine": None},
                    }
                },
            },
            "type",
        ),
        (
            {
                "model": "decider-4b-q4-k-m",
                "state": "evidence",
                "questions": {"priority": {"type": "choice"}},
            },
            "criteria",
        ),
        (
            {
                "model": "decider-4b-q4-k-m",
                "questions": {
                    "priority": {
                        "type": "choice",
                        "criteria": {"urgent": None, "routine": None},
                    }
                },
            },
            "state",
        ),
        (
            {
                "model": "decider-4b-q4-k-m",
                "state": "evidence",
                "questions": {"check": {"type": "noul", "criteria": ["no", "yes"]}},
            },
            "criteria",
        ),
        (
            {
                "model": "decider-4b-q4-k-m",
                "state": "evidence",
                "questions": {"check": {"type": "noul", "criteria": {"true": 1}}},
            },
            "true",
        ),
        (
            {
                "model": "decider-4b-q4-k-m",
                "state": "evidence",
                "questions": {"check": {"type": "noul", "instructions": True}},
            },
            "instructions",
        ),
    ],
)
async def test_invalid_contract_inputs_return_field_oriented_errors(
    metadata_directory: Path,
    payload: dict[str, Any],
    location: str,
) -> None:
    app = create_app(configured_settings(metadata_directory))

    async with service_client(app) as client:
        response = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer caller-key"},
            json=payload,
        )

    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(location in error["loc"] for error in errors)


@pytest.mark.anyio
async def test_unknown_model_and_configured_limits_fail_before_backend_work(
    metadata_directory: Path,
) -> None:
    backend_requests: list[httpx.Request] = []

    def backend(request: httpx.Request) -> httpx.Response:
        backend_requests.append(request)
        return completion_response({1: 0.5, 2: 0.5})

    app = create_app(
        configured_settings(
            metadata_directory,
            max_questions=1,
            max_options=2,
        ),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )
    base: dict[str, Any] = {
        "state": "evidence",
        "questions": {
            "first": {
                "type": "choice",
                "criteria": {"a": None, "b": None},
            }
        },
    }

    async with service_client(app) as client:
        unknown = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer caller-key"},
            json={"model": "not-configured", **base},
        )
        too_many_questions = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer caller-key"},
            json={
                "model": "decider-4b-q4-k-m",
                "state": "evidence",
                "questions": {
                    **base["questions"],
                    "second": {
                        "type": "choice",
                        "criteria": {"a": None, "b": None},
                    },
                },
            },
        )
        too_many_options = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer caller-key"},
            json={
                "model": "decider-4b-q4-k-m",
                "state": "evidence",
                "questions": {
                    "first": {
                        "type": "choice",
                        "criteria": {"a": None, "b": None, "c": None},
                    }
                },
            },
        )
        too_many_score_levels = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer caller-key"},
            json={
                "model": "decider-4b-q4-k-m",
                "state": "evidence",
                "questions": {
                    "rating": {
                        "type": "score",
                        "criteria": ["low", "medium", "high"],
                    }
                },
            },
        )

    assert unknown.status_code == 422
    assert unknown.json()["detail"][0]["loc"] == ["body", "model"]
    assert too_many_questions.status_code == 422
    assert too_many_questions.json()["detail"][0]["loc"] == [
        "body",
        "questions",
    ]
    assert too_many_options.status_code == 422
    assert too_many_options.json()["detail"][0]["loc"] == [
        "body",
        "questions",
        "first",
        "criteria",
    ]
    assert too_many_score_levels.status_code == 422
    assert too_many_score_levels.json()["detail"][0]["loc"] == [
        "body",
        "questions",
        "rating",
        "criteria",
    ]
    assert backend_requests == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    "criteria",
    [
        pytest.param([], id="empty"),
        pytest.param([None], id="null-level"),
        pytest.param({"0": "only"}, id="map"),
    ],
)
async def test_invalid_score_level_is_rejected_by_the_public_contract(
    metadata_directory: Path,
    criteria: object,
) -> None:
    app = create_app(configured_settings(metadata_directory))

    async with service_client(app) as client:
        invalid_score = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer caller-key"},
            json={
                "model": "decider-4b-q4-k-m",
                "state": "evidence",
                "questions": {"rating": {"type": "score", "criteria": criteria}},
            },
        )

    assert invalid_score.status_code == 422
    assert "criteria" in invalid_score.json()["detail"][0]["loc"]


@pytest.mark.anyio
async def test_score_level_count_above_the_pinned_limit_is_rejected(
    metadata_directory: Path,
) -> None:
    backend_requests: list[httpx.Request] = []

    def backend(request: httpx.Request) -> httpx.Response:
        backend_requests.append(request)
        return completion_response({1: 0.5, 2: 0.5})

    app = create_app(
        configured_settings(metadata_directory),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )

    async with service_client(app) as client:
        response = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer caller-key"},
            json={
                "model": "decider-4b-q4-k-m",
                "state": "evidence",
                "questions": {
                    "rating": {
                        "type": "score",
                        "criteria": [f"level {index}" for index in range(11)],
                    }
                },
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"][0] == {
        "loc": ["body", "questions", "rating", "criteria"],
        "msg": "option count exceeds configured capacity",
        "type": "option_capacity",
    }
    assert backend_requests == []


@pytest.mark.anyio
async def test_later_isolated_score_prompt_overflow_names_its_question(
    metadata_directory: Path,
) -> None:
    backend_requests: list[httpx.Request] = []

    def backend(request: httpx.Request) -> httpx.Response:
        backend_requests.append(request)
        return completion_response({1: 0.5, 2: 0.5})

    app = create_app(
        configured_settings(metadata_directory, context_capacity=64),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )

    async with (
        app.router.lifespan_context(app),
        service_client(app) as client,
    ):
        response = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer caller-key"},
            json={
                "model": "decider-4b-q4-k-m",
                "state": "evidence",
                "questions": {
                    "severity": {
                        "type": "score",
                        "instructions": "Rate severity.",
                        "criteria": [
                            "low",
                            "very " * 100,
                            "high",
                        ],
                    }
                },
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"][0] == {
        "loc": ["body", "questions", "severity"],
        "msg": "complete rendered prompt exceeds configured backend context capacity",
        "type": "context_capacity",
    }
    assert backend_requests == []


@pytest.mark.anyio
async def test_noul_accepts_optional_fields_and_preserves_structured_values(
    metadata_directory: Path,
) -> None:
    backend_prompts: list[list[int]] = []

    def backend(request: httpx.Request) -> httpx.Response:
        backend_prompts.append(json.loads(request.content)["prompt"])
        return completion_response({1: 0.6, 2: 0.4})

    app = create_app(
        configured_settings(metadata_directory),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )
    questions = [
        {"type": "noul", "instructions": "Is this true?"},
        {"type": "noul", "instructions": "Is this true?", "criteria": None},
        {
            "type": "noul",
            "instructions": ["instruction-marker"],
            "criteria": {
                "true": ["true-marker"],
                "false": ["false-marker"],
            },
        },
        {"type": "noul", "criteria": {"true": ["true-marker"]}},
        {"type": "noul", "criteria": {"false": ["false-marker"]}},
        {
            "type": "noul",
            "instructions": "Is this true?",
            "criteria": {"true": None, "false": None},
        },
        {"type": "noul", "instructions": None, "criteria": None},
        {"type": "noul"},
    ]

    async with (
        app.router.lifespan_context(app),
        service_client(app) as client,
    ):
        responses = [
            await client.post(
                "/v1/systemone",
                headers={"Authorization": "Bearer caller-key"},
                json={
                    "model": "decider-4b-q4-k-m",
                    "state": "evidence",
                    "questions": {"check": question},
                },
            )
            for question in questions
        ]

    assert [response.status_code for response in responses] == [200] * 8
    assert all(
        response.json()["answers"] == {"check": {"type": "noul", "noul": 0.4}}
        for response in responses
    )
    tokenizer = PreTrainedTokenizerFast.from_pretrained(metadata_directory)
    structured_prompt = tokenizer.decode(backend_prompts[2])
    assert '["instruction-marker"]' in structured_prompt
    assert '["true-marker"]' in structured_prompt
    assert '["false-marker"]' in structured_prompt
    assert '["true-marker"]' in tokenizer.decode(backend_prompts[3])
    assert '["false-marker"]' in tokenizer.decode(backend_prompts[4])


@pytest.mark.anyio
async def test_structured_fields_and_omitted_or_null_instructions_are_accepted(
    metadata_directory: Path,
) -> None:
    backend_requests: list[httpx.Request] = []

    def backend(request: httpx.Request) -> httpx.Response:
        backend_requests.append(request)
        return completion_response({1: 0.6, 2: 0.4})

    app = create_app(
        configured_settings(metadata_directory),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )
    requests: list[dict[str, Any]] = [
        {
            "state": "plain text",
            "question": {
                "type": "choice",
                "criteria": {"first": None, "second": {"reason": "fallback"}},
            },
        },
        {
            "state": {"message": "structured", "metadata": {"attempt": 2}},
            "question": {
                "type": "choice",
                "instructions": None,
                "criteria": {"first": ["primary"], "second": None},
            },
        },
        {
            "state": ["array", {"message": "state"}],
            "question": {
                "type": "choice",
                "instructions": {"task": "select"},
                "criteria": {"first": None, "second": "fallback"},
            },
        },
    ]

    async with (
        app.router.lifespan_context(app),
        service_client(app) as client,
    ):
        responses = [
            await client.post(
                "/v1/systemone",
                headers={"Authorization": "Bearer structured-key"},
                json={
                    "model": "decider-4b-q4-k-m",
                    "state": case["state"],
                    "questions": {"result": case["question"]},
                },
            )
            for case in requests
        ]

    assert [response.status_code for response in responses] == [200, 200, 200]
    choices = [response.json()["answers"]["result"]["choice"] for response in responses]
    assert choices == [
        "first",
        "first",
        "first",
    ]
    assert len(backend_requests) == 3


@pytest.mark.anyio
async def test_complete_prompt_over_context_capacity_is_rejected_without_truncation(
    metadata_directory: Path,
) -> None:
    backend_requests: list[httpx.Request] = []

    def backend(request: httpx.Request) -> httpx.Response:
        backend_requests.append(request)
        return completion_response({1: 0.5, 2: 0.5})

    app = create_app(
        configured_settings(metadata_directory, context_capacity=64),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )

    async with (
        app.router.lifespan_context(app),
        service_client(app) as client,
    ):
        response = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer caller-key"},
            json={
                "model": "decider-4b-q4-k-m",
                "state": "evidence " * 100,
                "questions": {
                    "priority": {
                        "type": "choice",
                        "criteria": {"urgent": None, "routine": None},
                    }
                },
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"][0] == {
        "loc": ["body", "questions", "priority"],
        "msg": "complete rendered prompt exceeds configured backend context capacity",
        "type": "context_capacity",
    }
    assert backend_requests == []


@pytest.mark.anyio
async def test_request_body_limit_returns_a_field_oriented_error(
    metadata_directory: Path,
) -> None:
    app = create_app(
        configured_settings(metadata_directory, max_request_bytes=100),
    )

    async with service_client(app) as client:
        response = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer caller-key"},
            json={
                "model": "decider-4b-q4-k-m",
                "state": "x" * 200,
                "questions": {
                    "priority": {
                        "type": "choice",
                        "criteria": {"urgent": None, "routine": None},
                    }
                },
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "backend_body",
    [
        {
            "probs": [{"top_logprobs": [{"id": 1, "logprob": -0.1}]}],
            "tokens_evaluated": 10,
            "tokens_predicted": 1,
        },
        {
            "probs": "invalid",
            "tokens_evaluated": 10,
            "tokens_predicted": 1,
        },
        {
            "probs": [
                {
                    "top_logprobs": [
                        {"id": 1, "logprob": float("nan")},
                        {"id": 2, "logprob": -0.1},
                    ]
                }
            ],
            "tokens_evaluated": 10,
            "tokens_predicted": 1,
        },
        {
            "probs": [
                {
                    "top_logprobs": [
                        {"id": 1, "logprob": -0.1},
                        {"id": 2, "logprob": -0.2},
                    ]
                }
            ],
            "tokens_evaluated": 10,
        },
    ],
)
async def test_malformed_backend_data_returns_a_sanitized_502(
    metadata_directory: Path,
    backend_body: dict[str, Any],
) -> None:
    def backend(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=json.dumps(backend_body, allow_nan=True),
            headers={"Content-Type": "application/json"},
        )

    app = create_app(
        configured_settings(metadata_directory),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )

    async with (
        app.router.lifespan_context(app),
        service_client(app) as client,
    ):
        response = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer secret-caller-key"},
            json={
                "model": "decider-4b-q4-k-m",
                "state": "private evidence",
                "questions": {
                    "priority": {
                        "type": "choice",
                        "criteria": {"urgent": None, "routine": None},
                    }
                },
            },
        )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Backend response did not satisfy the inference contract."
    }
    assert "secret-caller-key" not in response.text
    assert "private evidence" not in response.text


@pytest.mark.anyio
async def test_concurrent_noul_callers_keep_results_usage_and_credentials_isolated(
    metadata_directory: Path,
) -> None:
    slow_started = asyncio.Event()
    release_slow = asyncio.Event()
    backend_requests: list[httpx.Request] = []

    async def backend(request: httpx.Request) -> httpx.Response:
        backend_requests.append(request)
        if request.headers["Authorization"] == "Bearer slow-key":
            slow_started.set()
            await release_slow.wait()
            return completion_response({1: 0.8, 2: 0.2}, input_tokens=13)
        return completion_response({1: 0.1, 2: 0.9}, input_tokens=17)

    app = create_app(
        configured_settings(metadata_directory),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )
    payload = {
        "model": "decider-4b-q4-k-m",
        "state": "evidence",
        "questions": {
            "check": {
                "type": "noul",
                "instructions": "Is this true?",
            }
        },
    }

    async with (
        app.router.lifespan_context(app),
        service_client(app) as client,
    ):
        slow_task = asyncio.create_task(
            client.post(
                "/v1/systemone",
                headers={"Authorization": "Bearer slow-key"},
                json=payload,
            )
        )
        await asyncio.wait_for(slow_started.wait(), timeout=2)
        fast_response = await asyncio.wait_for(
            client.post(
                "/v1/systemone",
                headers={"Authorization": "Bearer fast-key"},
                json=payload,
            ),
            timeout=2,
        )
        release_slow.set()
        slow_response = await asyncio.wait_for(slow_task, timeout=2)

    assert fast_response.status_code == 200
    assert slow_response.status_code == 200
    assert fast_response.json()["answers"] == {"check": {"type": "noul", "noul": 0.9}}
    assert fast_response.json()["usage"] == {
        "input_tokens": 17,
        "output_tokens": 1,
    }
    assert slow_response.json()["answers"] == {"check": {"type": "noul", "noul": 0.2}}
    assert slow_response.json()["usage"] == {
        "input_tokens": 13,
        "output_tokens": 1,
    }
    assert [request.headers["Authorization"] for request in backend_requests] == [
        "Bearer slow-key",
        "Bearer fast-key",
    ]
    assert all(
        "slow-key" not in request.content.decode()
        and "fast-key" not in request.content.decode()
        for request in backend_requests
    )


@pytest.mark.anyio
async def test_concurrent_score_callers_keep_rows_results_and_usage_isolated(
    metadata_directory: Path,
) -> None:
    slow_rows_started = asyncio.Event()
    release_slow_rows = asyncio.Event()
    rows_by_credential: dict[str, int] = {}
    backend_requests: list[httpx.Request] = []

    async def backend(request: httpx.Request) -> httpx.Response:
        backend_requests.append(request)
        credential = request.headers["Authorization"]
        row_number = rows_by_credential.get(credential, 0)
        rows_by_credential[credential] = row_number + 1
        if credential == "Bearer slow-key":
            if rows_by_credential[credential] == 2:
                slow_rows_started.set()
            await release_slow_rows.wait()
            fit_probability = [0.8, 0.2][row_number]
            input_tokens = 13 + row_number
        else:
            fit_probability = [0.1, 0.9][row_number]
            input_tokens = 17 + row_number
        return completion_response(
            {1: 1.0 - fit_probability, 2: fit_probability},
            input_tokens=input_tokens,
        )

    app = create_app(
        configured_settings(metadata_directory, backend_slots=4),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )
    payload = {
        "model": "decider-4b-q4-k-m",
        "state": "evidence",
        "questions": {
            "severity": {
                "type": "score",
                "instructions": "Rate severity.",
                "criteria": ["low", "high"],
            }
        },
    }

    async with (
        app.router.lifespan_context(app),
        service_client(app) as client,
    ):
        slow_task = asyncio.create_task(
            client.post(
                "/v1/systemone",
                headers={"Authorization": "Bearer slow-key"},
                json=payload,
            )
        )
        await asyncio.wait_for(slow_rows_started.wait(), timeout=2)
        fast_response = await asyncio.wait_for(
            client.post(
                "/v1/systemone",
                headers={"Authorization": "Bearer fast-key"},
                json=payload,
            ),
            timeout=2,
        )
        release_slow_rows.set()
        slow_response = await asyncio.wait_for(slow_task, timeout=2)

    assert fast_response.status_code == 200
    assert slow_response.status_code == 200
    assert fast_response.json()["answers"]["severity"] == {
        "type": "score",
        "score": 0.9,
        "confidence": 0.8,
        "legend": {"0": "low", "1": "high"},
        "probabilities": {"0": 0.1, "1": 0.9},
    }
    assert fast_response.json()["usage"] == {
        "input_tokens": 35,
        "output_tokens": 2,
    }
    assert slow_response.json()["answers"]["severity"] == {
        "type": "score",
        "score": 0.2,
        "confidence": 0.6,
        "legend": {"0": "low", "1": "high"},
        "probabilities": {"0": 0.8, "1": 0.2},
    }
    assert slow_response.json()["usage"] == {
        "input_tokens": 27,
        "output_tokens": 2,
    }
    assert rows_by_credential == {"Bearer slow-key": 2, "Bearer fast-key": 2}


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("target_name", "target", "independent_name", "independent"),
    [
        pytest.param(
            "priority",
            {
                "type": "choice",
                "instructions": "Choose priority.",
                "criteria": {"urgent": None, "routine": None},
            },
            "owner",
            {
                "type": "choice",
                "instructions": "Choose owner using more words.",
                "criteria": {"support": None, "sales": None},
            },
            id="choice",
        ),
        pytest.param(
            "check",
            {
                "type": "noul",
                "instructions": "Is this request urgent?",
                "criteria": {"true": "urgent", "false": "routine"},
            },
            "owner",
            {
                "type": "choice",
                "instructions": "Choose an owner.",
                "criteria": {"support": None, "sales": None},
            },
            id="noul",
        ),
    ],
)
async def test_question_is_unchanged_when_independent_questions_are_reordered(
    metadata_directory: Path,
    target_name: str,
    target: dict[str, Any],
    independent_name: str,
    independent: dict[str, Any],
) -> None:
    backend_requests: list[httpx.Request] = []

    def backend(request: httpx.Request) -> httpx.Response:
        backend_requests.append(request)
        return completion_response({1: 0.3, 2: 0.7})

    app = create_app(
        configured_settings(metadata_directory),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )
    async with (
        app.router.lifespan_context(app),
        service_client(app) as client,
    ):
        first = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer caller-key"},
            json={
                "model": "decider-4b-q4-k-m",
                "state": "same evidence",
                "questions": {target_name: target},
            },
        )
        second = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer caller-key"},
            json={
                "model": "decider-4b-q4-k-m",
                "state": "same evidence",
                "questions": {
                    independent_name: independent,
                    target_name: target,
                },
            },
        )
        reordered = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer caller-key"},
            json={
                "model": "decider-4b-q4-k-m",
                "state": "same evidence",
                "questions": {
                    target_name: target,
                    independent_name: independent,
                },
            },
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert reordered.status_code == 200
    prompts = [json.loads(request.content)["prompt"] for request in backend_requests]
    assert prompts[0] == prompts[2] == prompts[3]
    assert first.json()["answers"][target_name] == second.json()["answers"][target_name]
    assert (
        first.json()["answers"][target_name] == reordered.json()["answers"][target_name]
    )


@pytest.mark.anyio
async def test_score_is_unchanged_when_a_choice_is_added_or_reordered(
    metadata_directory: Path,
) -> None:
    backend_requests: list[httpx.Request] = []

    def backend(request: httpx.Request) -> httpx.Response:
        backend_requests.append(request)
        return completion_response({1: 0.3, 2: 0.7})

    app = create_app(
        configured_settings(metadata_directory),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )
    score = {
        "type": "score",
        "instructions": "Rate severity.",
        "criteria": ["low", "high"],
    }
    choice = {
        "type": "choice",
        "instructions": "Choose owner.",
        "criteria": {"support": None, "sales": None},
    }

    async with (
        app.router.lifespan_context(app),
        service_client(app) as client,
    ):
        first = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer caller-key"},
            json={
                "model": "decider-4b-q4-k-m",
                "state": "same evidence",
                "questions": {"severity": score},
            },
        )
        added = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer caller-key"},
            json={
                "model": "decider-4b-q4-k-m",
                "state": "same evidence",
                "questions": {"owner": choice, "severity": score},
            },
        )
        reordered = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer caller-key"},
            json={
                "model": "decider-4b-q4-k-m",
                "state": "same evidence",
                "questions": {"severity": score, "owner": choice},
            },
        )

    assert first.status_code == added.status_code == reordered.status_code == 200
    first_answer = first.json()["answers"]["severity"]
    assert added.json()["answers"]["severity"] == first_answer
    assert reordered.json()["answers"]["severity"] == first_answer
    prompts = [json.loads(request.content)["prompt"] for request in backend_requests]
    assert prompts[0:2] == prompts[3:5] == prompts[5:7]


@pytest.mark.anyio
async def test_neutralized_option_maps_back_to_the_requested_label(
    metadata_directory: Path,
) -> None:
    backend_prompts: list[list[int]] = []

    def backend(request: httpx.Request) -> httpx.Response:
        backend_prompts.append(json.loads(request.content)["prompt"])
        return completion_response({1: 0.2, 2: 0.8})

    app = create_app(
        configured_settings(metadata_directory),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )

    async with (
        app.router.lifespan_context(app),
        service_client(app) as client,
    ):
        response = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer caller-key"},
            json={
                "model": "decider-4b-q4-k-m",
                "state": "evidence",
                "questions": {
                    "result": {
                        "type": "choice",
                        "criteria": {"match": None, "none": None},
                    }
                },
            },
        )

    tokenizer = PreTrainedTokenizerFast.from_pretrained(metadata_directory)
    rendered_prompt = tokenizer.decode(backend_prompts[0])
    assert response.status_code == 200
    assert response.json()["answers"]["result"]["choice"] == "none"
    assert response.json()["answers"]["result"]["probabilities"] == {
        "match": 0.2,
        "none": 0.8,
    }
    assert "not listed here" in rendered_prompt
    assert " none " not in f" {rendered_prompt} "


@pytest.mark.anyio
async def test_oversized_content_length_is_rejected_before_reading_the_body(
    metadata_directory: Path,
) -> None:
    emitted_chunks: list[bytes] = []

    async def body() -> Any:
        chunk = b"x" * 128
        emitted_chunks.append(chunk)
        yield chunk

    app = create_app(
        configured_settings(metadata_directory, max_request_bytes=64),
    )

    async with service_client(app) as client:
        response = await client.post(
            "/v1/systemone",
            headers={
                "Authorization": "Bearer caller-key",
                "Content-Length": "128",
                "Content-Type": "application/json",
            },
            content=body(),
        )

    assert response.status_code == 422
    assert emitted_chunks == []


@pytest.mark.anyio
async def test_chunked_oversized_body_stops_consuming_at_the_limit(
    metadata_directory: Path,
) -> None:
    chunks = [b"123456", b"789012", b"unread-tail"]
    emitted_chunks: list[bytes] = []

    async def body() -> Any:
        for chunk in chunks:
            emitted_chunks.append(chunk)
            yield chunk

    app = create_app(
        configured_settings(metadata_directory, max_request_bytes=10),
    )

    async with service_client(app) as client:
        response = await client.post(
            "/v1/systemone",
            headers={
                "Authorization": "Bearer caller-key",
                "Content-Type": "application/json",
            },
            content=body(),
        )

    assert response.status_code == 422
    assert emitted_chunks == chunks[:2]


@pytest.mark.anyio
@pytest.mark.parametrize("backend_status", [401, 403])
async def test_backend_credential_rejection_is_a_public_authentication_error(
    metadata_directory: Path,
    backend_status: int,
) -> None:
    def backend(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            backend_status,
            json={"error": "upstream secret diagnostic"},
        )

    app = create_app(
        configured_settings(metadata_directory),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )

    async with (
        app.router.lifespan_context(app),
        service_client(app) as client,
    ):
        response = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer rejected-key"},
            json=choice_request_payload(),
        )

    assert response.status_code == backend_status
    assert response.json() == {"detail": "Backend rejected caller credentials."}
    assert "upstream secret diagnostic" not in response.text
    assert "rejected-key" not in response.text


@pytest.mark.anyio
async def test_backend_slots_bound_rows_within_one_request(
    metadata_directory: Path,
) -> None:
    first_started = asyncio.Event()
    release = asyncio.Event()
    active = 0
    maximum_active = 0

    async def backend(_request: httpx.Request) -> httpx.Response:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        first_started.set()
        await release.wait()
        active -= 1
        return completion_response({1: 0.7, 2: 0.3})

    app = create_app(
        configured_settings(
            metadata_directory,
            backend_slots=1,
            admission_capacity=2,
        ),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )
    payload = multi_row_choice_request_payload()

    async def release_after_first_starts() -> None:
        await first_started.wait()
        await asyncio.sleep(0)
        release.set()

    async with (
        app.router.lifespan_context(app),
        service_client(app) as client,
    ):
        release_task = asyncio.create_task(release_after_first_starts())
        response = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer caller-key"},
            json=payload,
        )
        await release_task

    assert response.status_code == 200
    assert maximum_active == 1


@pytest.mark.anyio
async def test_backend_slots_are_global_across_multi_row_callers(
    metadata_directory: Path,
) -> None:
    global_limit_reached = asyncio.Event()
    release_backend = asyncio.Event()
    active = 0
    maximum_active = 0
    backend_requests = 0

    async def backend(_request: httpx.Request) -> httpx.Response:
        nonlocal active, maximum_active, backend_requests
        backend_requests += 1
        active += 1
        maximum_active = max(maximum_active, active)
        if active == 2:
            global_limit_reached.set()
        try:
            await release_backend.wait()
        finally:
            active -= 1
        return completion_response({1: 0.7, 2: 0.3})

    app = create_app(
        configured_settings(
            metadata_directory,
            backend_slots=2,
            admission_capacity=4,
        ),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )
    payload = multi_row_choice_request_payload()

    async with (
        app.router.lifespan_context(app),
        service_client(app) as client,
    ):
        callers = [
            asyncio.create_task(
                client.post(
                    "/v1/systemone",
                    headers={"Authorization": f"Bearer caller-{index}"},
                    json=payload,
                )
            )
            for index in range(2)
        ]
        await asyncio.wait_for(global_limit_reached.wait(), timeout=2)
        release_backend.set()
        responses = await asyncio.gather(*callers)

    assert [response.status_code for response in responses] == [200, 200]
    assert backend_requests == 4
    assert maximum_active == 2


@pytest.mark.anyio
async def test_exhausted_admission_rejects_without_backend_work(
    metadata_directory: Path,
) -> None:
    slow_started = asyncio.Event()
    release_slow = asyncio.Event()
    backend_keys: list[str] = []

    async def backend(request: httpx.Request) -> httpx.Response:
        key = request.headers["Authorization"]
        backend_keys.append(key)
        if key == "Bearer slow-key":
            slow_started.set()
            await release_slow.wait()
        return completion_response({1: 0.7, 2: 0.3})

    app = create_app(
        configured_settings(
            metadata_directory,
            backend_slots=1,
            admission_capacity=1,
        ),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )

    async with (
        app.router.lifespan_context(app),
        service_client(app) as client,
    ):
        slow_task = asyncio.create_task(
            client.post(
                "/v1/systemone",
                headers={"Authorization": "Bearer slow-key"},
                json=choice_request_payload(),
            )
        )
        await slow_started.wait()
        rejected = await asyncio.wait_for(
            client.post(
                "/v1/systemone",
                headers={"Authorization": "Bearer rejected-key"},
                json=choice_request_payload(),
            ),
            timeout=2,
        )
        release_slow.set()
        completed = await slow_task
        after_release = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer next-key"},
            json=choice_request_payload(),
        )

    assert completed.status_code == 200
    assert rejected.status_code == 503
    assert rejected.json() == {"detail": "Decision service is at capacity."}
    assert after_release.status_code == 200
    assert backend_keys == ["Bearer slow-key", "Bearer next-key"]


@pytest.mark.anyio
async def test_request_deadline_cancels_backend_work_and_releases_capacity(
    metadata_directory: Path,
) -> None:
    slow_started = asyncio.Event()
    slow_cancelled = asyncio.Event()
    backend_keys: list[str] = []
    transport_timeouts: list[dict[str, float]] = []

    async def backend(request: httpx.Request) -> httpx.Response:
        key = request.headers["Authorization"]
        backend_keys.append(key)
        transport_timeouts.append(request.extensions["timeout"])
        if key == "Bearer slow-key":
            slow_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                slow_cancelled.set()
        return completion_response({1: 0.7, 2: 0.3})

    app = create_app(
        configured_settings(
            metadata_directory,
            backend_slots=1,
            admission_capacity=1,
            request_deadline_seconds=0.05,
        ),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )
    payload = multi_row_choice_request_payload()

    async with (
        app.router.lifespan_context(app),
        service_client(app) as client,
    ):
        slow_task = asyncio.create_task(
            client.post(
                "/v1/systemone",
                headers={"Authorization": "Bearer slow-key"},
                json=payload,
            )
        )
        await asyncio.wait_for(slow_started.wait(), timeout=2)
        expired = await asyncio.wait_for(slow_task, timeout=2)
        await asyncio.wait_for(slow_cancelled.wait(), timeout=2)
        after_expiry = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer next-key"},
            json=choice_request_payload(),
        )

    assert expired.status_code == 504
    assert expired.json() == {"detail": "Decision request deadline expired."}
    assert after_expiry.status_code == 200
    assert backend_keys == ["Bearer slow-key", "Bearer next-key"]
    assert all(
        0 < timeout <= 0.05
        for transport_timeout in transport_timeouts
        for timeout in transport_timeout.values()
    )


@pytest.mark.anyio
async def test_early_backend_transport_timeout_returns_503(
    metadata_directory: Path,
) -> None:
    def backend(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("remaining request budget expired", request=request)

    app = create_app(
        configured_settings(metadata_directory),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )

    async with (
        app.router.lifespan_context(app),
        service_client(app) as client,
    ):
        response = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer caller-key"},
            json=choice_request_payload(),
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "Decision backend unavailable."}
    assert "remaining request budget expired" not in response.text


@pytest.mark.anyio
async def test_abandoned_preparation_does_not_overlap_shared_runtime(
    metadata_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = configured_settings(
        metadata_directory,
        backend_slots=1,
        admission_capacity=2,
        request_deadline_seconds=0.05,
    )
    runtime = DecisionRuntime(settings)
    original_prepare_items = runtime._decider._system_one_items
    preparation_started = threading.Event()
    preparations_finished = threading.Event()
    release_preparation = threading.Event()
    state_lock = threading.Lock()
    active_preparations = 0
    maximum_active_preparations = 0
    preparation_calls = 0
    finished_preparations = 0

    def gated_prepare_items(*args: Any, **kwargs: Any) -> Any:
        nonlocal active_preparations
        nonlocal finished_preparations
        nonlocal maximum_active_preparations
        nonlocal preparation_calls
        with state_lock:
            preparation_calls += 1
            call_number = preparation_calls
            active_preparations += 1
            maximum_active_preparations = max(
                maximum_active_preparations,
                active_preparations,
            )
        try:
            if call_number == 1:
                preparation_started.set()
                release_preparation.wait()
            return original_prepare_items(*args, **kwargs)
        finally:
            with state_lock:
                active_preparations -= 1
                finished_preparations += 1
                if finished_preparations == 2:
                    preparations_finished.set()

    monkeypatch.setattr(
        runtime._decider,
        "_system_one_items",
        gated_prepare_items,
    )

    app = create_app(
        settings,
        backend_transport=ready_backend_transport(
            metadata_directory, lambda _request: completion_response({1: 0.7, 2: 0.3})
        ),
    )

    try:
        async with (
            app.router.lifespan_context(app),
            service_client(app) as client,
        ):
            app.state.decision_runtime = runtime
            first_request = asyncio.create_task(
                client.post(
                    "/v1/systemone",
                    headers={"Authorization": "Bearer first-key"},
                    json=choice_request_payload(),
                )
            )
            await asyncio.wait_for(
                asyncio.to_thread(preparation_started.wait),
                timeout=2,
            )
            first_expired = await asyncio.wait_for(first_request, timeout=2)
            second_expired = await asyncio.wait_for(
                client.post(
                    "/v1/systemone",
                    headers={"Authorization": "Bearer second-key"},
                    json=choice_request_payload(),
                ),
                timeout=2,
            )
            calls_before_release = preparation_calls
            maximum_before_release = maximum_active_preparations
            release_preparation.set()
            await asyncio.wait_for(
                asyncio.to_thread(preparations_finished.wait),
                timeout=2,
            )

            assert first_expired.status_code == 504
            assert second_expired.status_code == 504
            assert calls_before_release == 1
            assert maximum_before_release == 1

            after_release = await client.post(
                "/v1/systemone",
                headers={"Authorization": "Bearer next-key"},
                json=choice_request_payload(),
            )

        assert after_release.status_code == 200
        assert preparation_calls == 3
        assert maximum_active_preparations == 1
    finally:
        release_preparation.set()


@pytest.mark.anyio
async def test_abandoned_preparation_remains_bounded_after_deadline(
    metadata_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = configured_settings(
        metadata_directory,
        backend_slots=1,
        admission_capacity=1,
        request_deadline_seconds=0.05,
    )
    runtime = DecisionRuntime(settings)
    original_prepare = runtime.prepare
    preparation_started = threading.Event()
    preparation_finished = threading.Event()
    release_preparation = threading.Event()
    preparation_calls = 0

    def gated_prepare(request: Any) -> Any:
        nonlocal preparation_calls
        preparation_calls += 1
        if request.state == "blocked preparation":
            preparation_started.set()
            release_preparation.wait()
            preparation_finished.set()
        return original_prepare(request)

    monkeypatch.setattr(runtime, "prepare", gated_prepare)

    def backend(_request: httpx.Request) -> httpx.Response:
        return completion_response({1: 0.7, 2: 0.3})

    app = create_app(
        settings,
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )
    blocked_payload = choice_request_payload()
    blocked_payload["state"] = "blocked preparation"

    try:
        async with (
            app.router.lifespan_context(app),
            service_client(app) as client,
        ):
            app.state.decision_runtime = runtime
            first_request = asyncio.create_task(
                client.post(
                    "/v1/systemone",
                    headers={"Authorization": "Bearer first-key"},
                    json=blocked_payload,
                )
            )
            await asyncio.wait_for(
                asyncio.to_thread(preparation_started.wait),
                timeout=2,
            )
            first_expired = await asyncio.wait_for(first_request, timeout=2)
            second_expired = await asyncio.wait_for(
                client.post(
                    "/v1/systemone",
                    headers={"Authorization": "Bearer second-key"},
                    json=blocked_payload,
                ),
                timeout=2,
            )

            assert first_expired.status_code == 504
            assert second_expired.status_code == 504
            assert preparation_calls == 1

            release_preparation.set()
            await asyncio.wait_for(
                asyncio.to_thread(preparation_finished.wait),
                timeout=2,
            )
            after_release = await client.post(
                "/v1/systemone",
                headers={"Authorization": "Bearer next-key"},
                json=choice_request_payload(),
            )

        assert after_release.status_code == 200
        assert preparation_calls == 2
    finally:
        release_preparation.set()


@pytest.mark.anyio
async def test_assembly_deadline_releases_request_capacity_but_bounds_offload(
    metadata_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = configured_settings(
        metadata_directory,
        backend_slots=1,
        admission_capacity=1,
        request_deadline_seconds=0.05,
    )
    runtime = DecisionRuntime(settings)
    original_assemble = decision_module._calibrate_and_assemble
    assembly_started = threading.Event()
    assembly_finished = threading.Event()
    release_assembly = threading.Event()
    assembly_calls = 0
    backend_keys: list[str] = []

    def gated_assemble(*args: Any, **kwargs: Any) -> Any:
        nonlocal assembly_calls
        assembly_calls += 1
        if assembly_calls == 1:
            assembly_started.set()
            release_assembly.wait()
            assembly_finished.set()
        return original_assemble(*args, **kwargs)

    monkeypatch.setattr(decision_module, "_calibrate_and_assemble", gated_assemble)

    def backend(request: httpx.Request) -> httpx.Response:
        backend_keys.append(request.headers["Authorization"])
        return completion_response({1: 0.7, 2: 0.3})

    app = create_app(
        settings,
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )

    try:
        async with (
            app.router.lifespan_context(app),
            service_client(app) as client,
        ):
            app.state.decision_runtime = runtime
            first_request = asyncio.create_task(
                client.post(
                    "/v1/systemone",
                    headers={"Authorization": "Bearer first-key"},
                    json=choice_request_payload(),
                )
            )
            await asyncio.wait_for(
                asyncio.to_thread(assembly_started.wait),
                timeout=2,
            )
            first_expired = await asyncio.wait_for(first_request, timeout=2)
            second_expired = await asyncio.wait_for(
                client.post(
                    "/v1/systemone",
                    headers={"Authorization": "Bearer second-key"},
                    json=choice_request_payload(),
                ),
                timeout=2,
            )

            assert first_expired.status_code == 504
            assert second_expired.status_code == 504
            assert assembly_calls == 1
            assert backend_keys == ["Bearer first-key"]

            release_assembly.set()
            await asyncio.wait_for(
                asyncio.to_thread(assembly_finished.wait),
                timeout=2,
            )
            after_release = await client.post(
                "/v1/systemone",
                headers={"Authorization": "Bearer next-key"},
                json=choice_request_payload(),
            )

        assert after_release.status_code == 200
        assert assembly_calls == 2
        assert backend_keys == ["Bearer first-key", "Bearer next-key"]
    finally:
        release_assembly.set()


@pytest.mark.anyio
async def test_shutdown_cancels_inflight_backend_work(
    metadata_directory: Path,
) -> None:
    backend_started = asyncio.Event()
    backend_cancelled = asyncio.Event()

    async def backend(_request: httpx.Request) -> httpx.Response:
        backend_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            backend_cancelled.set()
        raise AssertionError("cancelled backend request must not resume")

    app = create_app(
        configured_settings(metadata_directory),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    client = service_client(app)
    request_task: asyncio.Task[httpx.Response] | None = None
    try:
        request_task = asyncio.create_task(
            client.post(
                "/v1/systemone",
                headers={"Authorization": "Bearer caller-key"},
                json=choice_request_payload(),
            )
        )
        await asyncio.wait_for(backend_started.wait(), timeout=2)

        await asyncio.wait_for(lifespan.__aexit__(None, None, None), timeout=2)

        await asyncio.wait_for(backend_cancelled.wait(), timeout=2)
        assert request_task.done()
    finally:
        if request_task is not None and not request_task.done():
            request_task.cancel()
        if request_task is not None:
            await asyncio.gather(request_task, return_exceptions=True)
        await client.aclose()


@pytest.mark.anyio
async def test_shutdown_awaits_abandoned_offloaded_work(
    metadata_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = configured_settings(
        metadata_directory,
        backend_slots=1,
        admission_capacity=1,
        request_deadline_seconds=0.05,
    )
    runtime = DecisionRuntime(settings)
    original_prepare = runtime.prepare
    preparation_started = threading.Event()
    preparation_finished = threading.Event()
    release_preparation = threading.Event()

    def gated_prepare(request: Any) -> Any:
        preparation_started.set()
        release_preparation.wait()
        result = original_prepare(request)
        preparation_finished.set()
        return result

    monkeypatch.setattr(runtime, "prepare", gated_prepare)

    app = create_app(
        settings,
        backend_transport=ready_backend_transport(
            metadata_directory, lambda _request: completion_response({1: 0.7, 2: 0.3})
        ),
    )
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    app.state.decision_runtime = runtime
    original_capacity_close = app.state.decision_capacity.close
    shutdown_started = asyncio.Event()

    async def signal_shutdown_started() -> None:
        shutdown_started.set()
        await original_capacity_close()

    monkeypatch.setattr(
        app.state.decision_capacity,
        "close",
        signal_shutdown_started,
    )
    client = service_client(app)
    shutdown_task: asyncio.Task[bool | None] | None = None
    try:
        response = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer caller-key"},
            json=choice_request_payload(),
        )
        await asyncio.wait_for(
            asyncio.to_thread(preparation_started.wait),
            timeout=2,
        )
        assert response.status_code == 504

        shutdown_task = asyncio.create_task(lifespan.__aexit__(None, None, None))
        await asyncio.wait_for(shutdown_started.wait(), timeout=2)
        assert not shutdown_task.done()

        release_preparation.set()
        await asyncio.wait_for(shutdown_task, timeout=2)
        assert preparation_finished.is_set()
    finally:
        release_preparation.set()
        if shutdown_task is not None and not shutdown_task.done():
            await asyncio.wait_for(shutdown_task, timeout=2)
        await client.aclose()


@pytest.mark.anyio
async def test_caller_disconnect_stops_pending_rows_and_releases_capacity(
    metadata_directory: Path,
) -> None:
    backend_started = asyncio.Event()
    backend_cancelled = asyncio.Event()
    backend_keys: list[str] = []

    async def backend(request: httpx.Request) -> httpx.Response:
        key = request.headers["Authorization"]
        backend_keys.append(key)
        if key == "Bearer disconnected-key":
            backend_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                backend_cancelled.set()
        return completion_response({1: 0.7, 2: 0.3})

    app = create_app(
        configured_settings(
            metadata_directory,
            backend_slots=1,
            admission_capacity=1,
        ),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )
    payload = multi_row_choice_request_payload()
    body = json.dumps(payload).encode()
    incoming: asyncio.Queue[Message] = asyncio.Queue()
    sent: list[Message] = []
    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/systemone",
        "raw_path": b"/v1/systemone",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"authorization", b"Bearer disconnected-key"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8000),
    }

    async def receive() -> Message:
        return await incoming.get()

    async def send(message: Message) -> None:
        sent.append(message)

    async with app.router.lifespan_context(app):
        await incoming.put({"type": "http.request", "body": body, "more_body": False})
        disconnected_request = asyncio.create_task(app(scope, receive, send))
        await asyncio.wait_for(backend_started.wait(), timeout=2)
        await incoming.put({"type": "http.disconnect"})

        await asyncio.wait_for(backend_cancelled.wait(), timeout=2)
        await asyncio.gather(disconnected_request, return_exceptions=True)
        async with service_client(app) as client:
            after_disconnect = await client.post(
                "/v1/systemone",
                headers={"Authorization": "Bearer next-key"},
                json=choice_request_payload(),
            )

    assert sent == []
    assert after_disconnect.status_code == 200
    assert backend_keys == ["Bearer disconnected-key", "Bearer next-key"]


@pytest.mark.anyio
async def test_empty_preparation_for_evaluable_question_returns_a_sanitized_502(
    metadata_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = configured_settings(metadata_directory)
    runtime = DecisionRuntime(settings)
    monkeypatch.setattr(
        runtime._decider,
        "_system_one_items",
        lambda *_args, **_kwargs: ({}, [], []),
    )

    def backend(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("empty preparation must fail before backend work")

    app = create_app(
        settings,
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )

    async with (
        app.router.lifespan_context(app),
        service_client(app) as client,
    ):
        app.state.decision_runtime = runtime
        payload = choice_request_payload()
        payload["questions"]["rating"] = {
            "type": "score",
            "criteria": ["only level"],
        }
        response = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer caller-key"},
            json=payload,
        )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Backend response did not satisfy the inference contract."
    }


@pytest.mark.anyio
async def test_invalid_preparation_shape_returns_a_sanitized_502(
    metadata_directory: Path,
) -> None:
    class InvalidPreparation:
        def _system_one_items(self, *_args: object, **_kwargs: object) -> object:
            return (
                {"priority": {}},
                [("priority", "list", 0, 1)],
                [{"ids": [1], "slots": [], "nopts": [2], "types": ["choice"]}],
            )

    settings = configured_settings(metadata_directory)
    runtime = DecisionRuntime(settings)
    runtime._decider = cast(Any, InvalidPreparation())

    def backend(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("invalid preparation must fail before backend work")

    app = create_app(
        settings,
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )

    async with (
        app.router.lifespan_context(app),
        service_client(app) as client,
    ):
        app.state.decision_runtime = runtime
        response = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer caller-key"},
            json=choice_request_payload(),
        )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Backend response did not satisfy the inference contract."
    }


@pytest.mark.anyio
async def test_calibration_and_assembly_run_off_the_event_loop(
    metadata_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop_thread = threading.get_ident()
    worker_threads: list[int] = []
    original = decision_module._calibrate_and_assemble

    def record_thread(*args: Any, **kwargs: Any) -> Any:
        worker_threads.append(threading.get_ident())
        return original(*args, **kwargs)

    monkeypatch.setattr(decision_module, "_calibrate_and_assemble", record_thread)

    def backend(_request: httpx.Request) -> httpx.Response:
        return completion_response({1: 0.7, 2: 0.3})

    app = create_app(
        configured_settings(metadata_directory),
        backend_transport=ready_backend_transport(metadata_directory, backend),
    )

    async with (
        app.router.lifespan_context(app),
        service_client(app) as client,
    ):
        response = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer caller-key"},
            json=choice_request_payload(),
        )

    assert response.status_code == 200
    assert worker_threads
    assert worker_threads[0] != loop_thread
