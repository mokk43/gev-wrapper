from __future__ import annotations

import asyncio
import json
import math
import string
import threading
from datetime import date
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from fastapi import FastAPI
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

import decider_service.decision as decision_module
from decider_service.app import create_app
from decider_service.config import Settings
from decider_service.decision import DecisionRuntime


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
            for offset, word in enumerate(("none", "not", "listed", "here"))
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
    return tmp_path


def configured_settings(metadata_directory: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "backend_url": "http://127.0.0.1:8080",
        "backend_build": "llama.cpp-b1234",
        "gguf_revision": "b79f09d9ba7837f1b744295ea267b55d08e958ec",
        "gguf_quantization": "Q4_K_M",
        "metadata_directory": metadata_directory,
        "metadata_revision": "b79f09d9ba7837f1b744295ea267b55d08e958ec",
        "model_name": "decider-4b-q4-k-m",
        "model_description": "Decider 4B served by the configured backend.",
        "model_release_date": date(2025, 7, 4),
        "model_aliases": ("jev-latest",),
        "context_capacity": 32_768,
        "backend_slots": 2,
        "admission_capacity": 8,
        "max_request_bytes": 1_048_576,
        "max_questions": 32,
        "max_options": 255,
    }
    values.update(overrides)
    return Settings.model_validate(values)


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
        backend_transport=httpx.MockTransport(backend),
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
        backend_transport=httpx.MockTransport(backend),
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
    assert {
        request.headers["Authorization"] for request in backend_requests
    } == {"Bearer multi-row-key"}


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
        backend_transport=httpx.MockTransport(backend),
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
    assert backend_requests == []


@pytest.mark.anyio
async def test_other_question_types_are_validated_before_staged_rejection(
    metadata_directory: Path,
) -> None:
    app = create_app(configured_settings(metadata_directory))

    async with service_client(app) as client:
        valid_noul = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer caller-key"},
            json={
                "model": "decider-4b-q4-k-m",
                "state": "evidence",
                "questions": {"check": {"type": "noul", "criteria": None}},
            },
        )
        invalid_score = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer caller-key"},
            json={
                "model": "decider-4b-q4-k-m",
                "state": "evidence",
                "questions": {
                    "rating": {"type": "score", "criteria": [None]}
                },
            },
        )

    assert valid_noul.status_code == 422
    assert valid_noul.json()["detail"][0]["loc"] == [
        "body",
        "questions",
        "check",
        "type",
    ]
    assert valid_noul.json()["detail"][0]["type"] == "unsupported_question_type"
    assert invalid_score.status_code == 422
    assert "criteria" in invalid_score.json()["detail"][0]["loc"]


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
        backend_transport=httpx.MockTransport(backend),
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
    choices = [
        response.json()["answers"]["result"]["choice"] for response in responses
    ]
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
        configured_settings(metadata_directory, context_capacity=4),
        backend_transport=httpx.MockTransport(backend),
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
        backend_transport=httpx.MockTransport(backend),
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
async def test_slow_caller_does_not_block_another_caller_or_mix_credentials(
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
        return completion_response({1: 0.7, 2: 0.3})

    app = create_app(
        configured_settings(metadata_directory),
        backend_transport=httpx.MockTransport(backend),
    )
    payload = {
        "model": "decider-4b-q4-k-m",
        "state": "evidence",
        "questions": {
            "priority": {
                "type": "choice",
                "criteria": {"urgent": None, "routine": None},
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
async def test_question_prompt_is_unchanged_when_other_questions_are_reordered(
    metadata_directory: Path,
) -> None:
    backend_requests: list[httpx.Request] = []

    def backend(request: httpx.Request) -> httpx.Response:
        backend_requests.append(request)
        return completion_response({1: 0.7, 2: 0.3})

    app = create_app(
        configured_settings(metadata_directory),
        backend_transport=httpx.MockTransport(backend),
    )
    priority = {
        "type": "choice",
        "instructions": "Choose priority.",
        "criteria": {"urgent": None, "routine": None},
    }
    owner = {
        "type": "choice",
        "instructions": "Choose owner for this request using more words.",
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
                "questions": {"priority": priority},
            },
        )
        second = await client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer caller-key"},
            json={
                "model": "decider-4b-q4-k-m",
                "state": "same evidence",
                "questions": {"owner": owner, "priority": priority},
            },
        )

    assert first.status_code == 200
    assert second.status_code == 200
    prompts = [json.loads(request.content)["prompt"] for request in backend_requests]
    assert prompts[0] == prompts[2]
    assert first.json()["answers"]["priority"] == second.json()["answers"][
        "priority"
    ]


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
        backend_transport=httpx.MockTransport(backend),
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
        backend_transport=httpx.MockTransport(backend),
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
        backend_transport=httpx.MockTransport(backend),
    )
    payload = choice_request_payload()
    payload["questions"]["owner"] = {
        "type": "choice",
        "criteria": {"support": None, "sales": None},
    }

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
        backend_transport=httpx.MockTransport(backend),
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
        backend_transport=httpx.MockTransport(backend),
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
        backend_transport=httpx.MockTransport(backend),
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
