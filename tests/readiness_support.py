from __future__ import annotations

import hashlib
import inspect
import json
import math
import string
from collections.abc import Awaitable, Callable
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from decider_service.config import INVALID_PROBE_CREDENTIAL, Settings

PROBE_KEY = "operator-probe-key"
BACKEND_MODEL_ID = "decider-4b-test-artifact"
BACKEND_MODEL_PATH = "/models/decider-4b-test-Q4_K_M.gguf"
GGUF_REVISION = "b79f09d9ba7837f1b744295ea267b55d08e958ec"
METADATA_REVISION = "50d0be0d7cb43d2066965ce5fa7f3fe4e489a60f"


def write_test_metadata(directory: Path) -> Path:
    labels = list(string.ascii_uppercase) + [
        first + second
        for first in string.ascii_uppercase
        for second in string.ascii_uppercase
    ]
    vocabulary = {"[UNK]": 0, **{label: i + 1 for i, label in enumerate(labels)}}
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
    ).save_pretrained(directory)
    (directory / "decider_config.json").write_text(
        json.dumps(
            {
                "version": "test",
                "layout": "plain",
                "temperature": 1.0,
                "neutralize_none": True,
                "isolated_levels": True,
            }
        )
    )
    write_test_manifest(directory)
    return directory


def write_test_manifest(directory: Path) -> None:
    manifest_path = directory / "deployment-manifest.json"
    files = {
        path.relative_to(directory).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    manifest_path.write_text(
        json.dumps(
            {
                "metadata_revision": METADATA_REVISION,
                "gguf_revision": GGUF_REVISION,
                "gguf_quantization": "Q4_K_M",
                "model_name": "decider-4b-q4-k-m",
                "decider_config_version": "test",
                "decider_dependency_version": "1.8.1",
                "prompt_layout": "plain",
                "files": files,
            }
        )
    )


def configured_settings(metadata_directory: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "backend_url": "http://127.0.0.1:8080",
        "backend_build": "llama.cpp-b1234",
        "backend_model_id": BACKEND_MODEL_ID,
        "backend_model_path": BACKEND_MODEL_PATH,
        "gguf_revision": GGUF_REVISION,
        "gguf_quantization": "Q4_K_M",
        "metadata_directory": metadata_directory,
        "metadata_revision": METADATA_REVISION,
        "model_name": "decider-4b-q4-k-m",
        "model_description": "Decider 4B served by the configured llama.cpp backend.",
        "model_release_date": date(2025, 7, 4),
        "model_aliases": ("jev-latest",),
        "context_capacity": 32_768,
        "backend_slots": 2,
        "admission_capacity": 8,
        "max_request_bytes": 1_048_576,
        "max_questions": 32,
        "max_options": 255,
        "initial_probability_coverage": 256,
        "maximum_probability_coverage": 512,
        "operator_probe_api_key": PROBE_KEY,
        "startup_probe_attempts": 3,
        "startup_probe_timeout_seconds": 5.0,
    }
    values.update(overrides)
    return Settings.model_validate(values)


def assert_failure_was_logged(
    response: httpx.Response,
    caplog: pytest.LogCaptureFixture,
    category: str,
) -> str:
    request_id = response.headers["x-typesafe-request-id"]
    messages = "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.name == "decider_service"
    )
    assert request_id in messages
    assert f"category={category}" in messages
    return messages


def completion_response(
    probabilities: dict[int, float],
    *,
    input_tokens: int = 11,
    output_tokens: int = 1,
    cached_tokens: int = 0,
    coverage: int = 256,
    excluded_token_ids: tuple[int, ...] = (),
) -> httpx.Response:
    covered_token_ids = set(probabilities) | set(excluded_token_ids)
    filler_token_ids = (
        token_id
        for token_id in range(coverage + len(covered_token_ids))
        if token_id not in covered_token_ids
    )
    top_logprobs = [
        {"id": token_id, "logprob": math.log(probability)}
        for token_id, probability in probabilities.items()
    ]
    top_logprobs.extend(
        {"id": next(filler_token_ids), "logprob": -1000.0}
        for _ in range(coverage - len(top_logprobs))
    )
    return httpx.Response(
        200,
        json={
            "content": "generated text is ignored",
            "probs": [
                {
                    "top_logprobs": top_logprobs
                }
            ],
            "tokens_cached": cached_tokens,
            "tokens_evaluated": input_tokens,
            "tokens_predicted": output_tokens,
            "truncated": False,
        },
    )


RuntimeResponder = Callable[[httpx.Request], httpx.Response]
AsyncRuntimeResponder = Callable[
    [httpx.Request],
    httpx.Response | Awaitable[httpx.Response],
]


class ControlledBackend:
    def __init__(
        self,
        metadata_directory: Path,
        runtime_responder: RuntimeResponder | None = None,
    ) -> None:
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(metadata_directory)
        self.runtime_responder = runtime_responder or (
            lambda _request: completion_response({1: 0.7, 2: 0.3})
        )
        self.requests: list[httpx.Request] = []

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        authorization = request.headers.get("Authorization")
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if authorization != f"Bearer {PROBE_KEY}":
            if authorization == f"Bearer {INVALID_PROBE_CREDENTIAL}":
                return httpx.Response(
                    401,
                    json={"error": {"type": "authentication_error"}},
                )
            return self.runtime_responder(request)
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "id": BACKEND_MODEL_ID,
                            "object": "model",
                            "meta": {
                                "n_vocab": len(self.tokenizer),
                                "n_ctx_train": 32_768,
                                "ftype": "Q4_K - Medium",
                            },
                        }
                    ],
                },
            )
        if request.url.path == "/props":
            return httpx.Response(
                200,
                json={
                    "default_generation_settings": {"n_ctx": 32_768},
                    "total_slots": 256,
                    "model_path": BACKEND_MODEL_PATH,
                    "build_info": "llama.cpp-b1234",
                },
            )
        if request.url.path == "/tokenize":
            body = json.loads(request.content)
            backend_tokenizer = self.tokenizer.backend_tokenizer
            previous_parse_special = backend_tokenizer.encode_special_tokens
            try:
                backend_tokenizer.encode_special_tokens = body["parse_special"]
                tokens = self.tokenizer.encode(
                    body["content"],
                    add_special_tokens=body["add_special"],
                )
            finally:
                backend_tokenizer.encode_special_tokens = previous_parse_special
            return httpx.Response(200, json={"tokens": tokens})
        if request.url.path == "/detokenize":
            body = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "content": self.tokenizer.decode(
                        body["tokens"],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                },
            )
        if request.url.path == "/completion":
            completion_body: dict[str, Any] = json.loads(request.content)
            coverage = completion_body["n_probs"]
            return httpx.Response(
                200,
                json={
                    "probs": [
                        {
                            "top_logprobs": [
                                {
                                    "id": token_id,
                                    "logprob": -math.log(coverage),
                                }
                                for token_id in range(coverage)
                            ]
                        }
                    ],
                    "tokens_cached": 0,
                    "tokens_evaluated": len(completion_body["prompt"]),
                    "tokens_predicted": 1,
                    "truncated": False,
                },
            )
        raise AssertionError(f"unexpected backend route {request.url.path}")


def ready_backend_transport(
    metadata_directory: Path,
    runtime_responder: AsyncRuntimeResponder,
) -> httpx.MockTransport:
    backend = ControlledBackend(metadata_directory)

    async def handle(request: httpx.Request) -> httpx.Response:
        authorization = request.headers.get("Authorization")
        if request.url.path == "/health" or authorization in {
            f"Bearer {PROBE_KEY}",
            f"Bearer {INVALID_PROBE_CREDENTIAL}",
        }:
            return backend(request)
        response = runtime_responder(request)
        if inspect.isawaitable(response):
            return await response
        return response

    return httpx.MockTransport(handle)
