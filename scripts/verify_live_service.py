#!/usr/bin/env python3
"""Opt-in verification of a configured service through its public HTTP API."""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from collections.abc import Mapping
from typing import Any

import httpx

B64TOKEN = re.compile(r"^[A-Za-z0-9._~+/\-]+={0,}$")


class VerificationError(RuntimeError):
    """A public live-check expectation was not met."""


def _object(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise VerificationError(f"{label} was not a JSON object")
    return value


def _expect_status(response: httpx.Response, expected: int, *, label: str) -> None:
    if response.status_code != expected:
        raise VerificationError(
            f"{label} returned HTTP {response.status_code}; expected {expected}"
        )


def _probabilities(
    value: object,
    *,
    expected_keys: set[str],
    label: str,
) -> Mapping[str, Any]:
    probabilities = _object(value, label=label)
    if set(probabilities) != expected_keys:
        raise VerificationError(f"{label} did not preserve the expected labels")
    numbers = list(probabilities.values())
    if any(
        not isinstance(number, int | float)
        or isinstance(number, bool)
        or not math.isfinite(number)
        or not 0 <= number <= 1
        for number in numbers
    ):
        raise VerificationError(f"{label} contained an invalid probability")
    if not math.isclose(sum(numbers), 1.0, abs_tol=1e-5):
        raise VerificationError(f"{label} did not sum to one within 1e-5")
    return probabilities


def _verify_catalog(
    payload: object,
    *,
    requested_model: str | None,
    actual_model: str,
) -> str:
    catalog = _object(payload, label="model catalog")
    models = catalog.get("models")
    if not isinstance(models, list) or not models:
        raise VerificationError("model catalog contained no models")
    names: list[str] = []
    for entry in models:
        model = _object(entry, label="model catalog entry")
        if set(model) != {"name", "description", "release_date"}:
            raise VerificationError("model catalog entry had an unexpected shape")
        if not all(isinstance(model[field], str) for field in model):
            raise VerificationError("model catalog entry contained a non-string field")
        names.append(model["name"])
    if actual_model not in names:
        raise VerificationError(f"actual model {actual_model!r} was not catalogued")
    if requested_model is not None and requested_model not in names:
        raise VerificationError(
            f"requested model {requested_model!r} was not catalogued"
        )
    return requested_model or actual_model


def _representative_request(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "state": {
            "service": "checkout",
            "observation": "Requests are failing after a release.",
        },
        "questions": {
            "priority": {
                "type": "choice",
                "instructions": "Choose the response priority.",
                "criteria": {
                    "routine": "Can wait for normal review.",
                    "urgent": "Requires immediate investigation.",
                },
            },
            "is_outage": {
                "type": "noul",
                "instructions": "Is the service unavailable?",
                "criteria": None,
            },
            "urgency": {
                "type": "score",
                "instructions": "Rate the response urgency.",
                "criteria": ["Can wait", "Needs attention", "Act now"],
            },
        },
    }


def _number_in_range(value: object, lower: float, upper: float, *, label: str) -> None:
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not lower <= value <= upper
    ):
        raise VerificationError(f"{label} was outside {lower} through {upper}")


def _verify_decision(payload: object, *, expected_model: str) -> str:
    decision = _object(payload, label="decision response")
    if set(decision) != {"model", "answers", "usage"}:
        raise VerificationError("decision response had an unexpected shape")
    if not isinstance(decision["model"], str):
        raise VerificationError("decision response model was not a string")
    if decision["model"] != expected_model:
        raise VerificationError(
            f"decision resolved to {decision['model']!r}; expected {expected_model!r}"
        )

    answers = _object(decision["answers"], label="answer map")
    if set(answers) != {"priority", "is_outage", "urgency"}:
        raise VerificationError("answer map did not preserve question names")

    choice = _object(answers["priority"], label="Choice answer")
    if choice.get("type") != "choice" or choice.get("choice") not in {
        "routine",
        "urgent",
    }:
        raise VerificationError("Choice answer was invalid")
    _number_in_range(choice.get("confidence"), 0, 1, label="Choice confidence")
    _probabilities(
        choice.get("probabilities"),
        expected_keys={"routine", "urgent"},
        label="Choice probabilities",
    )

    noul = _object(answers["is_outage"], label="Noul answer")
    if noul.get("type") != "noul":
        raise VerificationError("Noul answer had the wrong type")
    _number_in_range(noul.get("noul"), 0, 1, label="Noul probability")

    score = _object(answers["urgency"], label="Score answer")
    if score.get("type") != "score":
        raise VerificationError("Score answer had the wrong type")
    _number_in_range(score.get("score"), 0, 2, label="Score")
    _number_in_range(score.get("confidence"), 0, 1, label="Score confidence")
    legend = _object(score.get("legend"), label="Score legend")
    expected_legend = {"0": "Can wait", "1": "Needs attention", "2": "Act now"}
    if dict(legend) != expected_legend:
        raise VerificationError("Score legend did not preserve the rubric")
    _probabilities(
        score.get("probabilities"),
        expected_keys=set(expected_legend),
        label="Score probabilities",
    )

    usage = _object(decision["usage"], label="usage")
    if set(usage) != {"input_tokens", "output_tokens"}:
        raise VerificationError("usage had an unexpected shape")
    if any(
        not isinstance(usage[field], int)
        or isinstance(usage[field], bool)
        or usage[field] <= 0
        for field in usage
    ):
        raise VerificationError("usage counters were not positive integers")
    return decision["model"]


def verify_live_service(
    *,
    base_url: str,
    api_key: str,
    actual_model: str,
    rejected_api_key: str | None,
    model: str | None,
    timeout: float,
) -> None:
    if not B64TOKEN.fullmatch(api_key):
        raise VerificationError("API key is not an RFC 6750 b64token")
    if rejected_api_key is not None:
        if not B64TOKEN.fullmatch(rejected_api_key):
            raise VerificationError("rejected API key is not an RFC 6750 b64token")
        if rejected_api_key == api_key:
            raise VerificationError("rejected API key must differ from the caller key")

    with httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout) as client:
        missing = client.get("/v1/models")
        _expect_status(missing, 401, label="unauthenticated model catalog")
        if missing.headers.get("www-authenticate") != "Bearer":
            raise VerificationError("authentication challenge was not Bearer")
        print("PASS missing bearer credential rejected with HTTP 401")

        malformed = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer invalid credential"},
        )
        _expect_status(malformed, 401, label="malformed bearer credential")
        print("PASS malformed bearer credential rejected with HTTP 401")

        if rejected_api_key is not None:
            rejected = client.get(
                "/v1/models",
                headers={"Authorization": f"Bearer {rejected_api_key}"},
            )
            if rejected.status_code not in {401, 403}:
                raise VerificationError(
                    "explicit rejected bearer credential returned HTTP "
                    f"{rejected.status_code}; expected 401 or 403"
                )
            print(
                "PASS backend rejected the explicit invalid credential with HTTP "
                f"{rejected.status_code}"
            )
        else:
            print(
                "UNVERIFIED selected-backend rejection of a syntactically valid "
                "credential: no rejected key was supplied"
            )

        headers = {"Authorization": f"Bearer {api_key}"}
        catalog_response = client.get("/v1/models", headers=headers)
        _expect_status(catalog_response, 200, label="authenticated model catalog")
        selected_model = _verify_catalog(
            catalog_response.json(),
            requested_model=model,
            actual_model=actual_model,
        )
        print(f"PASS authenticated model catalog; selected {selected_model}")

        decision_response = client.post(
            "/v1/systemone",
            headers=headers,
            json=_representative_request(selected_model),
        )
        _expect_status(decision_response, 200, label="mixed typed decision")
        resolved_model = _verify_decision(
            decision_response.json(), expected_model=actual_model
        )
        print(f"PASS mixed Choice/Noul/Score decision; resolved {resolved_model}")

    print(
        "UNVERIFIED issued live TypeSafe key format: the supplied key passed only "
        "the RFC 6750 syntax check"
    )
    print(
        "UNVERIFIED trusted Decider baseline and GGUF numerical equivalence: no "
        "baseline is an input to this check"
    )
    print(
        "UNMEASURED representative workload, concurrency, and latency: this is one "
        "smoke request, not a workload benchmark"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a running, fully configured service without provisioning its "
            "backend or downloading artifacts."
        )
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("DECIDER_LIVE_SERVICE_URL"),
        help="service URL (or DECIDER_LIVE_SERVICE_URL)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("TYPESAFE_API_KEY"),
        help="caller key (or TYPESAFE_API_KEY)",
    )
    parser.add_argument(
        "--actual-model",
        default=os.getenv("DECIDER_LIVE_ACTUAL_MODEL"),
        help="canonical configured model identity (or DECIDER_LIVE_ACTUAL_MODEL)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("DECIDER_LIVE_MODEL"),
        help=(
            "catalogued request model or alias "
            "(or DECIDER_LIVE_MODEL; default: actual model)"
        ),
    )
    parser.add_argument(
        "--rejected-api-key",
        default=os.getenv("DECIDER_REJECTED_CALLER_API_KEY") or None,
        help=(
            "known rejected backend key for live auth enforcement "
            "(or DECIDER_REJECTED_CALLER_API_KEY)"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=65.0,
        help="per-operation transport timeout in seconds (default: 65)",
    )
    args = parser.parse_args()

    if not args.base_url:
        parser.error("--base-url or DECIDER_LIVE_SERVICE_URL is required")
    if not args.api_key:
        parser.error("--api-key or TYPESAFE_API_KEY is required")
    if not args.actual_model:
        parser.error("--actual-model or DECIDER_LIVE_ACTUAL_MODEL is required")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    try:
        verify_live_service(
            base_url=args.base_url,
            api_key=args.api_key,
            actual_model=args.actual_model,
            rejected_api_key=args.rejected_api_key,
            model=args.model,
            timeout=args.timeout,
        )
    except (httpx.HTTPError, ValueError, VerificationError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
