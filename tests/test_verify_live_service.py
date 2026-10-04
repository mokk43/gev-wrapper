import sys

import pytest

from scripts import verify_live_service


def test_empty_rejected_key_environment_value_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[dict[str, object]] = []

    def capture_verification_arguments(**arguments: object) -> None:
        received.append(arguments)

    monkeypatch.setenv("DECIDER_LIVE_SERVICE_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("TYPESAFE_API_KEY", "caller-key")
    monkeypatch.setenv("DECIDER_LIVE_ACTUAL_MODEL", "decider-4b")
    monkeypatch.setenv("DECIDER_REJECTED_CALLER_API_KEY", "")
    monkeypatch.setattr(sys, "argv", ["verify_live_service.py"])
    monkeypatch.setattr(
        verify_live_service,
        "verify_live_service",
        capture_verification_arguments,
    )

    assert verify_live_service.main() == 0
    assert received[0]["rejected_api_key"] is None
