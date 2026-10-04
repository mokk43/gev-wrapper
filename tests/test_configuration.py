from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from decider_service.config import Settings, load_settings
from tests.readiness_support import configured_settings, write_test_metadata


@pytest.fixture
def dotenv_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    for name in tuple(os.environ):
        if name.startswith("DECIDER_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    metadata_directory = tmp_path / "metadata"
    metadata_directory.mkdir()
    write_test_metadata(metadata_directory)
    settings = configured_settings(
        metadata_directory,
        model_description="Décider dotenv fixture.",
        bind_host="127.0.0.2",
        bind_port=8088,
        operator_probe_api_key="dotenv-test-probe",
    )
    values = settings.model_dump(mode="json", exclude_none=True)
    values["operator_probe_api_key"] = (
        settings.operator_probe_api_key.get_secret_value()
    )
    (tmp_path / ".env").write_text(
        "\n".join(
            f"DECIDER_{name.upper()}='{json.dumps(value, ensure_ascii=False)}'"
            if isinstance(value, list)
            else f"DECIDER_{name.upper()}={json.dumps(value, ensure_ascii=False)}"
            for name, value in values.items()
        )
        + "\n",
        encoding="utf-8",
    )
    return settings


def test_load_settings_reads_complete_dotenv_without_exports(
    dotenv_settings: Settings,
) -> None:
    loaded = load_settings()

    assert loaded.model_dump() == dotenv_settings.model_dump()
    assert loaded.operator_probe_api_key.get_secret_value() == "dotenv-test-probe"


def test_exported_settings_override_dotenv(
    dotenv_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DECIDER_BIND_HOST", "127.0.0.3")
    monkeypatch.setenv("DECIDER_BIND_PORT", "9099")
    monkeypatch.setenv("DECIDER_OPERATOR_PROBE_API_KEY", "exported-test-probe")

    loaded = load_settings()

    assert str(loaded.bind_host) == "127.0.0.3"
    assert loaded.bind_port == 9099
    assert loaded.operator_probe_api_key.get_secret_value() == "exported-test-probe"
    assert loaded.metadata_directory == dotenv_settings.metadata_directory
    assert loaded.model_release_date == dotenv_settings.model_release_date
    assert loaded.model_aliases == dotenv_settings.model_aliases
