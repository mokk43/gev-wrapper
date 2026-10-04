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


def set_dotenv_timing_flag(value: str | None) -> None:
    dotenv = Path(".env")
    lines = [
        line for line in dotenv.read_text(encoding="utf-8").splitlines()
        if not line.startswith("DECIDER_TIMING_LOGGING_ENABLED=")
    ]
    if value is not None:
        lines.append(f"DECIDER_TIMING_LOGGING_ENABLED={value}")
    dotenv.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_timing_defaults_to_disabled_without_dotenv_setting(
    dotenv_settings: Settings,
) -> None:
    set_dotenv_timing_flag(None)

    loaded = load_settings()

    assert loaded.timing_logging_enabled is False
    assert loaded.metadata_directory == dotenv_settings.metadata_directory


@pytest.mark.parametrize("value", ["true", "false"])
def test_dotenv_controls_timing_without_exports(
    dotenv_settings: Settings,
    value: str,
) -> None:
    set_dotenv_timing_flag(value)

    loaded = load_settings()

    assert loaded.timing_logging_enabled is (value == "true")
    assert loaded.model_name == dotenv_settings.model_name


@pytest.mark.parametrize(
    ("dotenv_value", "exported_value"), [("false", "true"), ("true", "false")]
)
def test_exported_timing_setting_overrides_dotenv(
    dotenv_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    dotenv_value: str,
    exported_value: str,
) -> None:
    set_dotenv_timing_flag(dotenv_value)
    monkeypatch.setenv("DECIDER_TIMING_LOGGING_ENABLED", exported_value)

    loaded = load_settings()

    assert loaded.timing_logging_enabled is (exported_value == "true")
    assert loaded.metadata_directory == dotenv_settings.metadata_directory
