from __future__ import annotations

import re
from datetime import date
from ipaddress import IPv4Address
from typing import Annotated

from pydantic import (
    AfterValidator,
    AnyHttpUrl,
    DirectoryPath,
    Field,
    IPvAnyAddress,
    PositiveFloat,
    PositiveInt,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


def _nonempty(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("must not be empty")
    return stripped


NonemptyString = Annotated[str, AfterValidator(_nonempty)]
IMMUTABLE_REVISION = re.compile(
    r"(?:[0-9a-f]{40}|(?:sha256:)?[0-9a-f]{64})",
    re.IGNORECASE,
)
INVALID_PROBE_CREDENTIAL = "decider-readiness-invalid-credential"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DECIDER_",
        extra="ignore",
        frozen=True,
    )

    backend_url: AnyHttpUrl
    backend_build: NonemptyString | None = None
    backend_model_id: NonemptyString | None = None
    backend_model_path: NonemptyString | None = None
    gguf_revision: NonemptyString | None = None
    gguf_quantization: NonemptyString | None = None
    metadata_directory: DirectoryPath
    metadata_revision: NonemptyString | None = None

    model_name: NonemptyString
    model_description: NonemptyString
    model_release_date: date
    model_aliases: tuple[str, ...] = ()

    context_capacity: PositiveInt
    backend_slots: PositiveInt
    admission_capacity: PositiveInt
    max_request_bytes: PositiveInt
    max_questions: PositiveInt
    max_options: int = Field(ge=2, le=255)

    operator_probe_api_key: SecretStr
    bind_host: IPvAnyAddress = IPv4Address("127.0.0.1")
    bind_port: int = Field(default=8000, ge=1, le=65_535)
    request_deadline_seconds: PositiveFloat = 60.0
    initial_probability_coverage: PositiveInt = 256
    maximum_probability_coverage: PositiveInt
    startup_probe_attempts: PositiveInt = 3
    startup_probe_timeout_seconds: PositiveFloat = 30.0
    startup_tokenizer_probe_chunk_size: int = Field(default=4096, ge=1, le=8192)
    skip_deployment_identity_validation: bool = False

    @field_validator("model_aliases", mode="before")
    @classmethod
    def normalize_aliases(cls, value: object) -> object:
        if isinstance(value, list | tuple):
            if not all(isinstance(alias, str) for alias in value):
                raise ValueError("model_aliases must contain only strings")
            return tuple(_nonempty(alias) for alias in value)
        return value

    @field_validator("gguf_revision", "metadata_revision")
    @classmethod
    def revision_is_immutable(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if IMMUTABLE_REVISION.fullmatch(value) is None:
            raise ValueError("must identify an immutable revision")
        return value

    @field_validator("model_release_date")
    @classmethod
    def release_date_is_not_in_the_future(cls, value: date) -> date:
        if value > date.today():
            raise ValueError("must not be in the future")
        return value

    @model_validator(mode="after")
    def configuration_is_consistent(self) -> Settings:
        if (
            self.skip_deployment_identity_validation
            and not self.bind_host.is_loopback
        ):
            raise ValueError(
                "skip_deployment_identity_validation requires a loopback bind_host"
            )
        identity_values = {
            "backend_build": self.backend_build,
            "backend_model_id": self.backend_model_id,
            "backend_model_path": self.backend_model_path,
            "gguf_revision": self.gguf_revision,
            "gguf_quantization": self.gguf_quantization,
            "metadata_revision": self.metadata_revision,
        }
        if not self.skip_deployment_identity_validation:
            missing = sorted(
                name for name, value in identity_values.items() if value is None
            )
            if missing:
                raise ValueError(
                    "deployment identity settings are required: " + ", ".join(missing)
                )
        if not self.operator_probe_api_key.get_secret_value().strip():
            raise ValueError("operator_probe_api_key must not be empty")
        if len(set(self.model_aliases)) != len(self.model_aliases):
            raise ValueError("model_aliases must be unique")
        if self.model_name in self.model_aliases:
            raise ValueError("model_aliases must not repeat model_name")
        if self.admission_capacity < self.backend_slots:
            raise ValueError("admission_capacity must be at least backend_slots")
        if self.initial_probability_coverage > self.maximum_probability_coverage:
            raise ValueError(
                "initial_probability_coverage must not exceed "
                "maximum_probability_coverage"
            )
        if self.initial_probability_coverage != 256:
            raise ValueError("initial_probability_coverage must be 256")
        if self.operator_probe_api_key.get_secret_value() == INVALID_PROBE_CREDENTIAL:
            raise ValueError("operator_probe_api_key uses the reserved rejection probe")
        return self


def load_settings() -> Settings:
    # Exported environment values take precedence over the local .env file.
    return Settings(_env_file=".env", _env_file_encoding="utf-8")  # type: ignore[call-arg]
