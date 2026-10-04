from __future__ import annotations

import hashlib
import json
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

from decider.prompt import resolve_layout  # type: ignore[import-untyped]

from decider_service.config import Settings

_MANIFEST_NAME = "deployment-manifest.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class DeploymentValidationError(RuntimeError):
    pass


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentValidationError(
            f"Cannot load required {description}: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise DeploymentValidationError(f"{path.name} must contain a JSON object")
    return cast(dict[str, Any], value)


def _manifest_files(metadata_directory: Path) -> dict[str, str]:
    manifest_path = metadata_directory / _MANIFEST_NAME
    return {
        path.relative_to(metadata_directory).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(metadata_directory.rglob("*"))
        if path.is_file() and path != manifest_path
    }


def load_local_runtime_config(metadata_directory: Path) -> dict[str, Any]:
    config = _read_json_object(
        metadata_directory / "decider_config.json",
        "Decider configuration",
    )
    required_config_fields = {
        "version",
        "temperature",
        "neutralize_none",
        "isolated_levels",
    }
    missing_fields = sorted(required_config_fields - config.keys())
    if missing_fields:
        raise DeploymentValidationError(
            "decider_config.json must explicitly define " + ", ".join(missing_fields)
        )
    if not isinstance(config["version"], str) or not config["version"].strip():
        raise DeploymentValidationError(
            "decider_config.json version must be a nonempty string"
        )
    if not isinstance(config["neutralize_none"], bool) or not isinstance(
        config["isolated_levels"], bool
    ):
        raise DeploymentValidationError(
            "decider_config.json neutralize_none and isolated_levels must be booleans"
        )
    try:
        resolve_layout(config)
    except ValueError as exc:
        raise DeploymentValidationError(str(exc)) from exc
    return config


def validate_local_deployment(settings: Settings) -> dict[str, Any]:
    metadata_directory = Path(settings.metadata_directory)
    manifest = _read_json_object(
        metadata_directory / _MANIFEST_NAME,
        "deployment manifest",
    )
    expected_manifest_values = {
        "metadata_revision": settings.metadata_revision,
        "gguf_revision": settings.gguf_revision,
        "gguf_quantization": settings.gguf_quantization,
        "model_name": settings.model_name,
    }
    for field, expected in expected_manifest_values.items():
        if manifest.get(field) != expected:
            raise DeploymentValidationError(
                f"deployment manifest {field} does not match configured {field}"
            )

    dependency_version = manifest.get("decider_dependency_version")
    try:
        installed_dependency_version = version("decider-ai")
    except PackageNotFoundError as exc:
        raise DeploymentValidationError(
            "configured Decider dependency is not installed"
        ) from exc
    if dependency_version != installed_dependency_version:
        raise DeploymentValidationError(
            "deployment manifest Decider dependency does not match the "
            "installed version"
        )

    recorded_files = manifest.get("files")
    if not isinstance(recorded_files, dict) or not all(
        isinstance(path, str)
        and isinstance(digest, str)
        and _SHA256.fullmatch(digest) is not None
        for path, digest in recorded_files.items()
    ):
        raise DeploymentValidationError(
            "deployment manifest files must map relative paths to SHA-256 digests"
        )
    actual_files = _manifest_files(metadata_directory)
    if recorded_files != actual_files:
        raise DeploymentValidationError(
            "local tokenizer/config files do not match the deployment manifest"
        )

    config = load_local_runtime_config(metadata_directory)
    if manifest.get("decider_config_version") != config["version"]:
        raise DeploymentValidationError(
            "deployment manifest Decider config version does not match metadata"
        )
    prompt_layout = resolve_layout(config)
    if manifest.get("prompt_layout") != prompt_layout:
        raise DeploymentValidationError(
            "deployment manifest prompt layout does not match Decider metadata"
        )
    return config
