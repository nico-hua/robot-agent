"""Tests for project-wide provider configuration loading."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError
from src.config import loader
from src.config.loader import load_provider_config, load_workspace_path


def _provider_environment(**overrides: str) -> dict[str, str]:
    environment = {
        "PROVIDER_TYPE": "openai_compat",
        "PROVIDER_API_KEY": "test-key",
        "PROVIDER_API_BASE": "https://api.example.test/v1",
        "PROVIDER_MODEL": "test-model",
        "PROVIDER_MAX_TOKENS": "2048",
        "PROVIDER_TEMPERATURE": "0.25",
    }
    environment.update(overrides)
    return environment


def test_load_provider_config_converts_unified_environment_values() -> None:
    config = load_provider_config(_provider_environment())

    assert config.type == "openai_compat"
    assert config.api_key == "test-key"
    assert config.api_base == "https://api.example.test/v1"
    assert config.default_model == "test-model"
    assert config.default_max_tokens == 2048
    assert config.default_temperature == 0.25


def test_load_provider_config_uses_process_environment_over_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "PROVIDER_TYPE=openai_compat",
                "PROVIDER_API_KEY=dotenv-key",
                "PROVIDER_API_BASE=https://api.example.test",
                "PROVIDER_MODEL=dotenv-model",
                "PROVIDER_MAX_TOKENS=1024",
                "PROVIDER_TEMPERATURE=0.7",
            ]
        ),
        encoding="utf-8",
    )
    for name in _provider_environment():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(loader, "_PROJECT_ENV_FILE", env_file)
    monkeypatch.setenv("PROVIDER_MODEL", "process-model")

    config = load_provider_config()

    assert config.type == "openai_compat"
    assert config.default_model == "process-model"
    assert config.default_max_tokens == 1024
    assert "PROVIDER_TYPE" not in os.environ


def test_load_provider_config_treats_blank_api_key_as_optional() -> None:
    config = load_provider_config(_provider_environment(PROVIDER_API_KEY=""))

    assert config.api_key == ""


def test_load_provider_config_rejects_unsupported_provider_type() -> None:
    with pytest.raises(ValidationError):
        load_provider_config(_provider_environment(PROVIDER_TYPE="unsupported_compat"))


def test_load_provider_config_reports_missing_required_values() -> None:
    with pytest.raises(ValidationError):
        load_provider_config({"PROVIDER_TYPE": "openai_compat"})


def test_load_workspace_path_uses_explicit_environment(tmp_path: Path) -> None:
    workspace = load_workspace_path({"WORKSPACE_PATH": str(tmp_path)})

    assert workspace == tmp_path.resolve()


def test_load_workspace_path_uses_process_environment_over_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    dotenv_workspace = tmp_path / "dotenv-workspace"
    process_workspace = tmp_path / "process-workspace"
    env_file.write_text(
        f"WORKSPACE_PATH={dotenv_workspace}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(loader, "_PROJECT_ENV_FILE", env_file)
    monkeypatch.setenv("WORKSPACE_PATH", str(process_workspace))

    workspace = load_workspace_path()

    assert workspace == process_workspace.resolve()


@pytest.mark.parametrize("value", [None, "", "   "])
def test_load_workspace_path_rejects_missing_or_blank_value(value: str | None) -> None:
    environ = {} if value is None else {"WORKSPACE_PATH": value}

    with pytest.raises(ValueError, match="WORKSPACE_PATH"):
        load_workspace_path(environ)
