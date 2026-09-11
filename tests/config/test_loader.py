"""Tests for unified project configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from src.config import loader
from src.config.loader import load_agent_config


def _agent_environment(workspace: Path, **overrides: str) -> dict[str, str]:
    environment = {
        "WORKSPACE_PATH": str(workspace),
        "PROVIDER_TYPE": "openai_compat",
        "PROVIDER_API_KEY": "test-key",
        "PROVIDER_API_BASE": "https://api.example.test/v1",
        "PROVIDER_MODEL": "test-model",
        "PROVIDER_MAX_TOKENS": "2048",
        "PROVIDER_TEMPERATURE": "0.25",
        "PROVIDER_REQUEST_TIMEOUT_SECONDS": "45",
        "API_HOST": "127.0.0.1",
        "API_PORT": "8000",
        "API_REQUEST_TIMEOUT_SECONDS": "60",
    }
    environment.update(overrides)
    return environment


def test_load_agent_config_converts_all_project_environment_values(tmp_path: Path) -> None:
    config = load_agent_config(
        _agent_environment(
            tmp_path,
            API_HOST=" 0.0.0.0 ",
            API_PORT="9000",
            API_REQUEST_TIMEOUT_SECONDS="2.5",
        )
    )

    assert config.workspace_path == tmp_path.resolve()
    assert config.provider.type == "openai_compat"
    assert config.provider.api_key == "test-key"
    assert config.provider.api_base == "https://api.example.test/v1"
    assert config.provider.default_model == "test-model"
    assert config.provider.default_max_tokens == 2048
    assert config.provider.default_temperature == 0.25
    assert config.provider.request_timeout_seconds == 45.0
    assert config.api.host == "0.0.0.0"
    assert config.api.port == 9000
    assert config.api.request_timeout_seconds == 2.5


def test_load_agent_config_uses_defaults_for_blank_optional_values(tmp_path: Path) -> None:
    config = load_agent_config(
        _agent_environment(
            tmp_path,
            PROVIDER_API_KEY="",
            PROVIDER_REQUEST_TIMEOUT_SECONDS=" ",
            API_HOST=" ",
            API_PORT=" ",
            API_REQUEST_TIMEOUT_SECONDS=" ",
        )
    )

    assert config.provider.api_key == ""
    assert config.provider.request_timeout_seconds == 60.0
    assert config.api.host == "127.0.0.1"
    assert config.api.port == 8000
    assert config.api.request_timeout_seconds == 60.0


def test_load_agent_config_supports_native_ollama_with_an_empty_api_key(tmp_path: Path) -> None:
    config = load_agent_config(
        _agent_environment(
            tmp_path,
            PROVIDER_TYPE="ollama",
            PROVIDER_API_KEY="",
            PROVIDER_API_BASE="http://127.0.0.1:11434",
            PROVIDER_MODEL="qwen3-vl:4b",
        )
    )

    assert config.provider.type == "ollama"
    assert config.provider.api_key == ""
    assert config.provider.api_base == "http://127.0.0.1:11434"
    assert config.provider.default_model == "qwen3-vl:4b"


def test_load_agent_config_uses_process_environment_over_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dotenv_workspace = tmp_path / "dotenv-workspace"
    process_workspace = tmp_path / "process-workspace"
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                f"WORKSPACE_PATH={dotenv_workspace}",
                "PROVIDER_TYPE=openai_compat",
                "PROVIDER_API_KEY=dotenv-key",
                "PROVIDER_API_BASE=https://api.example.test/v1",
                "PROVIDER_MODEL=dotenv-model",
                "PROVIDER_MAX_TOKENS=1024",
                "PROVIDER_TEMPERATURE=0.7",
                "PROVIDER_REQUEST_TIMEOUT_SECONDS=15",
                "API_HOST=127.0.0.1",
                "API_PORT=8001",
                "API_REQUEST_TIMEOUT_SECONDS=10",
            ]
        ),
        encoding="utf-8",
    )
    for name in _agent_environment(tmp_path):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(loader, "_PROJECT_ENV_FILE", env_file)
    monkeypatch.setenv("WORKSPACE_PATH", str(process_workspace))
    monkeypatch.setenv("PROVIDER_MODEL", "process-model")
    monkeypatch.setenv("API_PORT", "9001")

    config = load_agent_config()

    assert config.workspace_path == process_workspace.resolve()
    assert config.provider.default_model == "process-model"
    assert config.provider.default_max_tokens == 1024
    assert config.provider.request_timeout_seconds == 15.0
    assert config.api.host == "127.0.0.1"
    assert config.api.port == 9001
    assert config.api.request_timeout_seconds == 10.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"PROVIDER_TYPE": "unsupported_compat"},
        {"PROVIDER_REQUEST_TIMEOUT_SECONDS": "0"},
        {"API_PORT": "0"},
        {"API_PORT": "65536"},
        {"API_REQUEST_TIMEOUT_SECONDS": "0"},
    ],
)
def test_load_agent_config_rejects_invalid_component_values(
    tmp_path: Path,
    overrides: dict[str, str],
) -> None:
    with pytest.raises(ValidationError):
        load_agent_config(_agent_environment(tmp_path, **overrides))


def test_load_agent_config_rejects_missing_or_blank_workspace_path(tmp_path: Path) -> None:
    missing_workspace = _agent_environment(tmp_path)
    missing_workspace.pop("WORKSPACE_PATH")

    with pytest.raises(ValueError, match="WORKSPACE_PATH"):
        load_agent_config(missing_workspace)
    with pytest.raises(ValueError, match="WORKSPACE_PATH"):
        load_agent_config(_agent_environment(tmp_path, WORKSPACE_PATH="   "))


def test_load_agent_config_rejects_missing_required_provider_values(tmp_path: Path) -> None:
    environment = _agent_environment(tmp_path)
    environment.pop("PROVIDER_API_BASE")

    with pytest.raises(ValidationError):
        load_agent_config(environment)
