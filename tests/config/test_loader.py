"""Tests for project-wide provider configuration loading."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError
from src.config import loader
from src.config.loader import load_provider_config


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
                "PROVIDER_TYPE=anthropic_compat",
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

    assert config.type == "anthropic_compat"
    assert config.default_model == "process-model"
    assert config.default_max_tokens == 1024
    assert "PROVIDER_TYPE" not in os.environ


def test_load_provider_config_treats_blank_api_key_as_optional() -> None:
    config = load_provider_config(_provider_environment(PROVIDER_API_KEY=""))

    assert config.api_key == ""


def test_load_provider_config_reports_missing_required_values() -> None:
    with pytest.raises(ValidationError):
        load_provider_config({"PROVIDER_TYPE": "openai_compat"})
