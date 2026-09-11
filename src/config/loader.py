"""Load complete project configuration from the root ``.env`` file."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from dotenv import dotenv_values

from .schema import AgentConfig, ApiConfig, ProviderConfig

_PROJECT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
_ENVIRONMENT_FIELDS = {
    "PROVIDER_TYPE": "type",
    "PROVIDER_API_KEY": "api_key",
    "PROVIDER_API_BASE": "api_base",
    "PROVIDER_MODEL": "model",
    "PROVIDER_MAX_TOKENS": "max_tokens",
    "PROVIDER_TEMPERATURE": "temperature",
    "PROVIDER_REQUEST_TIMEOUT_SECONDS": "request_timeout_seconds",
}
_API_ENVIRONMENT_FIELDS = {
    "API_HOST": "host",
    "API_PORT": "port",
    "API_REQUEST_TIMEOUT_SECONDS": "request_timeout_seconds",
}


def load_agent_config(environ: Mapping[str, str] | None = None) -> AgentConfig:
    """Return the complete application configuration from one environment source.

    The resulting object keeps provider, local HTTP API, and workspace settings
    together for application-level assembly. Existing process environment
    values take precedence over the root ``.env`` file. Passing ``environ``
    avoids file loading and does not modify ``os.environ``. Empty Provider/API
    values are treated as unset; ``WORKSPACE_PATH`` remains required.
    """

    environment = _load_environment(environ)
    return AgentConfig(
        provider=_load_provider_config(environment),
        api=_load_api_config(environment),
        workspace_path=_load_workspace_path(environment),
    )


def _load_provider_config(environ: Mapping[str, str]) -> ProviderConfig:
    values = {
        field_name: value
        for environment_name, field_name in _ENVIRONMENT_FIELDS.items()
        if (value := environ.get(environment_name)) is not None and value.strip()
    }
    return ProviderConfig.model_validate(values)


def _load_workspace_path(environ: Mapping[str, str]) -> Path:
    workspace_value = environ.get("WORKSPACE_PATH")
    if workspace_value is None or not workspace_value.strip():
        raise ValueError("WORKSPACE_PATH must be set to a non-empty path")
    return Path(workspace_value.strip()).expanduser().resolve()


def _load_api_config(environ: Mapping[str, str]) -> ApiConfig:
    values = {
        field_name: value
        for environment_name, field_name in _API_ENVIRONMENT_FIELDS.items()
        if (value := environ.get(environment_name)) is not None and value.strip()
    }
    return ApiConfig.model_validate(values)


def _load_environment(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    if environ is not None:
        return environ

    dotenv_environment = {
        name: value for name, value in dotenv_values(_PROJECT_ENV_FILE).items() if value is not None
    }
    return {**dotenv_environment, **os.environ}


__all__ = ["load_agent_config"]
