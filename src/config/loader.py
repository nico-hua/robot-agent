"""Load project provider configuration from the root ``.env`` file."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from dotenv import dotenv_values

from .schema import ProviderConfig

_PROJECT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
_ENVIRONMENT_FIELDS = {
    "PROVIDER_TYPE": "type",
    "PROVIDER_API_KEY": "api_key",
    "PROVIDER_API_BASE": "api_base",
    "PROVIDER_MODEL": "model",
    "PROVIDER_MAX_TOKENS": "max_tokens",
    "PROVIDER_TEMPERATURE": "temperature",
}


def load_provider_config(
    environ: Mapping[str, str] | None = None,
) -> ProviderConfig:
    """Return validated provider settings from ``.env`` and process variables.

    Existing process environment variables take precedence over values in the
    project root's ``.env`` file. Passing ``environ`` is useful for callers
    that already control their configuration source and avoids file loading.
    The function does not modify ``os.environ``. Empty values are treated as
    unset so optional API keys can remain blank for compatible local endpoints.
    """

    if environ is None:
        dotenv_environment = {
            name: value
            for name, value in dotenv_values(_PROJECT_ENV_FILE).items()
            if value is not None
        }
        environ = {**dotenv_environment, **os.environ}

    values = {
        field_name: value
        for environment_name, field_name in _ENVIRONMENT_FIELDS.items()
        if (value := environ.get(environment_name)) is not None and value.strip()
    }
    return ProviderConfig.model_validate(values)


__all__ = ["load_provider_config"]
