"""Factories for creating configured LLM providers."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from ..config.schema import ProviderConfig
from .base import LLMProvider
from .ollama_compact_provider import OllamaCompactProvider
from .openai_compat_provider import OpenAICompatProvider

ProviderConstructor = Callable[[ProviderConfig], LLMProvider]


class ProviderFactory:
    """Create LLM providers from registered provider type constructors."""

    def __init__(self, constructors: Mapping[str, ProviderConstructor] | None = None) -> None:
        self._constructors = dict(constructors or {})

    def register(self, provider_type: str, constructor: ProviderConstructor) -> None:
        """Register one constructor without silently replacing an existing type."""

        if not isinstance(provider_type, str) or not provider_type.strip():
            raise ValueError("Provider type must be a non-empty string")
        if provider_type in self._constructors:
            raise ValueError(f"Provider type is already registered: {provider_type}")
        self._constructors[provider_type] = constructor

    def create(self, config: ProviderConfig) -> LLMProvider:
        """Create the provider selected by the resolved configuration."""

        constructor = self._constructors.get(config.type)
        if constructor is None:
            raise ValueError(f"Unsupported configured provider: {config.type}")
        return constructor(config)


def create_default_provider_factory() -> ProviderFactory:
    """Return a factory with the provider types supported by this project."""

    return ProviderFactory(
        {
            "ollama": _create_ollama_compact_provider,
            "openai_compat": _create_openai_compat_provider,
        }
    )


def _create_openai_compat_provider(config: ProviderConfig) -> LLMProvider:
    return OpenAICompatProvider(**_provider_options(config))


def _create_ollama_compact_provider(config: ProviderConfig) -> LLMProvider:
    return OllamaCompactProvider(**_ollama_provider_options(config))


def _provider_options(config: ProviderConfig) -> dict[str, str | int | float | bool]:
    return {
        "api_key": config.api_key,
        "api_base": config.api_base,
        "default_model": config.default_model,
        "default_max_tokens": config.default_max_tokens,
        "default_temperature": config.default_temperature,
        "default_think": config.think,
        "timeout": config.request_timeout_seconds,
    }


def _ollama_provider_options(config: ProviderConfig) -> dict[str, str | int | float | bool]:
    return {
        "api_base": config.api_base,
        "default_model": config.default_model,
        "default_max_tokens": config.default_max_tokens,
        "default_temperature": config.default_temperature,
        "default_think": config.think,
        "timeout": config.request_timeout_seconds,
    }
