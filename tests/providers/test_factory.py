"""Tests for offline construction of configured LLM providers."""

from __future__ import annotations

from typing import Any

import pytest
import src.providers.ollama_compact_provider as ollama_compact_provider
import src.providers.openai_compat_provider as openai_compat_provider
from src.config.schema import ProviderConfig
from src.providers.base import LLMProvider
from src.providers.factory import create_default_provider_factory


class _FakeAsyncClient:
    """Capture adapter client options without connecting to a provider."""

    def __init__(self, **options: Any) -> None:
        self.options = options


class _FakeOllamaAsyncClient:
    """Capture native Ollama adapter options without connecting to a provider."""

    def __init__(self, host: str | None = None, **options: Any) -> None:
        self.host = host
        self.options = options


@pytest.mark.parametrize(
    ("provider_type", "api_base", "api_key", "expected_type", "expected_sdk_api_key"),
    [
        pytest.param(
            "openai_compat",
            "https://api.example.test/v1",
            "",
            openai_compat_provider.OpenAICompatProvider,
            "not-required",
            id="openai-compatible-without-key",
        ),
        pytest.param(
            "openai_compat",
            "https://api.example.test/v1",
            "test-key",
            openai_compat_provider.OpenAICompatProvider,
            "test-key",
            id="openai-compatible-with-key",
        ),
    ],
)
def test_default_factory_constructs_concrete_provider(
    monkeypatch: pytest.MonkeyPatch,
    provider_type: str,
    api_base: str,
    api_key: str,
    expected_type: type[LLMProvider],
    expected_sdk_api_key: str,
) -> None:
    monkeypatch.setattr(openai_compat_provider, "AsyncOpenAI", _FakeAsyncClient)
    config = ProviderConfig(
        type=provider_type,
        api_key=api_key,
        api_base=api_base,
        model="test-model",
    )

    provider = create_default_provider_factory().create(config)

    assert isinstance(provider, expected_type)
    assert isinstance(provider, LLMProvider)
    assert provider._client.options == {
        "api_key": expected_sdk_api_key,
        "base_url": api_base,
        "timeout": 60.0,
    }


def test_default_factory_constructs_native_ollama_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ollama_compact_provider, "AsyncClient", _FakeOllamaAsyncClient)
    config = ProviderConfig(
        type="ollama",
        api_base="http://127.0.0.1:11434",
        model="qwen3.5:4b",
        request_timeout_seconds=12.5,
    )

    provider = create_default_provider_factory().create(config)

    assert isinstance(provider, ollama_compact_provider.OllamaCompactProvider)
    assert isinstance(provider, LLMProvider)
    assert provider._client.host == "http://127.0.0.1:11434"
    assert provider._client.options == {"timeout": 12.5}
