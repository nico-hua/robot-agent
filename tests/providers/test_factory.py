"""Tests for offline construction of configured LLM providers."""

from __future__ import annotations

from typing import Any

import pytest
import src.providers.openai_compat_provider as openai_compat_provider
from src.config.schema import ProviderConfig
from src.providers.base import LLMProvider
from src.providers.factory import create_default_provider_factory


class _FakeAsyncClient:
    """Capture adapter client options without connecting to a provider."""

    def __init__(self, **options: Any) -> None:
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
    }
