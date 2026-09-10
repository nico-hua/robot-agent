"""Unit tests for the Ollama adapter.

These tests use local fakes only.  They must not require an Ollama server,
network access, or a GPU.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import ollama
import pytest
from src.llm.base import LLMClient, ModelError, ModelResponse
from src.llm.ollama import OllamaClient, OllamaConfig
from src.messages.message import Message


class FakeResponse:
    """A small stand-in for an Ollama SDK response model."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.model_dump_calls: list[dict[str, Any]] = []

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        self.model_dump_calls.append(kwargs)
        return self.payload


class FakeOllamaClient:
    """Async fake whose result or exception is controlled by a test."""

    def __init__(
        self, result: object | None = None, error: BaseException | None = None
    ) -> None:
        self.result = result
        self.error = error
        self.chat_calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> object:
        self.chat_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


def make_client(
    fake: FakeOllamaClient,
    config: OllamaConfig | None = None,
) -> tuple[OllamaClient, dict[str, Any]]:
    """Build an adapter with a factory whose constructor arguments are visible."""

    factory_arguments: dict[str, Any] = {}

    def factory(**kwargs: Any) -> FakeOllamaClient:
        factory_arguments.update(kwargs)
        return fake

    return OllamaClient(config, client_factory=factory), factory_arguments


def user_message(content: str = "你好") -> list[Message]:
    return [Message(role="user", content=content)]


def test_default_config() -> None:
    config = OllamaConfig()

    assert config.base_url == "http://127.0.0.1:11434"
    assert config.model == "qwen3.5:4b"
    assert config.timeout == 60.0
    assert config.stream is False
    assert config.think is False


def test_config_reads_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.example:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "custom-model")
    monkeypatch.setenv("OLLAMA_TIMEOUT", "12.5")
    monkeypatch.setenv("OLLAMA_STREAM", "true")
    monkeypatch.setenv("OLLAMA_THINK", "1")

    config = OllamaConfig.from_env()

    assert config.base_url == "http://ollama.example:11434"
    assert config.model == "custom-model"
    assert config.timeout == 12.5
    assert config.stream is True
    assert config.think is True


def test_config_reads_a_supplied_environment_mapping() -> None:
    config = OllamaConfig.from_env(
        {
            "OLLAMA_BASE_URL": "http://configured.example:11434",
            "OLLAMA_MODEL": "configured-model",
            "OLLAMA_TIMEOUT": "7",
            "OLLAMA_STREAM": "false",
            "OLLAMA_THINK": "off",
        }
    )

    assert config.base_url == "http://configured.example:11434"
    assert config.model == "configured-model"
    assert config.timeout == 7.0
    assert config.stream is False
    assert config.think is False


def test_chat_converts_messages_and_returns_plain_text_response() -> None:
    payload = {
        "model": "qwen3.5:4b",
        "message": {"role": "assistant", "content": "你好，有什么可以帮你？"},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 8,
        "eval_count": 5,
        "total_duration": 100,
        "load_duration": 10,
    }
    raw_response = FakeResponse(payload)
    fake = FakeOllamaClient(result=raw_response)
    config = OllamaConfig(model="test-model", timeout=30.0)
    client, factory_arguments = make_client(fake, config)
    messages = [
        Message(
            role="system", content="你是一个简洁的助手", metadata={"ignored": True}
        ),
        Message(role="user", content="你好", metadata={"request_id": "secret"}),
        Message(role="assistant", content="此前的回答", metadata={"ignored": True}),
    ]

    response = asyncio.run(client.chat(messages))

    assert isinstance(client, LLMClient)
    assert isinstance(response, ModelResponse)
    assert response.error is None
    assert response.content == "你好，有什么可以帮你？"
    assert response.finish_reason == "stop"
    assert response.thinking is None
    assert response.raw_response == payload
    assert response.usage == {
        "prompt_tokens": 8,
        "completion_tokens": 5,
        "total_tokens": 13,
        "total_duration_ns": 100,
        "load_duration_ns": 10,
    }
    assert factory_arguments == {
        "host": "http://127.0.0.1:11434",
        "timeout": 30.0,
    }
    assert fake.chat_calls == [
        {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "你是一个简洁的助手"},
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "此前的回答"},
            ],
            "stream": False,
            "think": False,
        }
    ]
    assert raw_response.model_dump_calls == [
        {"mode": "json", "exclude_none": True},
    ]


def test_chat_preserves_thinking_when_ollama_returns_it() -> None:
    payload = {
        "message": {
            "role": "assistant",
            "content": "最终回答",
            "thinking": "简短推理",
        },
        "done_reason": "stop",
    }
    fake = FakeOllamaClient(result=FakeResponse(payload))
    client, _ = make_client(fake)

    response = asyncio.run(client.chat(user_message()))

    assert response.error is None
    assert response.content == "最终回答"
    assert response.thinking == "简短推理"


def test_chat_reports_service_unavailable() -> None:
    fake = FakeOllamaClient(error=ConnectionError("connection refused"))
    client, _ = make_client(fake)

    response = asyncio.run(client.chat(user_message()))

    assert response.error is not None
    assert isinstance(response.error, ModelError)
    assert response.error.code == "service_unavailable"
    assert response.error.retryable is True


def test_chat_reports_timeout() -> None:
    class WaitingClient:
        async def chat(self, **kwargs: Any) -> object:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    waiting_client = WaitingClient()

    def factory(**kwargs: Any) -> WaitingClient:
        return waiting_client

    client = OllamaClient(client_factory=factory)

    async def run_chat() -> ModelResponse:
        return await asyncio.wait_for(
            client.chat(user_message(), timeout=0.01), timeout=0.2
        )

    response = asyncio.run(run_chat())

    assert response.error is not None
    assert response.error.code == "timeout"
    assert response.error.retryable is True


def test_chat_reports_http_error() -> None:
    fake = FakeOllamaClient(
        error=ollama.ResponseError("model not found", status_code=404)
    )
    client, _ = make_client(fake)

    response = asyncio.run(client.chat(user_message()))

    assert response.error is not None
    assert response.error.code == "http_error"
    assert response.error.status_code == 404
    assert response.error.retryable is False


def test_chat_reports_invalid_response() -> None:
    fake = FakeOllamaClient(result=object())
    client, _ = make_client(fake)

    response = asyncio.run(client.chat(user_message()))

    assert response.error is not None
    assert response.error.code == "invalid_response"
    assert response.error.retryable is False


def test_streaming_is_rejected_without_calling_ollama() -> None:
    fake_client = type("FakeClient", (), {"chat": AsyncMock()})()

    def factory(**kwargs: Any) -> object:
        return fake_client

    client = OllamaClient(OllamaConfig(stream=True), client_factory=factory)

    response = asyncio.run(client.chat(user_message()))

    assert response.error is not None
    assert response.error.code == "stream_not_supported"
    fake_client.chat.assert_not_awaited()


def test_chat_propagates_request_cancellation() -> None:
    async def run_chat() -> None:
        started = asyncio.Event()

        class BlockingClient:
            async def chat(self, **kwargs: Any) -> object:
                started.set()
                await asyncio.Event().wait()
                raise AssertionError("unreachable")

        blocking_client = BlockingClient()

        def factory(**kwargs: Any) -> BlockingClient:
            return blocking_client

        client = OllamaClient(client_factory=factory)
        task = asyncio.create_task(client.chat(user_message()))
        await started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run_chat())
