"""Offline lifecycle tests for the minimal CLI Application assembly."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest
from src.cli.application import Application
from src.config import AgentConfig, ApiConfig, ProviderConfig
from src.providers import BaseMessage, LLMProvider, LLMResponse
from src.session import SessionManager
from src.tools import Tool
from src.tools.builtin.capture_camera import CaptureCameraTool
from src.tools.registry import ToolRegistry


class _StaticProvider(LLMProvider):
    """Offline provider used only to construct Application dependencies."""

    async def chat(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[Tool] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        return LLMResponse(content="unused")

    async def stream_chat(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[Tool] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        on_delta: object = None,
    ) -> LLMResponse:
        raise AssertionError("The minimal Application does not use stream_chat")


class _FakeApiService:
    """Record lifecycle calls without binding a TCP port."""

    def __init__(self) -> None:
        self.start_calls = 0
        self.stop_calls = 0

    async def start(self) -> None:
        self.start_calls += 1

    async def stop(self) -> None:
        self.stop_calls += 1


class _FailingAgentLoop:
    """Fail immediately to verify HTTP is not exposed without a consumer."""

    def __init__(self) -> None:
        self.close_calls = 0

    async def run(self) -> None:
        raise RuntimeError("agent loop startup failure")

    async def close(self) -> None:
        self.close_calls += 1


def _provider_config() -> ProviderConfig:
    return ProviderConfig(
        type="openai_compat",
        api_base="https://api.example.test/v1",
        model="test-model",
    )


def _agent_config(workspace: Path) -> AgentConfig:
    return AgentConfig(
        provider=_provider_config(),
        api=ApiConfig(port=8001),
        workspace_path=workspace,
    )


def test_application_starts_agent_loop_before_http_and_closes_both(tmp_path: Path) -> None:
    provider = _StaticProvider()
    api_service = _FakeApiService()
    captured: dict[str, object] = {}

    def create_api_service(
        agent_loop: object,
        session_manager: SessionManager,
        config: ApiConfig,
    ) -> _FakeApiService:
        captured.update(
            {
                "agent_loop": agent_loop,
                "session_manager": session_manager,
                "config": config,
            }
        )
        return api_service

    application = Application(
        _agent_config(tmp_path),
        provider_factory=lambda config: provider,
        api_service_factory=create_api_service,
    )

    async def run_scenario() -> None:
        await application.start()
        assert application.agent_task is not None
        assert not application.agent_task.done()
        await application.close()

    asyncio.run(run_scenario())

    assert api_service.start_calls == 1
    assert api_service.stop_calls == 1
    assert captured["agent_loop"] is application.agent_loop
    assert captured["session_manager"] is application.session_manager
    assert captured["config"] == ApiConfig(port=8001)
    assert application.config == _agent_config(tmp_path)
    assert application.agent_task is None


def test_application_does_not_start_http_when_agent_loop_fails_immediately(tmp_path: Path) -> None:
    provider = _StaticProvider()
    api_service = _FakeApiService()
    failing_loop = _FailingAgentLoop()
    application = Application(
        _agent_config(tmp_path),
        provider_factory=lambda config: provider,
        agent_loop_factory=lambda *args: failing_loop,
        api_service_factory=lambda *args: api_service,
    )

    async def run_scenario() -> None:
        with pytest.raises(RuntimeError, match="agent loop startup failure"):
            await application.start()

    asyncio.run(run_scenario())

    assert api_service.start_calls == 0
    assert api_service.stop_calls == 1
    assert failing_loop.close_calls == 1
    assert application.agent_task is None


def test_application_loads_builtin_capture_camera_tool(tmp_path: Path) -> None:
    captured: dict[str, ToolRegistry] = {}
    provider = _StaticProvider()
    agent_loop = _FailingAgentLoop()

    def create_agent_loop(
        *dependencies: object,
    ) -> _FailingAgentLoop:
        tool_registry = dependencies[2]
        assert isinstance(tool_registry, ToolRegistry)
        captured["tool_registry"] = tool_registry
        return agent_loop

    Application(
        _agent_config(tmp_path),
        provider_factory=lambda config: provider,
        agent_loop_factory=create_agent_loop,
        api_service_factory=lambda *args: _FakeApiService(),
    )

    assert isinstance(captured["tool_registry"].get("capture_camera"), CaptureCameraTool)
