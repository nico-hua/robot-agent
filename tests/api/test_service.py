"""Offline HTTP API tests that invoke the AgentLoop directly."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from src.agent.context import ContextBuilder
from src.agent.loop import AgentLoop
from src.agent.runner import AgentRunner
from src.api.service import HttpApiService
from src.bus import InboundMessage, MessageBus, OutboundMessage
from src.config import ApiConfig
from src.providers import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    LLMProvider,
    LLMResponse,
    SystemMessage,
)
from src.session import SessionManager
from src.tools import Tool, ToolRegistry


class _StaticProvider(LLMProvider):
    """Return one offline response while recording request messages."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.chat_calls: list[tuple[BaseMessage, ...]] = []

    async def chat(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[Tool] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        self.chat_calls.append(tuple(messages))
        return LLMResponse(content=self._content, finish_reason="stop")

    async def stream_chat(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[Tool] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        raise AssertionError("The HTTP application uses the non-streaming AgentRunner path")


class _GateProvider(LLMProvider):
    """Block the first request to make same-session serialization observable."""

    def __init__(self) -> None:
        self.first_request_started = asyncio.Event()
        self.second_request_started = asyncio.Event()
        self.release_first_request = asyncio.Event()
        self.chat_calls: list[tuple[BaseMessage, ...]] = []

    async def chat(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[Tool] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        del tools, max_tokens, temperature
        request_messages = tuple(messages)
        self.chat_calls.append(request_messages)
        current_message = request_messages[-1]
        assert isinstance(current_message, HumanMessage)
        if current_message.content == "first":
            self.first_request_started.set()
            await self.release_first_request.wait()
        elif current_message.content == "second":
            self.second_request_started.set()
        else:
            raise AssertionError(f"Unexpected user message: {current_message.content}")
        return LLMResponse(
            content=f"{current_message.content} reply",
            finish_reason="stop",
        )

    async def stream_chat(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[Tool] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        raise AssertionError("The HTTP application uses the non-streaming AgentRunner path")


class _BlockingProvider(LLMProvider):
    """Wait until cancellation so the HTTP timeout can be verified."""

    def __init__(self) -> None:
        self.request_started = asyncio.Event()
        self.request_cancelled = asyncio.Event()
        self._never_release = asyncio.Event()

    async def chat(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[Tool] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        del messages, tools, max_tokens, temperature
        self.request_started.set()
        try:
            await self._never_release.wait()
        except asyncio.CancelledError:
            self.request_cancelled.set()
            raise
        raise AssertionError("The blocking provider must be cancelled by the HTTP timeout")

    async def stream_chat(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[Tool] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        raise AssertionError("The HTTP application uses the non-streaming AgentRunner path")


def _create_service(
    tmp_path: Path,
    *,
    provider: LLMProvider | None = None,
    request_timeout_seconds: float = 1.0,
) -> tuple[HttpApiService, AgentLoop, MessageBus, SessionManager]:
    message_bus = MessageBus()
    session_manager = SessionManager(tmp_path)
    agent_loop = AgentLoop(
        runner=AgentRunner(),
        provider=provider or _StaticProvider("assistant reply"),
        tool_registry=ToolRegistry(),
        session_manager=session_manager,
        context_builder=ContextBuilder(),
        message_bus=message_bus,
    )
    service = HttpApiService(
        agent_loop,
        session_manager,
        ApiConfig(request_timeout_seconds=request_timeout_seconds),
    )
    return service, agent_loop, message_bus, session_manager


async def _start_client(service: HttpApiService) -> TestClient:
    client = TestClient(TestServer(service.app))
    await client.start_server()
    return client


def test_post_message_calls_agent_loop_directly_without_using_message_bus(
    tmp_path: Path,
) -> None:
    provider = _StaticProvider("assistant reply")
    service, _agent_loop, message_bus, session_manager = _create_service(
        tmp_path,
        provider=provider,
    )

    async def run_scenario() -> dict[str, object]:
        client = await _start_client(service)
        try:
            response = await client.post(
                "/v1/messages",
                json={"session_id": "session-1", "content": "hello"},
            )
            assert response.status == 200
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(message_bus.consume_inbound(), timeout=0.05)
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(message_bus.consume_outbound(), timeout=0.05)
            return await response.json()
        finally:
            await client.close()

    payload = asyncio.run(run_scenario())

    assert payload == {"session_id": "session-1", "content": "assistant reply"}
    assert provider.chat_calls[-1][-1] == HumanMessage(content="hello")
    saved_session = session_manager.get("session-1")
    assert saved_session is not None
    assert saved_session.messages == (
        HumanMessage(content="hello"),
        AIMessage(content="assistant reply"),
    )


def test_post_message_rejects_invalid_request_bodies(tmp_path: Path) -> None:
    service, _agent_loop, _message_bus, _session_manager = _create_service(tmp_path)

    async def run_scenario() -> list[tuple[int, str]]:
        client = await _start_client(service)
        try:
            invalid_json = await client.post(
                "/v1/messages",
                data="{",
                headers={"Content-Type": "application/json"},
            )
            blank_session = await client.post(
                "/v1/messages",
                json={"session_id": " ", "content": "hello"},
            )
            invalid_content = await client.post(
                "/v1/messages",
                json={"session_id": "session-1", "content": 1},
            )
            return [
                (invalid_json.status, (await invalid_json.json())["error"]["code"]),
                (blank_session.status, (await blank_session.json())["error"]["code"]),
                (invalid_content.status, (await invalid_content.json())["error"]["code"]),
            ]
        finally:
            await client.close()

    outcomes = asyncio.run(run_scenario())

    assert outcomes == [
        (400, "invalid_json"),
        (400, "invalid_session_id"),
        (400, "invalid_content"),
    ]


def test_post_message_serializes_concurrent_direct_calls_for_the_same_session(
    tmp_path: Path,
) -> None:
    provider = _GateProvider()
    service, _agent_loop, _message_bus, session_manager = _create_service(
        tmp_path,
        provider=provider,
    )

    async def post(client: TestClient, content: str) -> dict[str, object]:
        response = await client.post(
            "/v1/messages",
            json={"session_id": "shared-session", "content": content},
        )
        assert response.status == 200
        return await response.json()

    async def run_scenario() -> tuple[dict[str, object], dict[str, object]]:
        client = await _start_client(service)
        try:
            first_request = asyncio.create_task(post(client, "first"))
            await asyncio.wait_for(provider.first_request_started.wait(), timeout=1)
            second_request = asyncio.create_task(post(client, "second"))
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(provider.second_request_started.wait(), timeout=0.05)

            provider.release_first_request.set()
            return await asyncio.gather(first_request, second_request)
        finally:
            await client.close()

    first_payload, second_payload = asyncio.run(run_scenario())

    assert first_payload == {"session_id": "shared-session", "content": "first reply"}
    assert second_payload == {"session_id": "shared-session", "content": "second reply"}
    assert provider.chat_calls[0][-1] == HumanMessage(content="first")
    assert provider.chat_calls[1][0].role == "system"
    assert provider.chat_calls[1][1:] == (
        HumanMessage(content="first"),
        AIMessage(content="first reply"),
        HumanMessage(content="second"),
    )
    saved_session = session_manager.get("shared-session")
    assert saved_session is not None
    assert saved_session.messages == (
        HumanMessage(content="first"),
        AIMessage(content="first reply"),
        HumanMessage(content="second"),
        AIMessage(content="second reply"),
    )


def test_post_message_times_out_and_cancels_the_direct_agent_turn(tmp_path: Path) -> None:
    provider = _BlockingProvider()
    service, _agent_loop, _message_bus, session_manager = _create_service(
        tmp_path,
        provider=provider,
        request_timeout_seconds=0.05,
    )

    async def run_scenario() -> tuple[int, dict[str, object], bool, bool]:
        client = await _start_client(service)
        try:
            response = await client.post(
                "/v1/messages",
                json={"session_id": "session-1", "content": "slow request"},
            )
            await asyncio.wait_for(provider.request_started.wait(), timeout=1)
            await asyncio.wait_for(provider.request_cancelled.wait(), timeout=1)
            return (
                response.status,
                await response.json(),
                provider.request_started.is_set(),
                provider.request_cancelled.is_set(),
            )
        finally:
            await client.close()

    status, payload, request_started, request_cancelled = asyncio.run(run_scenario())

    assert status == 504
    assert payload["error"]["code"] == "agent_timeout"
    assert request_started is True
    assert request_cancelled is True
    assert session_manager.get("session-1") is None


def test_post_message_returns_agent_error_when_the_direct_loop_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service, agent_loop, _message_bus, _session_manager = _create_service(tmp_path)

    async def fail(inbound: InboundMessage) -> OutboundMessage:
        del inbound
        raise RuntimeError("internal direct-loop detail")

    monkeypatch.setattr(agent_loop, "_process_inbound", fail)

    async def run_scenario() -> tuple[int, dict[str, object]]:
        client = await _start_client(service)
        try:
            response = await client.post(
                "/v1/messages",
                json={"session_id": "session-1", "content": "hello"},
            )
            return response.status, await response.json()
        finally:
            await client.close()

    status, payload = asyncio.run(run_scenario())

    assert status == 502
    assert payload["error"]["code"] == "agent_error"
    assert "internal direct-loop detail" not in payload["error"]["message"]


def test_post_message_rejects_a_mismatched_direct_loop_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service, agent_loop, _message_bus, _session_manager = _create_service(tmp_path)

    async def mismatched(inbound: InboundMessage) -> OutboundMessage:
        return OutboundMessage(
            session_id="another-session",
            content="unexpected reply",
            metadata=inbound.metadata,
        )

    monkeypatch.setattr(agent_loop, "_process_inbound", mismatched)

    async def run_scenario() -> tuple[int, dict[str, object]]:
        client = await _start_client(service)
        try:
            response = await client.post(
                "/v1/messages",
                json={"session_id": "session-1", "content": "hello"},
            )
            return response.status, await response.json()
        finally:
            await client.close()

    status, payload = asyncio.run(run_scenario())

    assert status == 502
    assert payload["error"]["code"] == "invalid_agent_response"


def test_clear_session_resets_persisted_history(tmp_path: Path) -> None:
    service, _agent_loop, _message_bus, session_manager = _create_service(tmp_path)
    session_manager.save(
        session_manager.get_or_create("session/with/slashes").with_messages(
            (
                HumanMessage(content="earlier user message"),
                AIMessage(content="earlier assistant reply"),
            )
        )
    )

    async def run_scenario() -> dict[str, object]:
        client = await _start_client(service)
        try:
            response = await client.post(
                "/v1/sessions/clear",
                json={"session_id": "session/with/slashes"},
            )
            assert response.status == 200
            return await response.json()
        finally:
            await client.close()

    payload = asyncio.run(run_scenario())

    assert payload["session_id"] == "session/with/slashes"
    assert payload["cleared"] is True
    saved_session = session_manager.get("session/with/slashes")
    assert saved_session is not None
    assert saved_session.messages == ()


def test_session_endpoints_return_only_persisted_user_and_assistant_messages(
    tmp_path: Path,
) -> None:
    service, _agent_loop, _message_bus, session_manager = _create_service(tmp_path)
    session_manager.save(
        session_manager.get_or_create("session-1").with_messages(
            (
                SystemMessage(content="transient system prompt"),
                HumanMessage(content="earlier user message"),
                AIMessage(content="earlier assistant reply"),
            )
        )
    )

    async def run_scenario() -> tuple[dict[str, object], dict[str, object], int]:
        client = await _start_client(service)
        try:
            list_response = await client.get("/v1/sessions")
            get_response = await client.get("/v1/sessions/session-1")
            missing_response = await client.get("/v1/sessions/missing")
            assert list_response.status == 200
            assert get_response.status == 200
            return (
                await list_response.json(),
                await get_response.json(),
                missing_response.status,
            )
        finally:
            await client.close()

    listed, detail, missing_status = asyncio.run(run_scenario())

    assert listed["sessions"][0]["session_id"] == "session-1"
    assert listed["sessions"][0]["message_count"] == 2
    assert listed["sessions"][0]["preview"] == "earlier assistant reply"
    assert detail["session_id"] == "session-1"
    assert detail["messages"] == [
        {"role": "user", "content": "earlier user message"},
        {"role": "assistant", "content": "earlier assistant reply"},
    ]
    assert missing_status == 404
