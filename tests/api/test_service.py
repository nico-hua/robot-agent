"""Offline HTTP API tests using an in-memory MessageBus."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from pathlib import Path

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
    """Return one offline response while recording the request messages."""

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


def _create_service(
    tmp_path: Path,
    *,
    request_timeout_seconds: float = 1.0,
) -> tuple[HttpApiService, MessageBus, SessionManager]:
    message_bus = MessageBus()
    session_manager = SessionManager(tmp_path)
    service = HttpApiService(
        message_bus,
        session_manager,
        ApiConfig(request_timeout_seconds=request_timeout_seconds),
    )
    return service, message_bus, session_manager


async def _start_client(service: HttpApiService) -> TestClient:
    client = TestClient(TestServer(service.app))
    await client.start_server()
    return client


async def _reply_once(message_bus: MessageBus, *, content: str) -> InboundMessage:
    inbound = await message_bus.consume_inbound()
    await message_bus.publish_outbound(
        OutboundMessage(
            session_id=inbound.session_id,
            content=content,
            metadata=inbound.metadata,
        )
    )
    return inbound


def test_post_message_publishes_current_inbound_message_and_returns_bus_response(
    tmp_path: Path,
) -> None:
    service, message_bus, _session_manager = _create_service(tmp_path)

    async def run_scenario() -> tuple[dict[str, object], InboundMessage]:
        client = await _start_client(service)
        try:
            reply_task = asyncio.create_task(_reply_once(message_bus, content="assistant reply"))
            response = await client.post(
                "/v1/messages",
                json={"session_id": "session-1", "content": "hello"},
                headers={"Authorization": "Bearer ignored"},
            )
            payload = await response.json()
            return payload, await reply_task
        finally:
            await client.close()

    payload, inbound = asyncio.run(run_scenario())

    assert payload == {"session_id": "session-1", "content": "assistant reply"}
    assert inbound.session_id == "session-1"
    assert inbound.content == "hello"
    assert set(inbound.metadata) == {"_http_request_id"}


def test_post_message_rejects_invalid_request_bodies(tmp_path: Path) -> None:
    service, _message_bus, _session_manager = _create_service(tmp_path)

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


def test_post_message_correlates_concurrent_outbound_responses(tmp_path: Path) -> None:
    service, message_bus, _session_manager = _create_service(tmp_path)

    async def post(client: TestClient, content: str):
        return await client.post(
            "/v1/messages",
            json={"session_id": "shared-session", "content": content},
        )

    async def respond_out_of_order() -> None:
        first = await message_bus.consume_inbound()
        second = await message_bus.consume_inbound()
        await message_bus.publish_outbound(
            OutboundMessage(
                session_id=second.session_id,
                content="reply to second",
                metadata=second.metadata,
            )
        )
        await message_bus.publish_outbound(
            OutboundMessage(
                session_id=first.session_id,
                content="reply to first",
                metadata=first.metadata,
            )
        )

    async def run_scenario() -> tuple[dict[str, object], dict[str, object]]:
        client = await _start_client(service)
        try:
            first_task = asyncio.create_task(post(client, "first"))
            second_task = asyncio.create_task(post(client, "second"))
            responder_task = asyncio.create_task(respond_out_of_order())
            await responder_task
            first_response, second_response = await asyncio.gather(first_task, second_task)
            return await first_response.json(), await second_response.json()
        finally:
            await client.close()

    first_payload, second_payload = asyncio.run(run_scenario())

    assert first_payload == {"session_id": "shared-session", "content": "reply to first"}
    assert second_payload == {"session_id": "shared-session", "content": "reply to second"}


def test_post_message_discards_a_late_timeout_response_before_the_next_request(
    tmp_path: Path,
) -> None:
    service, message_bus, _session_manager = _create_service(
        tmp_path,
        request_timeout_seconds=0.05,
    )

    async def run_scenario() -> tuple[dict[str, object], dict[str, object]]:
        client = await _start_client(service)
        try:
            timed_out = await client.post(
                "/v1/messages",
                json={"session_id": "session-1", "content": "first"},
            )
            first_payload = await timed_out.json()
            first_inbound = await message_bus.consume_inbound()
            await message_bus.publish_outbound(
                OutboundMessage(
                    session_id=first_inbound.session_id,
                    content="late reply",
                    metadata=first_inbound.metadata,
                )
            )
            await asyncio.sleep(0)

            reply_task = asyncio.create_task(_reply_once(message_bus, content="second reply"))
            second_response = await client.post(
                "/v1/messages",
                json={"session_id": "session-1", "content": "second"},
            )
            second_payload = await second_response.json()
            await reply_task
            return first_payload, second_payload
        finally:
            await client.close()

    first_payload, second_payload = asyncio.run(run_scenario())

    assert first_payload["error"]["code"] == "agent_timeout"
    assert second_payload == {"session_id": "session-1", "content": "second reply"}


def test_post_message_reaches_the_real_agent_loop_and_clear_is_persisted(tmp_path: Path) -> None:
    provider = _StaticProvider("agent reply")
    message_bus = MessageBus()
    session_manager = SessionManager(tmp_path)
    agent_loop = AgentLoop(
        runner=AgentRunner(),
        provider=provider,
        tool_registry=ToolRegistry(),
        session_manager=session_manager,
        context_builder=ContextBuilder(),
        message_bus=message_bus,
    )
    service = HttpApiService(
        message_bus,
        session_manager,
        ApiConfig(request_timeout_seconds=1),
    )

    async def run_scenario() -> tuple[dict[str, object], dict[str, object]]:
        loop_task = asyncio.create_task(agent_loop.run())
        client = await _start_client(service)
        try:
            message_response = await client.post(
                "/v1/messages",
                json={"session_id": "session-1", "content": "hello"},
            )
            clear_response = await client.post(
                "/v1/sessions/clear",
                json={"session_id": "session-1"},
            )
            assert message_response.status == 200
            assert clear_response.status == 200
            return await message_response.json(), await clear_response.json()
        finally:
            await client.close()
            loop_task.cancel()
            with suppress(asyncio.CancelledError):
                await loop_task

    message_payload, clear_payload = asyncio.run(run_scenario())

    assert message_payload == {"session_id": "session-1", "content": "agent reply"}
    assert clear_payload["session_id"] == "session-1"
    assert clear_payload["cleared"] is True
    assert provider.chat_calls[-1][-1] == HumanMessage(content="hello")
    saved_session = session_manager.get("session-1")
    assert saved_session is not None
    assert saved_session.messages == ()


def test_post_message_returns_service_unavailable_when_router_stops(tmp_path: Path) -> None:
    service, message_bus, _session_manager = _create_service(tmp_path)

    async def run_scenario() -> dict[str, object]:
        client = await _start_client(service)
        try:
            request_task = asyncio.create_task(
                client.post(
                    "/v1/messages",
                    json={"session_id": "session-1", "content": "hello"},
                )
            )
            await asyncio.wait_for(message_bus.consume_inbound(), timeout=1)
            await service.stop()
            response = await asyncio.wait_for(request_task, timeout=1)
            assert response.status == 503
            return await response.json()
        finally:
            await client.close()

    payload = asyncio.run(run_scenario())

    assert payload["error"]["code"] == "agent_unavailable"


def test_clear_session_resets_persisted_history(tmp_path: Path) -> None:
    service, _message_bus, session_manager = _create_service(tmp_path)
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
    service, _message_bus, session_manager = _create_service(tmp_path)
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
