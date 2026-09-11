"""Offline tests for message-bus AgentLoop orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

import pytest
from src.agent.context import SYSTEM_PROMPT, ContextBuilder
from src.agent.loop import (
    _AGENT_FAILURE_MESSAGE,
    _EMPTY_RESPONSE_MESSAGE,
    _INVALID_SESSION_ID_MESSAGE,
    _MAX_ITERATIONS_MESSAGE,
    _PROCESSING_FAILURE_MESSAGE,
    _PROVIDER_FAILURE_MESSAGE,
    _SESSION_SAVE_FAILURE_MESSAGE,
    AgentLoop,
)
from src.agent.runner import AgentRunner, AgentRunnerError, AgentRunResult, AgentRunSpec
from src.bus import InboundMessage, MessageBus, OutboundMessage
from src.providers import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    LLMProvider,
    LLMResponse,
    ProviderError,
    SystemMessage,
    ToolCallRequest,
    ToolMessage,
)
from src.session import Session, SessionManager
from src.tools import Tool, ToolRegistry
from src.tools.base import ToolResult


class _QueueProvider(LLMProvider):
    """Return queued offline outcomes while recording normal chat requests."""

    def __init__(self, outcomes: Sequence[LLMResponse | Exception]) -> None:
        self._outcomes = list(outcomes)
        self.chat_calls: list[tuple[BaseMessage, ...]] = []

    async def chat(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[Tool] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        self.chat_calls.append(tuple(messages))
        if not self._outcomes:
            raise AssertionError("No provider outcome was configured")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def stream_chat(self, *args: object, **kwargs: object) -> LLMResponse:
        raise AssertionError("AgentLoop must use the non-streaming AgentRunner path")


class _GateProvider(LLMProvider):
    """Block one selected user message while other requests can complete."""

    def __init__(self, gated_content: str) -> None:
        self.gated_content = gated_content
        self.gated_request_started = asyncio.Event()
        self.non_gated_request_started = asyncio.Event()
        self.release_gated_request = asyncio.Event()
        self.chat_calls: list[tuple[BaseMessage, ...]] = []

    async def chat(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[Tool] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        self.chat_calls.append(tuple(messages))
        current_message = messages[-1]
        assert isinstance(current_message, HumanMessage)
        if current_message.content == self.gated_content:
            self.gated_request_started.set()
            await self.release_gated_request.wait()
        else:
            self.non_gated_request_started.set()
        return LLMResponse(
            content=f"{current_message.content} reply",
            finish_reason="stop",
        )

    async def stream_chat(self, *args: object, **kwargs: object) -> LLMResponse:
        raise AssertionError("AgentLoop must use the non-streaming AgentRunner path")


class _StaticTool(Tool):
    """Return a fixed ToolResult without external dependencies."""

    def __init__(self) -> None:
        super().__init__(name="lookup", description="Return a fixed lookup result.")

    async def execute(self, **arguments: object) -> ToolResult:
        return ToolResult(content="tool result")


class _FailingRunner(AgentRunner):
    """Raise a controlled runner error after receiving one valid specification."""

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        raise AgentRunnerError("internal runner detail must not be exposed")


class _UnexpectedFailingRunner(AgentRunner):
    """Raise an unexpected exception without producing a partial result."""

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        raise RuntimeError("internal unexpected detail must not be exposed")


def _create_loop(
    tmp_path: Path,
    provider: LLMProvider,
    *,
    runner: AgentRunner | None = None,
    tool_registry: ToolRegistry | None = None,
    max_iterations: int = 30,
) -> tuple[AgentLoop, MessageBus, SessionManager]:
    bus = MessageBus()
    manager = SessionManager(tmp_path)
    loop = AgentLoop(
        runner=runner or AgentRunner(),
        provider=provider,
        tool_registry=tool_registry or ToolRegistry(),
        session_manager=manager,
        context_builder=ContextBuilder(),
        message_bus=bus,
        max_iterations=max_iterations,
    )
    return loop, bus, manager


async def _run_one(
    loop: AgentLoop,
    bus: MessageBus,
    inbound: InboundMessage,
) -> OutboundMessage:
    loop_task = asyncio.create_task(loop.run())
    outbound_task: asyncio.Task[OutboundMessage] | None = None
    try:
        await bus.publish_inbound(inbound)
        outbound_task = asyncio.create_task(bus.consume_outbound())
        done, _ = await asyncio.wait(
            {loop_task, outbound_task},
            timeout=1,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if loop_task in done:
            await loop_task
        assert outbound_task in done
        return outbound_task.result()
    finally:
        if outbound_task is not None and not outbound_task.done():
            outbound_task.cancel()
            with suppress(asyncio.CancelledError):
                await outbound_task
        loop_task.cancel()
        with suppress(asyncio.CancelledError):
            await loop_task


def test_loop_builds_request_from_history_persists_result_and_publishes_reply(
    tmp_path: Path,
) -> None:
    provider = _QueueProvider([LLMResponse(content="final reply", finish_reason="stop")])
    loop, bus, manager = _create_loop(tmp_path, provider)
    existing_session = manager.get_or_create("session-1").with_messages(
        (
            SystemMessage(content="legacy system prompt"),
            HumanMessage(content="earlier user message"),
            AIMessage(content="earlier assistant reply"),
        )
    )
    manager.save(existing_session)
    inbound = InboundMessage(
        session_id="session-1",
        content="current user message",
        metadata={"trace_id": "trace-1"},
    )

    outbound = asyncio.run(_run_one(loop, bus, inbound))

    assert outbound == OutboundMessage(
        session_id="session-1",
        content="final reply",
        metadata={"trace_id": "trace-1"},
    )
    assert provider.chat_calls == [
        (
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content="earlier user message"),
            AIMessage(content="earlier assistant reply"),
            HumanMessage(content="current user message"),
        )
    ]
    saved_session = manager.get("session-1")
    assert saved_session is not None
    assert saved_session.messages == (
        HumanMessage(content="earlier user message"),
        AIMessage(content="earlier assistant reply"),
        HumanMessage(content="current user message"),
        AIMessage(content="final reply"),
    )


def test_loop_does_not_persist_provider_failure_and_continues_consuming_messages(
    tmp_path: Path,
) -> None:
    provider = _QueueProvider(
        [
            ProviderError("sensitive provider detail"),
            LLMResponse(content="recovered reply", finish_reason="stop"),
        ]
    )
    loop, bus, manager = _create_loop(tmp_path, provider)

    async def run_scenario() -> tuple[OutboundMessage, OutboundMessage]:
        loop_task = asyncio.create_task(loop.run())
        try:
            await bus.publish_inbound(
                InboundMessage(
                    session_id="session-1",
                    content="first request",
                    metadata={"trace_id": "first"},
                )
            )
            first = await asyncio.wait_for(bus.consume_outbound(), timeout=1)
            await bus.publish_inbound(
                InboundMessage(
                    session_id="session-1",
                    content="second request",
                    metadata={"trace_id": "second"},
                )
            )
            second = await asyncio.wait_for(bus.consume_outbound(), timeout=1)
            return first, second
        finally:
            loop_task.cancel()
            with suppress(asyncio.CancelledError):
                await loop_task

    first, second = asyncio.run(run_scenario())

    assert first == OutboundMessage(
        session_id="session-1",
        content=_PROVIDER_FAILURE_MESSAGE,
        metadata={"trace_id": "first"},
    )
    assert "sensitive provider detail" not in first.content
    assert second == OutboundMessage(
        session_id="session-1",
        content="recovered reply",
        metadata={"trace_id": "second"},
    )
    saved_session = manager.get("session-1")
    assert saved_session is not None
    assert saved_session.messages == (
        HumanMessage(content="second request"),
        AIMessage(content="recovered reply"),
    )


def test_loop_processes_a_fast_session_while_a_queued_session_is_still_running(
    tmp_path: Path,
) -> None:
    provider = _GateProvider(gated_content="slow request")
    loop, bus, manager = _create_loop(tmp_path, provider)

    async def run_scenario() -> tuple[OutboundMessage, OutboundMessage]:
        loop_task = asyncio.create_task(loop.run())
        try:
            await bus.publish_inbound(
                InboundMessage(session_id="slow-session", content="slow request")
            )
            await asyncio.wait_for(provider.gated_request_started.wait(), timeout=1)
            await bus.publish_inbound(
                InboundMessage(session_id="fast-session", content="fast request")
            )
            fast_outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=1)
            assert fast_outbound == OutboundMessage(
                session_id="fast-session",
                content="fast request reply",
            )
            assert manager.get("slow-session") is None

            provider.release_gated_request.set()
            slow_outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=1)
            return fast_outbound, slow_outbound
        finally:
            loop_task.cancel()
            with suppress(asyncio.CancelledError):
                await loop_task

    fast_outbound, slow_outbound = asyncio.run(run_scenario())

    assert fast_outbound.session_id == "fast-session"
    assert slow_outbound == OutboundMessage(
        session_id="slow-session",
        content="slow request reply",
    )
    assert manager.get("fast-session") is not None
    assert manager.get("slow-session") is not None


def test_loop_serializes_queued_messages_for_the_same_session(tmp_path: Path) -> None:
    provider = _GateProvider(gated_content="first request")
    loop, bus, manager = _create_loop(tmp_path, provider)

    async def run_scenario() -> tuple[OutboundMessage, OutboundMessage]:
        loop_task = asyncio.create_task(loop.run())
        try:
            await bus.publish_inbound(
                InboundMessage(session_id="session-1", content="first request")
            )
            await asyncio.wait_for(provider.gated_request_started.wait(), timeout=1)
            await bus.publish_inbound(
                InboundMessage(session_id="session-1", content="second request")
            )

            async def wait_for_second_worker() -> None:
                while len(loop._inbound_tasks) < 2:
                    await asyncio.sleep(0)

            await asyncio.wait_for(wait_for_second_worker(), timeout=1)
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(
                    provider.non_gated_request_started.wait(),
                    timeout=0.05,
                )

            provider.release_gated_request.set()
            first_outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=1)
            second_outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=1)
            return first_outbound, second_outbound
        finally:
            loop_task.cancel()
            with suppress(asyncio.CancelledError):
                await loop_task

    first_outbound, second_outbound = asyncio.run(run_scenario())

    assert first_outbound.content == "first request reply"
    assert second_outbound.content == "second request reply"
    assert provider.chat_calls == [
        (
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content="first request"),
        ),
        (
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content="first request"),
            AIMessage(content="first request reply"),
            HumanMessage(content="second request"),
        ),
    ]
    saved_session = manager.get("session-1")
    assert saved_session is not None
    assert saved_session.messages == (
        HumanMessage(content="first request"),
        AIMessage(content="first request reply"),
        HumanMessage(content="second request"),
        AIMessage(content="second request reply"),
    )


def test_loop_cancellation_cancels_queued_processing_without_persisting_a_turn(
    tmp_path: Path,
) -> None:
    provider = _GateProvider(gated_content="slow request")
    loop, bus, manager = _create_loop(tmp_path, provider)

    async def run_scenario() -> None:
        loop_task = asyncio.create_task(loop.run())
        await bus.publish_inbound(InboundMessage(session_id="session-1", content="slow request"))
        await asyncio.wait_for(provider.gated_request_started.wait(), timeout=1)
        loop_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await loop_task

    asyncio.run(run_scenario())

    assert manager.get("session-1") is None
    assert not loop._inbound_tasks


def test_loop_close_stops_the_consumer_before_later_messages_are_processed(
    tmp_path: Path,
) -> None:
    provider = _QueueProvider([LLMResponse(content="unexpected reply", finish_reason="stop")])
    loop, bus, manager = _create_loop(tmp_path, provider)

    async def run_scenario() -> None:
        loop_task = asyncio.create_task(loop.run())

        async def wait_for_loop_start() -> None:
            while loop._run_task is None:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_loop_start(), timeout=1)
        await loop.close()
        with pytest.raises(asyncio.CancelledError):
            await loop_task

        await bus.publish_inbound(
            InboundMessage(session_id="session-1", content="must not be processed")
        )
        await asyncio.sleep(0)

    asyncio.run(run_scenario())

    assert provider.chat_calls == []
    assert manager.get("session-1") is None
    assert not loop._inbound_tasks


def test_loop_persists_completed_tool_round_after_max_iteration_result(
    tmp_path: Path,
) -> None:
    tool_call = ToolCallRequest(id="call-1", name="lookup", arguments={})
    provider = _QueueProvider([LLMResponse(tool_calls=(tool_call,))])
    loop, bus, manager = _create_loop(
        tmp_path,
        provider,
        tool_registry=ToolRegistry([_StaticTool()]),
        max_iterations=1,
    )

    outbound = asyncio.run(
        _run_one(
            loop,
            bus,
            InboundMessage(session_id="session-1", content="look something up"),
        )
    )

    assert outbound.content == _MAX_ITERATIONS_MESSAGE
    saved_session = manager.get("session-1")
    assert saved_session is not None
    assert saved_session.messages == (
        HumanMessage(content="look something up"),
        AIMessage(content="", tool_calls=(tool_call,)),
        ToolMessage(
            content="tool result",
            tool_call_id="call-1",
            tool_name="lookup",
        ),
    )


def test_loop_does_not_persist_agent_runner_failure_without_exposing_internal_detail(
    tmp_path: Path,
) -> None:
    provider = _QueueProvider([])
    loop, bus, manager = _create_loop(
        tmp_path,
        provider,
        runner=_FailingRunner(),
    )

    outbound = asyncio.run(
        _run_one(
            loop,
            bus,
            InboundMessage(session_id="session-1", content="run the agent"),
        )
    )

    assert outbound.content == _AGENT_FAILURE_MESSAGE
    assert "internal runner detail" not in outbound.content
    assert manager.get("session-1") is None


def test_loop_does_not_persist_unexpected_runner_exception(tmp_path: Path) -> None:
    provider = _QueueProvider([])
    loop, bus, manager = _create_loop(
        tmp_path,
        provider,
        runner=_UnexpectedFailingRunner(),
    )

    outbound = asyncio.run(
        _run_one(
            loop,
            bus,
            InboundMessage(session_id="session-1", content="run unexpectedly"),
        )
    )

    assert outbound.content == _PROCESSING_FAILURE_MESSAGE
    assert "internal unexpected detail" not in outbound.content
    assert manager.get("session-1") is None


def test_loop_rejects_blank_session_id_without_creating_a_session(tmp_path: Path) -> None:
    provider = _QueueProvider([])
    loop, bus, manager = _create_loop(tmp_path, provider)

    outbound = asyncio.run(
        _run_one(loop, bus, InboundMessage(session_id="", content="missing session"))
    )

    assert outbound.content == _INVALID_SESSION_ID_MESSAGE
    assert manager.list_sessions() == ()


def test_loop_publishes_a_save_failure_without_exposing_storage_detail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider = _QueueProvider([LLMResponse(content="reply", finish_reason="stop")])
    loop, bus, manager = _create_loop(tmp_path, provider)

    def fail_save(session: Session) -> Session:
        raise OSError("sensitive storage detail")

    monkeypatch.setattr(manager, "save", fail_save)

    outbound = asyncio.run(
        _run_one(loop, bus, InboundMessage(session_id="session-1", content="save this"))
    )

    assert outbound.content == _SESSION_SAVE_FAILURE_MESSAGE
    assert "sensitive storage detail" not in outbound.content
    assert manager.get("session-1") is None


def test_loop_persists_new_messages_when_agent_runner_returns_an_empty_response(
    tmp_path: Path,
) -> None:
    provider = _QueueProvider([LLMResponse(content="", finish_reason="stop")])
    loop, bus, manager = _create_loop(tmp_path, provider)

    outbound = asyncio.run(
        _run_one(
            loop,
            bus,
            InboundMessage(session_id="session-1", content="respond if you can"),
        )
    )

    assert outbound.content == _EMPTY_RESPONSE_MESSAGE
    saved_session = manager.get("session-1")
    assert saved_session is not None
    assert saved_session.messages == (
        HumanMessage(content="respond if you can"),
        AIMessage(content=""),
    )
