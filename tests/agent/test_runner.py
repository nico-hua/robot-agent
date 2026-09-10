"""Tests for the non-streaming agent execution loop."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from src.agent.runner import AgentRunner, AgentRunSpec
from src.providers import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    LLMProvider,
    LLMResponse,
    TokenUsage,
    ToolCallRequest,
    ToolMessage,
)
from src.tools import Tool, ToolRegistry
from src.tools.base import ToolResult


class _FakeProvider(LLMProvider):
    """Provide queued responses while rejecting any streaming request."""

    def __init__(self, responses: Sequence[LLMResponse]) -> None:
        self._responses = list(responses)
        self.chat_calls: list[tuple[BaseMessage, ...]] = []
        self.stream_chat_calls = 0

    async def chat(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[Tool] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        self.chat_calls.append(tuple(messages))
        if not self._responses:
            raise AssertionError("No fake response was configured")
        return self._responses.pop(0)

    async def stream_chat(self, *args: object, **kwargs: object) -> LLMResponse:
        self.stream_chat_calls += 1
        raise AssertionError("AgentRunner must not call stream_chat")


class _RecordingTool(Tool):
    """Return a fixed result and retain the requested arguments."""

    def __init__(self, name: str = "lookup") -> None:
        super().__init__(name=name, description="Look up a value.")
        self.calls: list[dict[str, Any]] = []

    async def execute(self, **arguments: Any) -> ToolResult:
        self.calls.append(arguments)
        return ToolResult(content="tool result")


def _spec(
    provider: LLMProvider,
    registry: ToolRegistry | None = None,
    *,
    max_iterations: int = 30,
    blocked_tool_names: Sequence[str] = (),
) -> AgentRunSpec:
    return AgentRunSpec(
        messages=(HumanMessage(content="hello"),),
        provider=provider,
        tool_registry=registry or ToolRegistry(),
        max_iterations=max_iterations,
        blocked_tool_names=blocked_tool_names,
    )


def test_runner_exposes_no_goal_injection_or_streaming_api() -> None:
    removed_fields = {"is_goal_mode", "injection_callback", "on_tool_call", "on_delta"}

    assert removed_fields.isdisjoint(AgentRunSpec.__dataclass_fields__)
    assert not hasattr(AgentRunner, "run_stream")
    assert not hasattr(AgentRunner, "_run")


def test_runner_uses_chat_for_a_final_response() -> None:
    usage = TokenUsage(prompt_tokens=3, completion_tokens=5, total_tokens=8)
    provider = _FakeProvider([LLMResponse(content="done", finish_reason="stop", usage=usage)])

    result = asyncio.run(AgentRunner().run(_spec(provider)))

    assert result.content == "done"
    assert result.messages == (
        HumanMessage(content="hello"),
        AIMessage(content="done"),
    )
    assert result.tools_used == ()
    assert result.token_usage == usage
    assert result.stop_reason == "stop"
    assert len(provider.chat_calls) == 1
    assert provider.stream_chat_calls == 0


def test_runner_executes_tool_calls_before_the_next_chat_request() -> None:
    tool_call = ToolCallRequest(id="call-1", name="lookup", arguments={})
    provider = _FakeProvider(
        [
            LLMResponse(
                tool_calls=(tool_call,),
                usage=TokenUsage(prompt_tokens=2, completion_tokens=3, total_tokens=5),
            ),
            LLMResponse(
                content="complete",
                finish_reason="stop",
                usage=TokenUsage(prompt_tokens=7, completion_tokens=11, total_tokens=18),
            ),
        ]
    )
    tool = _RecordingTool()

    result = asyncio.run(AgentRunner().run(_spec(provider, ToolRegistry([tool]))))

    assert result.content == "complete"
    assert result.tools_used == (tool_call,)
    assert result.token_usage == TokenUsage(
        prompt_tokens=9,
        completion_tokens=14,
        total_tokens=23,
    )
    assert tool.calls == [{}]
    assert len(provider.chat_calls) == 2
    assert provider.chat_calls[1] == (
        HumanMessage(content="hello"),
        AIMessage(content="", tool_calls=(tool_call,)),
        ToolMessage(content="tool result", tool_call_id="call-1"),
    )
    assert provider.stream_chat_calls == 0


def test_runner_returns_a_tool_error_for_a_blocked_request() -> None:
    tool_call = ToolCallRequest(id="call-1", name="blocked", arguments={})
    provider = _FakeProvider(
        [
            LLMResponse(tool_calls=(tool_call,)),
            LLMResponse(content="complete", finish_reason="stop"),
        ]
    )
    tool = _RecordingTool(name="blocked")

    result = asyncio.run(
        AgentRunner().run(
            _spec(
                provider,
                ToolRegistry([tool]),
                blocked_tool_names=("blocked",),
            )
        )
    )

    assert tool.calls == []
    assert isinstance(provider.chat_calls[1][-1], ToolMessage)
    assert provider.chat_calls[1][-1].content == (
        "Error: Tool is not available in this agent run: blocked"
    )
    assert result.content == "complete"
    assert provider.stream_chat_calls == 0


def test_runner_returns_a_durable_boundary_at_max_iterations() -> None:
    tool_call = ToolCallRequest(id="call-1", name="lookup", arguments={})
    provider = _FakeProvider([LLMResponse(tool_calls=(tool_call,))])
    tool = _RecordingTool()

    result = asyncio.run(AgentRunner().run(_spec(provider, ToolRegistry([tool]), max_iterations=1)))

    assert result.content is None
    assert result.stop_reason == "max_iterations"
    assert result.tools_used == (tool_call,)
    assert result.messages[-1] == ToolMessage(
        content="tool result",
        tool_call_id="call-1",
    )
    assert provider.stream_chat_calls == 0
