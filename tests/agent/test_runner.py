"""Tests for the non-streaming agent execution loop."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from src.agent import runner as agent_runner
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


class _ToolImageCapableFakeProvider(_FakeProvider):
    """Opt into the runner's transient OpenAI-style image handoff behavior."""

    @property
    def supports_tool_image_messages(self) -> bool:
        return True


class _RecordingTool(Tool):
    """Return a fixed result and retain the requested arguments."""

    def __init__(self, name: str = "lookup") -> None:
        super().__init__(name=name, description="Look up a value.")
        self.calls: list[dict[str, Any]] = []

    async def execute(self, **arguments: Any) -> ToolResult:
        self.calls.append(arguments)
        return ToolResult(content="tool result")


class _ImageRecordingTool(_RecordingTool):
    """Return a tool result that refers to one local image."""

    def __init__(self, image_path: Path) -> None:
        super().__init__()
        self._image_path = image_path

    async def execute(self, **arguments: Any) -> ToolResult:
        self.calls.append(arguments)
        return ToolResult(content="image result", image_path=self._image_path)


class _StaticResultTool(Tool):
    """Return one configured result and retain every invocation."""

    def __init__(self, name: str, result: ToolResult) -> None:
        super().__init__(name=name, description=f"Return a {name} result.")
        self._result = result
        self.calls: list[dict[str, Any]] = []

    async def execute(self, **arguments: Any) -> ToolResult:
        self.calls.append(arguments)
        return self._result


def _spec(
    provider: LLMProvider,
    registry: ToolRegistry | None = None,
    *,
    messages: Sequence[BaseMessage] | None = None,
    max_iterations: int = 30,
    blocked_tool_names: Sequence[str] = (),
) -> AgentRunSpec:
    return AgentRunSpec(
        messages=(HumanMessage(content="hello"),) if messages is None else messages,
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


def test_runner_replaces_historical_tool_image_only_in_an_image_capable_snapshot(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "historic-camera.jpg"
    historical_tool_call = ToolCallRequest(
        id="call-history",
        name="capture_camera",
        arguments={},
    )
    historical_tool_message = ToolMessage(
        content="Historic camera image result.",
        tool_call_id=historical_tool_call.id,
        tool_name=historical_tool_call.name,
        image_path=image_path,
    )
    history = (
        HumanMessage(content="Please inspect the earlier camera image."),
        AIMessage(content="", tool_calls=(historical_tool_call,)),
        historical_tool_message,
    )
    provider = _ToolImageCapableFakeProvider(
        [LLMResponse(content="complete", finish_reason="stop")]
    )

    result = asyncio.run(AgentRunner().run(_spec(provider, messages=history)))

    provider_tool_message = provider.chat_calls[0][-1]
    assert isinstance(provider_tool_message, ToolMessage)
    assert provider_tool_message is not historical_tool_message
    assert "outdated" in provider_tool_message.content.lower()
    assert "capture_camera" in provider_tool_message.content
    assert provider_tool_message.image_path == image_path
    assert historical_tool_message.content == "Historic camera image result."
    assert historical_tool_message.image_path == image_path
    assert result.messages == (*history, AIMessage(content="complete"))


def test_runner_keeps_historical_tool_image_for_a_non_image_provider(tmp_path: Path) -> None:
    image_path = tmp_path / "historic-camera.jpg"
    historical_tool_message = ToolMessage(
        content="Historic camera image result.",
        tool_call_id="call-history",
        tool_name="capture_camera",
        image_path=image_path,
    )
    history = (HumanMessage(content="hello"), historical_tool_message)
    provider = _FakeProvider([LLMResponse(content="complete", finish_reason="stop")])

    result = asyncio.run(AgentRunner().run(_spec(provider, messages=history)))

    assert provider.chat_calls[0] == history
    assert result.messages == (*history, AIMessage(content="complete"))


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
        ToolMessage(content="tool result", tool_call_id="call-1", tool_name="lookup"),
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
    assert provider.chat_calls[1][-1].tool_name == "blocked"
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
        tool_name="lookup",
    )
    assert provider.stream_chat_calls == 0


def test_runner_propagates_a_tool_result_image_path_to_the_next_provider_request(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "tool-result.png"
    image_bytes = b"\x89PNG\r\n\x1a\nimage-data"
    image_path.write_bytes(image_bytes)
    tool_call = ToolCallRequest(id="call-image", name="lookup", arguments={})
    provider = _ToolImageCapableFakeProvider(
        [
            LLMResponse(tool_calls=(tool_call,)),
            LLMResponse(content="complete", finish_reason="stop"),
        ]
    )
    tool = _ImageRecordingTool(image_path)

    result = asyncio.run(AgentRunner().run(_spec(provider, ToolRegistry([tool]))))

    request_messages = provider.chat_calls[1]
    assert request_messages[2] == ToolMessage(
        content="image result",
        tool_call_id="call-image",
        tool_name="lookup",
        image_path=image_path,
    )
    image_message = request_messages[3]
    assert isinstance(image_message, HumanMessage)
    assert "lookup" in image_message.content
    assert "call-image" in image_message.content
    assert image_message.metadata == {
        "source": "tool_image",
        "tool_name": "lookup",
        "tool_call_id": "call-image",
        "image_path": image_path,
    }
    assert image_message.image_bytes == image_bytes
    assert image_message.image_media_type == "image/png"
    assert result.messages[-3] == request_messages[2]
    assert result.messages[-2] == image_message


def test_runner_appends_tool_images_only_after_a_complete_partial_image_batch(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "camera.png"
    image_bytes = b"\x89PNG\r\n\x1a\ncamera"
    image_path.write_bytes(image_bytes)
    text_call = ToolCallRequest(id="call-text", name="lookup", arguments={})
    image_call = ToolCallRequest(id="call-image", name="capture_camera", arguments={})
    provider = _ToolImageCapableFakeProvider(
        [
            LLMResponse(tool_calls=(text_call, image_call)),
            LLMResponse(content="complete", finish_reason="stop"),
        ]
    )
    text_tool = _StaticResultTool("lookup", ToolResult(content="text result"))
    image_tool = _StaticResultTool(
        "capture_camera",
        ToolResult(content="camera result", image_path=image_path),
    )

    result = asyncio.run(AgentRunner().run(_spec(provider, ToolRegistry([text_tool, image_tool]))))

    request_messages = provider.chat_calls[1]
    assert [type(message) for message in request_messages] == [
        HumanMessage,
        AIMessage,
        ToolMessage,
        ToolMessage,
        HumanMessage,
    ]
    assert request_messages[2] == ToolMessage(
        content="text result",
        tool_call_id="call-text",
        tool_name="lookup",
    )
    assert request_messages[3] == ToolMessage(
        content="camera result",
        tool_call_id="call-image",
        tool_name="capture_camera",
        image_path=image_path,
    )
    image_message = request_messages[4]
    assert image_message.metadata == {
        "source": "tool_image",
        "tool_name": "capture_camera",
        "tool_call_id": "call-image",
        "image_path": image_path,
    }
    assert image_message.image_bytes == image_bytes
    assert image_message.image_media_type == "image/png"
    assert result.messages[-4:-1] == request_messages[2:]
    assert text_tool.calls == [{}]
    assert image_tool.calls == [{}]


def test_runner_appends_all_tool_images_after_all_tool_results(tmp_path: Path) -> None:
    first_image_path = tmp_path / "first.jpg"
    second_image_path = tmp_path / "second.png"
    first_image_bytes = b"first-image"
    second_image_bytes = b"second-image"
    first_image_path.write_bytes(first_image_bytes)
    second_image_path.write_bytes(second_image_bytes)
    first_call = ToolCallRequest(id="call-first", name="first_camera", arguments={})
    second_call = ToolCallRequest(id="call-second", name="second_camera", arguments={})
    provider = _ToolImageCapableFakeProvider(
        [
            LLMResponse(tool_calls=(first_call, second_call)),
            LLMResponse(content="complete", finish_reason="stop"),
        ]
    )
    first_tool = _StaticResultTool(
        "first_camera",
        ToolResult(content="first result", image_path=first_image_path),
    )
    second_tool = _StaticResultTool(
        "second_camera",
        ToolResult(content="second result", image_path=second_image_path),
    )

    result = asyncio.run(
        AgentRunner().run(_spec(provider, ToolRegistry([first_tool, second_tool])))
    )

    request_messages = provider.chat_calls[1]
    assert [type(message) for message in request_messages] == [
        HumanMessage,
        AIMessage,
        ToolMessage,
        ToolMessage,
        HumanMessage,
        HumanMessage,
    ]
    assert [message.tool_call_id for message in request_messages[2:4]] == [
        "call-first",
        "call-second",
    ]
    assert [message.metadata["tool_call_id"] for message in request_messages[4:]] == [
        "call-first",
        "call-second",
    ]
    assert [message.image_bytes for message in request_messages[4:]] == [
        first_image_bytes,
        second_image_bytes,
    ]
    assert result.messages[2:4] == request_messages[2:4]
    assert [message.content for message in result.messages[2:4]] == [
        "first result",
        "second result",
    ]


@pytest.mark.parametrize("path_kind", ["missing", "directory"])
def test_runner_keeps_a_tool_result_when_its_image_cannot_be_read(
    tmp_path: Path,
    path_kind: str,
) -> None:
    image_path = tmp_path / "unavailable-image.png"
    if path_kind == "directory":
        image_path.mkdir()
    tool_call = ToolCallRequest(id="call-image", name="capture_camera", arguments={})
    provider = _ToolImageCapableFakeProvider(
        [
            LLMResponse(tool_calls=(tool_call,)),
            LLMResponse(content="complete", finish_reason="stop"),
        ]
    )
    tool = _StaticResultTool(
        "capture_camera",
        ToolResult(content="camera result", image_path=image_path),
    )

    result = asyncio.run(AgentRunner().run(_spec(provider, ToolRegistry([tool]))))

    request_messages = provider.chat_calls[1]
    assert [type(message) for message in request_messages] == [
        HumanMessage,
        AIMessage,
        ToolMessage,
    ]
    failed_tool_message = request_messages[-1]
    assert failed_tool_message.image_path == image_path
    assert "image" in failed_tool_message.content.lower()
    assert failed_tool_message.content != "camera result"
    assert result.content == "complete"


def test_runner_converts_an_image_read_os_error_to_a_tool_result_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "camera.png"
    image_path.write_bytes(b"camera-image")
    tool_call = ToolCallRequest(id="call-image", name="capture_camera", arguments={})
    provider = _ToolImageCapableFakeProvider(
        [
            LLMResponse(tool_calls=(tool_call,)),
            LLMResponse(content="complete", finish_reason="stop"),
        ]
    )
    tool = _StaticResultTool(
        "capture_camera",
        ToolResult(content="camera result", image_path=image_path),
    )

    def raise_read_error(path: Path) -> bytes:
        assert path == image_path
        raise OSError("simulated local image read failure")

    monkeypatch.setattr(agent_runner, "_read_tool_image_bytes", raise_read_error)

    result = asyncio.run(AgentRunner().run(_spec(provider, ToolRegistry([tool]))))

    request_messages = provider.chat_calls[1]
    assert [type(message) for message in request_messages] == [
        HumanMessage,
        AIMessage,
        ToolMessage,
    ]
    failed_tool_message = request_messages[-1]
    assert failed_tool_message == ToolMessage(
        content="camera result\nImage attachment error: the local image could not be read.",
        tool_call_id="call-image",
        tool_name="capture_camera",
        image_path=image_path,
    )
    assert not any(
        isinstance(message, HumanMessage) and message.metadata.get("source") == "tool_image"
        for message in request_messages
    )
    assert result.content == "complete"
