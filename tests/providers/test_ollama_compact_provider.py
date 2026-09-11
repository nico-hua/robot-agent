"""Offline tests for the native Ollama provider adapter."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from src.providers import (
    AIMessage,
    HumanMessage,
    LLMResponse,
    ProviderError,
    SystemMessage,
    TokenUsage,
    ToolCallRequest,
    ToolMessage,
)
from src.providers.ollama_compact_provider import OllamaCompactProvider
from src.tools.base import Tool, ToolParameter, ToolResult


class _RecordingAsyncClient:
    """Return a fixed SDK-shaped response without making a network call."""

    def __init__(
        self,
        response: Any | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **request: Any) -> Any:
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        return self.response


class _StreamingAsyncClient:
    """Provide a small async iterator shaped like native Ollama streaming."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **request: Any) -> Any:
        self.calls.append(request)

        async def chunks() -> Any:
            yield _chat_response(
                content="你",
                done_reason=None,
                prompt_eval_count=None,
                eval_count=None,
            )
            yield _chat_response(
                content="好",
                done_reason="stop",
                prompt_eval_count=4,
                eval_count=2,
            )

        return chunks()


class _SchemaTool(Tool):
    """A small tool used to verify the native function schema conversion."""

    def __init__(self) -> None:
        super().__init__(
            name="capture_frame",
            description="Capture one camera frame.",
            parameters=(
                ToolParameter(
                    name="camera",
                    description="Camera name.",
                    type="string",
                    required=True,
                ),
                ToolParameter(
                    name="quality",
                    description="Requested quality.",
                    type="integer",
                ),
            ),
        )

    async def execute(self, **arguments: Any) -> ToolResult:
        return ToolResult(content=str(arguments))


def _provider(client: Any) -> OllamaCompactProvider:
    return OllamaCompactProvider(
        api_base="http://127.0.0.1:11434",
        default_model="qwen3-vl:4b",
        default_max_tokens=256,
        default_temperature=0.2,
        timeout=12.5,
        client=client,
    )


def _chat_response(
    *,
    content: str | None = "done",
    tool_calls: object = (),
    done_reason: str | None = "stop",
    prompt_eval_count: int | None = 3,
    eval_count: int | None = 5,
) -> SimpleNamespace:
    return SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=tool_calls),
        done_reason=done_reason,
        prompt_eval_count=prompt_eval_count,
        eval_count=eval_count,
    )


def _native_tool_call(name: str, arguments: object) -> SimpleNamespace:
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))


def test_chat_maps_tool_result_image_and_schema_to_native_ollama(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_bytes = b"example-image-bytes"
    image_path.write_bytes(image_bytes)
    client = _RecordingAsyncClient(_chat_response(content="已看到画面。"))
    tool_call = ToolCallRequest(
        id="call-1",
        name="capture_frame",
        arguments={"camera": "front"},
    )

    response = asyncio.run(
        _provider(client).chat(
            (
                SystemMessage(content="system prompt"),
                HumanMessage(content="inspect the frame"),
                AIMessage(content="", tool_calls=(tool_call,)),
                ToolMessage(
                    content="Captured one image.",
                    tool_call_id="call-1",
                    tool_name="capture_frame",
                    image_path=image_path,
                ),
            ),
            tools=(_SchemaTool(),),
        )
    )

    assert response == LLMResponse(
        content="已看到画面。",
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=3, completion_tokens=5, total_tokens=8),
    )
    assert client.calls == [
        {
            "model": "qwen3-vl:4b",
            "messages": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "inspect the frame"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "capture_frame",
                                "arguments": {"camera": "front"},
                            }
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "Captured one image.",
                    "tool_name": "capture_frame",
                    "images": [image_bytes],
                },
            ],
            "stream": False,
            "tools": [_SchemaTool().to_openai_tool()],
            "options": {"num_predict": 256, "temperature": 0.2},
        }
    ]


def test_chat_uses_history_to_name_a_legacy_text_tool_result() -> None:
    client = _RecordingAsyncClient(_chat_response())
    tool_call = ToolCallRequest(id="call-1", name="lookup", arguments={})

    asyncio.run(
        _provider(client).chat(
            (
                AIMessage(content="", tool_calls=(tool_call,)),
                ToolMessage(content="plain tool result", tool_call_id="call-1"),
            )
        )
    )

    assert client.calls[0]["messages"][-1] == {
        "role": "tool",
        "content": "plain tool result",
        "tool_name": "lookup",
    }


def test_chat_parses_native_tool_calls_with_internal_ids() -> None:
    client = _RecordingAsyncClient(
        _chat_response(
            content="",
            tool_calls=(
                _native_tool_call("capture_frame", {"camera": "front"}),
                _native_tool_call("analyze_frame", {"detail": "brief"}),
            ),
            done_reason="tool_calls",
        )
    )

    response = asyncio.run(_provider(client).chat((HumanMessage(content="look"),)))

    assert response.content == ""
    assert response.finish_reason == "tool_calls"
    assert [tool_call.name for tool_call in response.tool_calls] == [
        "capture_frame",
        "analyze_frame",
    ]
    assert [tool_call.arguments for tool_call in response.tool_calls] == [
        {"camera": "front"},
        {"detail": "brief"},
    ]
    assert all(tool_call.id for tool_call in response.tool_calls)
    assert len({tool_call.id for tool_call in response.tool_calls}) == 2


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(
            SimpleNamespace(message=None),
            id="missing-message",
        ),
        pytest.param(
            _chat_response(content=object()),
            id="non-text-content",
        ),
        pytest.param(
            _chat_response(tool_calls=(_native_tool_call("lookup", "{}"),)),
            id="non-object-arguments",
        ),
        pytest.param(
            _chat_response(tool_calls=(SimpleNamespace(function=None),)),
            id="missing-function",
        ),
    ],
)
def test_chat_rejects_invalid_native_responses(response: object) -> None:
    client = _RecordingAsyncClient(response)

    with pytest.raises(ProviderError, match="Ollama"):
        asyncio.run(_provider(client).chat((HumanMessage(content="hello"),)))


def test_chat_rejects_unreadable_local_image_before_calling_sdk(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing-image.jpg"
    client = _RecordingAsyncClient()

    with pytest.raises(ProviderError, match="image") as error:
        asyncio.run(
            _provider(client).chat(
                (
                    ToolMessage(
                        content="image result",
                        tool_call_id="call-1",
                        tool_name="capture_frame",
                        image_path=missing_path,
                    ),
                )
            )
        )

    assert str(missing_path) not in str(error.value)
    assert client.calls == []


def test_tool_message_accepts_one_local_path_and_rejects_non_paths(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"

    message = ToolMessage(
        content="image result",
        tool_call_id="call-1",
        tool_name="capture_frame",
        image_path=str(image_path),
    )

    assert message.image_path == image_path
    with pytest.raises(TypeError, match="image_path"):
        ToolMessage(content="image result", tool_call_id="call-1", image_path=b"image")
    with pytest.raises(TypeError, match="image_path"):
        ToolMessage(content="image result", tool_call_id="call-1", image_path=[image_path])
    with pytest.raises(ValueError, match="local path"):
        ToolMessage(
            content="image result",
            tool_call_id="call-1",
            image_path="https://example.test/frame.jpg",
        )


def test_stream_chat_aggregates_native_deltas() -> None:
    client = _StreamingAsyncClient()
    deltas: list[str] = []

    async def on_delta(content: str) -> None:
        deltas.append(content)

    response = asyncio.run(
        _provider(client).stream_chat(
            (HumanMessage(content="hello"),),
            on_delta=on_delta,
        )
    )

    assert client.calls[0]["stream"] is True
    assert deltas == ["你", "好"]
    assert response == LLMResponse(
        content="你好",
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=4, completion_tokens=2, total_tokens=6),
    )


def test_chat_preserves_cancellation() -> None:
    client = _RecordingAsyncClient(error=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_provider(client).chat((HumanMessage(content="hello"),)))
