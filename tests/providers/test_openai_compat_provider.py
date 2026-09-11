"""Offline tests for OpenAI-compatible tool-image message handling."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from src.providers import AIMessage, HumanMessage, ToolCallRequest, ToolMessage
from src.providers.openai_compat_provider import OpenAICompatProvider


class _RecordingCompletions:
    """Return one configured completion and retain the OpenAI request."""

    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **request: Any) -> Any:
        self.calls.append(request)
        return self._response


class _RecordingClient:
    """Expose the small AsyncOpenAI surface used by the adapter."""

    def __init__(self, response: Any) -> None:
        self.completions = _RecordingCompletions(response)
        self.chat = SimpleNamespace(completions=self.completions)


def _completion(
    *,
    content: str | None = "done",
    tool_calls: tuple[Any, ...] = (),
    finish_reason: str | None = "stop",
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=(
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            ),
        ),
        usage=None,
    )


def _stream(chunks: tuple[Any, ...]) -> Any:
    async def generate() -> Any:
        for chunk in chunks:
            yield chunk

    return generate()


def _stream_chunk(
    *,
    content: str | None = None,
    tool_calls: tuple[Any, ...] = (),
    finish_reason: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=(
            SimpleNamespace(
                delta=SimpleNamespace(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            ),
        ),
        usage=None,
    )


def _provider(
    client: _RecordingClient,
    *,
    default_think: bool = False,
) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        api_key="test-key",
        api_base="https://api.example.test/v1",
        default_model="test-model",
        default_think=default_think,
        client=client,
    )


def _tool_history_with_image(image_path: Path) -> tuple[AIMessage, ToolMessage]:
    tool_call = ToolCallRequest(id="call-camera", name="capture_camera", arguments={})
    tool_message = ToolMessage(
        content="The camera tool returned an image.",
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        image_path=image_path,
    )
    return AIMessage(content="", tool_calls=(tool_call,)), tool_message


def _assert_compatible_tool_image_message(
    request_message: dict[str, Any],
    *,
    content: str,
) -> None:
    assert request_message["role"] == "tool"
    assert request_message["content"] == content
    assert request_message["tool_call_id"] == "call-camera"
    assert "image_path" not in request_message


def test_chat_accepts_historical_tool_image_without_rewriting_it(tmp_path: Path) -> None:
    image_path = tmp_path / "historic-camera.jpg"
    assistant_message, tool_message = _tool_history_with_image(image_path)
    client = _RecordingClient(_completion())

    response = asyncio.run(_provider(client).chat((assistant_message, tool_message)))

    assert response.content == "done"
    assert tool_message.content == "The camera tool returned an image."
    assert tool_message.image_path == image_path
    _assert_compatible_tool_image_message(
        client.completions.calls[0]["messages"][-1],
        content=tool_message.content,
    )


def test_stream_chat_accepts_historical_tool_image_without_rewriting_it(tmp_path: Path) -> None:
    image_path = tmp_path / "historic-camera.jpg"
    assistant_message, tool_message = _tool_history_with_image(image_path)
    client = _RecordingClient(_stream((_stream_chunk(content="done", finish_reason="stop"),)))

    response = asyncio.run(_provider(client).stream_chat((assistant_message, tool_message)))

    assert response.content == "done"
    assert tool_message.content == "The camera tool returned an image."
    assert tool_message.image_path == image_path
    request = client.completions.calls[0]
    assert request["stream"] is True
    _assert_compatible_tool_image_message(
        request["messages"][-1],
        content=tool_message.content,
    )


def test_chat_encodes_an_auto_generated_tool_image_as_one_multimodal_user_message(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "camera.jpg"
    image_bytes = b"camera-image-bytes"
    image_path.write_bytes(image_bytes)
    message = HumanMessage(
        content="Image from tool capture_camera for tool call call-camera.",
        metadata={
            "source": "tool_image",
            "tool_name": "capture_camera",
            "tool_call_id": "call-camera",
            "image_path": image_path,
        },
        image_bytes=image_bytes,
        image_media_type="image/jpeg",
    )
    client = _RecordingClient(_completion())

    asyncio.run(_provider(client).chat((message,)))

    wire_message = client.completions.calls[0]["messages"][0]
    assert wire_message["role"] == "user"
    assert isinstance(wire_message["content"], list)
    assert wire_message["content"][0] == {
        "type": "text",
        "text": "Image from tool capture_camera for tool call call-camera.",
    }
    image_part = wire_message["content"][1]
    assert image_part["type"] == "image_url"
    assert base64.b64encode(image_bytes).decode("ascii") in image_part["image_url"]["url"]


def test_chat_preserves_a_fresh_tool_result_before_its_image_handoff(tmp_path: Path) -> None:
    image_path = tmp_path / "camera.jpg"
    image_bytes = b"camera-image-bytes"
    assistant_message, tool_message = _tool_history_with_image(image_path)
    image_message = HumanMessage(
        content="Image from tool capture_camera for tool call call-camera.",
        metadata={
            "source": "tool_image",
            "tool_name": "capture_camera",
            "tool_call_id": "call-camera",
            "image_path": image_path,
        },
        image_bytes=image_bytes,
        image_media_type="image/jpeg",
    )
    client = _RecordingClient(_completion())

    asyncio.run(_provider(client).chat((assistant_message, tool_message, image_message)))

    wire_messages = client.completions.calls[0]["messages"]
    assert wire_messages[1] == {
        "role": "tool",
        "content": "The camera tool returned an image.",
        "tool_call_id": "call-camera",
    }
    assert isinstance(wire_messages[2]["content"], list)
    assert tool_message.content == "The camera tool returned an image."


def test_stream_chat_encodes_an_auto_generated_tool_image_as_one_multimodal_user_message(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "camera.jpg"
    image_bytes = b"camera-image-bytes"
    message = HumanMessage(
        content="Image from tool capture_camera for tool call call-camera.",
        metadata={
            "source": "tool_image",
            "tool_name": "capture_camera",
            "tool_call_id": "call-camera",
            "image_path": image_path,
        },
        image_bytes=image_bytes,
        image_media_type="image/jpeg",
    )
    client = _RecordingClient(_stream((_stream_chunk(content="done", finish_reason="stop"),)))

    response = asyncio.run(_provider(client).stream_chat((message,)))

    assert response.content == "done"
    request = client.completions.calls[0]
    assert request["stream"] is True
    wire_message = request["messages"][0]
    assert wire_message["role"] == "user"
    assert wire_message["content"][0]["text"] == message.content
    assert (
        base64.b64encode(image_bytes).decode("ascii")
        in wire_message["content"][1]["image_url"]["url"]
    )


def test_stream_chat_collects_complete_multiple_tool_calls_before_returning_them() -> None:
    first_camera_delta = SimpleNamespace(
        index=0,
        id="call-camera",
        function=SimpleNamespace(name="capture_camera", arguments='{"camera":'),
    )
    first_lookup_delta = SimpleNamespace(
        index=1,
        id="call-lookup",
        function=SimpleNamespace(name="lookup", arguments='{"query":'),
    )
    second_camera_delta = SimpleNamespace(
        index=0,
        id=None,
        function=SimpleNamespace(name=None, arguments='"head"}'),
    )
    second_lookup_delta = SimpleNamespace(
        index=1,
        id=None,
        function=SimpleNamespace(name=None, arguments='"room"}'),
    )
    client = _RecordingClient(
        _stream(
            (
                _stream_chunk(tool_calls=(first_camera_delta, first_lookup_delta)),
                _stream_chunk(
                    tool_calls=(second_camera_delta, second_lookup_delta),
                    finish_reason="tool_calls",
                ),
            )
        )
    )

    response = asyncio.run(_provider(client).stream_chat((HumanMessage(content="look"),)))

    assert response.tool_calls == (
        ToolCallRequest(
            id="call-camera",
            name="capture_camera",
            arguments={"camera": "head"},
        ),
        ToolCallRequest(
            id="call-lookup",
            name="lookup",
            arguments={"query": "room"},
        ),
    )
    assert response.finish_reason == "tool_calls"
    assert len(client.completions.calls) == 1


@pytest.mark.parametrize(
    ("default_think", "think", "expected_reasoning_effort"),
    [
        pytest.param(False, None, "none", id="default-disabled"),
        pytest.param(True, None, "medium", id="default-enabled"),
        pytest.param(False, True, "medium", id="explicitly-enabled"),
        pytest.param(True, False, "none", id="explicitly-disabled"),
    ],
)
def test_chat_passes_resolved_thinking_mode_to_openai_compatible_api(
    default_think: bool,
    think: bool | None,
    expected_reasoning_effort: str,
) -> None:
    client = _RecordingClient(_completion())

    asyncio.run(
        _provider(client, default_think=default_think).chat(
            (HumanMessage(content="think about this"),),
            think=think,
        )
    )

    assert client.completions.calls[0]["reasoning_effort"] == expected_reasoning_effort


@pytest.mark.parametrize(
    ("default_think", "think", "expected_reasoning_effort"),
    [
        pytest.param(False, None, "none", id="default-disabled"),
        pytest.param(True, None, "medium", id="default-enabled"),
        pytest.param(False, True, "medium", id="explicitly-enabled"),
        pytest.param(True, False, "none", id="explicitly-disabled"),
    ],
)
def test_stream_chat_passes_resolved_thinking_mode_to_openai_compatible_api(
    default_think: bool,
    think: bool | None,
    expected_reasoning_effort: str,
) -> None:
    client = _RecordingClient(_stream((_stream_chunk(content="done", finish_reason="stop"),)))

    asyncio.run(
        _provider(client, default_think=default_think).stream_chat(
            (HumanMessage(content="think about this"),),
            think=think,
        )
    )

    request = client.completions.calls[0]
    assert request["stream"] is True
    assert request["reasoning_effort"] == expected_reasoning_effort
