"""Native Ollama SDK implementation of the provider interface.

This adapter keeps Ollama SDK types at the provider boundary. In particular,
``ToolMessage`` stores a local image path and the adapter reads that one file
only when it builds the request sent to Ollama.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from ollama import AsyncClient

from ..tools import Tool
from .base import LLMProvider, LLMResponse, ProviderError, TokenUsage
from .messages import AIMessage, BaseMessage, ToolCallRequest, ToolMessage

logger = logging.getLogger(__name__)
_MISSING = object()


class OllamaCompactProvider(LLMProvider):
    """Call a native Ollama ``/api/chat`` endpoint through its Python SDK."""

    def __init__(
        self,
        api_base: str,
        default_model: str,
        default_max_tokens: int | None = None,
        default_temperature: float | None = None,
        timeout: float = 60.0,
        *,
        client: Any | None = None,
    ) -> None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("Ollama timeout must be a positive number")

        self.api_base = api_base
        self.default_model = default_model
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature
        self.timeout = float(timeout)
        self._client = (
            client
            if client is not None
            else AsyncClient(
                host=api_base,
                timeout=self.timeout,
            )
        )

    async def chat(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[Tool] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Complete one non-streaming native Ollama chat request."""

        request = await self._build_request(
            messages,
            tools,
            max_tokens,
            temperature,
            stream=False,
        )
        logger.debug(
            "Ollama completion requested (model=%s, messages=%d, tools=%d)",
            self.default_model,
            len(messages),
            len(tools or ()),
        )
        try:
            response = await self._client.chat(**request)
            return _response_from_chat(response)
        except ProviderError:
            raise
        except Exception as exc:
            logger.exception("Ollama completion failed")
            raise ProviderError("Ollama completion failed") from exc

    async def stream_chat(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[Tool] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        """Stream native Ollama text deltas and return the aggregated response."""

        request = await self._build_request(
            messages,
            tools,
            max_tokens,
            temperature,
            stream=True,
        )
        logger.debug(
            "Ollama streaming requested (model=%s, messages=%d, tools=%d)",
            self.default_model,
            len(messages),
            len(tools or ()),
        )
        try:
            response = await self._client.chat(**request)
            if not hasattr(response, "__aiter__"):
                raise ProviderError("Ollama streaming response was not asynchronous")

            content_parts: list[str] = []
            tool_calls: tuple[ToolCallRequest, ...] = ()
            finish_reason: str | None = None
            usage: TokenUsage | None = None
            async for chunk in response:
                chunk_response = _response_from_chat(chunk)
                if chunk_response.content:
                    content_parts.append(chunk_response.content)
                    if on_delta is not None:
                        await on_delta(chunk_response.content)
                if chunk_response.tool_calls:
                    tool_calls = chunk_response.tool_calls
                finish_reason = chunk_response.finish_reason or finish_reason
                usage = chunk_response.usage or usage

            return LLMResponse(
                content="".join(content_parts) or None,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                usage=usage,
            )
        except ProviderError:
            raise
        except Exception as exc:
            logger.exception("Ollama streaming failed")
            raise ProviderError("Ollama streaming failed") from exc

    async def _build_request(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[Tool] | None,
        max_tokens: int | None,
        temperature: float | None,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.default_model,
            "messages": await _messages_to_ollama(messages),
            "stream": stream,
        }
        if tools is not None:
            request["tools"] = [tool.to_openai_tool() for tool in tools]

        options: dict[str, int | float] = {}
        resolved_max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        if resolved_max_tokens is not None:
            options["num_predict"] = resolved_max_tokens
        resolved_temperature = temperature if temperature is not None else self.default_temperature
        if resolved_temperature is not None:
            options["temperature"] = resolved_temperature
        if options:
            request["options"] = options
        return request


async def _messages_to_ollama(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """Translate core messages without exposing SDK message classes upstream."""

    conversation = tuple(messages)
    native_messages: list[dict[str, Any]] = []
    for index, message in enumerate(conversation):
        native_message: dict[str, Any] = {
            "role": message.role,
            "content": message.content,
        }
        if isinstance(message, AIMessage) and message.tool_calls:
            native_message["tool_calls"] = [
                {
                    "function": {
                        "name": tool_call.name,
                        "arguments": dict(tool_call.arguments),
                    }
                }
                for tool_call in message.tool_calls
            ]
        elif isinstance(message, ToolMessage):
            tool_name = message.tool_name or _tool_name_from_history(
                conversation[:index],
                message.tool_call_id,
            )
            if tool_name is None:
                raise ProviderError("Ollama tool result requires a tool_name")
            native_message["tool_name"] = tool_name
            if message.image_path is not None:
                native_message["images"] = [await _read_local_image(message.image_path)]
        native_messages.append(native_message)
    return native_messages


def _tool_name_from_history(
    messages: Sequence[BaseMessage],
    tool_call_id: str,
) -> str | None:
    """Recover a name for legacy tool messages that only stored an ID."""

    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        for tool_call in message.tool_calls:
            if tool_call.id == tool_call_id:
                return tool_call.name
    return None


async def _read_local_image(image_path: Path) -> bytes:
    """Read one local image file without exposing its path in public errors."""

    try:
        image_bytes = await asyncio.to_thread(_read_local_image_bytes, image_path)
    except OSError as exc:
        raise ProviderError("Ollama could not read the ToolMessage image") from exc
    if not image_bytes:
        raise ProviderError("Ollama ToolMessage image must not be empty")
    return image_bytes


def _read_local_image_bytes(image_path: Path) -> bytes:
    if not image_path.is_file():
        raise FileNotFoundError
    return image_path.read_bytes()


def _response_from_chat(response: Any) -> LLMResponse:
    """Normalize an Ollama SDK chat response at the adapter boundary."""

    message = _field(response, "message", _MISSING)
    if message is _MISSING or message is None:
        raise ProviderError("Ollama response contained no message")

    content = _field(message, "content", None)
    if content is not None and not isinstance(content, str):
        raise ProviderError("Ollama response content must be text")
    finish_reason = _field(response, "done_reason", None)
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise ProviderError("Ollama response finish reason must be text")

    return LLMResponse(
        content=content,
        tool_calls=_tool_calls_from_chat_message(message),
        finish_reason=finish_reason,
        usage=_usage_from_chat(response),
    )


def _tool_calls_from_chat_message(message: Any) -> tuple[ToolCallRequest, ...]:
    tool_calls = _field(message, "tool_calls", None)
    if tool_calls is None:
        return ()
    if isinstance(tool_calls, (str, bytes, Mapping)) or not isinstance(tool_calls, Sequence):
        raise ProviderError("Ollama response tool calls must be a sequence")

    normalized_calls: list[ToolCallRequest] = []
    for index, tool_call in enumerate(tool_calls):
        function = _field(tool_call, "function", None)
        if function is None:
            raise ProviderError("Ollama response tool call contained no function")
        name = _field(function, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise ProviderError("Ollama response tool call function name must be text")
        arguments = _field(function, "arguments", None)
        if not isinstance(arguments, Mapping):
            raise ProviderError("Ollama response tool call arguments must be an object")

        response_id = _field(tool_call, "id", None)
        tool_call_id = (
            response_id
            if isinstance(response_id, str) and response_id.strip()
            else f"ollama-{uuid4().hex}-{index}"
        )
        normalized_calls.append(
            ToolCallRequest(
                id=tool_call_id,
                name=name,
                arguments=dict(arguments),
            )
        )
    return tuple(normalized_calls)


def _usage_from_chat(response: Any) -> TokenUsage | None:
    prompt_tokens = _field(response, "prompt_eval_count", None)
    completion_tokens = _field(response, "eval_count", None)
    if prompt_tokens is None and completion_tokens is None:
        return None
    if not _is_token_count(prompt_tokens) or not _is_token_count(completion_tokens):
        raise ProviderError("Ollama response usage counts must be non-negative integers")
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _is_token_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _field(value: Any, name: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)
