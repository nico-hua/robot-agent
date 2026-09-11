"""OpenAI-compatible implementation of the LLM provider interface."""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from openai import AsyncOpenAI

from ..tools import Tool
from .base import LLMProvider, LLMResponse, ProviderError, TokenUsage
from .messages import AIMessage, BaseMessage, HumanMessage, ToolCallRequest, ToolMessage

logger = logging.getLogger(__name__)
_EMPTY_API_KEY_PLACEHOLDER = "not-required"
_TOOL_IMAGE_SOURCE = "tool_image"


class OpenAICompatProvider(LLMProvider):
    """Call chat-completions APIs that implement the OpenAI protocol."""

    def __init__(
        self,
        api_key: str,
        api_base: str,
        default_model: str,
        default_max_tokens: int | None = None,
        default_temperature: float | None = None,
        default_think: bool = False,
        timeout: float = 60.0,
        *,
        client: Any | None = None,
    ) -> None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("OpenAI-compatible timeout must be a positive number")
        if not isinstance(default_think, bool):
            raise TypeError("OpenAI-compatible default_think must be a boolean")

        self.api_base = api_base
        self.default_model = default_model
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature
        self.default_think = default_think
        self.timeout = float(timeout)
        self._client = (
            client
            if client is not None
            else AsyncOpenAI(
                # The SDK requires a non-empty value even for local
                # OpenAI-compatible endpoints that do not authenticate.
                api_key=api_key or _EMPTY_API_KEY_PLACEHOLDER,
                base_url=api_base,
                timeout=self.timeout,
            )
        )

    @property
    def supports_tool_image_messages(self) -> bool:
        """Request transient user-image handoffs after fresh tool results."""

        return True

    async def chat(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[Tool] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        think: bool | None = None,
    ) -> LLMResponse:
        request = self._build_request(messages, tools, max_tokens, temperature, think)
        logger.debug(
            "OpenAI-compatible completion requested (model=%s, messages=%d, tools=%d)",
            self.default_model,
            len(messages),
            len(tools or ()),
        )

        try:
            response = await self._client.chat.completions.create(**request)
            return _response_from_completion(response)
        except ProviderError:
            raise
        except Exception as exc:
            logger.exception("OpenAI-compatible completion failed")
            raise ProviderError("OpenAI-compatible completion failed") from exc

    async def stream_chat(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[Tool] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
        think: bool | None = None,
    ) -> LLMResponse:
        request = self._build_request(messages, tools, max_tokens, temperature, think)
        request["stream"] = True
        logger.debug(
            "OpenAI-compatible streaming requested (model=%s, messages=%d, tools=%d)",
            self.default_model,
            len(messages),
            len(tools or ()),
        )

        try:
            response = await self._client.chat.completions.create(**request)
            content_parts: list[str] = []
            tool_call_parts: dict[int, dict[str, str]] = {}
            finish_reason: str | None = None
            usage: TokenUsage | None = None

            async for chunk in response:
                usage = _usage_from_completion(getattr(chunk, "usage", None)) or usage
                choices = getattr(chunk, "choices", None) or ()
                if not choices:
                    continue

                choice = choices[0]
                finish_reason = getattr(choice, "finish_reason", None) or finish_reason
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue

                content = getattr(delta, "content", None)
                if content:
                    content_parts.append(content)
                    if on_delta is not None:
                        await on_delta(content)

                for tool_call in getattr(delta, "tool_calls", None) or ():
                    _append_tool_call(tool_call_parts, tool_call)

            tool_calls = tuple(
                _tool_call_from_parts(tool_call_parts[index]) for index in sorted(tool_call_parts)
            )
            return LLMResponse(
                content="".join(content_parts) or None,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                usage=usage,
            )
        except ProviderError:
            raise
        except Exception as exc:
            logger.exception("OpenAI-compatible streaming failed")
            raise ProviderError("OpenAI-compatible streaming failed") from exc

    def _build_request(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[Tool] | None,
        max_tokens: int | None,
        temperature: float | None,
        think: bool | None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.default_model,
            "messages": _messages_to_openai(messages),
        }
        if tools is not None:
            request["tools"] = [tool.to_openai_tool() for tool in tools]
        resolved_max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        if resolved_max_tokens is not None:
            request["max_tokens"] = resolved_max_tokens
        resolved_temperature = temperature if temperature is not None else self.default_temperature
        if resolved_temperature is not None:
            request["temperature"] = resolved_temperature
        request["reasoning_effort"] = _reasoning_effort(_resolve_think(think, self.default_think))
        return request


def _resolve_think(think: bool | None, default_think: bool) -> bool:
    """Use the per-request thinking override or the configured default."""

    if think is None:
        return default_think
    if not isinstance(think, bool):
        raise TypeError("OpenAI-compatible think must be a boolean or None")
    return think


def _reasoning_effort(think: bool) -> str:
    """Map the project boolean to OpenAI-compatible reasoning control."""

    return "medium" if think else "none"


def _messages_to_openai(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """Convert a message snapshot for either standard or streaming requests."""

    return [_message_to_dict(message) for message in tuple(messages)]


def _message_to_dict(message: BaseMessage) -> dict[str, Any]:
    result: dict[str, Any] = {
        "role": message.role,
        "content": message.content,
    }
    if isinstance(message, AIMessage) and message.tool_calls:
        result["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(tool_call.arguments),
                },
            }
            for tool_call in message.tool_calls
        ]
    elif isinstance(message, ToolMessage):
        result["tool_call_id"] = message.tool_call_id
    elif isinstance(message, HumanMessage) and _is_tool_image_message(message):
        result["content"] = _tool_image_content(message)
    return result


def _is_tool_image_message(message: HumanMessage) -> bool:
    return message.metadata.get("source") == _TOOL_IMAGE_SOURCE


def _tool_image_content(message: HumanMessage) -> str | list[dict[str, Any]]:
    """Translate one transient image handoff to OpenAI multimodal content."""

    image_bytes = message.image_bytes
    image_media_type = message.image_media_type
    if image_bytes is None or image_media_type is None:
        return message.content
    encoded_image = base64.b64encode(image_bytes).decode("ascii")
    return [
        {"type": "text", "text": message.content},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{image_media_type};base64,{encoded_image}",
            },
        },
    ]


def _response_from_completion(response: Any) -> LLMResponse:
    choices = getattr(response, "choices", None) or ()
    if not choices:
        raise ProviderError("OpenAI-compatible response contained no choices")

    choice = choices[0]
    message = choice.message
    tool_calls = tuple(
        _tool_call_from_response(tool_call)
        for tool_call in (getattr(message, "tool_calls", None) or ())
    )
    return LLMResponse(
        content=getattr(message, "content", None),
        tool_calls=tool_calls,
        finish_reason=getattr(choice, "finish_reason", None),
        usage=_usage_from_completion(getattr(response, "usage", None)),
    )


def _tool_call_from_response(tool_call: Any) -> ToolCallRequest:
    function = tool_call.function
    return _tool_call_from_parts(
        {
            "id": tool_call.id,
            "name": function.name,
            "arguments": function.arguments or "{}",
        }
    )


def _append_tool_call(
    tool_call_parts: dict[int, dict[str, str]],
    tool_call: Any,
) -> None:
    index = getattr(tool_call, "index", None)
    if index is None:
        index = len(tool_call_parts)
    parts = tool_call_parts.setdefault(
        index,
        {"id": "", "name": "", "arguments": ""},
    )
    parts["id"] += getattr(tool_call, "id", None) or ""
    function = getattr(tool_call, "function", None)
    if function is not None:
        parts["name"] += getattr(function, "name", None) or ""
        parts["arguments"] += getattr(function, "arguments", None) or ""


def _tool_call_from_parts(parts: Mapping[str, str]) -> ToolCallRequest:
    try:
        arguments = json.loads(parts["arguments"] or "{}")
    except json.JSONDecodeError as exc:
        raise ProviderError("Tool call arguments were not valid JSON") from exc
    if not isinstance(arguments, dict):
        raise ProviderError("Tool call arguments must be a JSON object")
    return ToolCallRequest(
        id=parts["id"],
        name=parts["name"],
        arguments=arguments,
    )


def _usage_from_completion(usage: Any) -> TokenUsage | None:
    if usage is None:
        return None
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    if prompt_tokens is None or completion_tokens is None or total_tokens is None:
        return None
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )
