"""Minimal agent execution loop for sequential tool calls."""

from __future__ import annotations

import asyncio
import logging
import mimetypes
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from ..providers import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    LLMProvider,
    TokenUsage,
    ToolCallRequest,
    ToolMessage,
)
from ..tools import ToolRegistry

logger = logging.getLogger(__name__)

_TOOL_IMAGE_SOURCE = "tool_image"
_TOOL_IMAGE_READ_ERROR = "Image attachment error: the local image could not be read."
_STALE_TOOL_IMAGE_CONTENT = (
    "The image is outdated. Call `capture_camera` to obtain the latest image."
)


class AgentRunnerError(RuntimeError):
    """Raised when an agent run cannot reach a final model response."""


@dataclass(frozen=True)
class AgentRunSpec:
    """All dependencies and input required for one agent run."""

    messages: Sequence[BaseMessage]
    provider: LLMProvider
    tool_registry: ToolRegistry
    max_iterations: int = 30
    blocked_tool_names: Sequence[str] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.messages, Sequence) or not all(
            isinstance(message, BaseMessage) for message in self.messages
        ):
            raise TypeError("messages must be a sequence of BaseMessage instances")
        if not isinstance(self.provider, LLMProvider):
            raise TypeError("AgentRunSpec provider must be an LLMProvider")
        if not isinstance(self.tool_registry, ToolRegistry):
            raise TypeError("AgentRunSpec tool_registry must be a ToolRegistry")
        if not isinstance(self.max_iterations, int) or isinstance(
            self.max_iterations,
            bool,
        ):
            raise TypeError("max_iterations must be an integer")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if isinstance(self.blocked_tool_names, (str, bytes)) or not isinstance(
            self.blocked_tool_names,
            Sequence,
        ):
            raise TypeError("blocked_tool_names must be a sequence of strings")
        if any(not isinstance(name, str) or not name.strip() for name in self.blocked_tool_names):
            raise ValueError("blocked_tool_names must not contain blank names")
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "blocked_tool_names", tuple(self.blocked_tool_names))


@dataclass(frozen=True)
class AgentRunResult:
    """The response boundary and conversation state from one agent run."""

    content: str | None
    messages: tuple[BaseMessage, ...]
    tools_used: tuple[ToolCallRequest, ...]
    token_usage: TokenUsage | None
    stop_reason: str | None


class AgentRunner:
    """Run provider completions and sequential tool calls to a response boundary."""

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        """Execute sequential non-streaming provider and tool-call rounds."""

        conversation = list(spec.messages)
        initial_message_count = len(conversation)
        tools_used: list[ToolCallRequest] = []
        token_usage: TokenUsage | None = None
        blocked_tool_names = set(spec.blocked_tool_names)
        tools = tuple(
            tool for tool in spec.tool_registry.tools if tool.name not in blocked_tool_names
        )
        logger.info(
            "Agent run started (messages=%d, tools=%d, max_iterations=%d)",
            len(conversation),
            len(tools),
            spec.max_iterations,
        )
        for iteration in range(spec.max_iterations):
            logger.debug("Requesting provider completion (iteration=%d)", iteration + 1)
            provider_messages = _provider_messages(
                conversation,
                initial_message_count=initial_message_count,
                replace_historical_tool_images=spec.provider.supports_tool_image_messages,
            )
            print(provider_messages)
            response = await spec.provider.chat(provider_messages, tools=tools or None)
            token_usage = _combine_token_usage(token_usage, response.usage)
            if not response.tool_calls:
                conversation.append(AIMessage(content=response.content or ""))
                logger.info(
                    "Agent run completed (iterations=%d, tool_calls=%d, stop_reason=%s)",
                    iteration + 1,
                    len(tools_used),
                    response.finish_reason,
                )
                return AgentRunResult(
                    content=response.content,
                    messages=tuple(conversation),
                    tools_used=tuple(tools_used),
                    token_usage=token_usage,
                    stop_reason=response.finish_reason,
                )

            conversation.append(
                AIMessage(
                    content=response.content or "",
                    tool_calls=response.tool_calls,
                )
            )
            tools_used.extend(response.tool_calls)
            tool_messages: list[ToolMessage] = []
            for tool_call in response.tool_calls:
                if tool_call.name in blocked_tool_names:
                    logger.warning("Blocked requested tool (name=%s)", tool_call.name)
                    tool_messages.append(
                        ToolMessage(
                            content=(
                                f"Error: Tool is not available in this agent run: {tool_call.name}"
                            ),
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.name,
                        )
                    )
                else:
                    logger.info("Executing requested tool (name=%s)", tool_call.name)
                    result = await spec.tool_registry.execute(
                        tool_call.name,
                        tool_call.arguments,
                    )
                    tool_messages.append(
                        ToolMessage(
                            content=result.content,
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.name,
                            image_path=result.image_path,
                        )
                    )
            conversation.extend(tool_messages)
            if tool_messages and spec.provider.supports_tool_image_messages:
                resolved_tool_messages, image_messages = await _prepare_tool_image_messages(
                    tool_messages
                )
                conversation[-len(tool_messages) :] = resolved_tool_messages
                conversation.extend(image_messages)
        # Every tool-call batch above is complete at this point: each assistant
        # tool request has its matching ToolMessage. Return that durable
        # boundary to the caller instead of raising.
        logger.warning("Agent run reached maximum iteration count (%d)", spec.max_iterations)
        return AgentRunResult(
            content=None,
            messages=tuple(conversation),
            tools_used=tuple(tools_used),
            token_usage=token_usage,
            stop_reason="max_iterations",
        )


def _combine_token_usage(
    current: TokenUsage | None,
    response_usage: TokenUsage | None,
) -> TokenUsage | None:
    if response_usage is None:
        return current
    if current is None:
        return response_usage
    return TokenUsage(
        prompt_tokens=current.prompt_tokens + response_usage.prompt_tokens,
        completion_tokens=current.completion_tokens + response_usage.completion_tokens,
        total_tokens=current.total_tokens + response_usage.total_tokens,
    )


def _provider_messages(
    conversation: Sequence[BaseMessage],
    *,
    initial_message_count: int,
    replace_historical_tool_images: bool,
) -> tuple[BaseMessage, ...]:
    """Build one provider-only snapshot without changing durable history."""

    if not replace_historical_tool_images:
        return tuple(conversation)
    return tuple(
        _stale_tool_image_message(message)
        if index < initial_message_count
        and isinstance(message, ToolMessage)
        and message.image_path is not None
        else message
        for index, message in enumerate(conversation)
    )


def _stale_tool_image_message(tool_message: ToolMessage) -> ToolMessage:
    """Tell the model to obtain a current image while retaining the reference."""

    content = _STALE_TOOL_IMAGE_CONTENT
    if "Image attachment error:" in tool_message.content:
        content = f"{content}\nTool result: {tool_message.content}"
    return replace(tool_message, content=content)


async def _prepare_tool_image_messages(
    tool_messages: Sequence[ToolMessage],
) -> tuple[list[ToolMessage], list[HumanMessage]]:
    """Create transient image handoffs after a complete tool-result batch.

    All ``ToolMessage`` instances are resolved first so that no generated user
    image message can appear between tool results required by one assistant
    tool-call response.
    """

    resolved_tool_messages: list[ToolMessage] = []
    image_messages: list[HumanMessage] = []
    for tool_message in tool_messages:
        image_path = tool_message.image_path
        if image_path is None:
            resolved_tool_messages.append(tool_message)
            continue

        image_bytes = await _read_tool_image(image_path)
        if image_bytes is None:
            logger.warning(
                "Tool result image could not be read (tool=%s, tool_call_id=%s)",
                tool_message.tool_name,
                tool_message.tool_call_id,
            )
            resolved_tool_messages.append(_tool_message_with_image_error(tool_message))
            continue

        resolved_tool_messages.append(tool_message)
        tool_name = tool_message.tool_name or "unknown"
        image_messages.append(
            HumanMessage(
                content=(
                    f"Image returned by tool '{tool_name}' for tool call "
                    f"'{tool_message.tool_call_id}'."
                ),
                metadata={
                    "source": _TOOL_IMAGE_SOURCE,
                    "tool_name": tool_name,
                    "tool_call_id": tool_message.tool_call_id,
                    "image_path": image_path,
                },
                image_bytes=image_bytes,
                image_media_type=_image_media_type(image_path),
            )
        )
    return resolved_tool_messages, image_messages


async def _read_tool_image(image_path: Path) -> bytes | None:
    """Read one local tool image without exposing filesystem details."""

    try:
        image_bytes = await asyncio.to_thread(_read_tool_image_bytes, image_path)
    except OSError:
        return None
    return image_bytes or None


def _read_tool_image_bytes(image_path: Path) -> bytes:
    if not image_path.is_file():
        raise FileNotFoundError
    return image_path.read_bytes()


def _image_media_type(image_path: Path) -> str:
    """Return a conservative media type for a local tool image."""

    media_type, _ = mimetypes.guess_type(image_path.name)
    if isinstance(media_type, str) and media_type.startswith("image/"):
        return media_type
    return "image/jpeg"


def _tool_message_with_image_error(tool_message: ToolMessage) -> ToolMessage:
    """Retain the result reference while making image-read failure visible."""

    content = tool_message.content
    updated_content = f"{content}\n{_TOOL_IMAGE_READ_ERROR}" if content else _TOOL_IMAGE_READ_ERROR
    return replace(tool_message, content=updated_content)
