"""Tests for provider request-context construction."""

from __future__ import annotations

import pytest
from src.agent.context import SYSTEM_PROMPT, ContextBuilder
from src.providers import AIMessage, HumanMessage, SystemMessage, ToolMessage


def test_build_system_prompt_returns_the_current_fixed_prompt() -> None:
    assert ContextBuilder().build_system_prompt() == SYSTEM_PROMPT


def test_build_request_messages_orders_system_history_and_current_user_message() -> None:
    builder = ContextBuilder()
    history = (
        SystemMessage(content="legacy system prompt"),
        HumanMessage(content="earlier user message"),
        AIMessage(content="earlier assistant response"),
        ToolMessage(content="earlier tool result", tool_call_id="call-1"),
    )
    current_message = HumanMessage(content="current user message")

    messages = builder.build_request_messages(history, current_message)

    assert messages == (
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content="earlier user message"),
        AIMessage(content="earlier assistant response"),
        ToolMessage(content="earlier tool result", tool_call_id="call-1"),
        current_message,
    )


def test_build_request_messages_rejects_invalid_inputs() -> None:
    builder = ContextBuilder()

    with pytest.raises(TypeError, match="history"):
        builder.build_request_messages(("not a message",), HumanMessage(content="current"))
    with pytest.raises(TypeError, match="current_message"):
        builder.build_request_messages((), AIMessage(content="not a user message"))
