"""Tests for validated, offline tool registration and execution."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest
from src.tools.base import Tool, ToolParameter, ToolResult
from src.tools.registry import ToolRegistry


class _RecordingTool(Tool):
    def __init__(self, name: str = "inspect") -> None:
        super().__init__(
            name=name,
            description=f"Inspect values for {name}.",
            parameters=(
                ToolParameter("text", "Text to inspect.", "string", required=True),
                ToolParameter("count", "An optional count.", "integer"),
                ToolParameter("weight", "An optional weight.", "number"),
                ToolParameter("enabled", "An optional flag.", "boolean"),
            ),
        )
        self.calls: list[dict[str, Any]] = []

    async def execute(self, **arguments: Any) -> ToolResult:
        self.calls.append(arguments)
        return ToolResult(content="completed")


class _FailingTool(_RecordingTool):
    async def execute(self, **arguments: Any) -> ToolResult:
        raise RuntimeError("test failure")


class _InvalidResultTool(_RecordingTool):
    async def execute(self, **arguments: Any) -> ToolResult:
        return cast(ToolResult, "not-a-tool-result")


class _CancelledTool(_RecordingTool):
    async def execute(self, **arguments: Any) -> ToolResult:
        raise asyncio.CancelledError


def test_registry_preserves_order_and_manages_registered_tools() -> None:
    first = _RecordingTool("first")
    second = _RecordingTool("second")
    registry = ToolRegistry([first, second])

    assert registry.tools == (first, second)
    assert registry.schemas == (
        {
            "name": "first",
            "description": "Inspect values for first.",
            "parameters": first.parameters_schema,
        },
        {
            "name": "second",
            "description": "Inspect values for second.",
            "parameters": second.parameters_schema,
        },
    )
    assert registry.get("first") is first
    assert registry.has("second") is True
    assert registry.remove("first") is first
    assert registry.get("first") is None
    assert registry.remove("missing") is None

    with pytest.raises(ValueError, match="already registered"):
        registry.register(_RecordingTool("second"))
    with pytest.raises(TypeError, match="Only Tool instances"):
        registry.register(cast(Tool, object()))


def test_registry_executes_validated_tool_call() -> None:
    tool = _RecordingTool()
    registry = ToolRegistry([tool])

    result = asyncio.run(
        registry.execute(
            "inspect",
            {"text": "hello", "count": 2, "weight": 1.5, "enabled": False},
        )
    )

    assert result == ToolResult(content="completed")
    assert tool.calls == [{"text": "hello", "count": 2, "weight": 1.5, "enabled": False}]


@pytest.mark.parametrize(
    ("arguments", "expected_error"),
    [
        pytest.param(
            [],
            "Tool arguments for inspect must be an object",
            id="non-object",
        ),
        pytest.param({}, "Missing required parameter for inspect: text", id="missing-required"),
        pytest.param(
            {"text": "hello", "unknown": "value"},
            "Unknown parameter for inspect: unknown",
            id="unknown-parameter",
        ),
        pytest.param(
            {"text": 3}, "Invalid string parameter for inspect: text", id="invalid-string"
        ),
        pytest.param(
            {"text": "hello", "count": True},
            "Invalid integer parameter for inspect: count",
            id="boolean-is-not-integer",
        ),
        pytest.param(
            {"text": "hello", "weight": True},
            "Invalid number parameter for inspect: weight",
            id="boolean-is-not-number",
        ),
        pytest.param(
            {"text": "hello", "enabled": 1},
            "Invalid boolean parameter for inspect: enabled",
            id="integer-is-not-boolean",
        ),
    ],
)
def test_registry_rejects_invalid_arguments_before_execution(
    arguments: object,
    expected_error: str,
) -> None:
    tool = _RecordingTool()
    registry = ToolRegistry([tool])

    result = asyncio.run(registry.execute("inspect", arguments))

    assert result.success is False
    assert result.error == expected_error
    assert result.content == f"Error: {expected_error}"
    assert tool.calls == []


def test_registry_returns_structured_error_for_unknown_tool() -> None:
    result = asyncio.run(ToolRegistry().execute("missing", {}))

    assert result == ToolResult(
        content="Error: Unknown tool: missing",
        success=False,
        error="Unknown tool: missing",
    )


@pytest.mark.parametrize(
    ("tool", "expected_error"),
    [
        pytest.param(
            _FailingTool("failing"),
            "Tool execution failed: failing (test failure)",
            id="tool-exception",
        ),
        pytest.param(
            _InvalidResultTool("invalid"),
            "Tool returned an invalid result: invalid",
            id="invalid-tool-result",
        ),
    ],
)
def test_registry_normalizes_tool_execution_failures(
    tool: Tool,
    expected_error: str,
) -> None:
    result = asyncio.run(ToolRegistry([tool]).execute(tool.name, {"text": "hello"}))

    assert result.success is False
    assert result.error == expected_error
    assert result.content == f"Error: {expected_error}"


def test_registry_propagates_cancelled_execution() -> None:
    registry = ToolRegistry([_CancelledTool("cancelled")])

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(registry.execute("cancelled", {"text": "hello"}))
