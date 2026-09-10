"""Tests for core tool types and provider-facing schemas."""

from __future__ import annotations

from typing import Any

import pytest
from src.tools.base import Tool, ToolParameter, ToolResult
from src.tools.context import ToolContext


class _SchemaTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="search_records",
            description="Search records by a query.",
            parameters=(
                ToolParameter(
                    name="query",
                    description="Text to search for.",
                    type="string",
                    required=True,
                ),
                ToolParameter(
                    name="limit",
                    description="Maximum number of records.",
                    type="integer",
                ),
                ToolParameter(
                    name="include_archived",
                    description="Whether archived records are included.",
                    type="boolean",
                ),
            ),
        )

    async def execute(self, **arguments: Any) -> ToolResult:
        return ToolResult(content=str(arguments))


class _NoArgumentTool(Tool):
    def __init__(self) -> None:
        super().__init__(name="no_arguments", description="A tool without parameters.")

    async def execute(self, **arguments: Any) -> ToolResult:
        return ToolResult(content="done")


class _MinimalTool(Tool):
    async def execute(self, **arguments: Any) -> ToolResult:
        return ToolResult(content="done")


@pytest.mark.parametrize(
    ("name", "description", "parameter_type", "required", "error_type", "match"),
    [
        pytest.param(" ", "Description", "string", False, ValueError, "name", id="blank-name"),
        pytest.param(
            "name", " ", "string", False, ValueError, "description", id="blank-description"
        ),
        pytest.param(
            "name", "Description", "array", False, ValueError, "Unsupported", id="bad-type"
        ),
        pytest.param(
            "name", "Description", "string", 1, TypeError, "boolean", id="non-bool-required"
        ),
    ],
)
def test_tool_parameter_rejects_invalid_metadata(
    name: object,
    description: object,
    parameter_type: object,
    required: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        ToolParameter(
            name=name,
            description=description,
            type=parameter_type,
            required=required,
        )


@pytest.mark.parametrize(
    ("name", "description", "parameters", "error_type", "match"),
    [
        pytest.param("", "Description", (), ValueError, "name", id="blank-name"),
        pytest.param("name", "", (), ValueError, "description", id="blank-description"),
        pytest.param("name", "Description", None, TypeError, "sequence", id="non-sequence"),
        pytest.param(
            "name",
            "Description",
            (ToolParameter("value", "A value.", "string"), "invalid"),
            TypeError,
            "ToolParameter",
            id="non-parameter-member",
        ),
        pytest.param(
            "name",
            "Description",
            (
                ToolParameter("value", "A value.", "string"),
                ToolParameter("value", "Another value.", "integer"),
            ),
            ValueError,
            "unique",
            id="duplicate-parameter-name",
        ),
    ],
)
def test_tool_rejects_invalid_definition(
    name: object,
    description: object,
    parameters: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        _MinimalTool(name=name, description=description, parameters=parameters)


def test_tool_generates_json_schema_and_openai_function_schema() -> None:
    tool = _SchemaTool()

    assert tool.parameters_schema == {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Text to search for."},
            "limit": {"type": "integer", "description": "Maximum number of records."},
            "include_archived": {
                "type": "boolean",
                "description": "Whether archived records are included.",
            },
        },
        "required": ["query"],
    }
    assert tool.to_openai_tool() == {
        "type": "function",
        "function": {
            "name": "search_records",
            "description": "Search records by a query.",
            "parameters": tool.parameters_schema,
        },
    }


def test_tool_default_lifecycle_creates_parameterless_tool() -> None:
    context = ToolContext()

    assert _NoArgumentTool.enabled(context) is True
    assert isinstance(_NoArgumentTool.create(context), _NoArgumentTool)
