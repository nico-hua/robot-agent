"""Tests for deterministic discovery and loading of built-in tools."""

from __future__ import annotations

import pkgutil
from types import ModuleType
from typing import Any

import pytest
import src.tools.loader as tool_loader_module
from src.tools.base import Tool, ToolResult
from src.tools.context import ToolContext
from src.tools.registry import ToolRegistry


class _EnabledTool(Tool):
    enabled_calls = 0
    create_calls = 0

    def __init__(self) -> None:
        super().__init__(name="loaded", description="An enabled discovered tool.")

    @classmethod
    def enabled(cls, context: ToolContext) -> bool:
        cls.enabled_calls += 1
        return True

    @classmethod
    def create(cls, context: ToolContext) -> "_EnabledTool":
        cls.create_calls += 1
        return cls()

    async def execute(self, **arguments: Any) -> ToolResult:
        return ToolResult(content="done")


class _DisabledTool(Tool):
    enabled_calls = 0
    create_calls = 0

    def __init__(self) -> None:
        super().__init__(name="disabled", description="A disabled discovered tool.")

    @classmethod
    def enabled(cls, context: ToolContext) -> bool:
        cls.enabled_calls += 1
        return False

    @classmethod
    def create(cls, context: ToolContext) -> "_DisabledTool":
        cls.create_calls += 1
        return cls()

    async def execute(self, **arguments: Any) -> ToolResult:
        return ToolResult(content="done")


class _DuplicateTool(Tool):
    create_calls = 0

    def __init__(self) -> None:
        super().__init__(name="existing", description="A duplicate discovered tool.")

    @classmethod
    def create(cls, context: ToolContext) -> "_DuplicateTool":
        cls.create_calls += 1
        return cls()

    async def execute(self, **arguments: Any) -> ToolResult:
        return ToolResult(content="done")


def _concrete_tool_class(class_name: str, module_name: str) -> type[Tool]:
    async def execute(self: Tool, **arguments: Any) -> ToolResult:
        return ToolResult(content="done")

    return type(
        class_name,
        (Tool,),
        {"__module__": module_name, "execute": execute},
    )


def test_tool_loader_discovers_concrete_classes_in_stable_class_name_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_name = tool_loader_module._BUILTIN_PACKAGE
    package = ModuleType(package_name)
    package.__path__ = ["in-memory-tools"]
    alpha_module = ModuleType(f"{package_name}.alpha")
    zulu_module = ModuleType(f"{package_name}.zulu")

    zulu_tool = _concrete_tool_class("ZuluTool", alpha_module.__name__)
    alpha_tool = _concrete_tool_class("AlphaTool", zulu_module.__name__)
    abstract_tool = type("AbstractTool", (Tool,), {"__module__": alpha_module.__name__})
    foreign_tool = _concrete_tool_class("ForeignTool", "other.module")
    alpha_module.ZuluTool = zulu_tool
    alpha_module.AbstractTool = abstract_tool
    alpha_module.ForeignTool = foreign_tool
    alpha_module.Tool = Tool
    zulu_module.AlphaTool = alpha_tool

    modules = {
        package_name: package,
        alpha_module.__name__: alpha_module,
        zulu_module.__name__: zulu_module,
    }
    imported: list[str] = []

    def fake_import_module(name: str) -> ModuleType:
        imported.append(name)
        return modules[name]

    module_infos = [
        pkgutil.ModuleInfo(None, f"{package_name}.zulu", False),
        pkgutil.ModuleInfo(None, f"{package_name}._private", False),
        pkgutil.ModuleInfo(None, f"{package_name}.alpha", False),
    ]
    monkeypatch.setattr(tool_loader_module.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(
        tool_loader_module.pkgutil,
        "iter_modules",
        lambda *args, **kwargs: module_infos,
    )

    discovered = tool_loader_module.ToolLoader()._discover_tool_classes()

    assert discovered == (alpha_tool, zulu_tool)
    assert imported == [package_name, alpha_module.__name__, zulu_module.__name__]


def test_tool_loader_rejects_a_builtin_module_without_package_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = ModuleType(tool_loader_module._BUILTIN_PACKAGE)
    monkeypatch.setattr(
        tool_loader_module.importlib,
        "import_module",
        lambda name: package,
    )

    with pytest.raises(ValueError, match="is not a package"):
        tool_loader_module.ToolLoader()._discover_tool_classes()


def test_tool_loader_registers_enabled_tools_and_skips_existing_or_disabled_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _EnabledTool.enabled_calls = 0
    _EnabledTool.create_calls = 0
    _DisabledTool.enabled_calls = 0
    _DisabledTool.create_calls = 0
    _DuplicateTool.create_calls = 0
    existing = _DuplicateTool()
    registry = ToolRegistry([existing])
    loader = tool_loader_module.ToolLoader()
    monkeypatch.setattr(
        loader,
        "_discover_tool_classes",
        lambda: (_DisabledTool, _EnabledTool, _DuplicateTool),
    )

    registered = loader.load(registry, ToolContext())

    assert registered == ("loaded",)
    assert registry.tools == (existing, registry.get("loaded"))
    assert _DisabledTool.enabled_calls == 1
    assert _DisabledTool.create_calls == 0
    assert _EnabledTool.enabled_calls == 1
    assert _EnabledTool.create_calls == 1
    assert _DuplicateTool.create_calls == 1


@pytest.mark.parametrize(
    ("registry", "context", "match"),
    [
        pytest.param(object(), ToolContext(), "ToolRegistry", id="invalid-registry"),
        pytest.param(ToolRegistry(), object(), "ToolContext", id="invalid-context"),
    ],
)
def test_tool_loader_rejects_invalid_load_dependencies(
    registry: object,
    context: object,
    match: str,
) -> None:
    with pytest.raises(TypeError, match=match):
        tool_loader_module.ToolLoader().load(registry, context)
