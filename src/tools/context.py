"""Shared dependencies used while creating tools."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolContext:
    """Dependencies available to tool factories.

    Workspace is optional so tools can decide whether they are enabled for a
    particular agent configuration. Runtime-owned services are injected here
    while the application assembles built-in tools.
    """
