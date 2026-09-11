"""Builtin tool that returns the local robot head-camera image path.

This initial implementation deliberately does not control a camera device. It
provides the agreed local image path so the existing tool-result image flow can
be exercised without a hardware dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..base import Tool, ToolResult

DEFAULT_CAPTURE_CAMERA_IMAGE_PATH = Path("./workspace/pictures/test.jpg")


class CaptureCameraTool(Tool):
    """Return one local image associated with the robot head camera."""

    def __init__(self) -> None:
        super().__init__(
            name="capture_camera",
            description="Return the robot head camera image as one local image path.",
        )

    async def execute(self, **arguments: Any) -> ToolResult:
        """Return the configured local sample image without accessing hardware."""

        return ToolResult(
            content="Robot head camera image path returned.",
            image_path=DEFAULT_CAPTURE_CAMERA_IMAGE_PATH,
        )
