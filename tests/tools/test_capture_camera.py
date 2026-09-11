"""Tests for the built-in camera capture placeholder tool."""

from __future__ import annotations

import asyncio
from pathlib import Path

from src.tools.builtin.capture_camera import (
    DEFAULT_CAPTURE_CAMERA_IMAGE_PATH,
    CaptureCameraTool,
)


def test_capture_camera_has_a_parameterless_tool_schema() -> None:
    """The placeholder camera capture tool exposes no model-call arguments."""

    tool = CaptureCameraTool()

    assert tool.name == "capture_camera"
    assert tool.parameters == ()
    assert tool.parameters_schema == {"type": "object", "properties": {}}


def test_capture_camera_returns_default_image_path_without_reading_file(
    monkeypatch,
) -> None:
    """Execution returns the configured local path without touching an image file."""

    def unexpected_file_access(*args: object, **kwargs: object) -> object:
        raise AssertionError("CaptureCameraTool must not read the image file")

    monkeypatch.setattr(Path, "read_bytes", unexpected_file_access)
    monkeypatch.setattr(Path, "open", unexpected_file_access)

    result = asyncio.run(CaptureCameraTool().execute())

    assert result.success is True
    assert result.error is None
    assert result.image_path == Path("./workspace/pictures/test.jpg")
    assert DEFAULT_CAPTURE_CAMERA_IMAGE_PATH == Path("./workspace/pictures/test.jpg")
