"""Tests for the configuration-backed CLI entry points."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src.cli.main import main


class _FakeApplication:
    """Record CLI lifecycle operations without starting real components."""

    def __init__(self) -> None:
        self.run_calls = 0
        self.close_calls = 0
        self.stop_requests = 0

    async def run(self) -> None:
        self.run_calls += 1

    async def close(self) -> None:
        self.close_calls += 1

    def request_stop(self) -> None:
        self.stop_requests += 1


def test_main_runs_the_injected_application_and_cleans_up_handlers() -> None:
    application = _FakeApplication()
    uninstall_calls = 0

    def install_signal_handlers(_loop: object, _application: object):
        def uninstall() -> None:
            nonlocal uninstall_calls
            uninstall_calls += 1

        return uninstall

    result = main(
        [],
        application_factory=lambda: application,
        signal_installer=install_signal_handlers,
    )

    assert result == 0
    assert application.run_calls == 1
    assert application.close_calls == 1
    assert uninstall_calls == 1


def test_main_returns_nonzero_when_application_construction_fails() -> None:
    def fail_to_create_application() -> _FakeApplication:
        raise RuntimeError("configuration failure")

    assert main([], application_factory=fail_to_create_application) == 1


def test_python_module_help_reaches_the_cli_without_loading_environment_configuration() -> None:
    project_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [sys.executable, "-m", "src", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "Run robot-agent" in completed.stdout
