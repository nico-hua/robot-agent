"""Command-line entry point for the configured robot-agent application."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from collections.abc import Callable, Sequence

from .application import Application

logger = logging.getLogger(__name__)

ApplicationFactory = Callable[[], Application]
SignalInstaller = Callable[[asyncio.AbstractEventLoop, Application], Callable[[], None]]


def main(
    argv: Sequence[str] | None = None,
    *,
    application_factory: ApplicationFactory = Application.from_environment,
    signal_installer: SignalInstaller | None = None,
) -> int:
    """Load ``.env`` configuration and run the application until it stops."""

    _create_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        application = application_factory()
        asyncio.run(_run_application(application, signal_installer or _install_signal_handlers))
    except KeyboardInterrupt:
        logger.info("Application interrupted")
        return 0
    except Exception:
        logger.exception("Application failed to start or run")
        return 1
    return 0


async def _run_application(
    application: Application,
    signal_installer: SignalInstaller,
) -> None:
    """Run one application while mapping process signals to a stop request."""

    uninstall = _noop
    try:
        uninstall = signal_installer(asyncio.get_running_loop(), application)
        await application.run()
    finally:
        try:
            uninstall()
        except Exception:
            logger.exception("Failed to remove application signal handlers")
        await application.close()


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run robot-agent using configuration from the root .env file."
    )
    parser.add_argument("--version", action="version", version="robot-agent 0.1.0")
    return parser


def _install_signal_handlers(
    loop: asyncio.AbstractEventLoop,
    application: Application,
) -> Callable[[], None]:
    """Install SIGINT/SIGTERM handlers with a Windows-compatible fallback."""

    removers: list[Callable[[], None]] = []
    for stop_signal in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(stop_signal, application.request_stop)
        except (NotImplementedError, RuntimeError):
            previous_handler = signal.getsignal(stop_signal)

            def request_stop(_signum: int, _frame: object) -> None:
                application.request_stop()

            signal.signal(stop_signal, request_stop)
            removers.append(
                lambda stop_signal=stop_signal, previous_handler=previous_handler: signal.signal(
                    stop_signal,
                    previous_handler,
                )
            )
        else:
            removers.append(lambda stop_signal=stop_signal: loop.remove_signal_handler(stop_signal))

    def uninstall() -> None:
        for remove in reversed(removers):
            remove()

    return uninstall


def _noop() -> None:
    """Provide a no-op signal cleanup callback before handlers are installed."""
