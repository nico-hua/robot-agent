"""Assembly and lifecycle management for the minimal HTTP agent application."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path

from ..agent.context import ContextBuilder
from ..agent.loop import AgentLoop
from ..agent.runner import AgentRunner
from ..api.service import HttpApiService
from ..bus import MessageBus
from ..config import (
    AgentConfig,
    ApiConfig,
    ProviderConfig,
    load_agent_config,
)
from ..providers import LLMProvider, create_default_provider_factory
from ..session import SessionManager
from ..tools import ToolRegistry

logger = logging.getLogger(__name__)

ProviderFactory = Callable[[ProviderConfig], LLMProvider]
AgentLoopFactory = Callable[
    [AgentRunner, LLMProvider, ToolRegistry, SessionManager, ContextBuilder, MessageBus],
    AgentLoop,
]
ApiServiceFactory = Callable[
    [MessageBus, SessionManager, ApiConfig],
    HttpApiService,
]


class Application:
    """Assemble and run the currently supported HTTP agent components."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        provider_factory: ProviderFactory | None = None,
        agent_loop_factory: AgentLoopFactory | None = None,
        api_service_factory: ApiServiceFactory | None = None,
    ) -> None:
        if not isinstance(config, AgentConfig):
            raise TypeError("Application requires an AgentConfig")

        self._config = config
        self._workspace = Path(config.workspace_path).expanduser().resolve()
        provider_factory = provider_factory or create_default_provider_factory().create
        agent_loop_factory = agent_loop_factory or _create_agent_loop
        api_service_factory = api_service_factory or _create_http_api_service

        self._message_bus = MessageBus()
        self._provider = provider_factory(config.provider)
        self._session_manager = SessionManager(self._workspace)
        self._tool_registry = ToolRegistry()
        self._agent_loop = agent_loop_factory(
            AgentRunner(),
            self._provider,
            self._tool_registry,
            self._session_manager,
            ContextBuilder(),
            self._message_bus,
        )
        self._api_service = api_service_factory(
            self._message_bus,
            self._session_manager,
            config.api,
        )
        self._agent_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._close_lock = asyncio.Lock()
        self._started = False
        self._closed = False

    @classmethod
    def from_environment(cls) -> Application:
        """Create the application from the root ``.env`` and process environment."""

        return cls(load_agent_config())

    @property
    def config(self) -> AgentConfig:
        """Return the root configuration used to assemble this application."""

        return self._config

    @property
    def message_bus(self) -> MessageBus:
        """Return the bus shared by HTTP requests and the AgentLoop."""

        return self._message_bus

    @property
    def session_manager(self) -> SessionManager:
        """Return the session manager shared by the loop and HTTP API."""

        return self._session_manager

    @property
    def agent_loop(self) -> AgentLoop:
        """Return the AgentLoop managed by this application."""

        return self._agent_loop

    @property
    def api_service(self) -> HttpApiService:
        """Return the HTTP API service managed by this application."""

        return self._api_service

    @property
    def agent_task(self) -> asyncio.Task[None] | None:
        """Return the AgentLoop task while it is managed by the application."""

        return self._agent_task

    async def run(self) -> None:
        """Run until a stop request, cancellation, or AgentLoop failure."""

        await self.start()
        stop_task = asyncio.create_task(self._stop_event.wait())
        try:
            agent_task = self._require_agent_task()
            done, _ = await asyncio.wait(
                (stop_task, agent_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if agent_task in done:
                agent_task.result()
                raise RuntimeError("AgentLoop stopped unexpectedly")
        finally:
            if not stop_task.done():
                stop_task.cancel()
                try:
                    await stop_task
                except asyncio.CancelledError:
                    pass
            await self.close()

    async def start(self) -> None:
        """Start the AgentLoop before accepting HTTP requests."""

        if self._closed:
            raise RuntimeError("Application has already been closed")
        if self._started:
            return

        logger.info("Application starting")
        self._agent_task = asyncio.create_task(self._agent_loop.run())
        try:
            # Let the consumer enter its run method before exposing the HTTP
            # listener, and propagate failures that happen during startup.
            await asyncio.sleep(0)
            agent_task = self._require_agent_task()
            if agent_task.done():
                agent_task.result()
            await self._api_service.start()
        except asyncio.CancelledError:
            await self.close()
            raise
        except Exception:
            logger.exception("Application failed to start")
            await self.close()
            raise

        self._started = True
        logger.info("Application started")

    def request_stop(self) -> None:
        """Request that :meth:`run` starts application shutdown."""

        self._stop_event.set()

    async def close(self) -> None:
        """Stop HTTP intake and cancel the AgentLoop exactly once."""

        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._stop_event.set()
            logger.info("Application stopping")
            await self._stop_http_service()
            await self._cancel_agent_task()
            await self._close_agent_loop()
            self._started = False
            logger.info("Application stopped")

    async def _stop_http_service(self) -> None:
        try:
            await self._api_service.stop()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Application failed while stopping the HTTP API service")

    async def _cancel_agent_task(self) -> None:
        task = self._agent_task
        self._agent_task = None
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _close_agent_loop(self) -> None:
        try:
            await self._agent_loop.close()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Application failed while closing the AgentLoop")

    def _require_agent_task(self) -> asyncio.Task[None]:
        if self._agent_task is None:
            raise RuntimeError("Application has not started an AgentLoop task")
        return self._agent_task


def _create_agent_loop(
    runner: AgentRunner,
    provider: LLMProvider,
    tool_registry: ToolRegistry,
    session_manager: SessionManager,
    context_builder: ContextBuilder,
    message_bus: MessageBus,
) -> AgentLoop:
    return AgentLoop(
        runner=runner,
        provider=provider,
        tool_registry=tool_registry,
        session_manager=session_manager,
        context_builder=context_builder,
        message_bus=message_bus,
    )


def _create_http_api_service(
    message_bus: MessageBus,
    session_manager: SessionManager,
    api_config: ApiConfig,
) -> HttpApiService:
    return HttpApiService(
        message_bus,
        session_manager,
        api_config,
    )
