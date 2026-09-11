"""Queued message-bus orchestration for the minimal agent runtime."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from ..bus import InboundMessage, MessageBus, OutboundMessage
from ..providers import (
    BaseMessage,
    HumanMessage,
    LLMProvider,
    ProviderError,
    SystemMessage,
)
from ..session import SessionManager
from ..tools import ToolRegistry
from .context import ContextBuilder
from .runner import AgentRunner, AgentRunnerError, AgentRunResult, AgentRunSpec

logger = logging.getLogger(__name__)

_INVALID_SESSION_ID_MESSAGE = "A non-empty session ID is required."
_SESSION_LOAD_FAILURE_MESSAGE = "The conversation could not be accessed."
_SESSION_SAVE_FAILURE_MESSAGE = "The conversation could not be saved."
_PROVIDER_FAILURE_MESSAGE = "The model provider could not complete the request."
_AGENT_FAILURE_MESSAGE = "The agent could not complete the request."
_PROCESSING_FAILURE_MESSAGE = "The request could not be processed."
_MAX_ITERATIONS_MESSAGE = (
    "The agent reached the maximum iteration limit before completing the request."
)
_EMPTY_RESPONSE_MESSAGE = "The agent did not produce a final response."


class AgentLoop:
    """Queue inbound messages and persist one complete agent turn at a time."""

    def __init__(
        self,
        runner: AgentRunner,
        provider: LLMProvider,
        tool_registry: ToolRegistry,
        session_manager: SessionManager,
        context_builder: ContextBuilder,
        message_bus: MessageBus,
        max_iterations: int = 30,
    ) -> None:
        if not isinstance(runner, AgentRunner):
            raise TypeError("AgentLoop requires an AgentRunner")
        if not isinstance(provider, LLMProvider):
            raise TypeError("AgentLoop requires an LLMProvider")
        if not isinstance(tool_registry, ToolRegistry):
            raise TypeError("AgentLoop requires a ToolRegistry")
        if not isinstance(session_manager, SessionManager):
            raise TypeError("AgentLoop requires a SessionManager")
        if not isinstance(context_builder, ContextBuilder):
            raise TypeError("AgentLoop requires a ContextBuilder")
        if not isinstance(message_bus, MessageBus):
            raise TypeError("AgentLoop requires a MessageBus")
        if not isinstance(max_iterations, int) or isinstance(max_iterations, bool):
            raise TypeError("AgentLoop max_iterations must be an integer")
        if max_iterations <= 0:
            raise ValueError("AgentLoop max_iterations must be positive")

        self._runner = runner
        self._provider = provider
        self._tool_registry = tool_registry
        self._session_manager = session_manager
        self._context_builder = context_builder
        self._message_bus = message_bus
        self._max_iterations = max_iterations
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._inbound_tasks: set[asyncio.Task[None]] = set()
        self._run_task: asyncio.Task[None] | None = None
        self._closed = False

    async def run(self) -> None:
        """Continuously consume inbound messages until the task is cancelled."""

        if self._closed:
            raise RuntimeError("AgentLoop is closed")
        if self._run_task is not None:
            raise RuntimeError("AgentLoop is already running")

        logger.info("Agent loop started")
        run_task = asyncio.current_task()
        if run_task is None:
            raise RuntimeError("AgentLoop requires a running asyncio task")
        self._run_task = run_task
        try:
            while True:
                try:
                    inbound = await asyncio.wait_for(
                        self._message_bus.consume_inbound(),
                        timeout=1,
                    )
                except TimeoutError:
                    continue
                if self._closed:
                    return
                self._enqueue_inbound(inbound)
        except asyncio.CancelledError:
            logger.info("Agent loop cancelled")
            await self.close()
            raise
        except Exception:
            logger.exception("Agent loop stopped while consuming inbound messages")
            await self.close()
            raise
        finally:
            if self._run_task is run_task:
                self._run_task = None

    async def close(self) -> None:
        """Cancel and await queued inbound-message workers."""

        if self._closed:
            return

        self._closed = True
        current_task = asyncio.current_task()
        tasks = set(self._inbound_tasks)
        if self._run_task is not None and self._run_task is not current_task:
            tasks.add(self._run_task)
        tasks.discard(current_task)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._inbound_tasks.clear()

    def _enqueue_inbound(self, inbound: InboundMessage) -> None:
        """Schedule one inbound queue item without blocking bus consumption."""

        if self._closed:
            return

        task = asyncio.create_task(self._process_queued_inbound(inbound))
        self._inbound_tasks.add(task)
        task.add_done_callback(self._inbound_tasks.discard)

    async def _process_queued_inbound(self, inbound: InboundMessage) -> None:
        """Process and publish one queued inbound message."""

        try:
            outbound = await self._process_inbound(inbound)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Agent loop could not process an inbound message")
            outbound = _outbound_message(inbound, _PROCESSING_FAILURE_MESSAGE)

        try:
            await self._message_bus.publish_outbound(outbound)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Agent loop could not publish an outbound message")

    async def _process_inbound(self, inbound: InboundMessage) -> OutboundMessage:
        """Execute, persist, and format one inbound user message."""

        if not inbound.session_id.strip():
            return _outbound_message(inbound, _INVALID_SESSION_ID_MESSAGE)

        async with self._lock_for(inbound.session_id):
            try:
                session = self._session_manager.get_or_create(inbound.session_id)
            except Exception:
                logger.error("Agent loop could not load a session")
                return _outbound_message(inbound, _SESSION_LOAD_FAILURE_MESSAGE)

            history = _without_system_messages(session.messages)
            current_message = HumanMessage(content=inbound.content)
            try:
                request_messages = self._context_builder.build_request_messages(
                    history,
                    current_message,
                )
                result = await self._runner.run(
                    AgentRunSpec(
                        messages=request_messages,
                        provider=self._provider,
                        tool_registry=self._tool_registry,
                        max_iterations=self._max_iterations,
                    )
                )
                new_messages = _new_messages(result, request_messages)
            except asyncio.CancelledError:
                raise
            except ProviderError:
                logger.warning("Model provider failed during an agent run")
                return _outbound_message(inbound, _PROVIDER_FAILURE_MESSAGE)
            except AgentRunnerError:
                logger.error("Agent runner failed during an agent run")
                return _outbound_message(inbound, _AGENT_FAILURE_MESSAGE)
            except Exception:
                logger.error("Agent processing failed before a completed result")
                return _outbound_message(inbound, _PROCESSING_FAILURE_MESSAGE)

            updated_session = session.with_messages((*history, current_message, *new_messages))
            try:
                self._session_manager.save(updated_session)
            except Exception:
                logger.error("Agent loop could not save a session")
                return _outbound_message(inbound, _SESSION_SAVE_FAILURE_MESSAGE)
            if result.stop_reason == "max_iterations":
                return _outbound_message(inbound, _MAX_ITERATIONS_MESSAGE)
            if result.content is None or not result.content.strip():
                return _outbound_message(inbound, _EMPTY_RESPONSE_MESSAGE)
            return _outbound_message(inbound, result.content)

    def _lock_for(self, session_id: str) -> asyncio.Lock:
        """Return the lock that preserves one session's turn order."""

        lock = self._session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[session_id] = lock
        return lock


def _new_messages(
    result: AgentRunResult,
    request_messages: Sequence[BaseMessage],
) -> tuple[BaseMessage, ...]:
    """Return result messages created after the request prefix."""

    request_prefix = tuple(request_messages)
    if tuple(result.messages[: len(request_prefix)]) != request_prefix:
        raise AgentRunnerError("Agent run result did not retain its request messages")
    return _without_system_messages(result.messages[len(request_prefix) :])


def _without_system_messages(
    messages: Sequence[BaseMessage],
) -> tuple[BaseMessage, ...]:
    """Return persisted history without transient system prompts."""

    return tuple(message for message in messages if not isinstance(message, SystemMessage))


def _outbound_message(inbound: InboundMessage, content: str) -> OutboundMessage:
    return OutboundMessage(
        session_id=inbound.session_id,
        content=content,
        metadata=inbound.metadata,
    )
