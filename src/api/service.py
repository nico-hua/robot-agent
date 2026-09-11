"""Local aiohttp API backed by the shared message bus."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlparse

from aiohttp import web

from ..bus import InboundMessage, MessageBus, OutboundMessage
from ..config import ApiConfig
from ..providers import AIMessage, BaseMessage
from ..session import Session, SessionManager

logger = logging.getLogger(__name__)

_HTTP_REQUEST_ID_METADATA_KEY = "_http_request_id"
_MAX_REQUEST_BODY_BYTES = 1_048_576
_SHUTDOWN_TIMEOUT_SECONDS = 1.0


class _OutboundRouterStoppedError(RuntimeError):
    """Raised when a request loses the service's outbound response router."""


class HttpApiService:
    """Serve local JSON requests through the application's ``MessageBus``.

    A single outbound router consumes the shared outbound queue and uses a
    private correlation ID in message metadata to resolve the matching HTTP
    request. This prevents concurrent handlers from consuming each other's
    responses.
    """

    def __init__(
        self,
        message_bus: MessageBus,
        session_manager: SessionManager,
        config: ApiConfig,
    ) -> None:
        if not isinstance(message_bus, MessageBus):
            raise TypeError("HttpApiService requires a MessageBus")
        if not isinstance(session_manager, SessionManager):
            raise TypeError("HttpApiService requires a SessionManager")
        if not isinstance(config, ApiConfig):
            raise TypeError("HttpApiService requires an ApiConfig")

        self._message_bus = message_bus
        self._session_manager = session_manager
        self._config = config
        self._pending_responses: dict[str, asyncio.Future[OutboundMessage]] = {}
        self._outbound_router_task: asyncio.Task[None] | None = None
        self._app = web.Application(
            client_max_size=_MAX_REQUEST_BODY_BYTES,
            middlewares=(_cors_middleware, _error_middleware),
        )
        self._app.on_startup.append(self._start_outbound_router)
        self._app.on_cleanup.append(self._stop_outbound_router)
        self._app.router.add_get("/health", self._health)
        self._app.router.add_post("/v1/messages", self._post_message)
        self._app.router.add_post("/v1/sessions/clear", self._clear_session_request)
        self._app.router.add_get("/v1/sessions", self._list_sessions)
        self._app.router.add_get("/v1/sessions/{session_id}", self._get_session)
        self._runner: web.AppRunner | None = None
        self._started = False

    @property
    def app(self) -> web.Application:
        """Return the aiohttp application for embedding or local test servers."""

        return self._app

    @property
    def started(self) -> bool:
        """Return whether this service owns an active TCP listener."""

        return self._started

    @property
    def port(self) -> int | None:
        """Return the bound port, including an OS-selected test port."""

        if self._runner is None:
            return None
        for address in self._runner.addresses:
            if isinstance(address, tuple) and len(address) >= 2:
                return int(address[1])
        return None

    async def start(self) -> None:
        """Bind the configured local HTTP listener once."""

        if self._started:
            return

        runner = web.AppRunner(
            self._app,
            access_log=None,
            shutdown_timeout=_SHUTDOWN_TIMEOUT_SECONDS,
        )
        try:
            await runner.setup()
            site = web.TCPSite(
                runner,
                host=self._config.host,
                port=self._config.port,
            )
            await site.start()
        except asyncio.CancelledError:
            await runner.cleanup()
            raise
        except Exception:
            await runner.cleanup()
            logger.exception("HTTP API service failed to start")
            raise

        self._runner = runner
        self._started = True
        logger.info(
            "HTTP API service started (host=%s, port=%s)",
            self._config.host,
            self.port,
        )

    async def stop(self) -> None:
        """Stop the listener and release all pending HTTP response waiters."""

        runner = self._runner
        self._runner = None
        if runner is not None:
            await runner.cleanup()
        else:
            await self._stop_outbound_router(self._app)

        if self._started:
            logger.info("HTTP API service stopped")
        self._started = False

    async def _start_outbound_router(self, app: web.Application) -> None:
        """Start the sole outbound consumer before accepting HTTP requests."""

        del app
        if self._outbound_router_task is None or self._outbound_router_task.done():
            self._outbound_router_task = asyncio.create_task(self._route_outbound_messages())

    async def _stop_outbound_router(self, app: web.Application) -> None:
        """Stop outbound routing and fail waiters that can no longer complete."""

        del app
        task = self._outbound_router_task
        self._outbound_router_task = None
        if task is not None and not task.done():
            task.cancel()
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("HTTP outbound response router stopped with an error")
        self._fail_pending_responses()

    async def _route_outbound_messages(self) -> None:
        """Dispatch correlated outbound messages to their waiting HTTP handlers."""

        try:
            while True:
                outbound = await self._message_bus.consume_outbound()
                request_id = outbound.metadata.get(_HTTP_REQUEST_ID_METADATA_KEY)
                if not isinstance(request_id, str):
                    logger.debug("Discarded uncorrelated outbound message from the HTTP queue")
                    continue
                response_future = self._pending_responses.pop(request_id, None)
                if response_future is None or response_future.done():
                    logger.debug("Discarded late or unknown HTTP outbound response")
                    continue
                response_future.set_result(outbound)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("HTTP outbound response router failed")
            self._fail_pending_responses()
            raise

    def _fail_pending_responses(self) -> None:
        """Wake all HTTP handlers when their response router has stopped."""

        for response_future in self._pending_responses.values():
            if not response_future.done():
                response_future.set_exception(_OutboundRouterStoppedError())
        self._pending_responses.clear()

    async def _health(self, request: web.Request) -> web.Response:
        del request
        return _json_response({"status": "ok"})

    async def _post_message(self, request: web.Request) -> web.Response:
        payload = await _read_json_object(request)
        if isinstance(payload, web.Response):
            return payload

        session_id = _required_session_id(payload)
        if isinstance(session_id, web.Response):
            return session_id
        content = payload.get("content")
        if not isinstance(content, str):
            return _error_response(400, "invalid_content", "content must be a string")

        try:
            outbound = await self._publish_and_wait(session_id, content)
        except TimeoutError:
            return _error_response(504, "agent_timeout", "Agent response timed out")
        except _OutboundRouterStoppedError:
            return _error_response(
                503, "agent_unavailable", "Agent response routing is unavailable"
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Could not publish an HTTP message to the AgentLoop")
            return _error_response(502, "agent_error", "Agent failed to process the message")

        if outbound.session_id != session_id:
            return _error_response(
                502,
                "invalid_agent_response",
                "Agent returned a mismatched response",
            )
        return _json_response({"session_id": session_id, "content": outbound.content})

    async def _publish_and_wait(self, session_id: str, content: str) -> OutboundMessage:
        """Publish one input and await only its correlated outbound response."""

        router_task = self._outbound_router_task
        if router_task is None or router_task.done():
            raise _OutboundRouterStoppedError()

        request_id = uuid.uuid4().hex
        response_future: asyncio.Future[OutboundMessage] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending_responses[request_id] = response_future
        inbound = InboundMessage(
            session_id=session_id,
            content=content,
            metadata={_HTTP_REQUEST_ID_METADATA_KEY: request_id},
        )
        try:
            await self._message_bus.publish_inbound(inbound)
            return await asyncio.wait_for(
                asyncio.shield(response_future),
                timeout=self._config.request_timeout_seconds,
            )
        finally:
            pending_response = self._pending_responses.pop(request_id, None)
            if pending_response is not None and not pending_response.done():
                pending_response.cancel()

    async def _clear_session_request(self, request: web.Request) -> web.Response:
        payload = await _read_json_object(request)
        if isinstance(payload, web.Response):
            return payload

        session_id = _required_session_id(payload)
        if isinstance(session_id, web.Response):
            return session_id
        try:
            session = self._session_manager.clear_messages(session_id)
        except Exception:
            logger.exception("Could not clear an HTTP session")
            return _error_response(500, "session_clear_failed", "Session could not be cleared")
        return _json_response(
            {
                "session_id": session.key,
                "cleared": True,
                "updated_at": session.updated_at.isoformat(),
            }
        )

    async def _list_sessions(self, request: web.Request) -> web.Response:
        """Return persisted sessions ordered by their most recent update."""

        del request
        sessions = sorted(
            self._session_manager.list_sessions(),
            key=lambda session: (session.updated_at, session.key),
            reverse=True,
        )
        return _json_response({"sessions": [_session_summary(session) for session in sessions]})

    async def _get_session(self, request: web.Request) -> web.Response:
        """Return the UI-visible history for one saved session."""

        session_id = request.match_info["session_id"]
        try:
            session = self._session_manager.get(session_id)
        except (TypeError, ValueError):
            return _error_response(
                400, "invalid_session_id", "session_id must be a non-empty string"
            )
        if session is None:
            return _error_response(404, "session_not_found", "Session was not found")
        return _json_response(
            {
                "session_id": session.key,
                "updated_at": session.updated_at.isoformat(),
                "messages": _visible_message_records(session.messages),
            }
        )


@web.middleware
async def _cors_middleware(
    request: web.Request,
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> web.StreamResponse:
    """Allow local browser clients to read local API responses."""

    origin = request.headers.get("Origin")
    if request.method == "OPTIONS" and _is_local_browser_origin(origin):
        response: web.StreamResponse = web.Response(status=204)
    else:
        response = await handler(request)
    if _is_local_browser_origin(origin):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@web.middleware
async def _error_middleware(
    request: web.Request,
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> web.StreamResponse:
    """Convert framework-level request failures into the API JSON contract."""

    try:
        return await handler(request)
    except asyncio.CancelledError:
        raise
    except web.HTTPException as error:
        code, message = _framework_error_details(error.status)
        return _error_response(error.status, code, message)
    except Exception:
        logger.exception("HTTP API request failed")
        return _error_response(500, "internal_error", "Internal server error")


async def _read_json_object(request: web.Request) -> dict[str, Any] | web.Response:
    """Return a JSON object or the stable error response for invalid input."""

    try:
        payload = await request.json()
    except web.HTTPRequestEntityTooLarge:
        return _error_response(413, "payload_too_large", "Request body is too large")
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return _error_response(400, "invalid_json", "Request body must be valid JSON")
    if not isinstance(payload, dict):
        return _error_response(400, "invalid_request", "Request JSON must be an object")
    return payload


def _required_session_id(payload: dict[str, Any]) -> str | web.Response:
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        return _error_response(400, "invalid_session_id", "session_id must be a non-empty string")
    return session_id.strip()


def _json_response(payload: dict[str, Any], *, status: int = 200) -> web.Response:
    return web.json_response(
        payload,
        status=status,
        dumps=lambda value: json.dumps(value, ensure_ascii=False),
    )


def _error_response(status: int, code: str, message: str) -> web.Response:
    return _json_response({"error": {"code": code, "message": message}}, status=status)


def _framework_error_details(status: int) -> tuple[str, str]:
    return {
        404: ("not_found", "Route not found"),
        405: ("method_not_allowed", "Method not allowed"),
        413: ("payload_too_large", "Request body is too large"),
    }.get(status, ("request_error", "Request could not be processed"))


def _session_summary(session: Session) -> dict[str, Any]:
    messages = _visible_message_records(session.messages)
    return {
        "session_id": session.key,
        "updated_at": session.updated_at.isoformat(),
        "message_count": len(messages),
        "preview": _preview(messages),
    }


def _visible_message_records(messages: tuple[BaseMessage, ...]) -> list[dict[str, Any]]:
    """Expose user and assistant messages while hiding transient tool state."""

    records: list[dict[str, Any]] = []
    for message in messages:
        if message.role not in {"user", "assistant"}:
            continue
        record: dict[str, Any] = {"role": message.role, "content": message.content}
        if isinstance(message, AIMessage) and message.tool_calls:
            record["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": dict(tool_call.arguments),
                }
                for tool_call in message.tool_calls
            ]
        records.append(record)
    return records


def _preview(messages: list[dict[str, Any]], *, limit: int = 120) -> str:
    for message in reversed(messages):
        content_value = message.get("content")
        if not isinstance(content_value, str):
            continue
        content = " ".join(content_value.split())
        if content:
            return content if len(content) <= limit else f"{content[: limit - 1]}…"
    return ""


def _is_local_browser_origin(origin: str | None) -> bool:
    if not isinstance(origin, str):
        return False
    parsed = urlparse(origin)
    return parsed.scheme in {"http", "https"} and parsed.hostname in {
        "localhost",
        "127.0.0.1",
        "::1",
    }
