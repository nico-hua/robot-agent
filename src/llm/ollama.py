"""Ollama adapter for the project's minimal non-streaming text interface."""

from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import httpx
from dotenv import load_dotenv
from ollama import AsyncClient, RequestError, ResponseError

from src.llm.base import LLMClient, ModelError, ModelResponse
from src.messages.message import Message

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3.5:4b"
DEFAULT_TIMEOUT = 60.0
_PROJECT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class _AsyncChatTransport(Protocol):
    async def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, str]],
        stream: bool,
        think: bool,
    ) -> Any:
        """Send a non-streaming chat request."""

    async def close(self) -> None:
        """Release transport resources."""


ClientFactory = Callable[..., _AsyncChatTransport]


@dataclass(frozen=True, slots=True)
class OllamaConfig:
    """Configuration for a local Ollama text chat request."""

    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout: float = DEFAULT_TIMEOUT
    stream: bool = False
    think: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("Ollama base_url must be a non-empty string")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("Ollama model must be a non-empty string")
        if (
            not isinstance(self.timeout, (int, float))
            or isinstance(self.timeout, bool)
            or self.timeout <= 0
        ):
            raise ValueError("Ollama timeout must be a positive number")
        if not isinstance(self.stream, bool):
            raise TypeError("Ollama stream must be a boolean")
        if not isinstance(self.think, bool):
            raise TypeError("Ollama think must be a boolean")

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "OllamaConfig":
        """Load root `.env` values, with process environment taking priority."""

        if environ is None:
            load_dotenv(dotenv_path=_PROJECT_ENV_FILE, override=False)
            environ = os.environ

        return cls(
            base_url=environ.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL),
            model=environ.get("OLLAMA_MODEL", DEFAULT_MODEL),
            timeout=_parse_timeout(environ.get("OLLAMA_TIMEOUT")),
            stream=_parse_bool(environ.get("OLLAMA_STREAM"), default=False),
            think=_parse_bool(environ.get("OLLAMA_THINK"), default=False),
        )


class OllamaClient(LLMClient):
    """Translate project messages to the official asynchronous Ollama client."""

    def __init__(
        self,
        config: OllamaConfig | None = None,
        *,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self.config = config if config is not None else OllamaConfig.from_env()
        self._client_factory = client_factory
        self._client: _AsyncChatTransport | None = None

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        timeout: float | None = None,
    ) -> ModelResponse:
        """Call `/api/chat` once and normalize its response or failure."""

        if self.config.stream:
            return _failure(
                code="stream_not_supported",
                message="Streaming is not supported by this text chat adapter yet.",
            )

        try:
            request_timeout = _resolve_timeout(timeout, self.config.timeout)
            payload = [_message_payload(message) for message in messages]
        except (TypeError, ValueError):
            return _failure(
                code="invalid_request",
                message="Messages or timeout are invalid for this model request.",
            )

        try:
            raw = await asyncio.wait_for(
                self._get_client().chat(
                    model=self.config.model,
                    messages=payload,
                    stream=False,
                    think=self.config.think,
                ),
                timeout=request_timeout,
            )
        except asyncio.CancelledError:
            raise
        except (asyncio.TimeoutError, httpx.TimeoutException):
            return _failure(
                code="timeout",
                message="The Ollama request timed out.",
                retryable=True,
            )
        except (ConnectionError, httpx.NetworkError):
            return _failure(
                code="service_unavailable",
                message="The Ollama service is unavailable.",
                retryable=True,
            )
        except ResponseError as error:
            return _failure(
                code="http_error",
                message="Ollama returned an HTTP error.",
                retryable=error.status_code >= 500,
                status_code=error.status_code,
            )
        except RequestError:
            return _failure(
                code="request_error",
                message="Ollama rejected the model request.",
            )
        except Exception:
            return _failure(
                code="request_failed",
                message="The Ollama request failed.",
            )

        try:
            return _normalize_response(raw)
        except (KeyError, TypeError, ValueError):
            return _failure(
                code="invalid_response",
                message="Ollama returned an invalid chat response.",
            )

    async def aclose(self) -> None:
        """Close the lazily created SDK client, if any."""

        if self._client is None:
            return

        close = getattr(self._client, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result
        self._client = None

    def _get_client(self) -> _AsyncChatTransport:
        if self._client is None:
            if self._client_factory is not None:
                self._client = self._client_factory(
                    host=self.config.base_url,
                    timeout=self.config.timeout,
                )
            else:
                self._client = AsyncClient(
                    host=self.config.base_url,
                    timeout=self.config.timeout,
                )
        return self._client


def _message_payload(message: Message) -> dict[str, str]:
    if not isinstance(message, Message):
        raise TypeError("messages must contain Message instances")
    return message.to_dict()


def _resolve_timeout(timeout: float | None, default: float) -> float:
    if timeout is None:
        return default
    if (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or timeout <= 0
    ):
        raise ValueError("timeout must be a positive number")
    return float(timeout)


def _parse_timeout(value: str | None) -> float:
    if value is None:
        return DEFAULT_TIMEOUT
    try:
        timeout = float(value)
    except ValueError as error:
        raise ValueError("OLLAMA_TIMEOUT must be a positive number") from error
    if timeout <= 0:
        raise ValueError("OLLAMA_TIMEOUT must be a positive number")
    return timeout


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("Ollama boolean environment values must be true or false")


def _normalize_response(response: Any) -> ModelResponse:
    raw = _raw_mapping(response)
    message = raw.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("response message must be a mapping")

    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("response content must be a string")

    thinking = message.get("thinking")
    if thinking is not None and not isinstance(thinking, str):
        raise ValueError("response thinking must be a string")

    finish_reason = raw.get("done_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise ValueError("response done_reason must be a string")

    return ModelResponse(
        content=content,
        finish_reason=finish_reason,
        thinking=thinking,
        usage=_extract_usage(raw),
        raw_response=raw,
    )


def _raw_mapping(response: Any) -> dict[str, Any]:
    if isinstance(response, Mapping):
        return dict(response)

    model_dump = getattr(response, "model_dump", None)
    if not callable(model_dump):
        raise TypeError("response must provide model_dump")
    raw = model_dump(mode="json", exclude_none=True)
    if not isinstance(raw, Mapping):
        raise TypeError("response model_dump must return a mapping")
    return dict(raw)


def _extract_usage(raw: Mapping[str, Any]) -> dict[str, int | float] | None:
    usage: dict[str, int | float] = {}
    prompt_tokens = _numeric_value(raw.get("prompt_eval_count"))
    completion_tokens = _numeric_value(raw.get("eval_count"))

    if prompt_tokens is not None:
        usage["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        usage["completion_tokens"] = completion_tokens
    if prompt_tokens is not None and completion_tokens is not None:
        usage["total_tokens"] = prompt_tokens + completion_tokens

    for source_name, usage_name in (
        ("total_duration", "total_duration_ns"),
        ("load_duration", "load_duration_ns"),
        ("prompt_eval_duration", "prompt_eval_duration_ns"),
        ("eval_duration", "eval_duration_ns"),
    ):
        value = _numeric_value(raw.get(source_name))
        if value is not None:
            usage[usage_name] = value

    return usage or None


def _numeric_value(value: Any) -> int | float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return None


def _failure(
    *,
    code: str,
    message: str,
    retryable: bool = False,
    status_code: int | None = None,
) -> ModelResponse:
    return ModelResponse(
        error=ModelError(
            code=code,
            message=message,
            retryable=retryable,
            status_code=status_code,
        )
    )


__all__ = ["OllamaClient", "OllamaConfig"]
