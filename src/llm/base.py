"""Provider-independent interfaces and value objects for text models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.messages.message import Message


@dataclass(frozen=True, slots=True)
class ModelError:
    """A stable, safe-to-display description of a model request failure."""

    code: str
    message: str
    retryable: bool = False
    status_code: int | None = None


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """A provider-independent response to a non-streaming text request."""

    content: str = ""
    finish_reason: str | None = None
    thinking: str | None = None
    usage: Mapping[str, int | float] | None = None
    raw_response: Mapping[str, Any] | None = None
    error: ModelError | None = None

    @property
    def is_success(self) -> bool:
        """Whether the request completed without an adapter error."""

        return self.error is None


class LLMClient(ABC):
    """The minimal asynchronous interface implemented by text model adapters."""

    @abstractmethod
    async def chat(
        self,
        messages: Sequence[Message],
        *,
        timeout: float | None = None,
    ) -> ModelResponse:
        """Return one non-streaming text response for the supplied messages."""


__all__ = ["LLMClient", "ModelError", "ModelResponse"]
