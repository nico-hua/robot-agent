"""Route-aware messages exchanged through the in-memory message bus."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class InboundMessage:
    """One user message waiting for agent processing."""

    session_id: str
    content: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str):
            raise TypeError("session_id must be a string")
        if not isinstance(self.content, str):
            raise TypeError("InboundMessage content must be a string")
        _validate_metadata(self.metadata)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class OutboundMessage:
    """One final agent response ready for delivery to a channel."""

    session_id: str
    content: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str):
            raise TypeError("session_id must be a string")
        if not isinstance(self.content, str):
            raise TypeError("OutboundMessage content must be a string")
        _validate_metadata(self.metadata)
        object.__setattr__(self, "metadata", dict(self.metadata))


def _validate_metadata(metadata: Mapping[str, Any]) -> None:
    if not isinstance(metadata, Mapping):
        raise TypeError("message metadata must be a mapping")
