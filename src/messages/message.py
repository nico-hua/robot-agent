"""Provider-independent text conversation message definitions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

MessageRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    """A provider-independent text conversation message."""

    role: MessageRole
    content: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.role, str):
            raise TypeError("message role must be a string")
        if self.role not in ("system", "user", "assistant"):
            raise ValueError("message role must be system, user, or assistant")
        if not isinstance(self.content, str):
            raise TypeError("message content must be a string")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("message metadata must be a mapping")

    def to_dict(self) -> dict[str, str]:
        """Return only the portable text fields accepted by a chat provider."""

        return {"role": self.role, "content": self.content}
