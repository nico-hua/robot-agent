"""High-level lifecycle operations for persisted sessions."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ..config import load_agent_config
from .models import Session, _validate_session_key
from .storage import JsonlSessionStorage


class SessionManager:
    """Create, save, clear, delete, and list sessions in a workspace directory."""

    def __init__(self, workspace: str | Path | None = None) -> None:
        """Create a manager using an explicit or configured workspace path."""

        self._workspace = (
            Path(workspace) if workspace is not None else load_agent_config().workspace_path
        )
        self._storage = JsonlSessionStorage(self._workspace / "sessions")

    @property
    def workspace(self) -> Path:
        """Return the workspace that owns this manager's session storage."""

        return self._workspace

    def get_or_create(self, session_key: str) -> Session:
        """Load a saved session or return a new empty session for the key."""

        _validate_session_key(session_key)
        return self._storage.load(session_key) or Session.create(session_key)

    def get(self, session_key: str) -> Session | None:
        """Load a saved session without creating or persisting a new one."""

        _validate_session_key(session_key)
        return self._storage.load(session_key)

    def save(self, session: Session) -> Session:
        """Persist one complete session and return its timestamped version."""

        if not isinstance(session, Session):
            raise TypeError("SessionManager.save requires a Session")
        saved_session = session.with_updated_at(datetime.now(timezone.utc))
        self._storage.save(saved_session)
        return saved_session

    def clear_messages(self, session_key: str) -> Session:
        """Reset and persist all messages for one session key."""

        session = self.get_or_create(session_key)
        return self.save(session.reset())

    def delete(self, session_key: str) -> bool:
        """Delete one saved session and report whether it existed."""

        return self._storage.delete(session_key)

    def list_sessions(self) -> tuple[Session, ...]:
        """Return all saved sessions in stable key order."""

        return self._storage.list_sessions()
