"""Offline tests for the simplified persisted session lifecycle."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from src.config import AgentConfig, ProviderConfig
from src.providers import AIMessage, HumanMessage, SystemMessage, ToolCallRequest, ToolMessage
from src.session import manager as session_manager
from src.session.manager import SessionManager
from src.session.models import Session
from src.session.storage import JsonlSessionStorage

_TIMESTAMP = datetime(2026, 9, 10, tzinfo=timezone.utc)


def _session(key: str = "session-key") -> Session:
    tool_call = ToolCallRequest(id="call-1", name="lookup", arguments={"query": "value"})
    return Session(
        key=key,
        created_at=_TIMESTAMP,
        updated_at=_TIMESTAMP,
        messages=(
            SystemMessage(content="system"),
            HumanMessage(content="user"),
            AIMessage(content="assistant", tool_calls=(tool_call,)),
            ToolMessage(
                content="tool result",
                tool_call_id="call-1",
                tool_name="lookup",
                image_path=Path("workspace/tool-output.png"),
            ),
        ),
    )


def test_session_has_no_summary_or_goal_state_and_reset_clears_messages() -> None:
    session = _session()

    assert {"summary", "summary_until", "goal_state"}.isdisjoint(Session.__dataclass_fields__)
    assert not hasattr(session, "with_summary")
    assert not hasattr(session, "with_goal_state")
    assert session.reset() == Session(
        key=session.key,
        created_at=_TIMESTAMP,
        updated_at=_TIMESTAMP,
    )


def test_storage_round_trips_messages_without_removed_header_fields(tmp_path) -> None:
    storage = JsonlSessionStorage(tmp_path)
    session = _session()

    storage.save(session)

    path = next(tmp_path.glob("*.jsonl"))
    header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert header == {
        "type": "session",
        "key": session.key,
        "created_at": _TIMESTAMP.isoformat(),
        "updated_at": _TIMESTAMP.isoformat(),
    }
    assert storage.load(session.key) == session


def test_storage_round_trips_tool_result_name_and_local_image_path(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path)
    session = _session()

    storage.save(session)

    records = [
        json.loads(line)
        for line in next(tmp_path.glob("*.jsonl")).read_text(encoding="utf-8").splitlines()
    ]
    assert records[-1] == {
        "type": "message",
        "role": "tool",
        "content": "tool result",
        "tool_call_id": "call-1",
        "tool_name": "lookup",
        "image_path": str(Path("workspace/tool-output.png")),
    }
    assert storage.load(session.key) == session


def test_storage_reads_legacy_tool_message_without_name_or_image_path(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path)
    path = storage._path_for("legacy-tool-message")
    records = (
        {
            "type": "session",
            "key": "legacy-tool-message",
            "created_at": _TIMESTAMP.isoformat(),
            "updated_at": _TIMESTAMP.isoformat(),
        },
        {
            "type": "message",
            "role": "tool",
            "content": "legacy result",
            "tool_call_id": "call-legacy",
        },
    )
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    assert storage.load("legacy-tool-message") == Session(
        key="legacy-tool-message",
        created_at=_TIMESTAMP,
        updated_at=_TIMESTAMP,
        messages=(ToolMessage(content="legacy result", tool_call_id="call-legacy"),),
    )


def test_storage_reads_legacy_header_and_strips_removed_fields_on_save(tmp_path) -> None:
    storage = JsonlSessionStorage(tmp_path)
    path = storage._path_for("legacy-key")
    legacy_header = {
        "type": "session",
        "key": "legacy-key",
        "created_at": _TIMESTAMP.isoformat(),
        "updated_at": _TIMESTAMP.isoformat(),
        "summary": "legacy summary",
        "summary_until": 1,
        "goal_state": {"legacy": True},
    }
    legacy_message = {"type": "message", "role": "user", "content": "hello"}
    path.write_text(
        "\n".join(json.dumps(record) for record in (legacy_header, legacy_message)) + "\n",
        encoding="utf-8",
    )

    session = storage.load("legacy-key")

    assert session == Session(
        key="legacy-key",
        created_at=_TIMESTAMP,
        updated_at=_TIMESTAMP,
        messages=(HumanMessage(content="hello"),),
    )
    storage.save(session)
    saved_header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert {"summary", "summary_until", "goal_state"}.isdisjoint(saved_header)


def test_manager_persists_and_deletes_sessions_without_goal_apis(tmp_path) -> None:
    manager = SessionManager(tmp_path)
    created = manager.get_or_create("managed-key")

    assert created.messages == ()
    assert not hasattr(manager, "create_goal")
    assert not hasattr(manager, "update_goal")

    saved = manager.save(created.with_messages((HumanMessage(content="hello"),)))

    assert manager.get("managed-key") == saved
    assert manager.list_sessions() == (saved,)
    assert manager.delete("managed-key") is True
    assert manager.delete("managed-key") is False


def test_manager_clears_and_persists_one_session_messages(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    manager.save(
        manager.get_or_create("managed-key").with_messages(
            (HumanMessage(content="hello"), AIMessage(content="reply"))
        )
    )

    cleared = manager.clear_messages("managed-key")

    assert cleared.key == "managed-key"
    assert cleared.messages == ()
    persisted = manager.get("managed-key")
    assert persisted is not None
    assert persisted.messages == ()


def test_manager_uses_configured_workspace_when_path_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        session_manager,
        "load_agent_config",
        lambda: AgentConfig(
            provider=ProviderConfig(
                type="openai_compat",
                api_base="https://api.example.test/v1",
                model="test-model",
            ),
            workspace_path=tmp_path,
        ),
    )

    manager = SessionManager()

    assert manager.workspace == tmp_path
