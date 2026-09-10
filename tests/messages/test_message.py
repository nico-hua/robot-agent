"""Tests for the canonical message module and package export."""

from src.messages import Message as ExportedMessage
from src.messages.message import Message


def test_package_reexports_the_canonical_message_type() -> None:
    assert ExportedMessage is Message
