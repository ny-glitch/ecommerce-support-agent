from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import BaseMessage


@dataclass(frozen=True)
class TurnRef:
    conversation_id: str
    turn_id: str


@dataclass(frozen=True)
class StoredTurn:
    turn_id: str
    messages: tuple[BaseMessage, ...]
