"""Bounded, process-local sessions for one event loop and one worker."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from uuid import uuid4

from app.config import Settings
from app.context import Turn
from app.errors import ServiceError


@dataclass
class Session:
    id: str
    turns: list[Turn]
    busy: bool
    touched_at: float


class SessionStore:
    def __init__(
        self, settings: Settings, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._sessions: dict[str, Session] = {}

    def acquire(self, session_id: str | None) -> Session:
        now = self._clock()
        expired = [
            sid
            for sid, session in self._sessions.items()
            if not session.busy
            and now - session.touched_at >= self._settings.session_ttl_seconds
        ]
        for sid in expired:
            del self._sessions[sid]

        if session_id is not None:
            session = self._sessions.get(session_id)
            if session is None:
                raise ServiceError("SESSION_NOT_FOUND", "会话不存在或已过期，请新建会话", 404)
            if session.busy:
                raise ServiceError("SESSION_BUSY", "该会话正在生成回复，请稍后重试", 409)
        else:
            if len(self._sessions) >= self._settings.max_sessions:
                raise ServiceError("SESSION_CAPACITY", "会话数量已达上限，请稍后重试", 503)
            session = Session(str(uuid4()), [], False, now)
            self._sessions[session.id] = session
        session.busy = True
        session.touched_at = now
        return session

    def commit(self, session: Session, turns: list[Turn]) -> None:
        session.turns = list(turns[-self._settings.max_history_turns :])

    def release(self, session: Session) -> None:
        session.busy = False
        session.touched_at = self._clock()
        # A failed first request must not reserve capacity until the TTL expires.
        if not session.turns:
            self._sessions.pop(session.id, None)


class SessionGuard:
    def __init__(self, max_active: int) -> None:
        self._max_active = max_active
        self._active: set[str] = set()

    def acquire(self, conversation_id: str) -> None:
        if conversation_id in self._active:
            raise ServiceError(
                "SESSION_BUSY", "该会话正在生成回复，请稍后重试", 409
            )
        if len(self._active) >= self._max_active:
            raise ServiceError(
                "SESSION_CAPACITY", "当前请求数量已达上限，请稍后重试", 503
            )
        self._active.add(conversation_id)

    def release(self, conversation_id: str) -> None:
        self._active.discard(conversation_id)
