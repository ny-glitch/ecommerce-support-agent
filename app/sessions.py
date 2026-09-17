"""Bounded process-local exclusion for active durable conversations."""

from app.errors import ServiceError


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
