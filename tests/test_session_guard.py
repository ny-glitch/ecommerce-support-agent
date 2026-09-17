from __future__ import annotations

import pytest

from app.errors import ServiceError
from app.sessions import SessionGuard


def test_guard_rejects_a_second_request_for_the_same_conversation() -> None:
    guard = SessionGuard(max_active=2)

    guard.acquire("conversation-1")

    with pytest.raises(ServiceError) as exc_info:
        guard.acquire("conversation-1")

    assert exc_info.value.code == "SESSION_BUSY"
    assert exc_info.value.status_code == 409

    guard.release("conversation-1")
    guard.acquire("conversation-1")


def test_guard_limits_active_requests_and_release_restores_capacity() -> None:
    guard = SessionGuard(max_active=1)

    guard.acquire("conversation-1")

    with pytest.raises(ServiceError) as exc_info:
        guard.acquire("conversation-2")

    assert exc_info.value.code == "SESSION_CAPACITY"
    assert exc_info.value.status_code == 503

    guard.release("conversation-1")
    guard.acquire("conversation-2")
