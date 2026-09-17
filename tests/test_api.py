from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from app.context import Turn
from app.errors import ServiceError
from helpers import RecordingGateway, parse_sse, settings


def make_app(**overrides):
    from app.main import create_app

    gateway = RecordingGateway()
    return create_app(settings(**overrides), gateway), gateway


def test_chat_reuses_complete_turns_and_returns_readable_sse():
    app, gateway = make_app()
    with TestClient(app) as client:
        first = client.post("/api/chat", json={"message": "我叫小林"})
        assert first.status_code == 200
        assert first.headers["content-type"].startswith("text/event-stream")
        assert first.headers["x-accel-buffering"] == "no"
        assert "你好" in first.text
        events = parse_sse(first.text)
        assert [e["event"] for e in events] == ["meta", "token", "token", "done"]
        metadata = events[0]["data"]
        sid = metadata["session_id"]
        assert str(UUID(sid)) == sid
        assert metadata["estimated_input_tokens"] > 0
        assert metadata["token_count_is_estimate"] is True
        assert metadata["dropped_turns"] == 0
        assert [e["data"]["content"] for e in events if e["event"] == "token"] == ["你好", "，小林"]
        assert events[-1]["data"] == {"session_id": sid}
        second = client.post("/api/chat", json={"message": "我叫什么？", "session_id": sid})
        assert second.status_code == 200
        assert [(m.type, m.content) for m in gateway.calls[-1]][1:] == [
            ("human", "我叫小林"), ("ai", "你好，小林"), ("human", "我叫什么？")
        ]
    assert gateway.closed


@pytest.mark.parametrize("body", [{}, {"message": " "}, {"message": 12}, {"message": "hi", "session_id": "bad"}, {"message": "hi", "extra": "private"}, {"message": "x" * 32001}])
def test_invalid_chat_requests_do_not_reach_upstream(body):
    app, gateway = make_app(max_sessions=1)
    with TestClient(app) as client:
        response = client.post("/api/chat", json=body)
        assert response.status_code == 422
        assert response.json() == {"error": {"code": "INVALID_REQUEST", "message": "请求参数无效"}}
        assert gateway.calls == []
        assert client.post("/api/chat", json={"message": "hi"}).status_code == 200


def test_long_input_does_not_consume_session_capacity_or_lock_existing_session():
    app, gateway = make_app(max_sessions=1)
    with TestClient(app) as client:
        long_message = "长" * 3000
        for _ in range(2):
            response = client.post("/api/chat", json={"message": long_message})
            assert response.status_code == 413
            assert response.json()["error"]["code"] == "INPUT_TOO_LONG"
        first = client.post("/api/chat", json={"message": "hi"})
        sid = parse_sse(first.text)[0]["data"]["session_id"]
        assert client.post("/api/chat", json={"message": long_message, "session_id": sid}).status_code == 413
        assert client.post("/api/chat", json={"message": "again", "session_id": sid}).status_code == 200
        assert len(gateway.calls) == 2


def test_session_http_unknown_busy_and_capacity_errors():
    app, _ = make_app(max_sessions=1)
    with TestClient(app) as client:
        unknown = client.post("/api/chat", json={"message": "hi", "session_id": str(uuid4())})
        assert unknown.status_code == 404
        assert unknown.json()["error"]["code"] == "SESSION_NOT_FOUND"
        session = app.state.sessions.acquire(None)
        busy = client.post("/api/chat", json={"message": "hi", "session_id": session.id})
        assert busy.status_code == 409
        assert busy.json()["error"]["code"] == "SESSION_BUSY"
        full = client.post("/api/chat", json={"message": "hi"})
        assert full.status_code == 503
        assert full.json()["error"]["code"] == "SESSION_CAPACITY"
        app.state.sessions.release(session)


@pytest.mark.parametrize("failure,code", [(RuntimeError("secret-key raw upstream body"), "UPSTREAM_ERROR"), (ServiceError("UPSTREAM_INCOMPLETE", "模型回复未正常完成，请重试", 502), "UPSTREAM_INCOMPLETE")])
def test_failed_turn_leaves_history_intact(failure, code):
    app, gateway = make_app()
    with TestClient(app) as client:
        sid = parse_sse(client.post("/api/chat", json={"message": "old"}).text)[0]["data"]["session_id"]
        gateway.error = failure
        response = client.post("/api/chat", json={"message": "failed", "session_id": sid})
        assert response.status_code == 200
        events = parse_sse(response.text)
        assert events[-1]["event"] == "error"
        assert events[-1]["data"]["code"] == code
        assert "secret-key" not in response.text
        assert not any(e["event"] == "done" for e in events)
        gateway.error = None
        client.post("/api/chat", json={"message": "retry", "session_id": sid})
        assert [m.content for m in gateway.calls[-1]][1:] == ["old", "你好，小林", "retry"]


def test_failed_new_session_does_not_exhaust_capacity():
    app, gateway = make_app(max_sessions=1)
    with TestClient(app) as client:
        gateway.error = RuntimeError("failed")
        failed = client.post("/api/chat", json={"message": "failed"})
        assert parse_sse(failed.text)[-1]["event"] == "error"
        gateway.error = None
        assert client.post("/api/chat", json={"message": "new"}).status_code == 200


def test_history_retention_is_bounded():
    app, gateway = make_app(max_history_turns=2)
    with TestClient(app) as client:
        sid = None
        for message in ["first", "second", "third", "fourth"]:
            response = client.post("/api/chat", json={"message": message, "session_id": sid})
            sid = parse_sse(response.text)[0]["data"]["session_id"]
        assert [m.content for m in gateway.calls[-1]][1:] == ["second", "你好，小林", "third", "你好，小林", "fourth"]
        session = app.state.sessions.acquire(sid)
        assert [t.user for t in session.turns] == ["third", "fourth"]
        app.state.sessions.release(session)


def test_idle_ttl_expires_but_busy_session_survives():
    from app.sessions import SessionStore

    now = [0.0]
    store = SessionStore(settings(session_ttl_seconds=10, max_sessions=2), clock=lambda: now[0])
    idle, busy = store.acquire(None), store.acquire(None)
    store.commit(idle, [Turn("u", "a")])
    store.release(idle)
    now[0] = 10.0
    with pytest.raises(ServiceError) as expired:
        store.acquire(idle.id)
    assert expired.value.status_code == 404
    with pytest.raises(ServiceError) as occupied:
        store.acquire(busy.id)
    assert occupied.value.status_code == 409
    replacement = store.acquire(None)
    store.commit(busy, [Turn("busy", "completed")])
    store.release(busy)
    now[0] = 19.0
    assert store.acquire(busy.id).turns == [Turn("busy", "completed")]
    store.release(replacement)


def test_extraction_returns_validated_result_without_creating_session():
    app, gateway = make_app(max_sessions=1)
    with TestClient(app) as client:
        response = client.post("/api/extract", json={"description": "  订单 ORDER-17，我要退款  "})
        assert response.status_code == 200
        assert response.json() == {"order_id": "ORDER-17", "request_type": "refund", "expected_resolution": "原路退款"}
        assert gateway.descriptions == ["订单 ORDER-17，我要退款"]
        assert client.post("/api/chat", json={"message": "hi"}).status_code == 200


@pytest.mark.parametrize("error,code,status", [(ServiceError("STRUCTURED_OUTPUT_ERROR", "售后信息提取失败，请重试", 502), "STRUCTURED_OUTPUT_ERROR", 502), (RuntimeError("secret-key"), "UPSTREAM_ERROR", 502), (ServiceError("INPUT_TOO_LONG", "输入内容超过可处理的上下文长度", 413), "INPUT_TOO_LONG", 413)])
def test_extraction_errors_are_safe_http_errors(error, code, status):
    app, gateway = make_app()
    gateway.extraction_error = error
    with TestClient(app) as client:
        response = client.post("/api/extract", json={"description": "hi"})
        assert response.status_code == status
        assert response.json()["error"]["code"] == code
        assert "secret-key" not in response.text


def test_invalid_extraction_does_not_echo_user_input():
    app, gateway = make_app()
    with TestClient(app) as client:
        response = client.post("/api/extract", json={"description": "secret-user-input", "extra": True})
        assert response.status_code == 422
        assert "secret-user-input" not in response.text
        assert gateway.descriptions == []


def test_expired_session_returns_http_404():
    from app.sessions import SessionStore

    app, gateway = make_app(session_ttl_seconds=10)
    with TestClient(app) as client:
        now = [0.0]
        app.state.sessions = SessionStore(app.state.settings, clock=lambda: now[0])
        first = client.post("/api/chat", json={"message": "old"})
        sid = parse_sse(first.text)[0]["data"]["session_id"]
        now[0] = 10.0
        expired = client.post("/api/chat", json={"message": "next", "session_id": sid})
        assert expired.status_code == 404
        assert expired.json()["error"]["code"] == "SESSION_NOT_FOUND"
        assert len(gateway.calls) == 1


def test_budget_trimming_is_reported_and_failed_attempt_keeps_original_history():
    app, gateway = make_app()
    with TestClient(app) as client:
        session = app.state.sessions.acquire(None)
        original = [Turn("长" * 2000, "旧答案"), Turn("recent", "recent answer")]
        app.state.sessions.commit(session, original)
        app.state.sessions.release(session)
        gateway.error = RuntimeError("failure")
        response = client.post("/api/chat", json={"message": "再" * 1000, "session_id": session.id})
        events = parse_sse(response.text)
        assert events[0]["data"]["dropped_turns"] == 1
        assert events[-1]["event"] == "error"
        assert [m.content for m in gateway.calls[-1]][1:-1] == ["recent", "recent answer"]
        stored = app.state.sessions.acquire(session.id)
        assert stored.turns == original
        app.state.sessions.release(stored)
