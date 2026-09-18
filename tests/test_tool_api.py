"""HTTP integration and startup lifecycle regressions for durable chat."""
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from helpers import RecordingGateway, http_app, parse_sse, settings
from tests.integration.conftest import mysql_db, load_test_database_url


def test_explicit_service_injection_bypasses_database_and_preserves_owner_404():
    app, gateway, conversations, _ = http_app()
    sid = conversations.seed(user_id="someone-else")
    with TestClient(app) as client:
        denied = client.post("/api/chat", json={"message": "hi", "session_id": sid})
        assert denied.status_code == 404
        assert denied.json() == {"error": {"code": "SESSION_NOT_FOUND", "message": "会话不存在"}}
        response = client.post("/api/chat", json={"message": "hi"})
        assert parse_sse(response.text)[-1]["event"] == "done"
        assert client.get("/").status_code == 200
    assert gateway.closed


def test_production_startup_requires_database_and_closes_gateway():
    gateway = RecordingGateway()
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        with TestClient(create_app(settings(), gateway)):
            pass
    assert gateway.closed


@pytest.mark.parametrize("failure", ["check", "assembly", None])
async def test_lifespan_closes_partial_and_successful_resources(monkeypatch, failure):
    import app.main as main
    events = []

    class Database:
        def __init__(self, url):
            self.sessions = object()
            events.append("created")

        async def check(self):
            events.append("checked")
            if failure == "check":
                raise RuntimeError("db unavailable")

        async def aclose(self):
            events.append("closed")

    monkeypatch.setattr(main, "Database", Database, raising=False)
    if failure == "assembly":
        def broken(*args, **kwargs):
            raise RuntimeError("assembly failed")
        monkeypatch.setattr(main, "ChatService", broken, raising=False)
    gateway = RecordingGateway()
    app = create_app(settings(database_url="mysql+asyncmy://local/test"), gateway)
    if failure:
        with pytest.raises(RuntimeError):
            async with app.router.lifespan_context(app):
                pytest.fail("Startup must fail")
    else:
        async with app.router.lifespan_context(app):
            assert app.state.chat_service.gateway is gateway
    assert events == ["created", "checked", "closed"]
    assert gateway.closed


def test_tool_failure_status_never_exposes_exception_or_database_url():
    from tests.ch02_helpers import selection
    app, gateway, _, service = http_app()

    async def choose(messages, tools):
        return selection("create_ticket", {
            "issue_description": "退货",
            "ticket_type": "return_refund",
        })

    class BrokenTickets:
        async def create_once(self, *args):
            raise RuntimeError("mysql+asyncmy://secret-user:secret-password@db/customer")

    gateway.select = choose
    service.tickets = BrokenTickets()
    with TestClient(app) as client:
        response = client.post("/api/chat", json={"message": "退货"})
    statuses = [e["data"] for e in parse_sse(response.text) if e["event"] == "tool_status"]
    assert [e["status"] for e in statuses] == ["running", "failed"]
    assert "secret" not in response.text
    assert "mysql" not in response.text
    assert "RuntimeError" not in response.text


def test_persistence_failure_after_metadata_is_safe_error_without_done():
    app, _, conversations, _ = http_app()

    async def failed_finish(*args):
        raise RuntimeError("secret database failure")

    conversations.finish_turn = failed_finish
    with TestClient(app) as client:
        response = client.post("/api/chat", json={"message": "hi"})
    events = parse_sse(response.text)
    assert response.status_code == 200
    assert events[0]["event"] == "meta"
    assert events[-1] == {"event": "error", "data": {"code": "DB_ERROR", "message": "服务暂时无法保存会话，请稍后重试"}}
    assert not any(e["event"] == "done" for e in events)
    assert "secret" not in response.text



async def test_production_http_assembly_persists_tool_turn_across_restart(mysql_db, request):
    # The shared fixture provisions/clears only the isolated test database.
    from tests.ch02_helpers import selection
    from tests.test_streaming import running_server

    configuration = settings(database_url=load_test_database_url(
        request.config.getoption("--require-mysql")
    ))
    gateway = RecordingGateway()

    async def choose(messages, tools):
        return selection("query_logistics", {"order_id": "1001"})

    gateway.select = choose
    first_app = create_app(configuration, gateway)
    async with running_server(first_app) as client:
        response = await client.post("/api/chat", json={"message": "订单1001物流"})
        events = parse_sse(response.text)
        assert events[-1]["event"] == "done"
        sid = events[0]["data"]["session_id"]
        audit = await first_app.state.chat_service.conversations.audit(sid, "demo")
        assert [row["role"] for row in audit] == ["user", "assistant", "tool", "assistant"]
        assert {row["turn_status"] for row in audit} == {"completed"}
    assert gateway.closed

    restored_gateway = RecordingGateway()
    restored_app = create_app(configuration, restored_gateway)
    async with running_server(restored_app) as client:
        response = await client.post("/api/chat", json={"message": "继续", "session_id": sid})
        assert parse_sse(response.text)[-1]["event"] == "done"
        assert [m.type for m in restored_gateway.calls[0]] == ["system", "human", "ai", "tool", "ai", "human"]
        assert restored_gateway.calls[0][1].content == "订单1001物流"
        assert restored_gateway.calls[0][-1].content == "继续"
    assert restored_gateway.closed


def test_http_conversations_keep_completed_history_separate():
    app, gateway, _, _ = http_app()
    with TestClient(app) as client:
        first = client.post("/api/chat", json={"message": "会话甲的内容"})
        sid = parse_sse(first.text)[0]["data"]["session_id"]
        client.post("/api/chat", json={"message": "会话乙的内容"})
        assert [m.content for m in gateway.calls[-1]][1:] == ["会话乙的内容"]
        client.post("/api/chat", json={"message": "继续甲", "session_id": sid})
        assert [m.content for m in gateway.calls[-1]][1:] == ["会话甲的内容", "你好，小林", "继续甲"]
