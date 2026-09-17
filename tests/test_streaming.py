"""Real loopback HTTP tests: ASGI test transports buffer the complete body."""

import asyncio
import socket
from contextlib import asynccontextmanager

import httpx
import pytest
import uvicorn

from app.context import Turn
from app.main import create_app
from helpers import GatedGateway, parse_sse, settings


@asynccontextmanager
async def running_server(app, asgi_spec=None):
    async def advertised_app(scope, receive, send):
        # Uvicorn currently advertises 2.3. Also exercise Starlette's >=2.4 path,
        # which relies on send errors and does not run a disconnect listener.
        if scope["type"] == "http" and asgi_spec:
            scope = {**scope, "asgi": {**scope["asgi"], "spec_version": asgi_spec}}
        await app(scope, receive, send)

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        config = uvicorn.Config(advertised_app, log_level="error", lifespan="on", timeout_graceful_shutdown=2)
        server = uvicorn.Server(config)
        task = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            async with asyncio.timeout(5):
                while not server.started:
                    if task.done():
                        await task
                        raise RuntimeError("Server exited before startup")
                    await asyncio.sleep(0.01)
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{sock.getsockname()[1]}", timeout=5, trust_env=False) as client:
                yield client
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, 5)


async def next_event(lines):
    block = []
    async with asyncio.timeout(2):
        async for line in lines:
            if not line:
                events = parse_sse("\n".join(block))
                if events:
                    return events[0]
                block = []
            else:
                block.append(line)
    raise AssertionError("Stream ended before event")


async def test_first_token_arrives_before_upstream_is_allowed_to_finish():
    gateway = GatedGateway()
    app = create_app(settings(), gateway)
    async with running_server(app) as client:
        try:
            async with client.stream("POST", "/api/chat", json={"message": "hi"}) as response:
                assert response.status_code == 200
                lines = response.aiter_lines()
                metadata = await next_event(lines)
                first = await next_event(lines)
                assert first == {"event": "token", "data": {"content": "第一段"}}
                assert gateway.waiting.is_set()
                assert not gateway.completed
                gateway.resume.set()
                assert await next_event(lines) == {"event": "token", "data": {"content": "第二段"}}
                assert await next_event(lines) == {"event": "done", "data": {"session_id": metadata["data"]["session_id"]}}
            assert gateway.completed
        finally:
            gateway.resume.set()


@pytest.mark.parametrize("existing", [True, False])
@pytest.mark.parametrize("asgi_spec", [None, "2.4"], ids=["uvicorn-native", "asgi-2.4"])
async def test_idle_disconnect_closes_upstream_releases_session_and_preserves_history(existing, asgi_spec):
    gateway = GatedGateway()
    app = create_app(settings(max_sessions=1), gateway)
    async with running_server(app, asgi_spec) as client:
        sid = None
        if existing:
            session = app.state.sessions.acquire(None)
            sid = session.id
            app.state.sessions.commit(session, [Turn("old", "old answer")])
            app.state.sessions.release(session)
        try:
            async with client.stream("POST", "/api/chat", json={"message": "cancelled", "session_id": sid}) as response:
                lines = response.aiter_lines()
                assert (await next_event(lines))["event"] == "meta"
                assert (await next_event(lines))["event"] == "token"
                await asyncio.wait_for(gateway.waiting.wait(), 1)
                conflict = await client.post("/api/chat", json={"message": "concurrent", "session_id": sid})
                assert conflict.status_code == (409 if existing else 503)

            # The gateway is still blocked. Disconnect alone must release resources.
            await asyncio.wait_for(gateway.stream_closed.wait(), 1)
            assert not gateway.completed
            gateway.resume.set()
            retry = await client.post("/api/chat", json={"message": "retry", "session_id": sid})
            assert retry.status_code == 200
            assert parse_sse(retry.text)[-1]["event"] == "done"
            assert [m.content for m in gateway.calls[-1]][1:] == (
                ["old", "old answer", "retry"] if existing else ["retry"]
            )
        finally:
            gateway.resume.set()


async def test_stalled_stream_times_out_without_committing_partial_turn():
    gateway = GatedGateway()
    app = create_app(settings(request_timeout_seconds=1, max_sessions=1), gateway)
    async with running_server(app) as client:
        try:
            async with asyncio.timeout(3):
                response = await client.post("/api/chat", json={"message": "timeout"})
            events = parse_sse(response.text)
            assert events[-1]["event"] == "error"
            assert events[-1]["data"]["code"] == "UPSTREAM_TIMEOUT"
            assert not any(e["event"] == "done" for e in events)
            assert gateway.stream_closed.is_set()
            assert not gateway.completed
            gateway.resume.set()
            assert (await client.post("/api/chat", json={"message": "new"})).status_code == 200
        finally:
            gateway.resume.set()


async def test_stalled_extraction_returns_safe_timeout_error():
    class StalledExtraction(GatedGateway):
        async def extract(self, description):
            try:
                await self.resume.wait()
            finally:
                self.stream_closed.set()

    gateway = StalledExtraction()
    app = create_app(settings(request_timeout_seconds=1), gateway)
    async with running_server(app) as client:
        try:
            async with asyncio.timeout(3):
                response = await client.post("/api/after-sales/extract", json={"description": "hi"})
            assert response.status_code == 504
            assert response.json()["error"]["code"] == "UPSTREAM_TIMEOUT"
            assert gateway.stream_closed.is_set()
        finally:
            gateway.resume.set()
