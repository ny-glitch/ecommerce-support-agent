import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass
from typing import Annotated

import anyio
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.config import Settings, load_settings
from app.context import ContextWindow, Turn, build_context
from app.errors import ServiceError
from app.model import ModelGateway, OpenAIModelGateway
from app.prompts import customer_system_prompt
from app.schemas import AfterSalesResult, ChatRequest, ExtractRequest
from app.sessions import Session, SessionStore


@dataclass
class PreparedChat:
    message: str
    session: Session
    window: ContextWindow
    upstream: AsyncIterator[str]


def _event(name: str, data: dict) -> ServerSentEvent:
    return ServerSentEvent(event=name, raw_data=json.dumps(data, ensure_ascii=False))


def _upstream_error() -> ServiceError:
    return ServiceError("UPSTREAM_ERROR", "模型服务暂时不可用", 502)


def _timeout_error() -> ServiceError:
    return ServiceError("UPSTREAM_TIMEOUT", "模型服务响应超时，请重试", 504)


async def _stream_tokens(
    request: Request, upstream: AsyncIterator[str]
) -> AsyncIterator[str]:
    """Check idle disconnects without cancelling the pending read on every poll."""
    deadline = (
        asyncio.get_running_loop().time()
        + request.app.state.settings.request_timeout_seconds
    )
    pending = None
    try:
        while not await request.is_disconnected():
            pending = asyncio.create_task(anext(upstream))
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise _timeout_error()
                ready, _ = await asyncio.wait({pending}, timeout=min(0.05, remaining))
                if await request.is_disconnected():
                    return
                if ready:
                    break
            try:
                text = pending.result()
            except StopAsyncIteration:
                return
            yield text
    finally:
        if pending is not None:
            pending.cancel()
            # FastAPI's producer task group can already be cancelled here.
            with anyio.CancelScope(shield=True):
                await asyncio.gather(pending, return_exceptions=True)


async def _prepare_chat(body: ChatRequest, request: Request) -> AsyncIterator[PreparedChat]:
    store = request.app.state.sessions
    session = store.acquire(str(body.session_id) if body.session_id else None)
    try:
        window = build_context(
            customer_system_prompt(), session.turns, body.message, request.app.state.settings
        )
        upstream = request.app.state.gateway.stream(window.messages)
        try:
            yield PreparedChat(body.message, session, window, upstream)
        finally:
            # The request dependency also closes generators suspended at a yield.
            with anyio.CancelScope(shield=True):
                await upstream.aclose()
    finally:
        store.release(session)


def create_app(
    settings: Settings | None = None, gateway: ModelGateway | None = None
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings if settings is not None else load_settings()
        app.state.gateway = (
            gateway if gateway is not None else OpenAIModelGateway(app.state.settings)
        )
        app.state.sessions = SessionStore(app.state.settings)
        try:
            yield
        finally:
            await app.state.gateway.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, error: ServiceError):
        return JSONResponse(
            {"error": {"code": error.code, "message": error.message}},
            status_code=error.status_code,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        # Validation details can echo private customer input; expose only the contract.
        return JSONResponse(
            {"error": {"code": "INVALID_REQUEST", "message": "请求参数无效"}},
            status_code=422,
        )

    @app.post("/api/chat", response_class=EventSourceResponse)
    async def chat(
        request: Request,
        prepared: Annotated[PreparedChat, Depends(_prepare_chat, scope="request")],
    ) -> AsyncIterator[ServerSentEvent]:
        session, window = prepared.session, prepared.window
        yield _event("meta", {
            "session_id": session.id,
            "estimated_input_tokens": window.estimated_input_tokens,
            "token_count_is_estimate": True,
            "dropped_turns": window.dropped_turns,
        })
        parts = []
        try:
            async with aclosing(_stream_tokens(request, prepared.upstream)) as stream:
                async for text in stream:
                    if text:
                        parts.append(text)
                        yield _event("token", {"content": text})
            if await request.is_disconnected():
                return
            request.app.state.sessions.commit(
                session, window.retained_turns + [Turn(prepared.message, "".join(parts))]
            )
            yield _event("done", {"session_id": session.id})
        except Exception as exc:
            error = exc if isinstance(exc, ServiceError) else _upstream_error()
            yield _event("error", {"code": error.code, "message": error.message})

    @app.post("/api/extract", response_model=AfterSalesResult)
    async def extract(body: ExtractRequest, request: Request) -> AfterSalesResult:
        try:
            async with asyncio.timeout(request.app.state.settings.request_timeout_seconds):
                return await request.app.state.gateway.extract(body.description)
        except TimeoutError as exc:
            raise _timeout_error() from exc
        except ServiceError:
            raise
        except Exception as exc:
            raise _upstream_error() from exc

    return app
