"""A persisted turn: one selection, at most one tool, one final text stream."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
import logging
from typing import Literal, TypeVar
from uuid import uuid4

from anyio import CancelScope

from app.config import Settings
from app.context import ToolContextWindow, build_tool_context
from app.db.contracts import TurnRef
from app.db.conversations import ConversationRepository
from app.db.faq import FaqRepository
from app.db.tickets import TicketRepository
from app.errors import ServiceError
from app.model import ModelGateway
from app.prompts import tool_chat_system_prompt
from app.services.events import ChatEvent
from app.sessions import SessionGuard
from app.tools.business import ToolContext, build_registry
from app.tools.executor import ToolExecutor, ToolOutcome, ToolProgress
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass
class PreparedTurn:
    ref: TurnRef
    deadline: float
    window: ToolContextWindow
    registry: ToolRegistry
    message: str
    _parts: list[str] = field(default_factory=list, repr=False)
    _iterators: list = field(default_factory=list, repr=False)
    _operations: set[asyncio.Task] = field(default_factory=set, repr=False)
    _mutations: set[asyncio.Task] = field(default_factory=set, repr=False)
    _started: bool = field(default=False, repr=False)
    _stream_started: bool = field(default=False, repr=False)
    _finalized: bool = field(default=False, repr=False)
    _cleanup_started: bool = field(default=False, repr=False)


async def _bounded(
    factory: Callable[[], Awaitable[T]], deadline: float,
    operations: set[asyncio.Task], *, mutations: set[asyncio.Task] | None = None,
) -> T:
    # Each scope enters/exits in the same Task; never hold a timeout over yield.
    if asyncio.get_running_loop().time() >= deadline:
        raise TimeoutError
    async def invoke() -> T:
        return await factory()

    def finished(task: asyncio.Task) -> None:
        operations.discard(task)
        if mutations is not None:
            mutations.discard(task)
        if not task.cancelled():
            task.exception()  # Also retrieve failures after the consumer has left.

    task = asyncio.create_task(invoke())
    operations.add(task)
    if mutations is not None:
        mutations.add(task)
    task.add_done_callback(finished)
    try:
        async with asyncio.timeout_at(deadline):
            return await asyncio.shield(task)
    except (asyncio.CancelledError, TimeoutError):
        # A cancelled operation may await resource close in its finally block.
        # Let bounded cleanup own that unwind instead of blocking this consumer.
        task.cancel()
        raise


def _safe_error(error: Exception) -> ServiceError:
    if isinstance(error, ServiceError):
        return error
    if isinstance(error, TimeoutError):
        return ServiceError("UPSTREAM_TIMEOUT", "模型服务响应超时，请重试", 504)
    return ServiceError("DB_ERROR", "服务暂时无法保存会话，请稍后重试", 503)


class ChatService:
    def __init__(
        self, settings: Settings, gateway: ModelGateway,
        conversations: ConversationRepository, faq: FaqRepository,
        tickets: TicketRepository, guard: SessionGuard, executor: ToolExecutor,
    ) -> None:
        self.settings = settings
        self.gateway = gateway
        self.conversations = conversations
        self.faq = faq
        self.tickets = tickets
        self.guard = guard
        self.executor = executor
        self._cleanup_tasks: set[asyncio.Task] = set()

    @asynccontextmanager
    async def prepare(
        self, message: str, session_id: str | None, user_id: str = "demo"
    ) -> AsyncIterator[PreparedTurn]:
        deadline = asyncio.get_running_loop().time() + self.settings.request_timeout_seconds
        if not message.strip():
            raise ServiceError("INVALID_REQUEST", "请求参数无效", 422)
        ref = TurnRef(session_id or str(uuid4()), str(uuid4()))
        self.guard.acquire(ref.conversation_id)
        prepared = None
        status: Literal["failed", "cancelled"] = "cancelled"
        try:
            registry = build_registry(
                ToolContext(ref, user_id, message, "TK-" + uuid4().hex),
                self.faq, self.tickets,
            )
            prompt = tool_chat_system_prompt()
            # Required inputs must fit before creating any durable state.
            window = build_tool_context(
                prompt, [], message, self.settings, tool_schemas=registry.schemas()
            )
            prepared = PreparedTurn(ref, deadline, window, registry, message)
            if session_id is not None:
                owner = await _bounded(
                    lambda: self.conversations.get(session_id, user_id), deadline,
                    prepared._operations,
                )
                if owner is None:
                    raise ServiceError("CONVERSATION_NOT_FOUND", "会话不存在", 404)
                history = await _bounded(
                    lambda: self.conversations.history(
                        session_id, user_id, self.settings.max_history_turns
                    ), deadline, prepared._operations,
                )
                prepared.window = build_tool_context(
                    prompt, history, message, self.settings,
                    tool_schemas=registry.schemas(),
                )
            else:
                await _bounded(
                    lambda: self.conversations.create(ref.conversation_id, user_id),
                    deadline, prepared._operations,
                    mutations=prepared._mutations,
                )
            # A commit may succeed before its acknowledgment is cancelled.
            prepared._started = True
            await _bounded(
                lambda: self.conversations.start_turn(ref, user_id, message), deadline,
                prepared._operations,
                mutations=prepared._mutations,
            )
            yield prepared
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except Exception as error:
            status = "failed"
            raise _safe_error(error) from error
        finally:
            try:
                if prepared is not None:
                    await self._cleanup(prepared, status)
            finally:
                self.guard.release(ref.conversation_id)

    async def _cleanup(self, prepared: PreparedTurn, status: Literal["failed", "cancelled"]) -> None:
        if prepared._cleanup_started:
            return
        prepared._cleanup_started = True

        async def close(iterator):
            try:
                await iterator.aclose()
            except Exception:
                logger.warning("chat cleanup failed: stream_close")

        async def finish():
            # Never compete with a write whose cancellation/commit is still
            # unwinding. If it cannot settle within the shared cleanup grace
            # period, leave the audit unresolved instead of starting a write.
            if prepared._mutations:
                await asyncio.gather(*tuple(prepared._mutations), return_exceptions=True)
            if prepared._started and not prepared._finalized:
                try:
                    await self.conversations.finish_turn(
                        prepared.ref, "".join(prepared._parts), status
                    )
                    prepared._finalized = True
                except Exception:
                    # Never log database exception text, prompts, or credentials.
                    logger.warning("chat cleanup failed: audit_finish")

        async def cleanup_work():
            # Closing one stalled upstream must not prevent the audit attempt.
            async def drain_and_close():
                # Do not call aclose while its anext is still unwinding.
                if prepared._operations:
                    await asyncio.gather(*tuple(prepared._operations), return_exceptions=True)
                await asyncio.gather(*(close(it) for it in prepared._iterators))

            jobs = [asyncio.create_task(drain_and_close()), asyncio.create_task(finish())]
            try:
                await asyncio.gather(*jobs)
            finally:
                for job in jobs:
                    if not job.done():
                        job.cancel()

        task = asyncio.create_task(cleanup_work())
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)
        # Shielding gives the caller a hard bound even if cleanup delays cancellation.
        deadline = asyncio.get_running_loop().time() + 1.0
        cancelled = False
        # The ASGI consumer may already be inside a cancelled AnyIO scope.
        # This scope is local to cleanup and never crosses a generator yield.
        with CancelScope(shield=True):
            while not task.done():
                try:
                    # Shield the cleanup task itself; do not create another
                    # operation task owned by the cleanup being awaited.
                    async with asyncio.timeout_at(deadline):
                        await asyncio.shield(task)
                except TimeoutError:
                    task.cancel()
                    logger.warning("chat cleanup failed: timeout")
                    break
                except asyncio.CancelledError:
                    cancelled = True
        if task.done() and not task.cancelled():
            task.exception()  # Retrieve any unexpected failure without exposing details.
        if cancelled:
            raise asyncio.CancelledError

    async def stream(self, prepared: PreparedTurn) -> AsyncIterator[ChatEvent]:
        if prepared._stream_started or prepared._cleanup_started:
            raise ServiceError("TURN_INVALID", "当前轮次不能重复生成", 409)
        prepared._stream_started = True
        try:
            yield ChatEvent("meta", {
                "session_id": prepared.ref.conversation_id,
                "turn_id": prepared.ref.turn_id,
                "estimated_input_tokens": prepared.window.estimated_input_tokens,
                "token_count_is_estimate": True,
                "dropped_turns": prepared.window.dropped_turns,
            })
            decision = await _bounded(
                lambda: self.gateway.select(prepared.window.messages, prepared.registry.tools),
                prepared.deadline, prepared._operations,
            )
            calls = decision.tool_calls
            if decision.invalid_tool_calls or len(calls) > 1 or any(not call.get("id") for call in calls):
                raise ServiceError("INVALID_TOOL_CALL", "工具调用格式无效，请重试", 502)
            current_tool_messages = []
            if calls:
                await _bounded(
                    lambda: self.conversations.append_call(prepared.ref, decision),
                    prepared.deadline, prepared._operations,
                    mutations=prepared._mutations,
                )
                execution = self.executor.run(calls[0], prepared.registry, deadline=prepared.deadline)
                prepared._iterators.append(execution)
                while True:
                    try:
                        event = await _bounded(
                            lambda: anext(execution), prepared.deadline, prepared._operations
                        )
                    except StopAsyncIteration:
                        break
                    if isinstance(event, ToolOutcome):
                        await _bounded(
                            lambda: self.conversations.append_result(prepared.ref, event.message),
                            prepared.deadline, prepared._operations,
                            mutations=prepared._mutations,
                        )
                        current_tool_messages = [decision, event.message]
                        progress = ToolProgress(
                            calls[0]["name"], calls[0]["id"], event.terminal_status,
                            event.attempt,
                            {"succeeded": "工具执行完成", "not_found": "未找到匹配信息"}.get(
                                event.terminal_status, "工具执行结果无法确认，请稍后核实"
                            ),
                        )
                        yield ChatEvent("tool_status", asdict(progress))
                    else:
                        yield ChatEvent("tool_status", asdict(event))
            final_window = build_tool_context(
                tool_chat_system_prompt(), prepared.window.retained_turns,
                prepared.message, self.settings, tool_schemas=[],
                current_tool_messages=current_tool_messages,
            )
            upstream = self.gateway.stream(final_window.messages)
            prepared._iterators.append(upstream)
            while True:
                try:
                    text = await _bounded(
                        lambda: anext(upstream), prepared.deadline, prepared._operations
                    )
                except StopAsyncIteration:
                    break
                if text:
                    prepared._parts.append(text)
                    yield ChatEvent("token", {"content": text})
            content = "".join(prepared._parts)
            if not content.strip():
                raise ServiceError("UPSTREAM_INCOMPLETE", "模型回复未正常完成，请重试", 502)
            await _bounded(
                lambda: self.conversations.finish_turn(prepared.ref, content, "completed"),
                prepared.deadline, prepared._operations,
                mutations=prepared._mutations,
            )
            prepared._finalized = True
            yield ChatEvent("done", {"session_id": prepared.ref.conversation_id})
        except (asyncio.CancelledError, GeneratorExit):
            await self._cleanup(prepared, "cancelled")
            raise
        except Exception as error:
            safe = _safe_error(error)
            await self._cleanup(prepared, "failed")
            yield ChatEvent("error", {"code": safe.code, "message": safe.message})
        finally:
            await self._cleanup(prepared, "cancelled")
