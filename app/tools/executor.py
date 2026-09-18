from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
import json
import time
from typing import Any

from langchain_core.messages import ToolMessage
from pydantic import ValidationError
from sqlalchemy.exc import (
    DBAPIError,
    IntegrityError,
    InterfaceError,
    OperationalError,
    TimeoutError as SQLAlchemyTimeoutError,
)

from app.errors import ServiceError
from app.tools.registry import ToolRegistry
from app.tools.results import (
    InvalidToolArguments,
    TransientToolError,
    bounded_result,
    validated_knowledge_result,
)


@dataclass(frozen=True)
class ToolProgress:
    name: str
    tool_call_id: str
    status: str
    attempt: int
    message: str


@dataclass(frozen=True)
class RetrievalProgress:
    tool_call_id: str
    stage: str
    message: str


@dataclass(frozen=True)
class ToolOutcome:
    message: ToolMessage
    terminal_status: str
    attempt: int


class ToolExecutor:
    def __init__(self, timeout_seconds: float = 5, max_attempts: int = 2) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_attempts not in (1, 2):
            raise ValueError("max_attempts must be 1 or 2")
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts

    @staticmethod
    def _outcome(
        call_id: str,
        name: str,
        payload: dict[str, Any],
        terminal_status: str,
        attempt: int,
    ) -> ToolOutcome:
        content = bounded_result(payload)
        final_status = json.loads(content).get("status")
        terminal_status = {
            "ok": "succeeded",
            "not_found": "not_found",
            "error": "failed",
        }.get(final_status, terminal_status)
        return ToolOutcome(
            ToolMessage(
                content=content,
                tool_call_id=call_id,
                name=name or None,
                status="error" if terminal_status == "failed" else "success",
            ),
            terminal_status,
            attempt,
        )

    @staticmethod
    def _database_code(error: DBAPIError) -> int | None:
        args = getattr(error.orig, "args", ())
        if args and isinstance(args[0], int):
            return args[0]
        return None

    @classmethod
    def _is_transient(cls, error: BaseException) -> bool:
        if isinstance(error, TransientToolError | SQLAlchemyTimeoutError):
            return True
        if isinstance(error, IntegrityError):
            return False
        if isinstance(error, DBAPIError) and error.connection_invalidated:
            return True
        if isinstance(error, OperationalError | InterfaceError):
            return cls._database_code(error) in {
                1205,
                1213,
                2002,
                2003,
                2006,
                2013,
                2055,
            }
        return False

    @classmethod
    def _successful_outcome(
        cls,
        call_id: str,
        name: str,
        raw_message: object,
        attempt: int,
        *,
        max_bytes: int,
    ) -> ToolOutcome:
        if isinstance(raw_message, ToolMessage):
            raw_content = raw_message.content
        else:
            raw_content = raw_message
        if not isinstance(raw_content, str):
            return cls._outcome(
                call_id,
                name,
                {"status": "error", "code": "INVALID_TOOL_RESULT"},
                "failed",
                attempt,
            )
        try:
            payload = json.loads(raw_content)
        except (json.JSONDecodeError, TypeError):
            return cls._outcome(
                call_id,
                name,
                {"status": "error", "code": "INVALID_TOOL_RESULT"},
                "failed",
                attempt,
            )
        if not isinstance(payload, dict) or payload.get("status") not in {
            "ok",
            "not_found",
            "error",
        }:
            return cls._outcome(
                call_id,
                name,
                {"status": "error", "code": "INVALID_TOOL_RESULT"},
                "failed",
                attempt,
            )
        terminal_status = {
            "ok": "succeeded",
            "not_found": "not_found",
            "error": "failed",
        }[payload["status"]]
        if max_bytes > 4096:
            if name != "query_faq":
                return cls._outcome(
                    call_id,
                    name,
                    {"status": "error", "code": "INVALID_TOOL_RESULT"},
                    "failed",
                    attempt,
                )
            try:
                content = validated_knowledge_result(payload)
            except ValueError:
                return cls._outcome(
                    call_id,
                    name,
                    {"status": "error", "code": "INVALID_TOOL_RESULT"},
                    "failed",
                    attempt,
                )
            return ToolOutcome(
                ToolMessage(
                    content=content,
                    tool_call_id=call_id,
                    name=name,
                    status="success",
                ),
                terminal_status,
                attempt,
            )
        return cls._outcome(call_id, name, payload, terminal_status, attempt)

    async def run(
        self,
        call: dict,
        registry: ToolRegistry,
        *,
        deadline: float,
        progress_queue: asyncio.Queue[RetrievalProgress] | None = None,
    ) -> AsyncIterator[ToolProgress | RetrievalProgress | ToolOutcome]:
        call_id = str(call.get("id", ""))
        name = call.get("name")
        name = name if isinstance(name, str) else ""
        business_tool = registry.get(name)
        if business_tool is None:
            yield self._outcome(
                call_id,
                name,
                {"status": "error", "code": "UNKNOWN_TOOL"},
                "failed",
                0,
            )
            return
        if not isinstance(call.get("args"), dict):
            yield self._outcome(
                call_id,
                name,
                {"status": "error", "code": "INVALID_TOOL_ARGUMENTS"},
                "failed",
                0,
            )
            return
        policy = registry.policy(name)
        max_attempts = policy.max_attempts or self.max_attempts
        for attempt in range(1, max_attempts + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                yield self._outcome(
                    call_id,
                    name,
                    {"status": "error", "code": "TOOL_DEADLINE_EXCEEDED"},
                    "failed",
                    attempt - 1,
                )
                return

            yield ToolProgress(
                name=name,
                tool_call_id=call_id,
                status="running" if attempt == 1 else "retrying",
                attempt=attempt,
                message="工具正在执行" if attempt == 1 else "工具正在重试",
            )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                yield self._outcome(
                    call_id,
                    name,
                    {"status": "error", "code": "TOOL_DEADLINE_EXCEEDED"},
                    "failed",
                    attempt,
                )
                return

            args_schema = business_tool.args_schema
            if isinstance(args_schema, type) and hasattr(args_schema, "model_validate"):
                try:
                    args_schema.model_validate(call["args"])
                except ValidationError:
                    yield self._outcome(
                        call_id,
                        name,
                        {"status": "error", "code": "INVALID_TOOL_ARGUMENTS"},
                        "failed",
                        attempt,
                    )
                    return

            invoke = asyncio.create_task(business_tool.ainvoke(call))
            progress_get: asyncio.Task | None = None
            terminal_outcome: ToolOutcome | None = None
            retry = False
            raw_message: object = None
            attempt_deadline = (
                deadline
                if policy.shared_deadline
                else min(deadline, time.monotonic() + self.timeout_seconds)
            )
            try:
                while not invoke.done():
                    remaining = attempt_deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError
                    if progress_queue is None:
                        done, _ = await asyncio.wait({invoke}, timeout=remaining)
                    else:
                        progress_get = asyncio.create_task(progress_queue.get())
                        done, _ = await asyncio.wait(
                            {invoke, progress_get},
                            timeout=remaining,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                    if not done:
                        raise TimeoutError
                    if progress_get is not None and progress_get in done:
                        progress = progress_get.result()
                        progress_get = None
                        yield progress
                        continue
                    if progress_get is not None:
                        progress_get.cancel()
                        await asyncio.gather(progress_get, return_exceptions=True)
                        progress_get = None
                raw_message = invoke.result()
                if progress_queue is not None:
                    while not progress_queue.empty():
                        yield progress_queue.get_nowait()
            except asyncio.CancelledError:
                raise
            except (ValidationError, InvalidToolArguments):
                terminal_outcome = self._outcome(
                    call_id,
                    name,
                    {"status": "error", "code": "INVALID_TOOL_ARGUMENTS"},
                    "failed",
                    attempt,
                )
            except TimeoutError:
                deadline_expired = time.monotonic() >= deadline
                if deadline_expired:
                    terminal_outcome = self._outcome(
                        call_id,
                        name,
                        {"status": "error", "code": "TOOL_DEADLINE_EXCEEDED"},
                        "failed",
                        attempt,
                    )
                elif attempt == max_attempts:
                    terminal_outcome = self._outcome(
                        call_id,
                        name,
                        {"status": "error", "code": "TOOL_TIMEOUT"},
                        "failed",
                        attempt,
                    )
                else:
                    retry = True
            except ServiceError as error:
                terminal_outcome = self._outcome(
                    call_id,
                    name,
                    {"status": "error", "code": error.code},
                    "failed",
                    attempt,
                )
            except Exception as error:
                if self._is_transient(error):
                    if time.monotonic() >= deadline:
                        code = "TOOL_DEADLINE_EXCEEDED"
                    elif attempt < max_attempts:
                        retry = True
                        code = ""
                    else:
                        code = "TOOL_TEMPORARY_FAILURE"
                else:
                    code = "TOOL_EXECUTION_FAILED"
                if not retry:
                    terminal_outcome = self._outcome(
                        call_id,
                        name,
                        {"status": "error", "code": code},
                        "failed",
                        attempt,
                    )
            finally:
                if progress_get is not None:
                    progress_get.cancel()
                    await asyncio.gather(progress_get, return_exceptions=True)
                if not invoke.done():
                    invoke.cancel()
                await asyncio.gather(invoke, return_exceptions=True)

            if terminal_outcome is not None:
                yield terminal_outcome
                return
            if retry:
                continue
            yield self._successful_outcome(
                call_id,
                name,
                raw_message,
                attempt,
                max_bytes=policy.max_bytes,
            )
            return
