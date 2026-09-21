from __future__ import annotations

import json
import time
from typing import Any

from pydantic import ValidationError

from app.db.contracts import TurnRef
from app.errors import ServiceError
from app.tools.business import ToolContext, build_registry
from app.tools.executor import ToolOutcome


_SAFE_TOOL_ERRORS: dict[str, tuple[str, int]] = {
    "CONVERSATION_NOT_FOUND": ("会话不存在", 404),
    "TICKET_CONFLICT": ("工单号已用于其他请求", 409),
    "INVALID_TOOL_ARGUMENTS": ("已保存的工单信息无效", 409),
    "TOOL_DEADLINE_EXCEEDED": ("工单创建超时，请重试", 504),
    "TOOL_TIMEOUT": ("工单创建超时，请重试", 504),
    "TOOL_TEMPORARY_FAILURE": ("工单服务暂时不可用，请重试", 503),
}


def _action_conflict() -> ServiceError:
    return ServiceError(
        "ACTION_CONFLICT",
        "操作建议不存在、不可执行或与已保存内容冲突",
        409,
    )


def _tool_error(code: object) -> ServiceError:
    safe_code = code if isinstance(code, str) and code else "INVALID_TOOL_RESULT"
    message, status = _SAFE_TOOL_ERRORS.get(
        safe_code, ("工单创建失败，请重试", 502)
    )
    return ServiceError(safe_code, message, status)


class ActionService:
    def __init__(self, actions, faq, tickets, executor, settings) -> None:
        self.actions = actions
        self.faq = faq
        self.tickets = tickets
        self.executor = executor
        self.settings = settings

    async def confirm(
        self,
        conversation_id: str,
        action_id: str,
        user_id: str = "demo",
    ) -> dict[str, str]:
        try:
            offer = await self.actions.get_confirmable(
                conversation_id, action_id, user_id
            )
        except ValidationError as exc:
            raise _action_conflict() from exc

        context = ToolContext(
            ref=TurnRef(offer.conversation_id, offer.turn_id),
            user_id=user_id,
            user_message=offer.draft.issue_description,
            ticket_no=offer.ticket_no,
        )
        registry = build_registry(context, self.faq, self.tickets)
        call = {
            "name": "create_ticket",
            "id": f"action-{offer.action_id}",
            "args": offer.draft.model_dump(),
            "type": "tool_call",
        }
        outcome: ToolOutcome | None = None
        deadline = time.monotonic() + self.settings.request_timeout_seconds
        async for event in self.executor.run(call, registry, deadline=deadline):
            if isinstance(event, ToolOutcome):
                outcome = event

        if outcome is None:
            raise _tool_error("INVALID_TOOL_RESULT")
        try:
            payload: Any = json.loads(outcome.message.content)
        except (TypeError, json.JSONDecodeError) as exc:
            raise _tool_error("INVALID_TOOL_RESULT") from exc
        if not isinstance(payload, dict) or payload.get("status") != "ok":
            code = payload.get("code") if isinstance(payload, dict) else None
            raise _tool_error(code)
        data = payload.get("data")
        if (
            outcome.terminal_status != "succeeded"
            or not isinstance(data, dict)
            or data.get("ticket_no") != offer.ticket_no
            or data.get("conversation_id") != offer.conversation_id
        ):
            raise _tool_error("INVALID_TOOL_RESULT")

        completed = await self.actions.mark_completed(
            offer.action_id, offer.ticket_no
        )
        if (
            completed.action_id != offer.action_id
            or completed.conversation_id != offer.conversation_id
            or completed.ticket_no != offer.ticket_no
            or completed.status != "completed"
            or completed.draft != offer.draft
        ):
            raise _action_conflict()
        return {
            "ticket_no": completed.ticket_no,
            "conversation_id": completed.conversation_id,
            "status": completed.status,
            "action_id": completed.action_id,
        }

