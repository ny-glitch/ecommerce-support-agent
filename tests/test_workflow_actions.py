from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.api.actions import router
from app.db.contracts import TurnRef
from app.errors import ServiceError
from app.services.actions import ActionService
from app.tools.executor import ToolExecutor
from app.tools.schemas import TicketInput
from app.workflow.contracts import ActionOffer


class MemoryActions:
    def __init__(self, offer: ActionOffer) -> None:
        self.offer = offer
        self.mark_calls = 0

    async def get_confirmable(
        self, conversation_id: str, action_id: str, user_id: str
    ) -> ActionOffer:
        if (
            conversation_id != self.offer.conversation_id
            or action_id != self.offer.action_id
            or user_id != "demo"
        ):
            raise ServiceError("ACTION_NOT_FOUND", "操作建议不存在", 409)
        return self.offer

    async def mark_completed(self, action_id: str, ticket_no: str) -> ActionOffer:
        self.mark_calls += 1
        if action_id != self.offer.action_id or ticket_no != self.offer.ticket_no:
            raise ServiceError("ACTION_CONFLICT", "操作建议冲突", 409)
        self.offer = self.offer.model_copy(update={"status": "completed"})
        return self.offer


class MemoryTickets:
    def __init__(self) -> None:
        self.rows: dict[str, tuple[str, str, str, str]] = {}

    async def create_once(
        self,
        ticket_no: str,
        conversation_id: str,
        user_id: str,
        issue_description: str,
        ticket_type: str,
    ) -> dict:
        candidate = (conversation_id, user_id, issue_description, ticket_type)
        existing = self.rows.get(ticket_no)
        if existing is not None and existing != candidate:
            raise ServiceError("TICKET_CONFLICT", "工单号已用于其他请求", 409)
        self.rows[ticket_no] = candidate
        return {
            "ticket_no": ticket_no,
            "conversation_id": conversation_id,
            "status": "pending",
        }


class UnusedFaq:
    pass


@pytest.fixture
def offered_action() -> ActionOffer:
    return ActionOffer(
        action_id=str(uuid4()),
        conversation_id=str(uuid4()),
        turn_id=str(uuid4()),
        ticket_no=f"TK-{uuid4().hex}",
        draft=TicketInput(
            issue_description="收到的商品外壳破损",
            ticket_type="complaint",
        ),
        status="offered",
    )


@pytest.fixture
def action_service(offered_action: ActionOffer) -> ActionService:
    actions = MemoryActions(offered_action)
    tickets = MemoryTickets()
    service = ActionService(
        actions,
        UnusedFaq(),
        tickets,
        ToolExecutor(timeout_seconds=1, max_attempts=1),
        SimpleNamespace(request_timeout_seconds=2),
    )
    service.test_actions = actions
    service.test_tickets = tickets
    return service


async def test_confirmation_is_the_only_point_that_creates_a_ticket(
    action_service: ActionService, offered_action: ActionOffer
) -> None:
    assert action_service.test_tickets.rows == {}

    result = await action_service.confirm(
        offered_action.conversation_id, offered_action.action_id
    )

    assert result == {
        "ticket_no": offered_action.ticket_no,
        "conversation_id": offered_action.conversation_id,
        "status": "completed",
        "action_id": offered_action.action_id,
    }
    assert action_service.test_tickets.rows == {
        offered_action.ticket_no: (
            offered_action.conversation_id,
            "demo",
            "收到的商品外壳破损",
            "complaint",
        )
    }


async def test_repeated_confirmation_returns_same_ticket(
    action_service: ActionService, offered_action: ActionOffer
) -> None:
    first = await action_service.confirm(
        offered_action.conversation_id, offered_action.action_id
    )
    second = await action_service.confirm(
        offered_action.conversation_id, offered_action.action_id
    )

    assert first["ticket_no"] == second["ticket_no"]
    assert len(action_service.test_tickets.rows) == 1


async def test_completed_confirmation_revalidates_the_persisted_ticket(
    action_service: ActionService, offered_action: ActionOffer
) -> None:
    await action_service.confirm(offered_action.conversation_id, offered_action.action_id)
    action_service.test_tickets.rows[offered_action.ticket_no] = (
        offered_action.conversation_id,
        "demo",
        "被篡改的问题",
        "other",
    )

    with pytest.raises(ServiceError) as caught:
        await action_service.confirm(
            offered_action.conversation_id, offered_action.action_id
        )

    assert caught.value.code == "TICKET_CONFLICT"
    assert caught.value.status_code == 409


def http_app(service: ActionService) -> FastAPI:
    app = FastAPI()
    app.state.action_service = service
    app.include_router(router)

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, error: ServiceError):
        return JSONResponse(
            {"error": {"code": error.code, "message": error.message}},
            status_code=error.status_code,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        return JSONResponse(
            {"error": {"code": "INVALID_REQUEST", "message": "请求参数无效"}},
            status_code=422,
        )

    return app


def test_confirmation_http_accepts_only_an_explicit_empty_json_object(
    action_service: ActionService, offered_action: ActionOffer
) -> None:
    path = (
        f"/api/conversations/{offered_action.conversation_id}/actions/"
        f"{offered_action.action_id}/confirm"
    )
    with TestClient(http_app(action_service)) as client:
        accepted = client.post(path, json={})
        assert accepted.status_code == 200
        assert accepted.json()["ticket_no"] == offered_action.ticket_no

        for body in (
            {"description": "客户端伪造"},
            {"tool": "query_order"},
            {"user_id": "other"},
            {"description": "x" * 100_000},
        ):
            rejected = client.post(path, json=body)
            assert rejected.status_code == 422
            assert rejected.json() == {
                "error": {"code": "INVALID_REQUEST", "message": "请求参数无效"}
            }

        assert client.post(path).status_code == 422
        assert client.post(path, content="[]", headers={"content-type": "application/json"}).status_code == 422


def test_confirmation_http_validates_uuid_paths_and_exposes_no_handoff_endpoint(
    action_service: ActionService, offered_action: ActionOffer
) -> None:
    with TestClient(http_app(action_service)) as client:
        bad_path = (
            f"/api/conversations/not-a-uuid/actions/{offered_action.action_id}/confirm"
        )
        assert client.post(bad_path, json={}).status_code == 422
        assert client.post(
            f"/api/conversations/{offered_action.conversation_id}/actions/not-a-uuid/confirm",
            json={},
        ).status_code == 422
        assert client.post(
            f"/api/conversations/{offered_action.conversation_id}/actions/"
            f"{offered_action.action_id}/handoff",
            json={},
        ).status_code == 404


async def test_tool_failure_returns_only_the_safe_business_code(
    action_service: ActionService, offered_action: ActionOffer
) -> None:
    class BrokenTickets:
        async def create_once(self, *args):
            raise ServiceError("TICKET_CONFLICT", "secret raw detail", 409)

    action_service.tickets = BrokenTickets()

    with pytest.raises(ServiceError) as caught:
        await action_service.confirm(
            offered_action.conversation_id, offered_action.action_id
        )

    assert caught.value.code == "TICKET_CONFLICT"
    assert caught.value.status_code == 409
    assert "secret" not in caught.value.message
