from __future__ import annotations

import json

from app.tools.business import ToolContext, build_registry
from app.tools.executor import ToolExecutor, ToolOutcome


async def execute(registry, name: str, args: dict, call_id: str = "call-1") -> ToolOutcome:
    events = [
        event
        async for event in ToolExecutor().run(
            {"name": name, "args": args, "id": call_id, "type": "tool_call"},
            registry,
            deadline=1e30,
        )
    ]
    outcome = events[-1]
    assert isinstance(outcome, ToolOutcome)
    return outcome


async def test_query_faq_requires_the_new_knowledge_dependency(repos, new_turn) -> None:
    _, faq, tickets = repos
    ctx = ToolContext(new_turn, "demo", "邮费是多少", "TK-test")
    registry = build_registry(ctx, faq, tickets)

    result = await execute(registry, "query_faq", {})

    assert json.loads(result.message.content)["code"] == "KNOWLEDGE_UNAVAILABLE"
    assert result.attempt == 1


async def test_create_ticket_is_idempotent_for_context_ticket_number(
    repos, new_turn
) -> None:
    conversations, faq, tickets = repos
    ctx = ToolContext(new_turn, "demo", "收到的商品损坏了", "TK-idempotent")
    registry = build_registry(ctx, faq, tickets)
    args = {"issue_description": "收到的商品损坏了", "ticket_type": "repair"}

    first = await execute(registry, "create_ticket", args, "ticket-1")
    second = await execute(registry, "create_ticket", args, "ticket-2")

    assert json.loads(first.message.content)["data"]["ticket_no"] == "TK-idempotent"
    assert json.loads(second.message.content) == json.loads(first.message.content)

    assert (await conversations.get(new_turn.conversation_id, "demo"))["status"] == "open"
