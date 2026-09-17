from __future__ import annotations

import json
import random

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.db.contracts import TurnRef
from app.tools.business import ToolContext, build_registry
from app.tools.executor import ToolExecutor, ToolOutcome
from app.tools.results import TransientToolError, bounded_result


class UnusedFaq:
    async def search(self, keyword: str) -> list[dict]:
        raise AssertionError("FAQ repository should not be called")


class RecordingTickets:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, str, str]] = []

    async def create_once(
        self,
        ticket_no: str,
        conversation_id: str,
        user_id: str,
        issue_description: str,
        ticket_type: str,
    ) -> dict:
        self.calls.append(
            (ticket_no, conversation_id, user_id, issue_description, ticket_type)
        )
        if len(self.calls) == 1:
            raise TransientToolError("temporary database failure")
        return {
            "ticket_no": ticket_no,
            "conversation_id": conversation_id,
            "status": "pending",
        }


def context(message: str = "请查询订单 ORDER_17") -> ToolContext:
    return ToolContext(TurnRef("conversation-1", "turn-1"), "demo", message, "TK-17")


async def invoke(name: str, args: dict, *, message: str = "测试") -> dict:
    registry = build_registry(
        context(message), UnusedFaq(), RecordingTickets(), rng=random.Random(1)
    )
    outcome = [
        event
        async for event in ToolExecutor().run(
            {"name": name, "args": args, "id": "call-1", "type": "tool_call"},
            registry,
            deadline=1e30,
        )
    ][-1]
    assert isinstance(outcome, ToolOutcome)
    return json.loads(outcome.message.content)


def test_registry_exposes_only_five_business_tool_schemas() -> None:
    registry = build_registry(context(), UnusedFaq(), RecordingTickets())

    assert [tool.name for tool in registry.tools] == [
        "query_order",
        "query_product",
        "query_logistics",
        "query_faq",
        "create_ticket",
    ]
    assert registry.get("run_shell") is None
    schemas = registry.schemas()
    assert {schema["function"]["name"] for schema in schemas} == {
        "query_order",
        "query_product",
        "query_logistics",
        "query_faq",
        "create_ticket",
    }
    assert all(schema["type"] == "function" for schema in schemas)
    assert all(
        "user_id" not in schema["function"]["parameters"].get("properties", {})
        for schema in schemas
    )


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("query_order", {"order_id": ""}),
        ("query_order", {"order_id": "A" * 65}),
        ("query_order", {"order_id": "ORDER/17"}),
        ("query_product", {"keyword": " "}),
        ("query_product", {"keyword": "商" * 101}),
        ("query_faq", {"keyword": "退"}),
        ("query_faq", {"keyword": "退" * 33}),
        ("create_ticket", {"issue_description": "", "ticket_type": "refund"}),
        (
            "create_ticket",
            {"issue_description": "x" * 2001, "ticket_type": "refund"},
        ),
        (
            "create_ticket",
            {"issue_description": "需要帮助", "ticket_type": "chargeback"},
        ),
        ("query_product", {"keyword": "鞋", "unknown": True}),
    ],
)
async def test_tool_inputs_reject_out_of_contract_values(name: str, args: dict) -> None:
    payload = await invoke(name, args, message="退换货物流商品鞋")

    assert payload["status"] == "error"
    assert payload["code"] == "INVALID_TOOL_ARGUMENTS"


async def test_inputs_are_stripped_before_business_use() -> None:
    payload = await invoke(
        "query_order", {"order_id": "  ORDER_17  "}, message="ORDER_17"
    )

    assert payload["status"] == "ok"
    assert payload["data"]["order_id"] == "ORDER_17"


async def test_simulated_order_preserves_requested_identifier() -> None:
    payload = await invoke(
        "query_order", {"order_id": "ORDER_17"}, message="ORDER_17"
    )

    assert payload["status"] == "ok"
    assert payload["simulated"] is True
    assert payload["data"]["order_id"] == "ORDER_17"
    assert payload["data"]["order_status"]


async def test_simulated_product_preserves_keyword() -> None:
    payload = await invoke("query_product", {"keyword": "运动鞋"}, message="运动鞋")

    assert payload["status"] == "ok"
    assert payload["simulated"] is True
    assert payload["data"][0]["keyword"] == "运动鞋"


async def test_simulated_logistics_has_ordered_timestamps() -> None:
    payload = await invoke(
        "query_logistics", {"order_id": "ORDER_17"}, message="ORDER_17"
    )

    assert payload["status"] == "ok"
    assert payload["simulated"] is True
    assert payload["data"]["order_id"] == "ORDER_17"
    timestamps = [item["time"] for item in payload["data"]["tracking"]]
    assert timestamps == sorted(timestamps)


async def test_ticket_retries_with_same_server_context() -> None:
    tickets = RecordingTickets()
    registry = build_registry(context("商品坏了"), UnusedFaq(), tickets)
    events = [
        event
        async for event in ToolExecutor(max_attempts=2).run(
            {
                "name": "create_ticket",
                "args": {
                    "issue_description": "  商品坏了  ",
                    "ticket_type": "repair",
                },
                "id": "ticket-call",
                "type": "tool_call",
            },
            registry,
            deadline=1e30,
        )
    ]

    outcome = events[-1]
    assert isinstance(outcome, ToolOutcome)
    assert outcome.terminal_status == "succeeded"
    assert len(tickets.calls) == 2
    assert tickets.calls[0] == tickets.calls[1]
    assert tickets.calls[0] == (
        "TK-17",
        "conversation-1",
        "demo",
        "商品坏了",
        "repair",
    )


async def test_ticket_type_is_stripped_before_literal_validation() -> None:
    payload = await invoke(
        "create_ticket",
        {"issue_description": "  需要退款  ", "ticket_type": "  refund  "},
        message="需要退款",
    )

    assert payload["status"] == "ok"
    assert payload["data"]["ticket_no"] == "TK-17"


def test_bounded_result_is_valid_utf8_json_and_preserves_essential_fields() -> None:
    result = bounded_result(
        {
            "status": "error",
            "code": "UPSTREAM_SAFE_CODE",
            "order_id": "ORDER_17",
            "description": "大" * 10_000,
        }
    )

    payload = json.loads(result)
    assert len(result.encode("utf-8")) <= 4096
    assert payload["status"] == "error"
    assert payload["code"] == "UPSTREAM_SAFE_CODE"
    assert payload["order_id"] == "ORDER_17"
    assert payload["truncated"] is True


def test_bounded_result_keeps_first_list_item_while_shortening_description() -> None:
    result = bounded_result(
        {
            "status": "ok",
            "data": [
                {"order_id": "ORDER-FIRST", "description": "大" * 10_000},
                {"order_id": "ORDER-SECOND", "description": "小" * 100},
            ],
        }
    )

    payload = json.loads(result)
    assert len(result.encode("utf-8")) <= 4096
    assert payload["status"] == "ok"
    assert payload["data"][0]["order_id"] == "ORDER-FIRST"
    assert payload["truncated"] is True


def test_tool_settings_default_and_attempt_validation() -> None:
    defaults = Settings(
        _env_file=None,
        llm_base_url="https://api.example/v1",
        llm_model="model",
        llm_api_key="key",
    )

    assert defaults.tool_timeout_seconds == 5
    assert defaults.tool_max_attempts == 2
    for invalid in (0, 3):
        with pytest.raises(ValidationError):
            defaults.model_copy(update={"tool_max_attempts": invalid}).model_validate(
                defaults.model_dump() | {"tool_max_attempts": invalid}
            )
