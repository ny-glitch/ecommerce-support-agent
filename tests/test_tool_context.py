from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import app.context as context_module
from app.config import Settings
from app.db.contracts import StoredTurn
from app.errors import ServiceError


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        llm_base_url="https://api.example.com/v1",
        llm_model="test-model",
        llm_api_key="test-key",
        context_window_tokens=2_000,
        max_output_tokens=200,
        token_safety_margin=100,
        max_history_turns=12,
    )


def stored_tool_turn(turn_id: str, user: str, result: str) -> StoredTurn:
    call_id = f"call-{turn_id}"
    return StoredTurn(
        turn_id=turn_id,
        messages=(
            HumanMessage(user),
            AIMessage(
                "",
                tool_calls=[
                    {
                        "id": call_id,
                        "name": "query_order",
                        "args": {"order_id": turn_id},
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(result, tool_call_id=call_id, name="query_order"),
            AIMessage(f"订单 {turn_id} 的演示结果"),
        ),
    )


def test_tool_context_drops_an_old_four_message_turn_as_a_unit(
    settings: Settings,
) -> None:
    old_turn = stored_tool_turn("old", "旧问题" * 500, "旧结果" * 500)
    latest_turn = stored_tool_turn("new", "新问题", '{"status":"ok"}')
    turns = [old_turn, latest_turn]

    result = context_module.build_tool_context(
        "客服",
        turns,
        "现在的问题",
        settings,
        tool_schemas=[],
    )

    assert result.retained_turns == [latest_turn]
    assert result.messages[1:5] == list(latest_turn.messages)
    assert result.dropped_turns == 1
    assert turns == [old_turn, latest_turn]


def test_tool_schema_and_tool_result_bytes_increase_estimate(
    settings: Settings,
) -> None:
    plain = context_module.build_tool_context(
        "客服", [], "查询", settings, tool_schemas=[]
    )
    with_schema = context_module.build_tool_context(
        "客服",
        [],
        "查询",
        settings,
        tool_schemas=[
            {
                "type": "function",
                "function": {
                    "name": "query_order",
                    "parameters": {
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}},
                    },
                },
            }
        ],
    )
    call = AIMessage(
        "",
        tool_calls=[
            {
                "id": "call-123",
                "name": "query_order",
                "args": {"order_id": "A1002"},
                "type": "tool_call",
            }
        ],
    )
    short_result = context_module.build_tool_context(
        "客服",
        [],
        "查询",
        settings,
        tool_schemas=[],
        current_tool_messages=[
            call,
            ToolMessage("ok", tool_call_id="call-123", name="query_order"),
        ],
    )
    long_result = context_module.build_tool_context(
        "客服",
        [],
        "查询",
        settings,
        tool_schemas=[],
        current_tool_messages=[
            call,
            ToolMessage(
                "模拟结果" * 30,
                tool_call_id="call-123",
                name="query_order",
            ),
        ],
    )

    assert with_schema.estimated_input_tokens > plain.estimated_input_tokens
    assert long_result.estimated_input_tokens > short_result.estimated_input_tokens


def test_tool_message_metadata_and_json_arguments_are_counted() -> None:
    compact = [
        AIMessage(
            "",
            tool_calls=[
                {
                    "id": "1",
                    "name": "q",
                    "args": {},
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage("ok", tool_call_id="1", name="q"),
    ]
    detailed = [
        AIMessage(
            "",
            tool_calls=[
                {
                    "id": "call-long-id",
                    "name": "query_logistics",
                    "args": {"order_id": "A1002"},
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            "ok", tool_call_id="call-long-id", name="query_logistics"
        ),
    ]

    assert context_module.estimate_tokens(detailed) > context_module.estimate_tokens(
        compact
    )


def test_tool_context_rejects_oversized_required_schema_or_input(
    settings: Settings,
) -> None:
    constrained = settings.model_copy(
        update={
            "context_window_tokens": 500,
            "max_output_tokens": 200,
            "token_safety_margin": 100,
        }
    )

    with pytest.raises(ServiceError) as exc_info:
        context_module.build_tool_context(
            "客服",
            [],
            "当前输入",
            constrained,
            tool_schemas=[{"description": "说明" * 100}],
        )

    assert exc_info.value.code == "INPUT_TOO_LONG"
    assert exc_info.value.status_code == 413


def test_final_phase_keeps_current_tool_pair_and_does_not_mutate_turns(
    settings: Settings,
) -> None:
    turns = [stored_tool_turn("history", "以前的问题", '{"status":"ok"}')]
    original = list(turns)
    current_call = AIMessage(
        "",
        tool_calls=[
            {
                "id": "call-current",
                "name": "query_logistics",
                "args": {"order_id": "1001"},
                "type": "tool_call",
            }
        ],
    )
    current_result = ToolMessage(
        '{"status":"ok"}',
        tool_call_id="call-current",
        name="query_logistics",
    )

    result = context_module.build_tool_context(
        "客服",
        turns,
        "订单 1001 的物流到哪了",
        settings,
        tool_schemas=[],
        current_tool_messages=[current_call, current_result],
    )

    assert result.messages[-2:] == [current_call, current_result]
    assert result.retained_turns is not turns
    assert turns == original
