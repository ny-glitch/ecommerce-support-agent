from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

try:
    from scripts.evaluate_tools import (
        evaluate_case,
        run_evaluation,
        validate_output_path,
    )
except ModuleNotFoundError:
    evaluate_case = None
    run_evaluation = None
    validate_output_path = None


SESSION_ID = "11111111-1111-4111-8111-111111111111"
TURN_ID = "22222222-2222-4222-8222-222222222222"


def event(name: str, data: dict) -> dict:
    return {"event": name, "data": data}


def successful_events(*, tool: bool = True) -> list[dict]:
    events = [event("meta", {"session_id": SESSION_ID, "turn_id": TURN_ID})]
    if tool:
        events.extend(
            [
                event(
                    "tool_status",
                    {
                        "name": "query_logistics",
                        "tool_call_id": "call-1",
                        "status": "running",
                        "attempt": 1,
                        "message": "工具正在执行",
                    },
                ),
                event(
                    "tool_status",
                    {
                        "name": "query_logistics",
                        "tool_call_id": "call-1",
                        "status": "succeeded",
                        "attempt": 1,
                        "message": "工具执行完成",
                    },
                ),
            ]
        )
    events.extend(
        [
            event("token", {"content": "模拟物流"}),
            event("token", {"content": "运输中"}),
            event("done", {"session_id": SESSION_ID}),
        ]
    )
    return events


def tool_audit(
    *,
    turn_id: str = TURN_ID,
    calls: list[dict] | None = None,
    result_status: str = "ok",
    result_call_id: str = "call-1",
) -> list[dict]:
    if calls is None:
        calls = [
            {
                "name": "query_logistics",
                "args": {"order_id": "1001"},
                "id": "call-1",
                "type": "tool_call",
            }
        ]
    return [
        {
            "turn_id": turn_id,
            "role": "user",
            "content": "订单 1001 的物流到哪了",
            "tool_calls": None,
            "tool_call_id": None,
            "turn_status": "completed",
        },
        {
            "turn_id": turn_id,
            "role": "assistant",
            "content": "",
            "tool_calls": calls,
            "tool_call_id": calls[0]["id"] if calls else None,
            "turn_status": "completed",
        },
        {
            "turn_id": turn_id,
            "role": "tool",
            "content": json.dumps(
                {"status": result_status, "data": {"order_id": "1001"}},
                ensure_ascii=False,
            ),
            "tool_calls": None,
            "tool_call_id": result_call_id,
            "turn_status": "completed",
        },
        {
            "turn_id": turn_id,
            "role": "assistant",
            "content": "模拟物流运输中",
            "tool_calls": None,
            "tool_call_id": None,
            "turn_status": "completed",
        },
    ]


def logistics_case() -> dict:
    return {
        "id": "logistics-1001",
        "message": "订单 1001 的物流到哪了",
        "tool": "query_logistics",
        "args": {"order_id": "1001"},
        "result_status": "ok",
        "rubric": "说明模拟性质，忠实于工具结果",
    }


def test_complete_tool_turn_passes_and_retains_review_evidence():
    assert callable(evaluate_case), "tool evaluator is not implemented"

    report = evaluate_case(logistics_case(), successful_events(), tool_audit())

    assert report == {
        "case_id": "logistics-1001",
        "question": "订单 1001 的物流到哪了",
        "session_id": SESSION_ID,
        "turn_id": TURN_ID,
        "events": successful_events(),
        "tool_calls": [
            {
                "name": "query_logistics",
                "args": {"order_id": "1001"},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
        "tool_results": [
            {
                "tool_call_id": "call-1",
                "content": {"status": "ok", "data": {"order_id": "1001"}},
            }
        ],
        "final_text": "模拟物流运输中",
        "protocol_pass": True,
        "protocol_errors": [],
        "semantic_review": "pending_manual_review",
    }


def test_audit_is_filtered_to_the_meta_turn_before_counting_calls():
    other_turn = tool_audit(turn_id="other-turn")

    report = evaluate_case(
        logistics_case(), successful_events(), other_turn + tool_audit()
    )

    assert report["protocol_pass"] is True
    assert len(report["tool_calls"]) == 1
    assert len(report["tool_results"]) == 1


def test_http_success_without_done_is_not_a_pass():
    case = {"id": "chat", "message": "你好", "tool": None, "rubric": "礼貌回应"}

    report = evaluate_case(
        case,
        [
            event("meta", {"session_id": SESSION_ID, "turn_id": TURN_ID}),
            event("token", {"content": "你好"}),
        ],
        [
            {
                "turn_id": TURN_ID,
                "role": "user",
                "content": "你好",
                "tool_calls": None,
                "tool_call_id": None,
                "turn_status": "completed",
            },
            {
                "turn_id": TURN_ID,
                "role": "assistant",
                "content": "你好",
                "tool_calls": None,
                "tool_call_id": None,
                "turn_status": "completed",
            },
        ],
    )

    assert report["protocol_pass"] is False
    assert "missing_done" in report["protocol_errors"]
    assert report["semantic_review"] == "pending_manual_review"


@pytest.mark.parametrize(
    ("audit", "error"),
    [
        (
            tool_audit(
                calls=[
                    {
                        "name": "query_logistics",
                        "args": {"order_id": "1001"},
                        "id": "call-1",
                        "type": "tool_call",
                    },
                    {
                        "name": "query_order",
                        "args": {"order_id": "1001"},
                        "id": "call-2",
                        "type": "tool_call",
                    },
                ]
            ),
            "tool_call_count_mismatch",
        ),
        (tool_audit(result_call_id="different-call"), "unpaired_tool_call_id"),
        (
            tool_audit(
                calls=[
                    {
                        "name": "query_logistics",
                        "args": {"order_id": "1002"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ]
            ),
            "tool_arguments_mismatch",
        ),
        (tool_audit(result_status="not_found"), "tool_result_status_mismatch"),
    ],
)
def test_tool_audit_rejects_duplicate_unpaired_or_mislabeled_evidence(audit, error):
    report = evaluate_case(logistics_case(), successful_events(), audit)

    assert report["protocol_pass"] is False
    assert error in report["protocol_errors"]


def test_keyword_must_be_one_of_the_labeled_literal_alternatives():
    case = {
        "id": "return-policy",
        "message": "退货政策是什么",
        "tool": "query_faq",
        "keywords": ["退货政策", "退货"],
        "result_status": "ok",
        "rubric": "与FAQ一致",
    }
    audit = tool_audit(
        calls=[
            {
                "name": "query_faq",
                "args": {"keyword": "售后"},
                "id": "call-1",
                "type": "tool_call",
            }
        ]
    )

    report = evaluate_case(case, successful_events(), audit)

    assert report["protocol_pass"] is False
    assert "tool_arguments_mismatch" in report["protocol_errors"]


def test_no_tool_case_rejects_a_model_tool_request():
    case = {"id": "clarify", "message": "查物流", "tool": None, "rubric": "询问订单号"}

    report = evaluate_case(case, successful_events(), tool_audit())

    assert report["protocol_pass"] is False
    assert "unexpected_tool_call" in report["protocol_errors"]


def test_sse_error_is_separate_from_pending_semantic_review():
    events = [
        event("meta", {"session_id": SESSION_ID, "turn_id": TURN_ID}),
        event("error", {"code": "UPSTREAM_ERROR", "message": "暂不可用"}),
    ]

    report = evaluate_case(logistics_case(), events, [])

    assert report["protocol_pass"] is False
    assert "sse_error" in report["protocol_errors"]
    assert report["semantic_review"] == "pending_manual_review"


def make_http_app(mode: str = "success") -> FastAPI:
    app = FastAPI()

    @app.post("/api/chat")
    async def chat(payload: dict):
        if mode == "http_error":
            return JSONResponse({"secret_detail": "must not be reported"}, status_code=503)
        if mode == "sse_error":
            body = (
                f"event: meta\ndata: {json.dumps({'session_id': SESSION_ID, 'turn_id': TURN_ID})}\n\n"
                'event: error\ndata: {"code":"UPSTREAM_ERROR","message":"暂不可用"}\n\n'
            )
        else:
            body = "".join(
                f"event: {item['event']}\ndata: {json.dumps(item['data'], ensure_ascii=False)}\n\n"
                for item in successful_events()
            )
        return StreamingResponse(iter([body]), media_type="text/event-stream")

    return app


@pytest.mark.asyncio
async def test_runner_reuses_session_for_followup_and_scores_each_turn():
    first = logistics_case()
    first["followup"] = {
        "id": "remember-order-1001",
        "message": "我刚才问的是哪个订单",
        "tool": None,
        "rubric": "回答1001",
    }
    seen_sessions: list[str] = []
    call_count = 0
    app = FastAPI()

    @app.post("/api/chat")
    async def chat(payload: dict):
        nonlocal call_count
        call_count += 1
        seen_sessions.append(payload.get("session_id"))
        current_turn = TURN_ID if call_count == 1 else "33333333-3333-4333-8333-333333333333"
        current = successful_events(tool=call_count == 1)
        for item in current:
            if item["event"] == "meta":
                item["data"]["turn_id"] = current_turn
            if item["event"] == "token" and call_count == 2:
                item["data"]["content"] = "1001"
        body = "".join(
            f"event: {item['event']}\ndata: {json.dumps(item['data'], ensure_ascii=False)}\n\n"
            for item in current
        )
        return StreamingResponse(iter([body]), media_type="text/event-stream")

    second_turn = [
        {
            "turn_id": "33333333-3333-4333-8333-333333333333",
            "role": "user",
            "content": "我刚才问的是哪个订单",
            "tool_calls": None,
            "tool_call_id": None,
            "turn_status": "completed",
        },
        {
            "turn_id": "33333333-3333-4333-8333-333333333333",
            "role": "assistant",
            "content": "10011001",
            "tool_calls": None,
            "tool_call_id": None,
            "turn_status": "completed",
        },
    ]

    async def audit_reader(session_id: str) -> list[dict]:
        assert session_id == SESSION_ID
        return tool_audit() + second_turn

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        report = await run_evaluation(
            "http://test", [first], audit_reader, timeout=2, client=client
        )

    assert seen_sessions == [None, SESSION_ID]
    assert [result["protocol_pass"] for result in report["results"]] == [True, True]
    assert [result["case_id"] for result in report["results"]] == [
        "logistics-1001",
        "remember-order-1001",
    ]
    assert report["status"] == "pending_manual_review"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected_error", "expected_audit_sessions"),
    [
        ("http_error", "http_error", []),
        ("sse_error", "sse_error", [SESSION_ID]),
    ],
)
async def test_runner_records_safe_protocol_failures(
    mode, expected_error, expected_audit_sessions
):
    audit_sessions: list[str] = []

    async def audit_reader(session_id: str) -> list[dict]:
        audit_sessions.append(session_id)
        return []

    transport = httpx.ASGITransport(app=make_http_app(mode))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        report = await run_evaluation(
            "http://test", [logistics_case()], audit_reader, timeout=2, client=client
        )

    result = report["results"][0]
    assert report["status"] == "failed"
    assert result["protocol_pass"] is False
    assert expected_error in result["protocol_errors"]
    assert audit_sessions == expected_audit_sessions
    assert "secret_detail" not in json.dumps(report, ensure_ascii=False)


def test_output_path_must_be_inside_eval_reports(tmp_path):
    assert callable(validate_output_path), "tool evaluator is not implemented"

    with pytest.raises(ValueError, match="evals/reports"):
        validate_output_path(tmp_path / "outside.json")

    root = Path(__file__).resolve().parents[1]
    allowed = root / "evals" / "reports" / "controlled.json"
    assert validate_output_path(allowed) == allowed.resolve()
