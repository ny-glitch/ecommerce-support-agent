import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from scripts.evaluate import score_extraction


ROOT = Path(__file__).resolve().parents[1]


class EvaluationHandler(BaseHTTPRequestHandler):
    sessions: dict[str, list[str]] = {}
    chat_mode = "success"

    def do_POST(self):  # noqa: N802 - stdlib HTTP handler API
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        if body.get("description") == "触发错误":
            self._json(502, {"error": {"code": "UPSTREAM_ERROR", "message": "暂不可用"}})
            return
        if self.path == "/api/after-sales/extract":
            actual = {
                "order_id": "A123",
                "request_type": "exchange",
                "expected_resolution": "换货",
            }
            if body["description"] == "模型漏字段":
                actual.pop("order_id")
            self._json(200, actual)
            return
        if self.path == "/api/chat":
            if self.chat_mode == "http_error":
                self._json(502, {"error": {"code": "UPSTREAM_ERROR", "message": "暂不可用"}})
                return
            session_id = body.get("session_id") or "11111111-1111-4111-8111-111111111111"
            self.sessions.setdefault(session_id, []).append(body["message"])
            answer = "你叫小林，耳机有杂音。" if len(self.sessions[session_id]) == 2 else "请提供订单号。"
            reported_session_id = session_id
            done_session_id = session_id
            if self.chat_mode == "invalid_uuid":
                reported_session_id = "not-a-uuid"
                done_session_id = reported_session_id
            elif self.chat_mode == "mismatched_done":
                done_session_id = "22222222-2222-4222-8222-222222222222"
            elif self.chat_mode == "changed_meta" and body.get("session_id"):
                reported_session_id = "33333333-3333-4333-8333-333333333333"
                done_session_id = reported_session_id
            meta = (
                f'event: meta\ndata: {{"session_id":"{reported_session_id}","estimated_input_tokens":8,'
                '"token_count_is_estimate":true,"dropped_turns":0}\n\n'
            )
            if self.chat_mode == "stream_error":
                events_text = meta + 'event: error\ndata: {"code":"UPSTREAM_ERROR","message":"暂不可用"}\n\n'
            elif self.chat_mode == "missing_done":
                events_text = meta + f'event: token\ndata: {json.dumps({"content": answer}, ensure_ascii=False)}\n\n'
            else:
                events_text = (
                    meta
                    + f'event: token\ndata: {json.dumps({"content": answer}, ensure_ascii=False)}\n\n'
                    + f'event: done\ndata: {{"session_id":"{done_session_id}"}}\n\n'
                )
            events = events_text.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(events)))
            self.end_headers()
            self.wfile.write(events)
            return
        self._json(404, {"error": {"code": "NOT_FOUND", "message": "不存在"}})

    def _json(self, status, body):
        payload = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        return


@pytest.fixture
def evaluation_server():
    EvaluationHandler.sessions = {}
    EvaluationHandler.chat_mode = "success"
    server = ThreadingHTTPServer(("127.0.0.1", 0), EvaluationHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize(
    ("expected", "actual", "want"),
    [
        (
            {"order_id": "A123", "request_type": "exchange", "expected_resolution": ["换货", "更换新品"]},
            {"order_id": "A123", "request_type": "exchange", "expected_resolution": "换货"},
            True,
        ),
        (
            {"order_id": None, "request_type": "unknown", "expected_resolution": None},
            {"order_id": "FAKE", "request_type": "unknown", "expected_resolution": None},
            False,
        ),
        (
            {"order_id": "A123", "request_type": "exchange", "expected_resolution": "换货"},
            {"request_type": "exchange", "expected_resolution": "换货"},
            False,
        ),
    ],
)
def test_score_extraction_requires_all_fields_and_accepts_labeled_equivalents(expected, actual, want):
    assert score_extraction(expected, actual) is want


def run_cli(tmp_path, base_url, cases):
    cases_path = tmp_path / "cases.json"
    report_path = tmp_path / "report.json"
    cases_path.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate.py"),
            "--base-url",
            base_url,
            "--cases",
            str(cases_path),
            "--output",
            str(report_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    return completed, json.loads(report_path.read_text(encoding="utf-8"))


def test_cli_records_actual_outputs_and_leaves_chat_for_manual_review(tmp_path, evaluation_server):
    cases = {
        "extraction": [
            {
                "id": "exchange",
                "description": "订单 A123 到货破损，希望换一个新的",
                "expected": {
                    "order_id": "A123",
                    "request_type": "exchange",
                    "expected_resolution": ["换货", "更换新品"],
                },
            }
        ],
        "chat": [
            {
                "id": "memory",
                "turns": ["我叫小林，耳机有杂音", "我叫什么，商品有什么问题？"],
                "manual_rubric": ["准确记住姓名和商品问题", "不编造订单状态"],
            }
        ],
    }

    completed, report = run_cli(tmp_path, evaluation_server, cases)

    assert completed.returncode == 0, completed.stderr
    assert report["summary"] == {
        "extraction_total": 1,
        "extraction_passed": 1,
        "extraction_failed": 0,
        "chat_total": 1,
        "chat_pending_manual_review": 1,
        "http_failures": 0,
    }
    assert report["status"] == "pending_manual_review"
    assert report["results"][0]["actual"]["expected_resolution"] == "换货"
    chat = report["results"][1]
    assert chat["status"] == "pending_manual_review"
    assert chat["manual_rubric"] == ["准确记住姓名和商品问题", "不编造订单状态"]
    assert [turn["response"] for turn in chat["actual_turns"]] == ["请提供订单号。", "你叫小林，耳机有杂音。"]


def test_cli_returns_nonzero_for_structured_mismatch(tmp_path, evaluation_server):
    cases = {
        "extraction": [
            {
                "id": "wrong-label",
                "description": "订单 A123 到货破损，希望换一个新的",
                "expected": {"order_id": "A123", "request_type": "refund", "expected_resolution": "退款"},
            }
        ],
        "chat": [],
    }

    completed, report = run_cli(tmp_path, evaluation_server, cases)

    assert completed.returncode == 1
    assert report["status"] == "failed"
    assert report["summary"]["extraction_failed"] == 1


@pytest.mark.parametrize("description", ["模型漏字段", "触发错误"])
def test_cli_counts_malformed_or_error_responses_as_failures(tmp_path, evaluation_server, description):
    cases = {
        "extraction": [
            {
                "id": "failure",
                "description": description,
                "expected": {"order_id": None, "request_type": "unknown", "expected_resolution": None},
            }
        ],
        "chat": [],
    }

    completed, report = run_cli(tmp_path, evaluation_server, cases)

    assert completed.returncode == 1
    assert report["status"] == "failed"
    assert report["results"][0]["status"] == "failed"
    if description == "触发错误":
        assert report["summary"]["http_failures"] == 1
        assert report["results"][0]["error"]["type"] == "http_error"


@pytest.mark.parametrize(
    ("mode", "turns"),
    [
        ("invalid_uuid", ["你好"]),
        ("mismatched_done", ["你好"]),
        ("changed_meta", ["我叫小林", "我叫什么？"]),
    ],
)
def test_cli_rejects_invalid_or_inconsistent_session_metadata(
    tmp_path, evaluation_server, mode, turns
):
    EvaluationHandler.chat_mode = mode
    cases = {
        "extraction": [],
        "chat": [
            {
                "id": mode,
                "turns": turns,
                "manual_rubric": ["只在完整且一致的会话协议后进入人工评审"],
            }
        ],
    }

    completed, report = run_cli(tmp_path, evaluation_server, cases)

    assert completed.returncode == 1
    assert report["status"] == "failed"
    assert report["summary"]["chat_pending_manual_review"] == 0
    assert report["summary"]["http_failures"] == 1
    assert report["results"][0]["status"] == "failed"
    assert report["results"][0]["error"]["type"] == "protocol_error"


def run_demo(base_url):
    return subprocess.run(
        ["bash", str(ROOT / "scripts" / "demo.sh")],
        cwd=ROOT,
        env={**os.environ, "BASE_URL": base_url, "PYTHON": sys.executable},
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )


def test_demo_reuses_meta_session_and_runs_extraction(evaluation_server):
    completed = run_demo(evaluation_server)

    assert completed.returncode == 0, completed.stderr
    assert EvaluationHandler.sessions == {
        "11111111-1111-4111-8111-111111111111": [
            "我叫小林，刚买的耳机有杂音",
            "我叫什么，商品出了什么问题？",
        ]
    }
    assert '"request_type": "exchange"' in completed.stdout


@pytest.mark.parametrize("mode", ["http_error", "stream_error", "missing_done"])
def test_demo_returns_nonzero_when_chat_does_not_complete(evaluation_server, mode):
    EvaluationHandler.chat_mode = mode

    completed = run_demo(evaluation_server)

    assert completed.returncode != 0
