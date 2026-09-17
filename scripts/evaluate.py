#!/usr/bin/env python3
"""Run the labeled evaluation set against a running HTTP service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib import error, request
from uuid import UUID


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "evals" / "cases.json"
REQUIRED_EXTRACTION_FIELDS = ("order_id", "request_type", "expected_resolution")


class RequestFailure(Exception):
    def __init__(self, kind: str, message: str, status_code: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.status_code = status_code

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"type": self.kind, "message": self.message}
        if self.status_code is not None:
            result["status_code"] = self.status_code
        return result


def score_extraction(expected: dict[str, Any], actual: Any) -> bool:
    """Compare all labeled fields, allowing listed resolution equivalents."""
    if not isinstance(actual, dict):
        return False
    if any(field not in expected or field not in actual for field in REQUIRED_EXTRACTION_FIELDS):
        return False
    if actual["order_id"] != expected["order_id"]:
        return False
    if actual["request_type"] != expected["request_type"]:
        return False
    resolutions = expected["expected_resolution"]
    if isinstance(resolutions, list):
        return actual["expected_resolution"] in resolutions
    return actual["expected_resolution"] == resolutions


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[str, bytes]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
        method="POST",
    )
    # Call evaluation targets directly so localhost is not diverted through a
    # desktop proxy. HTTPS certificate verification remains enabled by urllib.
    opener = request.build_opener(request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as response:
            return response.headers.get_content_type(), response.read()
    except error.HTTPError as exc:
        raise RequestFailure("http_error", f"HTTP {exc.code}", exc.code) from exc
    except (error.URLError, TimeoutError, OSError) as exc:
        raise RequestFailure("transport_error", "无法连接评估服务") from exc


def _extract(base_url: str, description: str, timeout: float) -> Any:
    content_type, body = _post_json(
        f"{base_url.rstrip('/')}/api/after-sales/extract",
        {"description": description},
        timeout,
    )
    try:
        actual = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RequestFailure("protocol_error", "提取接口未返回有效 JSON") from exc
    if content_type != "application/json":
        raise RequestFailure("protocol_error", "提取接口 Content-Type 不是 application/json")
    return actual


def _parse_sse(body: bytes) -> list[tuple[str, Any]]:
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RequestFailure("protocol_error", "聊天流不是 UTF-8") from exc
    events: list[tuple[str, Any]] = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        name = None
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if not name or not data_lines:
            continue
        try:
            data = json.loads("\n".join(data_lines))
        except json.JSONDecodeError as exc:
            raise RequestFailure("protocol_error", "SSE data 不是有效 JSON") from exc
        events.append((name, data))
    return events


def _chat_turn(
    base_url: str, message: str, session_id: str | None, timeout: float
) -> tuple[str, str]:
    payload: dict[str, Any] = {"message": message}
    if session_id is not None:
        payload["session_id"] = session_id
    content_type, body = _post_json(f"{base_url.rstrip('/')}/api/chat", payload, timeout)
    if content_type != "text/event-stream":
        raise RequestFailure("protocol_error", "聊天接口 Content-Type 不是 text/event-stream")
    events = _parse_sse(body)
    meta = next((data for name, data in events if name == "meta"), None)
    stream_error = next((data for name, data in events if name == "error"), None)
    done = next((data for name, data in events if name == "done"), None)
    if stream_error is not None:
        code = stream_error.get("code", "UNKNOWN") if isinstance(stream_error, dict) else "UNKNOWN"
        raise RequestFailure("stream_error", f"聊天流返回 error 事件：{code}")
    if not isinstance(meta, dict) or not isinstance(meta.get("session_id"), str):
        raise RequestFailure("protocol_error", "聊天流缺少 meta 或 done 事件")
    returned_session_id = meta["session_id"]
    try:
        UUID(returned_session_id)
    except ValueError as exc:
        raise RequestFailure("protocol_error", "聊天流 meta 包含无效 session_id") from exc
    if (
        not isinstance(done, dict)
        or done.get("session_id") != returned_session_id
        or (session_id is not None and returned_session_id != session_id)
    ):
        raise RequestFailure("protocol_error", "聊天流 session_id 不一致")
    answer_parts = [
        data["content"]
        for name, data in events
        if name == "token" and isinstance(data, dict) and isinstance(data.get("content"), str)
    ]
    return returned_session_id, "".join(answer_parts)


def _validate_cases(cases: Any) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(cases, dict):
        raise ValueError("评估集根节点必须是对象")
    extraction = cases.get("extraction")
    chat = cases.get("chat")
    if not isinstance(extraction, list) or not isinstance(chat, list):
        raise ValueError("评估集必须包含 extraction 和 chat 数组")
    for case in extraction:
        if not isinstance(case, dict) or not all(key in case for key in ("id", "description", "expected")):
            raise ValueError("提取样例缺少 id、description 或 expected")
        expected = case["expected"]
        if not isinstance(expected, dict) or any(field not in expected for field in REQUIRED_EXTRACTION_FIELDS):
            raise ValueError(f"提取样例 {case.get('id', '<unknown>')} 的 expected 缺少字段")
    for case in chat:
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("turns"), list)
            or not case["turns"]
            or not isinstance(case.get("manual_rubric"), list)
            or not case["manual_rubric"]
        ):
            raise ValueError("客服样例必须包含非空 turns 和 manual_rubric")
    return {"extraction": extraction, "chat": chat}


def evaluate(base_url: str, cases: dict[str, list[dict[str, Any]]], timeout: float) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    extraction_passed = 0
    extraction_failed = 0
    chat_pending = 0
    http_failures = 0

    for case in cases["extraction"]:
        result: dict[str, Any] = {
            "id": case["id"],
            "kind": "extraction",
            "description": case["description"],
            "expected": case["expected"],
        }
        try:
            actual = _extract(base_url, case["description"], timeout)
            result["actual"] = actual
            passed = score_extraction(case["expected"], actual)
            result["status"] = "passed" if passed else "failed"
        except RequestFailure as exc:
            passed = False
            http_failures += 1
            result.update(status="failed", error=exc.as_dict())
        extraction_passed += int(passed)
        extraction_failed += int(not passed)
        results.append(result)

    for case in cases["chat"]:
        result = {
            "id": case["id"],
            "kind": "chat",
            "manual_rubric": case["manual_rubric"],
            "actual_turns": [],
        }
        session_id = None
        try:
            for message in case["turns"]:
                session_id, answer = _chat_turn(base_url, message, session_id, timeout)
                result["actual_turns"].append({"message": message, "response": answer})
            result["status"] = "pending_manual_review"
            chat_pending += 1
        except RequestFailure as exc:
            http_failures += 1
            result.update(status="failed", error=exc.as_dict())
        results.append(result)

    status = "failed" if extraction_failed or http_failures else (
        "pending_manual_review" if chat_pending else "passed"
    )
    return {
        "status": status,
        "summary": {
            "extraction_total": len(cases["extraction"]),
            "extraction_passed": extraction_passed,
            "extraction_failed": extraction_failed,
            "chat_total": len(cases["chat"]),
            "chat_pending_manual_review": chat_pending,
            "http_failures": http_failures,
        },
        "results": results,
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行第一章真实 HTTP 标注评估")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=65.0)
    args = parser.parse_args(argv)

    try:
        loaded = json.loads(args.cases.read_text(encoding="utf-8"))
        cases = _validate_cases(loaded)
        report = evaluate(args.base_url, cases, args.timeout)
        exit_code = 1 if report["status"] == "failed" else 0
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        report = {
            "status": "failed",
            "summary": {
                "extraction_total": 0,
                "extraction_passed": 0,
                "extraction_failed": 0,
                "chat_total": 0,
                "chat_pending_manual_review": 0,
                "http_failures": 0,
            },
            "results": [],
            "error": {"type": "invalid_cases", "message": str(exc)},
        }
        exit_code = 2
    _write_report(args.output, report)
    print(json.dumps(report["summary"], ensure_ascii=False))
    print(f"status={report['status']} report={args.output}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
