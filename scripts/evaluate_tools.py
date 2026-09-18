#!/usr/bin/env python3
"""Evaluate tool-chat protocol evidence from SSE and the durable audit log."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable
import json
from pathlib import Path
import sys
from typing import Any
from uuid import UUID

import httpx


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "evals" / "ch02-cases.json"
REPORTS_DIR = (ROOT / "evals" / "reports").resolve()
AuditReader = Callable[[str], Awaitable[list[dict[str, Any]]]]


class EvaluationFailure(Exception):
    """A safe, categorical failure that may be written to an evaluation report."""

    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


def _append_error(errors: list[str], error: str) -> None:
    if error not in errors:
        errors.append(error)


def _event_data(events: list[dict[str, Any]], name: str) -> list[Any]:
    return [item.get("data") for item in events if item.get("event") == name]


def _parse_tool_result(content: object) -> object:
    if not isinstance(content, str):
        return content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content


def _arguments_match(case: dict[str, Any], actual: object) -> bool:
    if not isinstance(actual, dict):
        return False
    expected = case.get("args", {})
    if not isinstance(expected, dict):
        return False
    if any(actual.get(key) != value for key, value in expected.items()):
        return False
    keywords = case.get("keywords")
    if keywords is not None:
        if not isinstance(keywords, list) or actual.get("keyword") not in keywords:
            return False
    if case.get("tool") == "create_ticket":
        description = actual.get("issue_description")
        if not isinstance(description, str) or not description.strip():
            return False
    return True


def evaluate_case(
    case: dict[str, Any],
    events: list[dict[str, Any]],
    audit: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check one streamed turn against only its own durable audit rows."""
    errors: list[str] = []
    metas = _event_data(events, "meta")
    dones = _event_data(events, "done")
    tokens = _event_data(events, "token")
    stream_errors = _event_data(events, "error")

    meta = metas[0] if len(metas) == 1 and isinstance(metas[0], dict) else {}
    session_id = meta.get("session_id") if isinstance(meta.get("session_id"), str) else None
    turn_id = meta.get("turn_id") if isinstance(meta.get("turn_id"), str) else None

    if len(metas) != 1 or not events or events[0].get("event") != "meta":
        _append_error(errors, "invalid_meta")
    for value, error in ((session_id, "invalid_session_id"), (turn_id, "invalid_turn_id")):
        if value is None:
            _append_error(errors, error)
        else:
            try:
                UUID(value)
            except ValueError:
                _append_error(errors, error)
    if stream_errors:
        _append_error(errors, "sse_error")
    if len(dones) != 1:
        _append_error(errors, "missing_done" if not dones else "duplicate_done")
    elif not isinstance(dones[0], dict) or dones[0].get("session_id") != session_id:
        _append_error(errors, "done_session_mismatch")
    if dones and (not events or events[-1].get("event") != "done"):
        _append_error(errors, "done_not_terminal")
    if any(
        item.get("event") not in {"meta", "tool_status", "token", "done", "error"}
        for item in events
    ):
        _append_error(errors, "unknown_sse_event")

    final_parts: list[str] = []
    for token in tokens:
        if not isinstance(token, dict) or not isinstance(token.get("content"), str):
            _append_error(errors, "invalid_token")
        else:
            final_parts.append(token["content"])
    final_text = "".join(final_parts)
    if not final_text.strip():
        _append_error(errors, "empty_final_text")

    turn_rows = [row for row in audit if turn_id is not None and row.get("turn_id") == turn_id]
    if not turn_rows:
        _append_error(errors, "audit_turn_missing")
    if turn_rows and any(row.get("turn_status") != "completed" for row in turn_rows):
        _append_error(errors, "audit_turn_incomplete")

    raw_calls: list[object] = []
    for row in turn_rows:
        calls = row.get("tool_calls")
        if isinstance(calls, list):
            raw_calls.extend(calls)
        elif calls is not None:
            raw_calls.append(calls)
    tool_calls = [call for call in raw_calls if isinstance(call, dict)]
    if len(tool_calls) != len(raw_calls):
        _append_error(errors, "invalid_tool_call")

    tool_results = [
        {
            "tool_call_id": row.get("tool_call_id"),
            "content": _parse_tool_result(row.get("content")),
        }
        for row in turn_rows
        if row.get("role") == "tool"
    ]
    roles = [row.get("role") for row in turn_rows]
    expected_tool = case.get("tool")
    tool_statuses = _event_data(events, "tool_status")

    if expected_tool is None:
        if raw_calls or tool_results or tool_statuses:
            _append_error(errors, "unexpected_tool_call")
        if turn_rows and roles != ["user", "assistant"]:
            _append_error(errors, "invalid_audit_shape")
    else:
        if len(raw_calls) != 1 or len(tool_calls) != 1:
            _append_error(errors, "tool_call_count_mismatch")
        if len(tool_results) != 1:
            _append_error(errors, "tool_result_count_mismatch")
        if turn_rows and roles != ["user", "assistant", "tool", "assistant"]:
            _append_error(errors, "invalid_audit_shape")
        if tool_calls:
            call = tool_calls[0]
            call_id = call.get("id")
            call_rows = [row for row in turn_rows if row.get("tool_calls")]
            result_id = tool_results[0].get("tool_call_id") if tool_results else None
            recorded_call_id = call_rows[0].get("tool_call_id") if len(call_rows) == 1 else None
            if (
                not isinstance(call_id, str)
                or not call_id
                or recorded_call_id != call_id
                or result_id != call_id
            ):
                _append_error(errors, "unpaired_tool_call_id")
            if call.get("name") != expected_tool:
                _append_error(errors, "tool_name_mismatch")
            if not _arguments_match(case, call.get("args")):
                _append_error(errors, "tool_arguments_mismatch")
            valid_statuses = [status for status in tool_statuses if isinstance(status, dict)]
            if len(valid_statuses) != len(tool_statuses):
                _append_error(errors, "invalid_tool_status")
            if not valid_statuses or any(
                status.get("name") != expected_tool
                or status.get("tool_call_id") != call_id
                for status in valid_statuses
            ):
                _append_error(errors, "tool_status_mismatch")
        if tool_results:
            content = tool_results[0]["content"]
            if not isinstance(content, dict) or content.get("status") != case.get("result_status"):
                _append_error(errors, "tool_result_status_mismatch")

    if turn_rows:
        if turn_rows[0].get("role") != "user" or turn_rows[0].get("content") != case.get("message"):
            _append_error(errors, "audit_question_mismatch")
        if turn_rows[-1].get("role") != "assistant" or turn_rows[-1].get("content") != final_text:
            _append_error(errors, "audit_final_text_mismatch")

    return {
        "case_id": case.get("id"),
        "question": case.get("message"),
        "session_id": session_id,
        "turn_id": turn_id,
        "events": events,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "final_text": final_text,
        "protocol_pass": not errors,
        "protocol_errors": errors,
        "semantic_review": "pending_manual_review",
    }


def _failure_report(
    case: dict[str, Any], category: str, events: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    report = evaluate_case(case, events or [], [])
    report["protocol_pass"] = False
    _append_error(report["protocol_errors"], category)
    return report


async def _stream_turn(
    client: httpx.AsyncClient,
    base_url: str,
    case: dict[str, Any],
    session_id: str | None,
    timeout: float,
) -> list[dict[str, Any]]:
    payload: dict[str, Any] = {"message": case["message"]}
    if session_id is not None:
        payload["session_id"] = session_id
    try:
        async with asyncio.timeout(timeout):
            async with client.stream(
                "POST", f"{base_url.rstrip('/')}/api/chat", json=payload
            ) as response:
                if response.status_code < 200 or response.status_code >= 300:
                    await response.aread()
                    raise EvaluationFailure("http_error")
                content_type = response.headers.get("content-type", "").split(";", 1)[0]
                if content_type.strip().lower() != "text/event-stream":
                    await response.aread()
                    raise EvaluationFailure("invalid_content_type")
                events: list[dict[str, Any]] = []
                event_name: str | None = None
                data_lines: list[str] = []

                def flush() -> None:
                    nonlocal event_name, data_lines
                    if event_name is not None and data_lines:
                        try:
                            data = json.loads("\n".join(data_lines))
                        except json.JSONDecodeError as exc:
                            raise EvaluationFailure("invalid_sse_json") from exc
                        events.append({"event": event_name, "data": data})
                    event_name = None
                    data_lines = []

                async for line in response.aiter_lines():
                    if not line:
                        flush()
                    elif line.startswith("event:"):
                        event_name = line[6:].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                flush()
                return events
    except EvaluationFailure:
        raise
    except TimeoutError as exc:
        raise EvaluationFailure("request_timeout") from exc
    except httpx.HTTPError as exc:
        raise EvaluationFailure("transport_error") from exc


def _session_from_events(events: list[dict[str, Any]]) -> str | None:
    metas = _event_data(events, "meta")
    if (
        len(metas) != 1
        or not isinstance(metas[0], dict)
        or not isinstance(metas[0].get("session_id"), str)
    ):
        return None
    session_id = metas[0]["session_id"]
    try:
        UUID(session_id)
    except ValueError:
        return None
    return session_id


def _session_from_complete_events(events: list[dict[str, Any]]) -> str | None:
    session_id = _session_from_events(events)
    dones = _event_data(events, "done")
    if (
        session_id is not None
        and len(dones) == 1
        and isinstance(dones[0], dict)
        and dones[0].get("session_id") == session_id
        and not _event_data(events, "error")
    ):
        return session_id
    return None


async def run_evaluation(
    base_url: str,
    cases: list[dict[str, Any]],
    audit_reader: AuditReader,
    *,
    timeout: float,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=min(timeout, 10.0)),
            trust_env=False,
        )
    try:
        for group in cases:
            session_id: str | None = None
            turns = [group]
            if isinstance(group.get("followup"), dict):
                turns.append(group["followup"])
            for index, case in enumerate(turns):
                if index and session_id is None:
                    results.append(_failure_report(case, "missing_parent_session"))
                    continue
                try:
                    events = await _stream_turn(client, base_url, case, session_id, timeout)
                except EvaluationFailure as exc:
                    results.append(_failure_report(case, exc.category))
                    session_id = None
                    continue
                observed_session = _session_from_events(events)
                returned_session = _session_from_complete_events(events)
                if returned_session is None:
                    audit: list[dict[str, Any]] = []
                    if observed_session is not None:
                        try:
                            audit = await audit_reader(observed_session)
                        except Exception:
                            results.append(
                                _failure_report(case, "audit_unavailable", events)
                            )
                            session_id = None
                            continue
                    results.append(evaluate_case(case, events, audit))
                    session_id = None
                    continue
                if session_id is not None and returned_session != session_id:
                    results.append(_failure_report(case, "session_changed", events))
                    session_id = None
                    continue
                session_id = returned_session
                try:
                    audit = await audit_reader(session_id)
                except Exception:
                    results.append(_failure_report(case, "audit_unavailable", events))
                    continue
                results.append(evaluate_case(case, events, audit))
    finally:
        if owns_client:
            await client.aclose()

    passed = sum(bool(item["protocol_pass"]) for item in results)
    failed = len(results) - passed
    return {
        "status": "failed" if failed else "pending_manual_review",
        "summary": {
            "total": len(results),
            "protocol_passed": passed,
            "protocol_failed": failed,
            "pending_manual_review": len(results),
        },
        "results": results,
    }


def _validate_turn(case: object, *, label: str) -> dict[str, Any]:
    if not isinstance(case, dict):
        raise ValueError(f"{label} must be an object")
    if not all(isinstance(case.get(key), str) and case[key].strip() for key in ("id", "message", "rubric")):
        raise ValueError(f"{label} requires non-empty id, message, and rubric")
    tool = case.get("tool")
    if tool is not None and (not isinstance(tool, str) or not tool):
        raise ValueError(f"{label} tool must be null or a non-empty string")
    if tool is not None and case.get("result_status") not in {"ok", "not_found", "error"}:
        raise ValueError(f"{label} requires a labeled result_status")
    if "args" in case and not isinstance(case["args"], dict):
        raise ValueError(f"{label} args must be an object")
    if "keywords" in case and (
        not isinstance(case["keywords"], list)
        or not case["keywords"]
        or not all(isinstance(item, str) and item for item in case["keywords"])
    ):
        raise ValueError(f"{label} keywords must be a non-empty string list")
    return case


def validate_cases(loaded: object) -> list[dict[str, Any]]:
    if not isinstance(loaded, list) or not loaded:
        raise ValueError("evaluation cases must be a non-empty array")
    cases: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, value in enumerate(loaded):
        case = _validate_turn(value, label=f"case[{index}]")
        candidates = [case]
        if "followup" in case:
            candidates.append(_validate_turn(case["followup"], label=f"case[{index}].followup"))
        for candidate in candidates:
            case_id = candidate["id"]
            if case_id in ids:
                raise ValueError(f"duplicate case id: {case_id}")
            ids.add(case_id)
        cases.append(case)
    return cases


def validate_output_path(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(REPORTS_DIR)
    except ValueError as exc:
        raise ValueError("--output must be inside evals/reports/") from exc
    if resolved == REPORTS_DIR:
        raise ValueError("--output must name a file inside evals/reports/")
    return resolved


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


async def _run_cli(base_url: str, cases: list[dict[str, Any]], timeout: float) -> dict[str, Any]:
    from app.config import load_settings
    from app.db.conversations import ConversationRepository
    from app.db.database import Database

    settings = load_settings()
    if settings.database_url is None:
        raise EvaluationFailure("database_not_configured")
    database = Database(settings.database_url.get_secret_value())
    try:
        await database.check()
        conversations = ConversationRepository(database.sessions)

        async def read_audit(session_id: str) -> list[dict[str, Any]]:
            return await conversations.audit(session_id, "demo")

        return await run_evaluation(
            base_url, cases, read_audit, timeout=timeout
        )
    finally:
        await database.aclose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行第二章工具调用标注评估")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=65.0)
    args = parser.parse_args(argv)
    try:
        output = validate_output_path(args.output)
        if args.timeout <= 0:
            raise ValueError("--timeout must be positive")
        cases = validate_cases(json.loads(args.cases.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"configuration_error: {exc}", file=sys.stderr)
        return 2

    try:
        report = asyncio.run(_run_cli(args.base_url, cases, args.timeout))
    except Exception:
        report = {
            "status": "failed",
            "summary": {
                "total": 0,
                "protocol_passed": 0,
                "protocol_failed": 1,
                "pending_manual_review": 0,
            },
            "results": [],
            "failure": {"category": "configuration_or_database_error"},
        }
    _write_report(output, report)
    print(json.dumps(report["summary"], ensure_ascii=False))
    print(f"status={report['status']} report={output}")
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
