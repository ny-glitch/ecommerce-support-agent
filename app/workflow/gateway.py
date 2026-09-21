"""Model-backed workflow protocol with application-side validation."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import aclosing
from typing import Any, TypeVar

import openai
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.errors import ServiceError
from app.knowledge.contracts import Citation, EvidenceAssessment
from app.knowledge.gateway import validate_assessment
from app.workflow.contracts import FinalControl, IntentResult
from app.workflow.prompts import build_workflow_messages


BeforeRequest = Callable[[str, list[BaseMessage], list[dict]], None]
RecordUsage = Callable[[str, dict[str, int] | None], None]
_ALLOWED_TOOL_NAMES = frozenset(
    {"query_order", "query_product", "query_logistics"}
)
_StructuredValue = TypeVar("_StructuredValue", bound=BaseModel)


def _noop_before_request(
    _stage: str, _messages: list[BaseMessage], _tool_schemas: list[dict]
) -> None:
    return None


def _noop_record_usage(_stage: str, _usage: dict[str, int] | None) -> None:
    return None


def _usage_counts(message: BaseMessage) -> dict[str, int] | None:
    metadata = getattr(message, "usage_metadata", None)
    if not isinstance(metadata, dict):
        return None
    safe: dict[str, int] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = metadata.get(key)
        if type(value) is int and value >= 0:
            safe[key] = value
    return safe or None


def _validated_structured(
    result: Any,
    model_type: type[_StructuredValue],
    *,
    code: str,
    message: str,
) -> _StructuredValue:
    try:
        raw = result["raw"]
        content = raw.content
        parsed = result["parsed"]
        if (
            raw.response_metadata.get("finish_reason") != "stop"
            or not isinstance(content, str)
            or not content.strip()
            or result["parsing_error"] is not None
            or not isinstance(parsed, model_type)
        ):
            raise ValueError("incomplete or unparseable structured response")
        decoded = json.loads(content)
        if not isinstance(decoded, dict):
            raise ValueError("structured response must be a JSON object")
        validated = model_type.model_validate(decoded, strict=True)
        if validated != parsed:
            raise ValueError("parsed structured output does not match raw JSON")
        return validated
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, ValidationError) as exc:
        raise ServiceError(code, message, 502) from exc


class WorkflowGateway:
    def __init__(
        self,
        model: ChatOpenAI,
        settings: Settings,
        *,
        chat_extra_body: dict[str, Any],
        before_request: BeforeRequest | None = None,
        record_usage: RecordUsage | None = None,
    ) -> None:
        self._model = model
        self._settings = settings
        self._chat_extra_body = dict(chat_extra_body)
        self._before_request = before_request or _noop_before_request
        self._record_usage = record_usage or _noop_record_usage
        self._intent_model = model.with_structured_output(
            IntentResult,
            method="json_mode",
            include_raw=True,
            extra_body=dict(chat_extra_body),
        )
        self._assessment_model = model.with_structured_output(
            EvidenceAssessment,
            method="json_mode",
            include_raw=True,
            extra_body=dict(chat_extra_body),
        )

    async def classify(self, messages: list[BaseMessage]) -> IntentResult:
        self._before_request("intent", messages, [])
        try:
            result = await self._intent_model.ainvoke(messages)
        except (
            openai.ContentFilterFinishReasonError,
            openai.LengthFinishReasonError,
        ) as exc:
            raise ServiceError(
                "INTENT_PROTOCOL_ERROR", "意图识别结果无效，请重试", 502
            ) from exc
        except Exception as exc:
            if isinstance(exc, ServiceError):
                raise
            raise ServiceError(
                "UPSTREAM_ERROR", "模型服务暂时不可用", 502
            ) from exc
        raw = result.get("raw") if isinstance(result, dict) else None
        if isinstance(raw, BaseMessage):
            self._record_usage("intent", _usage_counts(raw))
        value = _validated_structured(
            result,
            IntentResult,
            code="INTENT_PROTOCOL_ERROR",
            message="意图识别结果无效，请重试",
        )
        return value

    async def decide(
        self,
        messages: list[BaseMessage],
        tools: list[BaseTool],
    ) -> AIMessage | FinalControl:
        names = [business_tool.name for business_tool in tools]
        if (
            len(names) != len(set(names))
            or not names
            or not set(names).issubset(_ALLOWED_TOOL_NAMES)
        ):
            raise ServiceError(
                "INVALID_TOOL_CALL", "工具调用格式无效", 502
            )
        schemas = [convert_to_openai_tool(business_tool) for business_tool in tools]
        selector = self._model.bind_tools(
            tools,
            tool_choice="auto",
            parallel_tool_calls=False,
            extra_body=dict(self._chat_extra_body),
        )
        self._before_request("agent", messages, schemas)
        try:
            reply = await selector.ainvoke(messages)
        except (
            openai.ContentFilterFinishReasonError,
            openai.LengthFinishReasonError,
        ) as exc:
            raise ServiceError(
                "UPSTREAM_INCOMPLETE", "模型回复未正常完成，请重试", 502
            ) from exc
        except Exception as exc:
            if isinstance(exc, ServiceError):
                raise
            raise ServiceError(
                "UPSTREAM_ERROR", "模型服务暂时不可用", 502
            ) from exc

        self._record_usage("agent", _usage_counts(reply))
        finish_reason = reply.response_metadata.get("finish_reason")
        if (
            finish_reason not in {"stop", "tool_calls"}
            or reply.invalid_tool_calls
            or len(reply.tool_calls) > 1
            or (finish_reason == "tool_calls" and not reply.tool_calls)
        ):
            raise ServiceError(
                "INVALID_TOOL_CALL", "工具调用格式无效", 502
            )
        if reply.tool_calls:
            if reply.tool_calls[0]["name"] not in names:
                raise ServiceError(
                    "INVALID_TOOL_CALL", "工具调用格式无效", 502
                )
            return reply
        if finish_reason != "stop" or not isinstance(reply.content, str):
            raise ServiceError(
                "AGENT_CONTROL_ERROR", "Agent 控制结果无效", 502
            )
        try:
            return FinalControl.model_validate_json(reply.content, strict=True)
        except (ValueError, ValidationError) as exc:
            raise ServiceError(
                "AGENT_CONTROL_ERROR", "Agent 控制结果无效", 502
            ) from exc

    async def assess(
        self,
        question: str,
        sources: Sequence[Citation],
        *,
        normalized_question: str,
        intent: IntentResult,
    ) -> EvidenceAssessment:
        messages = build_workflow_messages(
            "evidence",
            settings=self._settings,
            question=question,
            sources=sources,
            intent=intent,
            normalized_question=normalized_question,
        ).messages
        self._before_request("evidence", messages, [])
        try:
            result = await self._assessment_model.ainvoke(messages)
        except (
            openai.ContentFilterFinishReasonError,
            openai.LengthFinishReasonError,
        ) as exc:
            raise ServiceError(
                "EVIDENCE_ASSESSMENT_ERROR", "证据充分性校验失败，请重试", 502
            ) from exc
        except Exception as exc:
            if isinstance(exc, ServiceError):
                raise
            raise ServiceError(
                "KNOWLEDGE_UNAVAILABLE", "知识服务暂时不可用，请稍后重试", 502
            ) from exc

        raw = result.get("raw") if isinstance(result, dict) else None
        if isinstance(raw, BaseMessage):
            self._record_usage("evidence", _usage_counts(raw))
        value = _validated_structured(
            result,
            EvidenceAssessment,
            code="EVIDENCE_ASSESSMENT_ERROR",
            message="证据充分性校验失败，请重试",
        )
        try:
            return validate_assessment(value, sources)
        except ValueError as exc:
            raise ServiceError(
                "EVIDENCE_ASSESSMENT_ERROR", "证据充分性校验失败，请重试", 502
            ) from exc

    async def stream_final(
        self, messages: list[BaseMessage]
    ) -> AsyncIterator[str]:
        self._before_request("answer", messages, [])
        emitted_text = False
        finish_reason: str | None = None
        usage: dict[str, int] | None = None
        try:
            async with aclosing(
                self._model.astream(
                    messages,
                    extra_body=dict(self._chat_extra_body),
                )
            ) as chunks:
                async for chunk in chunks:
                    current_finish = chunk.response_metadata.get("finish_reason")
                    if isinstance(current_finish, str):
                        finish_reason = current_finish
                    current_usage = _usage_counts(chunk)
                    if current_usage is not None:
                        usage = current_usage
                    text = chunk.text
                    if text:
                        emitted_text = True
                        yield text
        except Exception as exc:
            if isinstance(exc, ServiceError):
                raise
            raise ServiceError(
                "UPSTREAM_ERROR", "模型服务暂时不可用", 502
            ) from exc

        self._record_usage("answer", usage)
        if not emitted_text or finish_reason != "stop":
            raise ServiceError(
                "UPSTREAM_INCOMPLETE", "模型回复未正常完成，请重试", 502
            )
