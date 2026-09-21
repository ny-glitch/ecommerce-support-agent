from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Annotated

import openai
from langchain_core.messages import BaseMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StringConstraints,
    ValidationError,
)

from app.config import Settings
from app.context import build_context
from app.errors import ServiceError
from app.knowledge.contracts import Citation, EvidenceAssessment
from app.prompts import evidence_assessment_system_prompt


_PROMPT_PATH = Path(__file__).parents[1] / "prompts" / "query_normalization.txt"
_FAITHFULNESS_PROMPT_PATH = (
    Path(__file__).parents[1] / "prompts" / "faithfulness_judge.txt"
)
_Normalized = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=512),
]
_Synonym = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=32),
]
BeforeRequest = Callable[[str, list[BaseMessage], list[dict]], None]
RecordUsage = Callable[[str, dict[str, int] | None], None]


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


class NormalizationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    normalized: _Normalized
    synonyms: tuple[_Synonym, ...] = Field(default=(), max_length=3)


class NormalizationRequestError(RuntimeError):
    pass


class NormalizationResponseError(ValueError):
    pass


class FaithfulnessClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    statement: str = Field(min_length=1, max_length=1_000)
    supported: bool
    source_ids: list[int] = Field(max_length=10)
    reason: str = Field(min_length=1, max_length=600)


class FaithfulnessJudgement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claims: list[FaithfulnessClaim] = Field(max_length=100)
    _raw_response: str = PrivateAttr(default="")

    @property
    def raw_response(self) -> str:
        return self._raw_response


class FaithfulnessRequestError(RuntimeError):
    pass


class FaithfulnessResponseError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        diagnostic_raw_response: str | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostic_raw_response = diagnostic_raw_response


def validate_assessment(
    value: EvidenceAssessment,
    sources: Sequence[Citation],
) -> EvidenceAssessment:
    """Validate the assessment's semantic relationship to supplied evidence."""
    source_ids = value.supporting_chunk_ids
    available_ids = {source.chunk_id for source in sources}
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("assessment supporting chunk IDs must be unique")
    if value.sufficient:
        if value.reason_code != "supported":
            raise ValueError("sufficient assessment must be supported")
        if not source_ids or not set(source_ids).issubset(available_ids):
            raise ValueError("sufficient assessment has invalid supporting IDs")
    elif value.reason_code == "supported" or source_ids:
        raise ValueError("insufficient assessment cannot name supporting evidence")
    return value


class KnowledgeGateway:
    def __init__(
        self,
        model: ChatOpenAI,
        *,
        chat_extra_body: dict[str, Any],
        settings: Settings,
        before_request: BeforeRequest | None = None,
        record_usage: RecordUsage | None = None,
    ) -> None:
        template = PromptTemplate.from_template(_PROMPT_PATH.read_text(encoding="utf-8"))
        self._system_prompt = template.format(
            schema_json=json.dumps(
                NormalizationOutput.model_json_schema(),
                ensure_ascii=False,
            )
        )
        self._normalizer = model.with_structured_output(
            NormalizationOutput,
            method="json_mode",
            include_raw=True,
            extra_body=dict(chat_extra_body),
        )
        assessment_prompt = evidence_assessment_system_prompt(
            json.dumps(
                EvidenceAssessment.model_json_schema(),
                ensure_ascii=False,
            )
        )
        self._assessment_prompt = assessment_prompt
        self._assessor = model.with_structured_output(
            EvidenceAssessment,
            method="json_mode",
            include_raw=True,
            extra_body=dict(chat_extra_body),
        )
        faithfulness_template = PromptTemplate.from_template(
            _FAITHFULNESS_PROMPT_PATH.read_text(encoding="utf-8")
        )
        self._faithfulness_prompt = faithfulness_template.format(
            schema_json=json.dumps(
                FaithfulnessJudgement.model_json_schema(),
                ensure_ascii=False,
            )
        )
        self._judge = model.with_structured_output(
            FaithfulnessJudgement,
            method="json_mode",
            include_raw=True,
            extra_body=dict(chat_extra_body),
        )
        self._settings = settings
        self._before_request = before_request or _noop_before_request
        self._record_usage = record_usage or _noop_record_usage

    async def normalize(self, question: str) -> NormalizationOutput:
        messages = build_context(
            self._system_prompt,
            [],
            question,
            self._settings,
        ).messages
        self._before_request("normalize", messages, [])
        try:
            result = await self._normalizer.ainvoke(messages)
        except (
            openai.ContentFilterFinishReasonError,
            openai.LengthFinishReasonError,
        ) as exc:
            raise NormalizationResponseError(
                "invalid structured normalization response"
            ) from exc
        except Exception as exc:
            raise NormalizationRequestError("normalization request failed") from exc
        raw = result.get("raw") if isinstance(result, dict) else None
        if isinstance(raw, BaseMessage):
            self._record_usage("normalize", _usage_counts(raw))
        try:
            raw = result["raw"]
            content = raw.content
            finish_reason = raw.response_metadata.get("finish_reason")
            parsed = result["parsed"]
            if (
                finish_reason != "stop"
                or not isinstance(content, str)
                or not content.strip()
                or result["parsing_error"] is not None
                or not isinstance(parsed, NormalizationOutput)
            ):
                raise ValueError("incomplete or unparseable structured normalization")
            decoded = json.loads(content)
            if not isinstance(decoded, dict):
                raise ValueError("structured normalization must be a JSON object")
            validated = NormalizationOutput.model_validate(decoded)
            if validated != parsed:
                raise ValueError("parsed normalization does not match raw JSON")
            return validated
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, ValidationError) as exc:
            raise NormalizationResponseError(
                "invalid structured normalization response"
            ) from exc

    async def assess(
        self,
        question: str,
        sources: tuple[Citation, ...],
        *,
        normalized_question: str,
    ) -> EvidenceAssessment:
        messages = build_assessment_messages(
            question,
            sources,
            normalized_question=normalized_question,
            settings=self._settings,
            system_prompt=self._assessment_prompt,
        )
        self._before_request("evidence", messages, [])
        try:
            result = await self._assessor.ainvoke(messages)
        except (
            openai.ContentFilterFinishReasonError,
            openai.LengthFinishReasonError,
        ) as exc:
            raise ServiceError(
                "EVIDENCE_ASSESSMENT_ERROR",
                "证据充分性校验失败，请重试",
                502,
            ) from exc
        except Exception as exc:
            if isinstance(exc, ServiceError):
                raise
            raise ServiceError(
                "KNOWLEDGE_UNAVAILABLE",
                "知识服务暂时不可用，请稍后重试",
                502,
            ) from exc

        raw = result.get("raw") if isinstance(result, dict) else None
        if isinstance(raw, BaseMessage):
            self._record_usage("evidence", _usage_counts(raw))

        try:
            raw = result["raw"]
            content = raw.content
            finish_reason = raw.response_metadata.get("finish_reason")
            parsed = result["parsed"]
            if (
                finish_reason != "stop"
                or not isinstance(content, str)
                or not content.strip()
                or result["parsing_error"] is not None
                or not isinstance(parsed, EvidenceAssessment)
            ):
                raise ValueError("incomplete or unparseable evidence assessment")
            decoded = json.loads(content)
            if not isinstance(decoded, dict):
                raise ValueError("evidence assessment must be a JSON object")
            validated = EvidenceAssessment.model_validate(decoded)
            if validated != parsed:
                raise ValueError("parsed assessment does not match raw JSON")
            return validate_assessment(validated, sources)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, ValidationError) as exc:
            raise ServiceError(
                "EVIDENCE_ASSESSMENT_ERROR",
                "证据充分性校验失败，请重试",
                502,
            ) from exc

    async def judge(
        self,
        question: str,
        answer: str,
        sources: tuple[Citation, ...],
    ) -> FaithfulnessJudgement:
        payload = json.dumps(
            {
                "question": question,
                "answer": answer,
                "sources": [source.model_dump(mode="json") for source in sources],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        messages = build_context(
            self._faithfulness_prompt,
            [],
            payload,
            self._settings,
        ).messages
        self._before_request("judge", messages, [])
        try:
            result = await self._judge.ainvoke(messages)
        except (
            openai.ContentFilterFinishReasonError,
            openai.LengthFinishReasonError,
        ) as exc:
            raise FaithfulnessResponseError(
                "invalid structured faithfulness response"
            ) from exc
        except Exception as exc:
            raise FaithfulnessRequestError("faithfulness request failed") from exc

        raw = result.get("raw") if isinstance(result, dict) else None
        if isinstance(raw, BaseMessage):
            self._record_usage("judge", _usage_counts(raw))

        diagnostic_raw_response: str | None = None
        try:
            raw = result["raw"]
            content = raw.content
            if isinstance(content, str):
                diagnostic_raw_response = content
            finish_reason = raw.response_metadata.get("finish_reason")
            parsed = result["parsed"]
            if (
                finish_reason != "stop"
                or not isinstance(content, str)
                or not content.strip()
                or result["parsing_error"] is not None
                or not isinstance(parsed, FaithfulnessJudgement)
            ):
                raise ValueError("incomplete or unparseable faithfulness response")
            decoded = json.loads(content)
            if not isinstance(decoded, dict):
                raise ValueError("faithfulness response must be a JSON object")
            validated = FaithfulnessJudgement.model_validate(decoded)
            if validated != parsed:
                raise ValueError("parsed faithfulness response does not match raw JSON")
            available_ids = {source.chunk_id for source in sources}
            for claim in validated.claims:
                source_ids = claim.source_ids
                if len(source_ids) != len(set(source_ids)):
                    raise ValueError("faithfulness source IDs must be unique")
                if claim.supported:
                    if not source_ids or not set(source_ids).issubset(available_ids):
                        raise ValueError("supported claim has invalid source IDs")
                elif source_ids:
                    raise ValueError("unsupported claim cannot name supporting sources")
            validated._raw_response = content
            return validated
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, ValidationError) as exc:
            raise FaithfulnessResponseError(
                "invalid structured faithfulness response",
                diagnostic_raw_response=diagnostic_raw_response,
            ) from exc


def build_assessment_messages(
    question: str,
    sources: tuple[Citation, ...],
    *,
    normalized_question: str,
    settings: Settings,
    system_prompt: str | None = None,
) -> list[BaseMessage]:
    prompt = system_prompt or evidence_assessment_system_prompt(
        json.dumps(EvidenceAssessment.model_json_schema(), ensure_ascii=False)
    )
    payload = json.dumps(
        {
            "original_question": question,
            "normalized_question": normalized_question,
            "sources": [source.model_dump(mode="json") for source in sources],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return build_context(prompt, [], payload, settings).messages
