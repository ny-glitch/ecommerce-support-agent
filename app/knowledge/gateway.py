from __future__ import annotations

import json
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
    pass


class KnowledgeGateway:
    def __init__(
        self,
        model: ChatOpenAI,
        *,
        chat_extra_body: dict[str, Any],
        settings: Settings,
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

    async def normalize(self, question: str) -> NormalizationOutput:
        messages = build_context(
            self._system_prompt,
            [],
            question,
            self._settings,
        ).messages
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
            return validated
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
                "invalid structured faithfulness response"
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
