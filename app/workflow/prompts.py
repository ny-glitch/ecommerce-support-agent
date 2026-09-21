"""Workflow prompt rendering and bounded request-message construction."""
from __future__ import annotations

import json
from importlib import resources
from typing import Literal, Sequence

from langchain_core.messages import BaseMessage
from langchain_core.prompts import PromptTemplate

from app.config import Settings
from app.context import ToolContextWindow, build_tool_context
from app.db.contracts import StoredTurn
from app.knowledge.contracts import Citation, EvidenceAssessment
from app.workflow.contracts import FinalControl, IntentResult


WorkflowPurpose = Literal["intent", "agent", "evidence", "answer"]


def _load_template(filename: str) -> str:
    return (
        resources.files("app")
        .joinpath("prompts")
        .joinpath(filename)
        .read_text(encoding="utf-8")
    )


def _system_prompt(purpose: WorkflowPurpose) -> str:
    filename = f"workflow_{purpose}.txt"
    template = PromptTemplate.from_template(_load_template(filename))
    values: dict[str, str] = {}
    if purpose == "intent":
        values["schema_json"] = json.dumps(
            IntentResult.model_json_schema(), ensure_ascii=False
        )
    elif purpose == "agent":
        values["schema_json"] = json.dumps(
            FinalControl.model_json_schema(), ensure_ascii=False
        )
    elif purpose == "evidence":
        values["schema_json"] = json.dumps(
            EvidenceAssessment.model_json_schema(), ensure_ascii=False
        )
    return template.format(**values)


def build_workflow_messages(
    purpose: WorkflowPurpose,
    *,
    settings: Settings,
    question: str,
    history: Sequence[StoredTurn] = (),
    sources: Sequence[Citation] = (),
    tool_messages: Sequence[BaseMessage] = (),
    intent: IntentResult | None = None,
    control: FinalControl | None = None,
    normalized_question: str | None = None,
    tool_schemas: Sequence[dict] = (),
) -> ToolContextWindow:
    if purpose not in {"intent", "agent", "evidence", "answer"}:
        raise ValueError("unknown workflow prompt purpose")

    payload: dict[str, object] = {"question": question}
    if normalized_question is not None:
        payload["normalized_question"] = normalized_question
    if sources:
        payload["sources"] = [source.model_dump(mode="json") for source in sources]
    if intent is not None:
        payload["intent"] = intent.model_dump(mode="json")
    if control is not None:
        payload["control"] = control.model_dump(mode="json")

    # Evidence sufficiency is deliberately independent of conversation history.
    retained_history = () if purpose == "evidence" else history
    retained_tool_messages = () if purpose == "evidence" else tool_messages
    return build_tool_context(
        _system_prompt(purpose),
        retained_history,
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        settings,
        tool_schemas=list(tool_schemas),
        current_tool_messages=retained_tool_messages,
    )
