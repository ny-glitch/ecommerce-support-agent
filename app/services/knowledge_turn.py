from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol

from langchain_core.messages import AIMessage

from app.config import Settings
from app.knowledge.contracts import KnowledgeDecision
from app.knowledge.evidence import EvidenceBudget
from app.services.turn_operations import bounded

if TYPE_CHECKING:
    from app.services.chat import PreparedTurn


class _KnowledgePipeline(Protocol):
    async def run(
        self,
        question: str,
        category: str | None,
        budget: EvidenceBudget,
        *,
        deadline: float,
        emit: Callable[[str], Awaitable[None]],
    ) -> KnowledgeDecision: ...


class _LowConfidence(Protocol):
    async def record_once(
        self,
        ref,
        question: str,
        reason_code: str,
        reason: str,
        entry_point: str = "chat",
    ) -> int: ...


class KnowledgeTurnRunner:
    def __init__(
        self,
        pipeline: _KnowledgePipeline,
        low_confidence: _LowConfidence,
        settings: Settings,
    ) -> None:
        self._pipeline = pipeline
        self._low_confidence = low_confidence
        self._settings = settings

    async def execute(
        self,
        prepared: PreparedTurn,
        call: AIMessage,
        emit: Callable[[str], Awaitable[None]],
    ) -> KnowledgeDecision:
        budget = EvidenceBudget(
            self._settings,
            prepared.window.retained_turns,
            prepared.message,
            call,
        )
        decision = await self._pipeline.run(
            prepared.message,
            prepared.category,
            budget,
            deadline=prepared.deadline,
            emit=emit,
        )
        if decision.status == "not_found":
            if not decision.reason_code or not decision.refusal:
                raise ValueError("refusal decision requires reason and template")
            reason = (
                decision.assessment.reason
                if decision.assessment is not None
                else decision.refusal
            )
            await bounded(
                lambda: self._low_confidence.record_once(
                    prepared.ref,
                    prepared.message,
                    decision.reason_code,
                    reason,
                    entry_point="chat",
                ),
                prepared.deadline,
                prepared._operations,
                mutations=prepared._mutations,
            )
        return decision
