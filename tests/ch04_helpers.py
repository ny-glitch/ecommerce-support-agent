from __future__ import annotations

import asyncio
from dataclasses import replace

from app.knowledge.contracts import (
    Citation,
    EvidenceAssessment,
    KnowledgeChunk,
    KnowledgeDecision,
    QueryPlan,
)


def make_chunk(**changes: object) -> KnowledgeChunk:
    return replace(
        KnowledgeChunk(
            910001,
            "数码配件/充电器",
            "C65-Pro 支持什么协议？",
            "支持 PD 3.0。",
            "商品手册/C65-Pro/协议",
            "manual",
        ),
        **changes,
    )


def make_decision(*, status: str = "ok", answer_size: int = 20) -> KnowledgeDecision:
    query = QueryPlan("C65-Pro支持什么协议？", "C65-Pro支持什么协议？", (), "数码配件")
    source = Citation(
        number=1,
        chunk_id=910001,
        category="数码配件/充电器",
        section_path="商品手册/C65-Pro/协议",
        questions="C65-Pro支持什么协议？",
        answer="支" * answer_size,
        content_hash="a" * 64,
        url="/api/knowledge/chunks/910001?expected_hash=" + "a" * 64,
        score=0.91,
    )
    if status == "not_found":
        return KnowledgeDecision(
            query=query,
            status="not_found",
            sources=(source,),
            assessment=EvidenceAssessment(
                sufficient=False,
                reason_code="insufficient_evidence",
                reason="现有证据没有覆盖用户询问的条件。",
                supporting_chunk_ids=[],
            ),
            reason_code="insufficient_evidence",
            refusal="现有知识不足以确认您询问的事项，建议补充信息或联系人工客服核实。",
        )
    return KnowledgeDecision(
        query=query,
        status="ok",
        sources=(source,),
        assessment=EvidenceAssessment(
            sufficient=True,
            reason_code="supported",
            reason="当前证据完整支持回答。",
            supporting_chunk_ids=[source.chunk_id],
        ),
        reason_code=None,
        refusal=None,
    )


class FakeKnowledgePipeline:
    def __init__(self, *decisions: KnowledgeDecision) -> None:
        self.decisions = list(decisions or (make_decision(),))
        self.calls: list[tuple[str, str | None, object, float]] = []
        self.error: Exception | None = None
        self.gate: asyncio.Event | None = None
        self.waiting = asyncio.Event()
        self.closed = asyncio.Event()

    async def run(self, question, category, budget, *, deadline, emit):
        self.calls.append((question, category, budget, deadline))
        try:
            await emit("normalizing")
            await emit("retrieving")
            await emit("reranking")
            await emit("checking_evidence")
            if self.gate is not None:
                self.waiting.set()
                await self.gate.wait()
            if self.error is not None:
                raise self.error
            return self.decisions.pop(0)
        finally:
            self.closed.set()


class RecordingLowConfidence:
    def __init__(self, trace: list[str] | None = None) -> None:
        self.calls = []
        self.trace = trace
        self.error: Exception | None = None

    async def record_once(self, *args, **kwargs):
        if self.error is not None:
            raise self.error
        self.calls.append((*args, kwargs))
        if self.trace is not None:
            self.trace.append("pool_saved")
        return 1
