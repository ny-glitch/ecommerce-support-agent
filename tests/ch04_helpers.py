from __future__ import annotations

from dataclasses import replace

from app.knowledge.contracts import KnowledgeChunk


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
