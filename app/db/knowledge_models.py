from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.mysql import BIGINT, ENUM, TINYINT
from sqlalchemy.orm import Mapped, mapped_column

from app.db.models import Base


class KnowledgeChunkRecord(Base):
    __tablename__ = "knowledge_chunks"
    __table_args__ = (
        Index("idx_category", "category"),
        Index("idx_vectorize_status", "vectorize_status"),
        {
            "mysql_engine": "InnoDB",
            "mysql_charset": "utf8mb4",
            "comment": "知识库 chunk 原文权威源",
        },
    )

    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        primary_key=True,
        autoincrement=True,
        comment="chunk 主键,与 Milvus 集合主键对齐",
    )
    category: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="分类 / 上级标题路径,进向量化文本",
    )
    questions: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="问法或本节标题,多个问法换行分隔,进向量化文本",
    )
    answer: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="正文答案,进向量化文本",
    )
    section_path: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        comment="章节路径,元数据,溯源用,不进向量",
    )
    content_type: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        comment="内容类型:faq / policy / manual 等,元数据",
    )
    is_key_clause: Mapped[bool] = mapped_column(
        TINYINT(1),
        nullable=False,
        default=False,
        server_default=text("0"),
        comment="是否关键条款,0 否 1 是,元数据",
    )
    prev_chunk_id: Mapped[int | None] = mapped_column(
        BIGINT(unsigned=True),
        ForeignKey(
            "knowledge_chunks.id",
            name="fk_chunks_prev",
            ondelete="SET NULL",
        ),
        nullable=True,
        comment="前一块指针,元数据",
    )
    next_chunk_id: Mapped[int | None] = mapped_column(
        BIGINT(unsigned=True),
        ForeignKey(
            "knowledge_chunks.id",
            name="fk_chunks_next",
            ondelete="SET NULL",
        ),
        nullable=True,
        comment="后一块指针,元数据",
    )
    vector_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="Milvus 集合 knowledge 里的主键,写入后回填",
    )
    vectorize_status: Mapped[str] = mapped_column(
        ENUM("pending", "done"),
        nullable=False,
        default="pending",
        server_default=text("'pending'"),
        comment="待向量化 / 已向量化,双写幂等靠它",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.current_timestamp(),
        comment="创建时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"),
        comment="更新时间",
    )


class QAExtractionStaging(Base):
    __tablename__ = "qa_extraction_staging"
    __table_args__ = (
        Index("idx_batch_no", "batch_no"),
        Index("idx_status", "status"),
        {
            "mysql_engine": "InnoDB",
            "mysql_charset": "utf8mb4",
            "comment": "历史对话抽 QA 的离线中转暂存表:分批抽取、"
            "整体去重,保留项入 knowledge_chunks,建库完成可清空",
        },
    )

    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        primary_key=True,
        autoincrement=True,
        comment="暂存行主键",
    )
    batch_no: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment=(
            "抽取批次号,一批几十个会话跑一次,"
            "分批防串味、按批追溯"
        ),
    )
    source_ref: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="来源会话 / 导出文件标识,溯源用,不入最终知识库",
    )
    question: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="LLM 从会话抽出的用户问法",
    )
    answer: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="LLM 从会话抽出的客服答案",
    )
    status: Mapped[str] = mapped_column(
        ENUM("extracted", "kept", "discarded"),
        nullable=False,
        default="extracted",
        server_default=text("'extracted'"),
        comment="已抽出待去重 / 去重保留 / 去重丢弃",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.current_timestamp(),
        comment="抽取写入时间",
    )


class LowConfidenceQuestion(Base):
    __tablename__ = "low_confidence_questions"
    __table_args__ = (
        UniqueConstraint(
            "conversation_id",
            "turn_id",
            name="uq_low_confidence_conversation_turn",
        ),
        Index("idx_low_confidence_created_at", "created_at"),
        Index("idx_low_confidence_reason_code", "reason_code"),
        {
            "mysql_engine": "InnoDB",
            "mysql_charset": "utf8mb4",
            "comment": "在线低置信度问题池",
        },
    )

    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        primary_key=True,
        autoincrement=True,
        comment="问题池记录主键",
    )
    original_question: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="本轮用户原话",
    )
    conversation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("conversations.id"),
        nullable=False,
        comment="关联会话标识",
    )
    turn_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="轮次幂等标识",
    )
    entry_point: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="写入入口",
    )
    reason_code: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="可统计的拒答原因编码",
    )
    reason: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="缺少的证据说明",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.current_timestamp(),
        comment="UTC 创建时间",
    )
