"""Durable, user-confirmable offers; inserting one never creates a ticket."""
from datetime import datetime
from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from app.db.models import Base, utc_now


class ConversationAction(Base):
    __tablename__ = 'conversation_actions'
    __table_args__ = (
        Index('uq_actions_turn_type', 'conversation_id', 'turn_id', 'action_type', unique=True),
        Index('uq_actions_ticket_no', 'ticket_no', unique=True),
        CheckConstraint("action_type = 'create_ticket'", name='ck_actions_type'),
        CheckConstraint("status IN ('offered', 'completed')", name='ck_actions_status'),
        {'mysql_charset':'utf8mb4', 'mysql_collate':'utf8mb4_0900_ai_ci'},
    )
    action_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String(36), ForeignKey('conversations.id'), nullable=False)
    turn_id: Mapped[str] = mapped_column(String(64), nullable=False)
    action_type: Mapped[str] = mapped_column(String(32), nullable=False)
    issue_description: Mapped[str] = mapped_column(Text, nullable=False)
    ticket_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    ticket_no: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now, server_default=func.now())
