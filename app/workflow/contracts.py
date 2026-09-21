"""Validated control values; persist their JSON dumps, never model instances."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.tools.schemas import TicketInput
from app.knowledge.contracts import KnowledgeDecision

Intent = Literal[
    'logistics', 'order', 'product', 'return_refund',
    'after_sales', 'complaint', 'chitchat',
]
Action = Literal['handoff', 'create_ticket']
KnowledgeBand = Literal['low', 'middle', 'high']
KnowledgeTarget = Literal[
    'workflow_answer', 'agent_tools', 'agent_generate', 'fallback'
]
INTENT_LABELS: dict[Intent, str] = {
    'logistics': '物流', 'order': '订单', 'product': '商品咨询',
    'return_refund': '退款退货', 'after_sales': '售后',
    'complaint': '投诉', 'chitchat': '闲聊',
}


class _StrictDTO(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', hide_input_in_errors=True)


class IntentResult(_StrictDTO):
    intent: Intent
    needs_business_data: bool


class FinalControl(_StrictDTO):
    kind: Literal['respond', 'clarify']
    actions: list[Action] = Field(default_factory=list, max_length=2)
    ticket: TicketInput | None = None

    @model_validator(mode='after')
    def validate_actions(self) -> FinalControl:
        if len(self.actions) != len(set(self.actions)):
            raise ValueError('duplicate action')
        if ('create_ticket' in self.actions) != (self.ticket is not None):
            raise ValueError('ticket is required only for create_ticket')
        return self


class ActionOffer(_StrictDTO):
    action_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    turn_id: str = Field(min_length=1)
    ticket_no: str = Field(min_length=1)
    draft: TicketInput
    status: Literal['offered', 'completed']


@dataclass(frozen=True)
class WorkflowKnowledgeResult:
    decision: KnowledgeDecision
    score: float | None
    band: KnowledgeBand | None
    target: KnowledgeTarget
