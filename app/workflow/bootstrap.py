from __future__ import annotations

from typing import Any

from app.config import Settings
from app.db.actions import ActionRepository
from app.db.conversations import ConversationRepository
from app.db.database import Database
from app.db.faq import FaqRepository
from app.db.tickets import TicketRepository
from app.knowledge.bootstrap import KnowledgeComponents
from app.knowledge.query import QueryNormalizer
from app.tools.executor import ToolExecutor
from app.workflow.agent import AgentDependencies
from app.workflow.knowledge import KnowledgeStage
from app.workflow.nodes import WorkflowDependencies
from app.workflow.state import TurnRuntime


def _request_hooks(runtime: TurnRuntime, settings: Settings):
    def before_request(stage, messages, tool_schemas):
        runtime.budget.reserve(
            stage,
            messages,
            output_tokens=settings.max_output_tokens,
            tool_schemas=tool_schemas,
        )

    def record_usage(stage, usage):
        runtime.budget.record_usage(stage, usage)

    return before_request, record_usage


def build_workflow_dependencies(
    settings: Settings,
    database: Database,
    model_gateway: Any,
    components: KnowledgeComponents,
) -> WorkflowDependencies:
    conversations = ConversationRepository(database.sessions)
    faq = FaqRepository(database.sessions)
    tickets = TicketRepository(database.sessions)
    actions = ActionRepository(database.sessions)
    executor = ToolExecutor(
        settings.tool_timeout_seconds,
        settings.tool_max_attempts,
    )

    workflow_factory = getattr(model_gateway, "create_workflow_gateway", None)
    if workflow_factory is None or not callable(workflow_factory):
        raise RuntimeError("model gateway cannot create the workflow gateway")

    def gateway_factory(runtime: TurnRuntime):
        before_request, record_usage = _request_hooks(runtime, settings)
        return workflow_factory(before_request, record_usage)

    def normalizer_factory(runtime: TurnRuntime) -> QueryNormalizer:
        before_request, record_usage = _request_hooks(runtime, settings)
        return QueryNormalizer(
            components.knowledge_gateway_factory(
                before_request=before_request,
                record_usage=record_usage,
            )
        )

    knowledge_stage = KnowledgeStage(
        normalizer_factory,
        components.retriever,
        gateway_factory,
        settings,
    )
    agent_dependencies = AgentDependencies(
        settings,
        gateway_factory,
        conversations,
        faq,
        tickets,
        executor,
    )
    return WorkflowDependencies(
        settings,
        gateway_factory,
        knowledge_stage,
        agent_dependencies,
        conversations,
        actions,
        components.low_confidence,
    )
