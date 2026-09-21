"""Fixed outer-workflow nodes; terminal SSE events belong to the chat adapter."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
import time

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from app.config import Settings
from app.db.actions import ActionRepository
from app.db.contracts import StoredTurn
from app.db.conversations import ConversationRepository
from app.db.low_confidence import LowConfidenceRepository
from app.errors import ServiceError
from app.knowledge.contracts import Citation, EvidenceAssessment, KnowledgeDecision
from app.knowledge.pipeline import REFUSALS
from app.tools.business import ToolContext, build_readonly_registry
from app.tools.schemas import TicketInput
from app.workflow.agent import AgentDependencies
from app.workflow.contracts import FinalControl, IntentResult, WorkflowKnowledgeResult
from app.workflow.knowledge import (
    KnowledgeStage, deserialize_retrieval, render_workflow_answer,
    serialize_retrieval,
)
from app.workflow.prompts import build_workflow_messages
from app.workflow.routing import route_intent
from app.workflow.state import (
    TurnRuntime, WorkflowState, dump_turns, load_tool_messages, load_turns,
)


logger = logging.getLogger(__name__)

_BUDGET_REPLY = (
    '本轮查询未能全部完成，暂时无法确认剩余信息。'
    '您可以补充问题后重试，或选择联系人工客服。'
)
_CHITCHAT_REPLY = '您好，我是客服助手，可以帮您查询商品、订单、物流和售后问题。'
_COMPLAINT_REPLY = '很抱歉给您带来不好的体验。您可以选择联系人工客服，或确认创建投诉工单。'
_COMPLAINT_MARKER = '…（已截断，完整描述见本会话）'


@dataclass(frozen=True)
class WorkflowDependencies:
    settings: Settings
    gateway_factory: Callable[[TurnRuntime], object]
    knowledge_stage: KnowledgeStage
    agent_dependencies: AgentDependencies
    conversations: ConversationRepository
    actions: ActionRepository
    low_confidence: LowConfidenceRepository


def _emit(runtime: Runtime[TurnRuntime], name: str, data: dict) -> None:
    runtime.stream_writer({'name': name, 'data': data})


def _status(state: WorkflowState, runtime: Runtime[TurnRuntime], node: str,
            stage: str | None = None) -> None:
    intent = state.get('intent') or {}
    _emit(runtime, 'workflow_status', {
        'session_id': state['conversation_id'], 'turn_id': state['turn_id'],
        'node': node, 'stage': stage or node,
        'message': {
            'resolve': '正在准备本轮请求', 'classify': '正在识别问题类型',
            'retrieve': '正在检索知识', 'normalizing': '正在理解检索问题',
            'retrieving': '正在查找相关知识', 'reranking': '正在核对相关内容',
            'evidence_gate': '正在检查证据', 'checking_evidence': '正在检查证据',
            'agent': '正在查询业务信息', 'workflow_answer': '正在整理知识答案',
            'fallback': '当前证据不足', 'complaint': '正在准备投诉处理选项',
            'chitchat': '正在回复', 'budget_reply': '本轮模型预算已用尽',
            'persist': '正在保存本轮结果',
        }.get(stage or node, '正在处理'),
        'intent': intent.get('intent'), 'route': state.get('route'),
        'band': state.get('band'),
    })


def _trace(state: WorkflowState, node: str, **data) -> list[dict]:
    return [*state['trace'], {'stage': node, **data}]


def _budget_update(state: WorkflowState, runtime: Runtime[TurnRuntime], node: str) -> dict:
    return {
        'budget_exhausted': True,
        'budget': runtime.context.budget.snapshot(),
        'trace': _trace(state, node, status='limited', reason='TURN_BUDGET_EXHAUSTED'),
    }


def _is_budget(error: BaseException) -> bool:
    return isinstance(error, ServiceError) and error.code == 'TURN_BUDGET_EXHAUSTED'


def _ticket_description(question: str) -> str:
    if len(question) <= 2000:
        return question
    return question[:2000 - len(_COMPLAINT_MARKER)] + _COMPLAINT_MARKER


def _knowledge_result(state: WorkflowState) -> WorkflowKnowledgeResult:
    retrieval = deserialize_retrieval(state['retrieval'])
    assessment = (
        None if state['assessment'] is None
        else EvidenceAssessment.model_validate(state['assessment'])
    )
    reason_code = state['refusal_reason']
    decision = KnowledgeDecision(
        query=retrieval.query,
        status=state['knowledge_status'],
        sources=tuple(Citation.model_validate(source) for source in state['sources']),
        assessment=assessment,
        reason_code=reason_code,
        refusal=None if reason_code is None else REFUSALS[reason_code],
    )
    return WorkflowKnowledgeResult(decision, state['score'], state['band'],
                                   state['knowledge_target'])


def build_nodes(deps: WorkflowDependencies) -> dict[str, Callable]:
    async def resolve(state: WorkflowState, runtime: Runtime[TurnRuntime]):
        _status(state, runtime, 'resolve')
        return {'question': state['original_question'],
                'trace': _trace(state, 'resolve', status='ok')}

    async def classify(state: WorkflowState, runtime: Runtime[TurnRuntime]):
        _status(state, runtime, 'classify')
        context = runtime.context
        messages = build_workflow_messages(
            'intent', settings=deps.settings, question=state['question'],
            history=load_turns(state['history']),
        ).messages
        try:
            intent = await context.operations.run(
                lambda: deps.gateway_factory(context).classify(messages),
                context.deadline,
            )
        except ServiceError as error:
            if not _is_budget(error):
                raise
            return _budget_update(state, runtime, 'classify')
        intent = IntentResult.model_validate(intent.model_dump(mode='json'), strict=True)
        route = route_intent(intent)
        timeout = (
            deps.settings.knowledge_request_timeout_seconds
            if route == 'knowledge' else deps.settings.request_timeout_seconds
        )
        context.deadline = context.started_at + timeout
        return {
            'intent': intent.model_dump(mode='json'), 'route': route,
            'agent_mode': 'tools' if route == 'business' else None,
            'budget': context.budget.snapshot(),
            'trace': _trace(state, 'classify', intent=intent.intent, route=route),
        }

    async def retrieve(state: WorkflowState, runtime: Runtime[TurnRuntime]):
        _status(state, runtime, 'retrieve')

        async def emit(stage: str) -> None:
            _status(state, runtime, 'retrieve', stage)

        try:
            result = await deps.knowledge_stage.retrieve(
                state['question'], state['category'], runtime=runtime.context, emit=emit,
            )
        except ServiceError as error:
            if not _is_budget(error):
                raise
            return _budget_update(state, runtime, 'retrieve')
        return {
            'retrieval': serialize_retrieval(result),
            'query': result.query.normalized,
            'trace': _trace(state, 'retrieve', status='ok'),
        }

    async def evidence_gate(state: WorkflowState, runtime: Runtime[TurnRuntime]):
        _status(state, runtime, 'evidence_gate')
        context = runtime.context

        async def emit(stage: str) -> None:
            _status(state, runtime, 'evidence_gate', stage)

        registry = build_readonly_registry(
            ToolContext(context.ref, context.user_id, '', ''),
            deps.agent_dependencies.faq, deps.agent_dependencies.tickets,
        )
        try:
            result = await deps.knowledge_stage.assess(
                deserialize_retrieval(state['retrieval']),
                IntentResult.model_validate(state['intent']),
                load_turns(state['history']), registry.schemas(),
                runtime=context, emit=emit,
            )
        except ServiceError as error:
            if not _is_budget(error):
                raise
            return _budget_update(state, runtime, 'evidence_gate')
        decision = result.decision
        sources = [source.model_dump(mode='json') for source in decision.sources]
        if sources:
            _emit(runtime, 'sources', {'sources': sources})
        return {
            'score': result.score, 'band': result.band,
            'sources': sources,
            'assessment': None if decision.assessment is None else decision.assessment.model_dump(mode='json'),
            'knowledge_status': decision.status,
            'knowledge_target': result.target,
            'agent_mode': (
                'tools' if result.target == 'agent_tools'
                else 'generate_only' if result.target == 'agent_generate'
                else None
            ),
            'refusal_reason': decision.reason_code,
            'budget': context.budget.snapshot(),
            'trace': _trace(state, 'evidence_gate', status=decision.status,
                            band=result.band, target=result.target),
        }

    async def workflow_answer(state: WorkflowState, runtime: Runtime[TurnRuntime]):
        _status(state, runtime, 'workflow_answer')
        result = _knowledge_result(state)
        answer = render_workflow_answer(result)
        supported = set(result.decision.assessment.supporting_chunk_ids)
        used = sorted(source.number for source in result.decision.sources
                      if source.chunk_id in supported)
        _emit(runtime, 'message', {'content': answer, 'kind': 'workflow'})
        return {'answer': answer, 'used_citations': used,
                'trace': _trace(state, 'workflow_answer', status='rendered')}

    async def fallback(state: WorkflowState, runtime: Runtime[TurnRuntime]):
        _status(state, runtime, 'fallback')
        reason_code = state['refusal_reason']
        if reason_code not in REFUSALS:
            raise ValueError('invalid refusal reason')
        assessment = state['assessment'] or {}
        await runtime.context.operations.run(
            lambda: deps.low_confidence.record_once(
                runtime.context.ref, state['original_question'], reason_code,
                assessment.get('reason') or reason_code, entry_point='workflow'),
            runtime.context.deadline, mutation=True,
        )
        answer = REFUSALS[reason_code]
        _emit(runtime, 'refusal', {'content': answer, 'reason_code': reason_code})
        return {'answer': answer, 'used_citations': [], 'suggestions': ['handoff'],
                'trace': _trace(state, 'fallback', status='recorded', reason=reason_code)}

    async def complaint(state: WorkflowState, runtime: Runtime[TurnRuntime]):
        _status(state, runtime, 'complaint')
        draft = TicketInput(issue_description=_ticket_description(state['original_question']),
                            ticket_type='complaint')
        control = FinalControl(kind='respond', actions=['handoff', 'create_ticket'], ticket=draft)
        _emit(runtime, 'message', {'content': _COMPLAINT_REPLY, 'kind': 'complaint'})
        return {'answer': _COMPLAINT_REPLY, 'used_citations': [],
                'suggestions': list(control.actions),
                'control': control.model_dump(mode='json'),
                'trace': _trace(state, 'complaint', status='offered')}

    async def chitchat(state: WorkflowState, runtime: Runtime[TurnRuntime]):
        _status(state, runtime, 'chitchat')
        _emit(runtime, 'message', {'content': _CHITCHAT_REPLY, 'kind': 'chitchat'})
        return {'answer': _CHITCHAT_REPLY, 'used_citations': [], 'suggestions': [],
                'trace': _trace(state, 'chitchat', status='fixed')}

    async def budget_reply(state: WorkflowState, runtime: Runtime[TurnRuntime]):
        _status(state, runtime, 'budget_reply')
        _emit(runtime, 'message', {'content': _BUDGET_REPLY, 'kind': 'budget'})
        return {'answer': _BUDGET_REPLY, 'used_citations': [],
                'suggestions': ['handoff'], 'budget_exhausted': True,
                'budget': runtime.context.budget.snapshot(),
                'trace': _trace(state, 'budget_reply', status='fixed')}

    async def persist(state: WorkflowState, runtime: Runtime[TurnRuntime]):
        _status(state, runtime, 'persist')
        context = runtime.context
        answer = state['answer']
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError('completed workflow requires an answer')
        offers: list[dict] = []
        control = FinalControl.model_validate(state['control']) if state['control'] else None
        for suggestion in state['suggestions']:
            if suggestion == 'handoff':
                offers.append({'type': 'handoff'})
            elif suggestion == 'create_ticket':
                if control is None or control.ticket is None:
                    raise ValueError('ticket suggestion requires a draft')
                offer = await context.operations.run(
                    lambda: deps.actions.offer_once(context.ref, context.user_id, control.ticket),
                    context.deadline, mutation=True,
                )
                offers.append({'type': 'create_ticket', **offer.model_dump(mode='json')})
            else:
                raise ValueError('invalid suggestion')
        budget = context.budget.snapshot()
        event_data = {
            'session_id': state['conversation_id'], 'turn_id': state['turn_id'],
            'node_path': [item.get('stage') for item in state['trace']],
            'intent': state['intent'], 'route': state['route'],
            'score': state['score'], 'band': state['band'],
            'assessment': state['assessment'],
            'sources': state['sources'], 'used_citations': state['used_citations'],
            'suggestions': state['suggestions'], 'offers': offers,
            'tools': [
                {key: item.get(key) for key in ('name', 'tool_call_id', 'status', 'attempts')}
                for item in state['trace'] if item.get('stage') == 'tool'
            ],
            'budget': budget,
            'elapsed_ms': max(0, int((time.monotonic() - context.started_at) * 1000)),
            'status': 'completed',
        }
        await context.operations.run(
            lambda: deps.conversations.finish_turn(
                context.ref, answer, 'completed', event_data=event_data),
            context.deadline, mutation=True,
        )
        messages = (
            HumanMessage(state['original_question']),
            *load_tool_messages(state['tool_messages']),
            AIMessage(answer),
        )
        history = [*load_turns(state['history']), StoredTurn(state['turn_id'], messages)]
        history = history[-deps.settings.max_history_turns:]
        logger.info('workflow turn completed', extra={'workflow': {
            'session_id': state['conversation_id'], 'turn_id': state['turn_id'],
            'node_path': event_data['node_path'], 'intent': state['intent'],
            'route': state['route'], 'score': state['score'], 'band': state['band'],
            'tool_count': state['tool_count'], 'budget': budget,
            'elapsed_ms': event_data['elapsed_ms'], 'status': 'completed',
        }})
        return {'offers': offers, 'history': dump_turns(history),
                'budget': budget, 'status': 'completed',
                'trace': _trace(state, 'persist', status='completed')}

    return {
        'resolve': resolve, 'classify': classify, 'retrieve': retrieve,
        'evidence_gate': evidence_gate, 'workflow_answer': workflow_answer,
        'fallback': fallback, 'complaint': complaint, 'chitchat': chitchat,
        'budget_reply': budget_reply, 'persist': persist,
    }
