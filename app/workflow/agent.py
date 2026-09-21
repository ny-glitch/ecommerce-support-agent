"""Bounded ReAct subgraph; terminal events and turn finalization belong to its parent.

Only final text deltas become token events. Consumers must accumulate those
before forwarding them, including on exceptions/cancellation, then drain the
runtime operations before persisting failed/cancelled partial-answer audits.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import time

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from app.config import Settings
from app.db.conversations import ConversationRepository
from app.db.faq import FaqRepository
from app.db.tickets import TicketRepository
from app.errors import ServiceError
from app.knowledge.contracts import Citation
from app.knowledge.evidence import validate_citation_numbers
from app.tools.business import ToolContext, build_readonly_registry
from app.tools.executor import ToolExecutor, ToolOutcome, ToolProgress
from app.workflow.contracts import FinalControl, IntentResult
from app.workflow.gateway import WorkflowGateway
from app.workflow.prompts import build_workflow_messages
from app.workflow.state import (
    TurnRuntime, WorkflowState, dump_tool_messages, load_tool_messages, load_turns,
)


_ALLOWED_TOOLS = frozenset({'query_order', 'query_product', 'query_logistics'})
_BUDGET_REPLY = '本轮查询未能全部完成，暂时无法确认剩余信息。您可以补充问题后重试，或选择联系人工客服。'


@dataclass(frozen=True)
class AgentDependencies:
    settings: Settings
    gateway_factory: Callable[[TurnRuntime], WorkflowGateway]
    conversations: ConversationRepository
    faq: FaqRepository
    tickets: TicketRepository
    executor: ToolExecutor


def build_agent_graph(deps: AgentDependencies) -> CompiledStateGraph:
    """Compile per-invocation state, inheriting the parent saver when nested."""

    def registry_for(context):
        return build_readonly_registry(
            ToolContext(context.ref, context.user_id, '', ''), deps.faq, deps.tickets,
        )

    def emit(runtime, name, data):
        runtime.stream_writer({'name': name, 'data': data})

    def messages_for(state, purpose, registry=None):
        return build_workflow_messages(
            purpose, settings=deps.settings, question=state['question'],
            history=load_turns(state['history']),
            sources=[Citation.model_validate(source) for source in state['sources']],
            tool_messages=load_tool_messages(state['tool_messages']),
            intent=IntentResult.model_validate(state['intent']) if state['intent'] else None,
            control=FinalControl.model_validate(state['control']) if state['control'] else None,
            tool_schemas=registry.schemas() if registry else (),
        ).messages

    def exhausted(state, runtime, reason):
        # Drop the unexecuted proposal; only admitted/settled pairs reach final.
        return {'pending_call': None, 'budget_exhausted': True,
                'control': FinalControl(kind='respond', actions=['handoff']).model_dump(mode='json'),
                'budget': runtime.context.budget.snapshot(),
                'trace': [*state['trace'], {'stage': 'agent', 'status': 'limited', 'reason': reason}]}

    async def decide(state: WorkflowState, runtime: Runtime[TurnRuntime]):
        context = runtime.context
        if time.monotonic() >= context.deadline:
            return exhausted(state, runtime, 'deadline')
        if state['decision_count'] >= deps.settings.agent_max_decisions:
            return exhausted(state, runtime, 'decisions')
        registry = registry_for(context)
        try:
            messages = messages_for(state, 'agent', registry)
            decision = await context.operations.run(
                lambda: deps.gateway_factory(context).decide(messages, registry.tools),
                context.deadline,
            )
        except TimeoutError:
            return exhausted(state, runtime, 'deadline')
        except ServiceError as error:
            if error.code not in {'TURN_BUDGET_EXHAUSTED', 'INPUT_TOO_LONG'}:
                raise
            return exhausted(state, runtime, error.code)
        update = {'decision_count': state['decision_count'] + 1,
                  'budget': context.budget.snapshot(),
                  'trace': [*state['trace'], {'stage': 'agent', 'decision': state['decision_count'] + 1}]}
        if isinstance(decision, FinalControl):
            # Revalidate even a scripted/custom gateway's constructed objects.
            control = FinalControl.model_validate(decision.model_dump(mode='json'))
            return update | {'pending_call': None, 'control': control.model_dump(mode='json')}
        if (not isinstance(decision, AIMessage) or decision.invalid_tool_calls
                or len(decision.tool_calls) != 1):
            raise ServiceError('INVALID_TOOL_CALL', '工具调用格式无效，请重试', 502)
        call = decision.tool_calls[0]
        previous_ids = {m['tool_call_id'] for m in state['tool_messages'] if m['role'] == 'tool'}
        if (call.get('name') not in _ALLOWED_TOOLS or not call.get('id')
                or call['id'] in previous_ids or not isinstance(call.get('args'), dict)):
            raise ServiceError('INVALID_TOOL_CALL', '工具调用格式无效，请重试', 502)
        # Never retain model intermediate text, reasoning or provider metadata.
        return update | {'pending_call': dict(call)}

    def after_decision(state):
        if state['budget_exhausted']:
            return 'budget_reply'
        if state['pending_call'] is None:
            return 'final'
        if state['tool_count'] >= deps.settings.agent_max_tool_calls:
            return 'budget_reply'
        return 'tools'

    async def execute_tools(state: WorkflowState, runtime: Runtime[TurnRuntime]):
        context = runtime.context
        if time.monotonic() >= context.deadline:
            return exhausted(state, runtime, 'deadline')
        call = state['pending_call']
        # Independent server boundary; schemas supplied to a model are not authorization.
        if not call or call['name'] not in _ALLOWED_TOOLS:
            raise ServiceError('INVALID_TOOL_CALL', '工具调用格式无效，请重试', 502)
        step = state['tool_count']
        decision = AIMessage('', tool_calls=[call])
        await context.operations.run(
            lambda: deps.conversations.append_call(context.ref, decision, step=step),
            context.deadline, mutation=True,
        )
        execution = context.operations.track_iterator(deps.executor.run(
            call, registry_for(context), deadline=context.deadline,
        ))
        outcome = None
        while True:
            try:
                event = await context.operations.run(lambda: anext(execution), context.deadline)
            except StopAsyncIteration:
                break
            if isinstance(event, ToolOutcome):
                if outcome is not None:
                    raise ServiceError('INVALID_TOOL_RESULT', '工具结果无效，请重试', 502)
                outcome = event
            elif isinstance(event, ToolProgress):
                emit(runtime, 'tool_status', asdict(event))
        if outcome is None:
            raise ServiceError('INVALID_TOOL_RESULT', '工具结果无效，请重试', 502)
        messages = load_tool_messages(state['tool_messages']) + [decision, outcome.message]
        wire = dump_tool_messages(messages)
        # Original executor has already physically settled the invocation. The
        # entire iterator is exhausted before audit and terminal progress.
        await context.operations.run(
            lambda: deps.conversations.append_result(context.ref, outcome.message, step=step),
            context.deadline, mutation=True,
        )
        emit(runtime, 'tool_status', asdict(ToolProgress(
            call['name'], call['id'], outcome.terminal_status, outcome.attempt,
            {'succeeded': '工具执行完成', 'not_found': '未找到匹配信息'}.get(
                outcome.terminal_status, '工具执行结果无法确认，请稍后核实'),
        )))
        return {'tool_messages': wire, 'tool_count': step + 1, 'pending_call': None,
                'budget': context.budget.snapshot(),
                'trace': [*state['trace'], {'stage': 'tool', 'step': step,
                    'name': call['name'], 'tool_call_id': call['id'],
                    'status': outcome.terminal_status, 'attempts': outcome.attempt}]}

    async def budget_reply(state: WorkflowState, runtime: Runtime[TurnRuntime]):
        reason = 'tools' if state['tool_count'] >= deps.settings.agent_max_tool_calls else 'budget'
        return exhausted(state, runtime, reason)

    def fixed_reply(state, runtime):
        emit(runtime, 'refusal', {'content': _BUDGET_REPLY, 'reason_code': 'TURN_BUDGET_EXHAUSTED'})
        return {'answer': _BUDGET_REPLY, 'used_citations': [], 'suggestions': ['handoff'],
                'pending_call': None, 'budget_exhausted': True,
                'budget': runtime.context.budget.snapshot()}

    async def final(state: WorkflowState, runtime: Runtime[TurnRuntime]):
        context = runtime.context
        if time.monotonic() >= context.deadline:
            return fixed_reply(state, runtime)
        parts = []
        try:
            messages = messages_for(state, 'answer')
            upstream = context.operations.track_iterator(
                deps.gateway_factory(context).stream_final(messages))
            while True:
                try:
                    text = await context.operations.run(lambda: anext(upstream), context.deadline)
                except StopAsyncIteration:
                    break
                if text:
                    parts.append(text)
                    emit(runtime, 'token', {'content': text})
        except TimeoutError as error:
            raise ServiceError('TURN_DEADLINE_EXCEEDED', '本轮请求超时，请重试', 504) from error
        except ServiceError as error:
            if not parts and error.code in {'TURN_BUDGET_EXHAUSTED', 'INPUT_TOO_LONG'}:
                return fixed_reply(state, runtime)
            raise
        answer = ''.join(parts)
        if not answer.strip():
            raise ServiceError('UPSTREAM_INCOMPLETE', '模型回复未正常完成，请重试', 502)
        try:
            used_citations = sorted(validate_citation_numbers(
                answer, {source['number'] for source in state['sources']},
                require_citation=bool(state['sources'])))
        except ValueError as error:
            raise ServiceError('INVALID_CITATION', '知识回答引用无效，请重试', 502) from error
        return {'answer': answer, 'used_citations': used_citations,
                'suggestions': (state['control'] or {}).get('actions', []),
                'budget': context.budget.snapshot(),
                'trace': [*state['trace'], {'stage': 'answer', 'status': 'validated'}]}

    def entry(state):
        if state['agent_mode'] not in {'tools', 'generate_only'}:
            raise ValueError('invalid agent mode')
        return 'final' if state['agent_mode'] == 'generate_only' else 'decide'

    graph = StateGraph(WorkflowState, context_schema=TurnRuntime)
    graph.add_node('decide', decide)
    graph.add_node('tools', execute_tools)
    graph.add_node('final', final)
    graph.add_node('budget_reply', budget_reply)
    graph.add_conditional_edges(START, entry, ['decide', 'final'])
    graph.add_conditional_edges('decide', after_decision, ['tools', 'final', 'budget_reply'])
    graph.add_conditional_edges('tools', lambda state: 'budget_reply' if state['budget_exhausted'] else 'decide', ['decide', 'budget_reply'])
    graph.add_edge('budget_reply', 'final')
    graph.add_edge('final', END)
    return graph.compile()
