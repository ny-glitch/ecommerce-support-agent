"""Agent boundary tests: real graph, budget, registry and original executor."""
import asyncio
import importlib
import json
import time
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.config import Settings
from app.db.contracts import TurnRef
from app.errors import ServiceError
from app.services.turn_operations import TurnOperations
from app.tools.executor import ToolExecutor
from app.tools.results import TransientToolError
from app.workflow.budget import RequestBudget
from app.workflow.contracts import FinalControl
from app.workflow.state import TurnRuntime, fresh_state, validate_turn_messages
from tests.ch05_helpers import RecordingConversations, ScriptedWorkflowGateway


def proposal(name='query_order', call_id='c1', args=None):
    return AIMessage('untrusted intermediate text', tool_calls=[{
        'name': name, 'id': call_id, 'args': {'order_id': 'O1'} if args is None else args,
    }], additional_kwargs={'reasoning_content': 'private'})


def setup_agent(*, decisions=(), tokens=('已查询演示数据。',), mode='tools', budget=49152,
                seconds=60, settings=None, repo=None, executor=None):
    module = importlib.import_module('app.workflow.agent')
    settings = settings or Settings(_env_file=None, llm_base_url='http://localhost:9999/v1',
        llm_model='test', llm_api_key='test')
    ref = TurnRef('conversation', 'turn')
    runtime = TurnRuntime(ref, 'user', time.monotonic(), time.monotonic() + seconds,
        RequestBudget(budget), TurnOperations())
    gateway = ScriptedWorkflowGateway(decisions=decisions, tokens=tokens).bind(runtime, settings)
    repo = repo or RecordingConversations()
    deps = module.AgentDependencies(settings, lambda runtime: gateway, repo,
        AsyncMock(), AsyncMock(), executor or ToolExecutor())
    graph = module.build_agent_graph(deps)
    state = fresh_state(ref, 'user', '先核对订单 O1 状态，再查物流', None, [], budget)
    state['agent_mode'] = mode
    return graph, state, runtime, gateway, repo


async def collect(graph, state, runtime, repo, events=None):
    events, result = ([] if events is None else events), None
    try:
        async for kind, value in graph.astream(state, context=runtime,
                                             stream_mode=['custom', 'values']):
            if kind == 'custom':
                events.append(value)
                repo.trace.append(('event', value))
            else:
                result = value
    finally:
        await runtime.operations.drain()
    return result, events


@pytest.mark.asyncio
async def test_order_then_logistics_feedback_is_paired_and_only_final_text_streams():
    graph, state, runtime, gateway, repo = setup_agent(decisions=[
        proposal(), proposal('query_logistics', 'c2'), FinalControl(kind='respond')])
    result, events = await collect(graph, state, runtime, repo)
    assert [m['name'] for m in result['tool_messages'] if m['role'] == 'tool'] == ['query_order', 'query_logistics']
    assert result['tool_count'] == 2 and result['decision_count'] == 3
    assert [m.tool_call_id for m in gateway.calls[1]['messages'] if isinstance(m, ToolMessage)] == ['c1']
    assert [m.tool_call_id for m in gateway.calls[2]['messages'] if isinstance(m, ToolMessage)] == ['c1', 'c2']
    assert result['answer'] == '已查询演示数据。'
    assert 'private' not in json.dumps(result, ensure_ascii=False)
    assert 'untrusted intermediate text' not in json.dumps(result, ensure_ascii=False)
    for index, item in enumerate(repo.trace):
        if item[0] == 'event' and item[1]['name'] == 'tool_status' and item[1]['data']['status'] == 'succeeded':
            step = 0 if item[1]['data']['tool_call_id'] == 'c1' else 1
            assert ('result', step) in repo.trace[:index]
    token_index = next(i for i, e in enumerate(events) if e['name'] == 'token')
    assert all(e['name'] != 'token' for e in events[:token_index])
    assert [c['stage'] for c in gateway.calls] == ['agent', 'agent', 'agent', 'answer']
    assert len(result['budget']['reservations']) == 4
    assert not any(e['name'] == 'done' for e in events)
    assert repo.finished == [] and graph.checkpointer is None


@pytest.mark.parametrize('decisions,count', [
    ([proposal('query_logistics'), FinalControl(kind='respond')], 1),
    ([FinalControl(kind='clarify')], 0),
])
@pytest.mark.asyncio
async def test_simple_query_and_missing_order_need_no_extra_tools(decisions, count):
    graph, state, runtime, gateway, repo = setup_agent(decisions=decisions)
    if not count:
        state['question'] = '请查物流'
    result, _ = await collect(graph, state, runtime, repo)
    assert result['tool_count'] == count
    assert len(repo.calls) == count


@pytest.mark.asyncio
async def test_generate_only_skips_all_decisions():
    graph, state, runtime, gateway, repo = setup_agent(mode='generate_only')
    result, _ = await collect(graph, state, runtime, repo)
    assert result['decision_count'] == result['tool_count'] == 0
    assert [c['stage'] for c in gateway.calls] == ['answer']


@pytest.mark.parametrize('bad', [proposal('create_ticket'), proposal('query_faq'), proposal('unknown'),
    AIMessage('', tool_calls=[{'name': 'query_order', 'id': '1', 'args': {}}, {'name': 'query_product', 'id': '2', 'args': {}}])])
@pytest.mark.asyncio
async def test_disallowed_or_multiple_tools_are_rejected_before_executor(bad):
    graph, state, runtime, gateway, repo = setup_agent(decisions=[bad])
    with pytest.raises(ServiceError, match='工具'):
        await collect(graph, state, runtime, repo)
    assert repo.calls == repo.results == {}
    assert [c['stage'] for c in gateway.calls] == ['agent']


@pytest.mark.parametrize('args', [{}, {'order_id': 1}, {'order_id': 'O1', 'extra': 'bad'}])
@pytest.mark.asyncio
async def test_original_executor_parameter_error_reaches_next_decision(args):
    graph, state, runtime, gateway, repo = setup_agent(decisions=[proposal(args=args), FinalControl(kind='clarify')])
    result, _ = await collect(graph, state, runtime, repo)
    feedback = [m for m in gateway.calls[1]['messages'] if isinstance(m, ToolMessage)]
    assert len(feedback) == 1 and feedback[0].tool_call_id == 'c1'
    assert json.loads(feedback[0].content)['code'] == 'INVALID_TOOL_ARGUMENTS'
    assert feedback[0].status == 'error' and result['tool_count'] == 1


@pytest.mark.asyncio
async def test_fifth_tool_proposal_never_executes_and_sixth_decision_never_requested():
    graph, state, runtime, gateway, repo = setup_agent(decisions=[proposal(call_id=f'c{i}') for i in range(5)])
    result, _ = await collect(graph, state, runtime, repo)
    assert result['tool_count'] == 4 and result['decision_count'] == 5
    assert len(repo.calls) == len(repo.results) == 4
    assert [c['stage'] for c in gateway.calls] == ['agent'] * 5 + ['answer']
    assert result['budget_exhausted'] is True
    assert len([m for m in gateway.calls[-1]['messages'] if isinstance(m, ToolMessage)]) == 4


@pytest.mark.asyncio
async def test_configured_decision_limit_and_exhausted_budget_never_dispatch_extra_requests():
    settings = Settings(_env_file=None, llm_base_url='http://localhost:9999/v1', llm_model='test', llm_api_key='test', agent_max_decisions=1)
    graph, state, runtime, gateway, repo = setup_agent(decisions=[proposal()], settings=settings)
    result, _ = await collect(graph, state, runtime, repo)
    assert result['decision_count'] == 1 and result['tool_count'] == 1
    assert [c['stage'] for c in gateway.calls] == ['agent', 'answer']
    graph, state, runtime, gateway, repo = setup_agent(budget=1)
    result, events = await collect(graph, state, runtime, repo)
    assert gateway.calls == [] and repo.calls == {}
    assert result['budget_exhausted'] is True and result['answer']
    assert not any(e['name'] == 'token' for e in events)


@pytest.mark.asyncio
async def test_expired_deadline_does_not_open_upstream():
    graph, state, runtime, gateway, repo = setup_agent(seconds=-1)
    result, _ = await collect(graph, state, runtime, repo)
    assert gateway.calls == [] and repo.calls == {} and result['budget_exhausted']


@pytest.mark.asyncio
async def test_continuous_tokens_cannot_reset_total_deadline_and_partial_answer_is_retained():
    async def tick():
        await asyncio.sleep(.01)
        return '片'
    graph, state, runtime, gateway, repo = setup_agent(mode='generate_only', seconds=.055, tokens=[tick] * 100)
    events = []
    with pytest.raises(ServiceError) as failure:
        await collect(graph, state, runtime, repo, events)
    partial = ''.join(e['data']['content'] for e in events if e['name'] == 'token')
    assert 0 < len(partial) < 100 and gateway.closed
    assert failure.value.code == 'TURN_DEADLINE_EXCEEDED'
    assert not repo.finished and not any(e['name'] == 'done' for e in events)


@pytest.mark.asyncio
async def test_citation_failure_returns_partial_answer_with_failed_status():
    graph, state, runtime, gateway, repo = setup_agent(mode='generate_only', tokens=['错误来源[2]'])
    state['sources'] = [dict(number=1, chunk_id=10, category='退款退货', section_path=None,
        questions='如何退款', answer='七天', content_hash='abc', url='/test', score=.5)]
    events = []
    with pytest.raises(ServiceError) as failure:
        await collect(graph, state, runtime, repo, events)
    assert failure.value.code == 'INVALID_CITATION'
    assert ''.join(e['data']['content'] for e in events if e['name'] == 'token') == '错误来源[2]'
    assert not any(e['name'] in {'error', 'done', 'actions'} for e in events) and not repo.finished


@pytest.mark.asyncio
async def test_incomplete_upstream_keeps_emitted_answer_and_suppresses_suggestions():
    graph, state, runtime, gateway, repo = setup_agent(decisions=[FinalControl(kind='respond', actions=['handoff'])],
        tokens=['部分', ServiceError('UPSTREAM_INCOMPLETE', '模型回复未正常完成，请重试', 502)])
    events = []
    with pytest.raises(ServiceError) as failure:
        await collect(graph, state, runtime, repo, events)
    assert failure.value.code == 'UPSTREAM_INCOMPLETE'
    assert ''.join(e['data']['content'] for e in events if e['name'] == 'token') == '部分'
    assert not any(e['name'] in {'done', 'actions', 'error'} for e in events) and gateway.closed


@pytest.mark.asyncio
async def test_result_audit_failure_never_emits_success_or_generates_answer():
    class FailingRepository(RecordingConversations):
        async def append_result(self, *args, **kwargs):
            raise ServiceError('AUDIT_FAILED', '审计写入失败', 503)
    repo = FailingRepository()
    graph, state, runtime, gateway, repo = setup_agent(decisions=[proposal()], repo=repo)
    events = []
    try:
        with pytest.raises(ServiceError):
            async for event in graph.astream(state, context=runtime, stream_mode='custom'):
                events.append(event)
    finally:
        await runtime.operations.drain()
    assert not any(e['name'] == 'tool_status' and e['data']['status'] == 'succeeded' for e in events)
    assert [c['stage'] for c in gateway.calls] == ['agent']


@pytest.mark.asyncio
async def test_cancellation_leaves_physical_close_owned_until_drain():
    entered, closing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def blocked():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            closing.set()
            await release.wait()
    graph, state, runtime, gateway, repo = setup_agent(mode='generate_only', tokens=[blocked])
    task = asyncio.create_task(graph.ainvoke(state, context=runtime))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await closing.wait()
    drain = asyncio.create_task(runtime.operations.drain())
    await asyncio.sleep(0)
    assert not drain.done() and not gateway.closed
    release.set()
    await drain
    assert gateway.closed and not repo.finished


@pytest.mark.asyncio
async def test_transient_retry_uses_original_executor_and_same_absolute_deadline(monkeypatch):
    # Replace only the original tool's physical invocation, keep its schema,
    # registry and executor. No invented alternate business implementation.
    from langchain_core.tools import BaseTool
    original = BaseTool.ainvoke
    attempts, deadlines = [], []
    async def transient(self, *args, **kwargs):
        if self.name == 'query_order':
            attempts.append(time.monotonic())
            raise TransientToolError('temporary')
        return await original(self, *args, **kwargs)
    monkeypatch.setattr(BaseTool, 'ainvoke', transient)
    class ObservedExecutor(ToolExecutor):
        def run(self, *args, deadline, **kwargs):
            deadlines.append(deadline)
            return super().run(*args, deadline=deadline, **kwargs)
    graph, state, runtime, gateway, repo = setup_agent(decisions=[proposal(), FinalControl(kind='respond')], executor=ObservedExecutor())
    result, events = await collect(graph, state, runtime, repo)
    assert len(attempts) == 2 and deadlines == [runtime.deadline]
    assert result['tool_count'] == 1
    assert any(e['name'] == 'tool_status' and e['data']['status'] == 'retrying' for e in events)
    assert json.loads(next(m for m in gateway.calls[1]['messages'] if isinstance(m, ToolMessage)).content)['code'] == 'TOOL_TEMPORARY_FAILURE'


def test_in_progress_tool_wire_is_validated_without_weakening_completed_history():
    from app.workflow.state import dump_tool_messages, load_tool_messages
    call = proposal()
    call.content = ''
    messages = [call, ToolMessage('result', tool_call_id='c1', name='query_order')]
    wire = dump_tool_messages(messages)
    assert len(load_tool_messages(wire)) == 2 and 'private' not in json.dumps(wire)
    with pytest.raises(ValueError):
        validate_turn_messages([HumanMessage('q'), *messages])
    for invalid in ([wire[0]], [wire[1]], [*wire, *wire]):
        with pytest.raises(ValueError):
            load_tool_messages(invalid)


@pytest.mark.asyncio
async def test_small_remaining_budget_falls_back_after_one_tool_without_extra_requests():
    graph, state, runtime, gateway, repo = setup_agent(decisions=[proposal()], budget=5000)
    result, events = await collect(graph, state, runtime, repo)
    assert len(repo.calls) == len(repo.results) == result['tool_count'] == 1
    assert [c['stage'] for c in gateway.calls] == ['agent']
    assert len(result['budget']['reservations']) == 1 and result['budget_exhausted']
    assert not any(e['name'] == 'token' for e in events)
    assert len([e for e in events if e['name'] == 'refusal']) == 1


@pytest.mark.asyncio
async def test_no_more_tool_budget_can_still_generate_from_settled_facts():
    graph, state, runtime, gateway, repo = setup_agent(decisions=[proposal()], budget=6500)
    result, _ = await collect(graph, state, runtime, repo)
    assert [c['stage'] for c in gateway.calls] == ['agent', 'answer']
    assert result['answer'] == '已查询演示数据。' and result['budget_exhausted']
    assert len(result['budget']['reservations']) == 2
    assert len([m for m in gateway.calls[-1]['messages'] if isinstance(m, ToolMessage)]) == 1


@pytest.mark.asyncio
async def test_duplicate_call_identity_is_rejected_before_second_execution():
    graph, state, runtime, gateway, repo = setup_agent(decisions=[proposal(), proposal()])
    with pytest.raises(ServiceError) as failure:
        await collect(graph, state, runtime, repo)
    assert failure.value.code == 'INVALID_TOOL_CALL'
    assert len(repo.calls) == len(repo.results) == 1


@pytest.mark.asyncio
async def test_tool_cancellation_drain_waits_for_real_invocation_before_closing_iterator(monkeypatch):
    from langchain_core.tools import BaseTool
    entered, closing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def delayed_close(self, *args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            closing.set()
            await release.wait()
    monkeypatch.setattr(BaseTool, 'ainvoke', delayed_close)
    graph, state, runtime, gateway, repo = setup_agent(decisions=[proposal()])
    events = []
    async def stream():
        async for event in graph.astream(state, context=runtime, stream_mode='custom'):
            events.append(event)
    task = asyncio.create_task(stream())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await closing.wait()
    drain = asyncio.create_task(runtime.operations.drain())
    await asyncio.sleep(0)
    assert not drain.done() and repo.results == {}
    release.set()
    await drain
    assert repo.results == {} and not repo.finished
    assert not any(e['name'] == 'tool_status' and e['data']['status'] == 'succeeded' for e in events)


@pytest.mark.asyncio
async def test_readonly_tools_preserve_original_results_with_deterministic_rng(monkeypatch):
    import random
    from app.tools import business
    original_random = random.Random
    monkeypatch.setattr(business.random, 'Random', lambda: original_random(7))
    graph, state, runtime, gateway, repo = setup_agent(decisions=[proposal(), proposal('query_logistics', 'c2'), FinalControl(kind='respond')])
    result, _ = await collect(graph, state, runtime, repo)
    first = next(m for m in gateway.calls[1]['messages'] if isinstance(m, ToolMessage))
    assert json.loads(first.content)['data']['order_status'] == '运输中'
    assert {s['function']['name'] for s in gateway.calls[0]['schemas']} == {'query_order', 'query_product', 'query_logistics'}
    assert result['tool_count'] == 2


@pytest.mark.asyncio
async def test_required_context_exceeding_window_falls_back_without_model_request():
    settings = Settings(_env_file=None, llm_base_url='http://localhost:9999/v1',
        llm_model='test', llm_api_key='test', context_window_tokens=1600)
    graph, state, runtime, gateway, repo = setup_agent(settings=settings)
    result, events = await collect(graph, state, runtime, repo)
    assert gateway.calls == [] and repo.calls == {}
    assert result['budget_exhausted'] and result['answer']
    assert len([e for e in events if e['name'] == 'refusal']) == 1
