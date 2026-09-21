"""Adapter boundary tests; durable database behavior lives in integration."""
from copy import deepcopy
import math
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import pytest

from app.db.contracts import StoredTurn, TurnRef, TurnSnapshot
from app.errors import ServiceError
from app.workflow.state import dump_turns


def test_adapter_rejects_forged_terminal_events_and_private_fields():
    from app.services.workflow_chat import checked_chat_event
    for name in ('done', 'error', 'actions', 'unknown'):
        with pytest.raises(ServiceError):
            checked_chat_event({'name': name, 'data': {}})
    with pytest.raises(ServiceError):
        checked_chat_event({'name': 'token', 'data': {'content': 'x', 'reasoning': 'secret'}})
    assert checked_chat_event({'name': 'token', 'data': {'content': '答案'}}).data == {'content': '答案'}


def completed_pair():
    ref = TurnRef('session', 'turn')
    messages = (HumanMessage('问题'), AIMessage('', tool_calls=[{
        'name': 'query_order', 'id': 'call-1', 'args': {'order_id': 'x'}}]),
        ToolMessage('{"error":"INVALID_TOOL_ARGUMENTS"}', tool_call_id='call-1',
                    name='query_order', status='error'), AIMessage('答案'))
    metadata = dict(session_id='session', turn_id='turn', status='completed',
        intent=None, route='business', score=None, band=None, assessment=None,
        sources=[], used_citations=[], suggestions=['handoff'], offers=[{'type': 'handoff'}],
        budget={}, node_path=['agent', 'persist'], tools=[], elapsed_ms=12)
    state = dict(conversation_id='session', turn_id='turn', status='completed',
        original_question='问题', answer='答案', trace=[{'stage': 'agent'}, {'stage': 'persist'}], history=dump_turns([StoredTurn('turn', messages)]),
        **{k: deepcopy(metadata[k]) for k in ('intent','route','score','band','assessment',
            'sources','used_citations','suggestions','offers','budget')})
    audit_messages = (*messages[:2], ToolMessage(messages[2].content, tool_call_id='call-1'), messages[-1])
    return SimpleNamespace(values=state, next=()), TurnSnapshot(ref, '问题', 'completed', '答案', metadata, audit_messages)


def test_completion_barrier_compares_persisted_business_fields_without_optional_tool_defaults():
    from app.services.workflow_chat import verify_completed_turn
    snapshot, audit = completed_pair()
    assert verify_completed_turn(snapshot, audit) == {
        'session_id': 'session', 'turn_id': 'turn', 'status': 'completed', 'refused': False, 'citations': []}
    for field, value in [('answer', '别的答案'), ('offers', []), ('used_citations', [1])]:
        bad = deepcopy(snapshot)
        bad.values[field] = value
        with pytest.raises(ServiceError):
            verify_completed_turn(bad, audit)
    bad = deepcopy(snapshot)
    bad.values['history'][0]['messages'][1]['tool_calls'][0]['args'] = {'order_id': 'other'}
    with pytest.raises(ServiceError):
        verify_completed_turn(bad, audit)
    snapshot.next = ('persist',)
    with pytest.raises(ServiceError):
        verify_completed_turn(snapshot, audit)


def test_completion_barrier_tolerates_only_machine_precision_score_roundtrip():
    from app.services.workflow_chat import verify_completed_turn
    checkpoint_score = 0.10740864967980515
    mysql_score = 0.10740864967980517
    snapshot, audit = completed_pair()
    snapshot.values['score'] = checkpoint_score
    audit.event_data['score'] = mysql_score
    source = {'chunk_id': 910071, 'score': checkpoint_score, 'category': '个护电器'}
    snapshot.values['sources'] = [source]
    audit.event_data['sources'] = [{**source, 'score': mysql_score}]
    assert verify_completed_turn(snapshot, audit)['status'] == 'completed'
    snapshot.values['score'], audit.event_data['score'] = mysql_score, checkpoint_score
    snapshot.values['sources'][0]['score'] = mysql_score
    audit.event_data['sources'][0]['score'] = checkpoint_score
    assert verify_completed_turn(snapshot, audit)['status'] == 'completed'
    snapshot.values['score'] = checkpoint_score
    snapshot.values['sources'][0]['score'] = checkpoint_score
    audit.event_data['sources'][0]['score'] = mysql_score
    two_ulps = math.nextafter(mysql_score, math.inf)

    for field, value in (
        ('score', two_ulps),
        ('score', checkpoint_score + 0.01),
        ('score', True),
        ('score', float('inf')),
        ('score', float('nan')),
        ('sources', [{**source, 'category': '不同分类', 'score': mysql_score}]),
        ('sources', [{**source, 'score': True}]),
    ):
        conflicting = deepcopy(audit)
        conflicting.event_data[field] = value
        with pytest.raises(ServiceError):
            verify_completed_turn(snapshot, conflicting)


async def test_oversized_intent_rejected_before_creating_durable_state():
    from unittest.mock import AsyncMock
    from app.services.workflow_chat import WorkflowChatService
    from app.sessions import SessionGuard
    from tests.test_workflow_graph import settings
    repo = AsyncMock()
    service = WorkflowChatService(settings(context_window_tokens=256), AsyncMock(), repo, SessionGuard(2))
    with pytest.raises(ServiceError) as error:
        async with service.prepare('问题' * 1000, None):
            pass
    assert error.value.code == 'INPUT_TOO_LONG'
    repo.create.assert_not_awaited()
    repo.start_turn.assert_not_awaited()


def test_actual_not_found_tool_progress_is_allowed():
    from app.services.workflow_chat import checked_chat_event
    event = checked_chat_event({'name': 'tool_status', 'data': {
        'name': 'query_order', 'tool_call_id': 'call', 'status': 'not_found',
        'attempt': 1, 'message': '未找到匹配信息'}})
    assert event.data['status'] == 'not_found'


def test_completion_rejects_conflicting_trace_metadata():
    from app.services.workflow_chat import verify_completed_turn
    snapshot, audit = completed_pair()
    snapshot.values['trace'] = [{'stage': 'resolve'}, {'stage': 'persist'}]
    with pytest.raises(ServiceError):
        verify_completed_turn(snapshot, audit)


async def test_cancellation_winning_at_cleanup_completion_is_not_swallowed():
    import asyncio
    from app.services.workflow_chat import _settled
    gate = asyncio.Event()
    physical = asyncio.create_task(gate.wait())
    waiter = asyncio.create_task(_settled(physical))
    await asyncio.sleep(0)
    physical.add_done_callback(lambda _: waiter.cancel())
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert physical.done() and not physical.cancelled()


async def test_service_close_settles_every_cleanup_before_shared_owners_on_failure():
    import asyncio

    from app.resource_lifecycle import close_resources
    from app.services.workflow_chat import WorkflowChatService
    from app.sessions import SessionGuard
    from tests.test_workflow_graph import settings

    start = asyncio.Event()
    first_failed = asyncio.Event()
    other_draining = asyncio.Event()
    release_other = asyncio.Event()
    terminal_audit = asyncio.Event()
    original_error = RuntimeError("first cleanup failed")
    first_task = None

    async def cleanup():
        await start.wait()
        if asyncio.current_task() is first_task:
            first_failed.set()
            raise original_error
        other_draining.set()
        await release_other.wait()
        terminal_audit.set()

    service = WorkflowChatService(
        settings(), object(), object(), SessionGuard(2)
    )
    tasks = [asyncio.create_task(cleanup()) for _ in range(2)]
    service._cleanup_tasks.update(tasks)
    for task in tasks:
        task.add_done_callback(service._cleanup_tasks.discard)
    first_task = tuple(service._cleanup_tasks)[0]
    other_task = next(task for task in tasks if task is not first_task)

    observations = []

    class SharedOwner:
        async def aclose(self):
            observations.append((other_task.done(), terminal_audit.is_set()))

    closing = asyncio.create_task(close_resources([SharedOwner(), service]))
    await asyncio.sleep(0)
    start.set()
    try:
        await first_failed.wait()
        await other_draining.wait()
        await asyncio.sleep(0)
        closing.cancel()
        closing.cancel()
        await asyncio.sleep(0)
        assert not closing.done()
        assert observations == []
    finally:
        release_other.set()

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await closing
    assert exc_info.value.__cause__ is original_error
    assert observations == [(True, True)]
