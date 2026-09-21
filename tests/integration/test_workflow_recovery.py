"""Real PostgreSQL + MySQL acceptance; scripted model only, no external calls."""
import asyncio
from contextlib import asynccontextmanager
from uuid import uuid4
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage
import pytest

from app.db.actions import ActionRepository
from app.db.contracts import TurnRef
from app.db.conversations import ConversationRepository
from app.errors import ServiceError
from app.sessions import SessionGuard
from app.workflow.agent import AgentDependencies
from app.workflow.checkpoints import CheckpointStore
from app.workflow.contracts import FinalControl, IntentResult
from app.workflow.graph import WorkflowDependencies, build_workflow
from app.workflow.state import fresh_state
from tests.ch05_helpers import RecordingLowConfidence, RecordingToolExecutor, ScriptedWorkflowGateway
from tests.test_workflow_graph import ScriptedKnowledgeStage, knowledge_result, settings


@asynccontextmanager
async def service_case(mysql_db, checkpoint_settings, *, intents=('chitchat',), decisions=(), tokens=(), guard=None, knowledge=None):
    from app.services.workflow_chat import WorkflowChatService
    selected = settings()
    conversations = ConversationRepository(mysql_db.sessions)
    gateway = ScriptedWorkflowGateway(intents=[IntentResult(intent=i, needs_business_data=False) for i in intents],
        decisions=decisions, tokens=tokens)
    factory = lambda runtime: gateway.bind(runtime, selected)
    executor = RecordingToolExecutor(gateway)
    agent = AgentDependencies(selected, factory, conversations, AsyncMock(), AsyncMock(), executor)
    store = CheckpointStore(checkpoint_settings, test_mode=True)
    saver = await store.open()
    deps = WorkflowDependencies(selected, factory, ScriptedKnowledgeStage(knowledge or knowledge_result('workflow_answer')),
        agent, conversations, ActionRepository(mysql_db.sessions), RecordingLowConfidence())
    graph = build_workflow(deps, saver)
    service = WorkflowChatService(selected, graph, conversations, guard or SessionGuard(20))
    try:
        yield service, graph, conversations, gateway, saver, executor
    finally:
        await service.aclose()
        await store.aclose()


async def create_session(repository):
    cid = 'ch05_test_' + uuid4().hex[:24]
    await repository.create(cid, 'demo')
    return cid


async def run_turn(service, cid, question='你好', **kwargs):
    async with service.prepare(question, cid, **kwargs) as prepared:
        events = [event async for event in service.stream(prepared)]
        return prepared, events


async def test_migration_only_completed_pending_interrupted_and_owner_checked(mysql_db, checkpoint_settings):
    from app.workflow.recovery import recover_conversation
    async with service_case(mysql_db, checkpoint_settings) as (service, graph, repo, gateway, _, _):
        cid = await create_session(repo)
        good, pending = TurnRef(cid, 'good'), TurnRef(cid, 'pending')
        await repo.start_turn(good, 'demo', '旧问题')
        await repo.finish_turn(good, '旧答案', 'completed')
        await repo.start_turn(pending, 'demo', '中断问题')
        with pytest.raises(ServiceError) as error:
            await recover_conversation(graph, repo, cid, 'other')
        assert error.value.code == 'CONVERSATION_NOT_FOUND'
        history = await recover_conversation(graph, repo, cid, 'demo')
        assert [turn['turn_id'] for turn in history] == ['good']
        assert (await repo.get_turn(pending, 'demo')).status == 'cancelled'
        assert await recover_conversation(graph, repo, cid, 'demo') == history
        assert gateway.model_calls == gateway.tool_calls == 0


async def test_restart_history_isolation_fresh_evidence_and_busy(mysql_db, checkpoint_settings):
    async with service_case(mysql_db, checkpoint_settings, intents=('product',)) as (service, graph, repo, *_):
        cid = await create_session(repo)
        other = await create_session(repo)
        first, events = await run_turn(service, cid, '支持什么协议？', category='第一类')
        assert events[-1].name == 'done'
        assert events[0].data['token_count_is_estimate'] is True
        assert events[-1].data['citations'] == [1]
        assert first.deadline == first.runtime.started_at + 240
    async with service_case(mysql_db, checkpoint_settings) as (service, graph, repo, gateway, *_):
        async with service.prepare('继续', cid) as prepared:
            assert prepared.initial_state['category'] is None
            assert prepared.initial_state['sources'] == []
            assert prepared.initial_state['offers'] == []
            assert [v['turn_id'] for v in prepared.initial_state['history']] == [first.ref.turn_id]
            with pytest.raises(ServiceError) as error:
                async with service.prepare('并发', cid):
                    pass
            assert error.value.code == 'SESSION_BUSY' and error.value.status_code == 409
            events = [e async for e in service.stream(prepared)]
            assert events[-1].name == 'done'
        async with service.prepare('独立', other) as prepared:
            assert prepared.initial_state['history'] == []


@pytest.mark.parametrize('bad_args', [False, True])
@pytest.mark.parametrize('fault', ['checkpoint_commit', 'audit_ack'])
async def test_pg_final_commit_fault_repair_does_not_replay_tools_or_model(mysql_db, checkpoint_settings, monkeypatch, bad_args, fault):
    from app.workflow.recovery import recover_conversation
    call = AIMessage('', tool_calls=[{'name': 'query_order', 'id': 'call-1',
        'args': {'unexpected': True} if bad_args else {'order_id': 'O1'}}])
    async with service_case(mysql_db, checkpoint_settings, intents=('order',),
            decisions=(call, FinalControl(kind='respond', actions=['handoff'])), tokens=('已核对。',)) as (service, graph, repo, gateway, saver, executor):
        cid = await create_session(repo)
        original_put = saver.aput
        fired = []
        async def fail_final(config, checkpoint, metadata, new_versions):
            if checkpoint['channel_values'].get('status') == 'completed':
                fired.append(True)
                raise RuntimeError('injected checkpoint commit failure')
            return await original_put(config, checkpoint, metadata, new_versions)
        if fault == 'checkpoint_commit':
            monkeypatch.setattr(saver, 'aput', fail_final)
        else:
            original_finish = repo.finish_turn
            async def fail_ack(ref, content, status, **kwargs):
                await original_finish(ref, content, status, **kwargs)
                if status == 'completed':
                    fired.append(True)
                    raise RuntimeError('injected audit acknowledgement failure')
            monkeypatch.setattr(repo, 'finish_turn', fail_ack)
        completed, events = await run_turn(service, cid, '查询订单')
        assert fired and events[-1].name == 'error'
        assert not {'actions','done'} & {e.name for e in events}
        assert ''.join(e.data['content'] for e in events if e.name == 'token') == '已核对。'
        statuses = [e for e in events if e.name == 'workflow_status']
        assert next(i for i,e in enumerate(events) if e in statuses and e.data['node'] == 'agent') < next(i for i,e in enumerate(events) if e.name == 'tool_status')
        assert gateway.closed and gateway.tool_calls == 1
        gold = await repo.get_turn(completed.ref, 'demo')
        assert gold.status == 'completed'
    async with service_case(mysql_db, checkpoint_settings, intents=()) as (service, graph, repo, recorder, _, executor):
        config = {'configurable': {'thread_id': cid}}
        before = await graph.aget_state(config)
        assert [task.name for task in before.tasks] == ['persist']
        if fault == 'checkpoint_commit':
            assert before.values['status'] == 'completed' and before.next == ()
            assert before.tasks[0].result['status'] == 'completed'
        else:
            assert before.values['status'] == 'pending' and before.next == ('persist',)
        async with service.prepare('再次提问', cid) as prepared:
            recovered_history = prepared.initial_state['history']
            assert recovered_history[-1]['turn_id'] == completed.ref.turn_id
            repaired = await graph.aget_state(config)
            assert repaired.next == ()
            assert repaired.tasks == ()
            assert recorder.model_calls == 0
            assert recorder.tool_calls == 0
        assert (await repo.get_turn(completed.ref, 'demo')) == gold
        assert (await recover_conversation(graph, repo, cid, 'demo'))[-1]['turn_id'] == completed.ref.turn_id


async def test_store_content_conflict_refuses_new_turn(mysql_db, checkpoint_settings):
    async with service_case(mysql_db, checkpoint_settings) as (service, graph, repo, gateway, *_):
        cid = await create_session(repo)
        prepared, _ = await run_turn(service, cid)
        config = {'configurable': {'thread_id': cid}}
        await graph.aupdate_state(config, {'answer': '冲突答案'}, as_node='persist')
        count = len(await repo.audit(cid, 'demo'))
        with pytest.raises(ServiceError) as error:
            async with service.prepare('新问题', cid):
                pass
        assert error.value.code == 'WORKFLOW_RECOVERY_CONFLICT'
        assert len(await repo.audit(cid, 'demo')) == count
        assert gateway.model_calls == 1


async def test_partial_failure_audits_observed_text_after_drain(mysql_db, checkpoint_settings):
    async with service_case(mysql_db, checkpoint_settings, intents=('order',),
            decisions=(FinalControl(kind='respond'),), tokens=('部分答案', ServiceError('MODEL_PROTOCOL_INVALID', '协议错误', 502))) as (service, graph, repo, gateway, *_):
        cid = await create_session(repo)
        prepared, events = await run_turn(service, cid)
        assert events[-1].name == 'error'
        audit = await repo.get_turn(prepared.ref, 'demo')
        assert audit.status == 'failed' and audit.final_content == '部分答案'
        assert 'agent' in audit.event_data['node_path']
        assert audit.event_data['intent'] == 'order'
        assert audit.event_data['route'] == 'business'
        assert gateway.closed and not any(e.name in ('done','actions') for e in events)


async def test_pending_checkpoint_cleared_without_automatic_replay(mysql_db, checkpoint_settings):
    from app.workflow.recovery import recover_conversation
    async with service_case(mysql_db, checkpoint_settings, intents=()) as (_, graph, repo, recorder, *_):
        cid = await create_session(repo)
        ref = TurnRef(cid, 'crashed-turn')
        await repo.start_turn(ref, 'demo', '崩溃前问题')
        config = {'configurable': {'thread_id': cid}}
        state = fresh_state(ref, 'demo', '崩溃前问题', '旧分类', [], 49152)
        state['sources'] = [{'old': 'evidence'}]
        await graph.aupdate_state(config, state, as_node='resolve')
        assert (await graph.aget_state(config)).next == ('classify',)
        assert await recover_conversation(graph, repo, cid, 'demo') == []
        snapshot = await graph.aget_state(config)
        assert snapshot.next == snapshot.tasks == ()
        assert snapshot.values['sources'] == [] and snapshot.values['category'] is None
        assert (await repo.get_turn(ref, 'demo')).status == 'cancelled'
        assert recorder.model_calls == recorder.tool_calls == 0
        count = len(await repo.audit(cid, 'demo'))
        await recover_conversation(graph, repo, cid, 'demo')
        assert len(await repo.audit(cid, 'demo')) == count
        assert (await graph.aget_state(config)).config == snapshot.config


async def test_disconnect_holds_guard_through_physical_close_and_postdrain_audit(mysql_db, checkpoint_settings, monkeypatch):
    entered, closing, release, auditing, audit_release = (asyncio.Event() for _ in range(5))
    async def blocked_token():
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            closing.set()
            await release.wait()
            raise
    async with service_case(mysql_db, checkpoint_settings, intents=('order',),
            decisions=(FinalControl(kind='respond'),), tokens=('已经显示', blocked_token)) as (service, graph, repo, gateway, *_):
        cid = await create_session(repo)
        finish = repo.finish_turn
        async def delayed_finish(ref, content, status, **kwargs):
            if status == 'cancelled':
                auditing.set()
                await audit_release.wait()
            await finish(ref, content, status, **kwargs)
        monkeypatch.setattr(repo, 'finish_turn', delayed_finish)
        observed, refs = [], []
        displayed = asyncio.Event()
        async def consume():
            async with service.prepare('问题', cid) as prepared:
                refs.append(prepared.ref)
                async for event in service.stream(prepared):
                    observed.append(event)
                    if event.name == 'token':
                        displayed.set()
        task = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(entered.wait(), 3)
            await asyncio.wait_for(displayed.wait(), 3)
            task.cancel()
            await asyncio.wait_for(closing.wait(), 3)
            task.cancel()  # Repeated cancellation cannot release ownership.
            with pytest.raises(ServiceError) as busy:
                async with service.prepare('竞态请求', cid):
                    pass
            assert busy.value.code == 'SESSION_BUSY' and not task.done()
            release.set()
            await asyncio.wait_for(auditing.wait(), 3)
            assert gateway.closed and not task.done()
            with pytest.raises(ServiceError) as busy:
                async with service.prepare('审计未提交', cid):
                    pass
            assert busy.value.code == 'SESSION_BUSY'
            audit_release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            audit = await repo.get_turn(refs[0], 'demo')
            assert audit.status == 'cancelled' and audit.final_content == '已经显示'
            assert 'agent' in audit.event_data['node_path']
            assert len([r for r in await repo.audit(cid, 'demo') if r['role'] == 'assistant']) == 1
            assert not any(e.name in ('actions', 'done') for e in observed)
            async with service.prepare('释放后请求', cid) as prepared:
                assert prepared.initial_state['history'] == []
        finally:
            release.set()
            audit_release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_pending_metadata_conflict_is_not_silently_repaired(mysql_db, checkpoint_settings):
    from app.workflow.recovery import recover_conversation
    async with service_case(mysql_db, checkpoint_settings) as (service, graph, repo, gateway, *_):
        cid = await create_session(repo)
        prepared, _ = await run_turn(service, cid)
        config = {'configurable': {'thread_id': cid}}
        await graph.aupdate_state(config, {'status': 'pending', 'route': 'knowledge'}, as_node='chitchat')
        before = await repo.get_turn(prepared.ref, 'demo')
        with pytest.raises(ServiceError) as error:
            await recover_conversation(graph, repo, cid, 'demo')
        assert error.value.code == 'WORKFLOW_RECOVERY_CONFLICT'
        assert await repo.get_turn(prepared.ref, 'demo') == before


async def test_verified_actions_envelope_matches_persisted_offer(mysql_db, checkpoint_settings):
    async with service_case(mysql_db, checkpoint_settings, intents=('complaint',)) as (service, graph, repo, *_):
        cid = await create_session(repo)
        prepared, events = await run_turn(service, cid, '我要投诉货物破损')
        assert [e.name for e in events[-2:]] == ['actions', 'done']
        payload = events[-2].data
        audit = await repo.get_turn(prepared.ref, 'demo')
        assert payload == {'session_id': cid, 'turn_id': prepared.ref.turn_id,
                           'actions': audit.event_data['offers']}
        assert payload['actions'][0] == {'type': 'handoff'}
        assert payload['actions'][1]['draft']['issue_description'] == '我要投诉货物破损'
        assert payload['actions'][1]['turn_id'] == prepared.ref.turn_id
        assert (await graph.aget_state({'configurable': {'thread_id': cid}})).tasks == ()


async def test_cancel_waits_for_inflight_mysql_write_before_final_audit(mysql_db, checkpoint_settings, monkeypatch):
    entered, cancelled, release = (asyncio.Event() for _ in range(3))
    call = AIMessage('', tool_calls=[{'name': 'query_order', 'id': 'delayed-call', 'args': {'order_id': 'O1'}}])
    async with service_case(mysql_db, checkpoint_settings, intents=('order',), decisions=(call,)) as (service, graph, repo, *_):
        cid = await create_session(repo)
        append = repo.append_call
        async def delayed_append(ref, message, **kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
                await append(ref, message, **kwargs)
        monkeypatch.setattr(repo, 'append_call', delayed_append)
        refs = []
        async def consume():
            async with service.prepare('查订单', cid) as prepared:
                refs.append(prepared.ref)
                async for _ in service.stream(prepared):
                    pass
        task = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(entered.wait(), 3)
            task.cancel()
            await asyncio.wait_for(cancelled.wait(), 3)
            with pytest.raises(ServiceError) as busy:
                async with service.prepare('竞态', cid):
                    pass
            assert busy.value.code == 'SESSION_BUSY' and not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            audit = await repo.audit(cid, 'demo')
            assert [row['role'] for row in audit] == ['user', 'assistant', 'assistant']
            assert audit[1]['tool_call_id'] == 'delayed-call'
            assert {row['turn_status'] for row in audit} == {'cancelled'}
            assert (await repo.history(cid, 'demo', 20)) == []
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_terminal_verification_read_failure_emits_one_safe_error(mysql_db, checkpoint_settings, monkeypatch):
    async with service_case(mysql_db, checkpoint_settings) as (service, graph, repo, gateway, *_):
        cid = await create_session(repo)
        get_turn = repo.get_turn
        async def broken_final_read(ref, user_id):
            turn = await get_turn(ref, user_id)
            if turn and turn.status == 'completed':
                raise RuntimeError('PRIVATE_DRIVER_DETAILS')
            return turn
        monkeypatch.setattr(repo, 'get_turn', broken_final_read)
        prepared, events = await run_turn(service, cid)
        assert [e.name for e in events].count('error') == 1
        assert events[-1].name == 'error'
        assert not any(e.name in ('actions', 'done') for e in events)
        assert 'PRIVATE' not in str(events)
        assert (await get_turn(prepared.ref, 'demo')).status == 'completed'
        service.guard.acquire(cid)
        service.guard.release(cid)


async def test_actual_knowledge_child_entry_precedes_final_token(mysql_db, checkpoint_settings):
    async with service_case(mysql_db, checkpoint_settings, intents=('product',),
            knowledge=knowledge_result('agent_tools', score=.75),
            decisions=(FinalControl(kind='respond'),), tokens=('支持 PD 3.0。[1]',)) as (service, graph, repo, *_):
        cid = await create_session(repo)
        _, events = await run_turn(service, cid, '支持什么协议？')
        names = [event.name for event in events]
        entry = next(i for i, event in enumerate(events)
                     if event.name == 'workflow_status' and event.data['node'] == 'agent')
        assert names.index('sources') < entry < names.index('token')
        assert events[-1].name == 'done' and events[-1].data['citations'] == [1]


@pytest.mark.parametrize('conflict', ['empty_tools', 'prefix_tools', 'node_path', 'tool_summary', 'missing_answer'])
async def test_pending_repair_rejects_missing_pairs_and_conflicting_trace_without_mutation(
        mysql_db, checkpoint_settings, conflict):
    from copy import deepcopy
    calls = tuple(AIMessage('', tool_calls=[{
        'name': 'query_order', 'id': f'call-{index}', 'args': {'order_id': f'O{index}'},
    }]) for index in (1, 2))
    async with service_case(mysql_db, checkpoint_settings, intents=('order',),
            decisions=(*calls, FinalControl(kind='respond')), tokens=('已完成核对。',)) as (
            service, graph, repo, recorder, *_):
        cid = await create_session(repo)
        completed, events = await run_turn(service, cid, '查询两个订单')
        assert events[-1].name == 'done'
        config = {'configurable': {'thread_id': cid}}
        state = deepcopy((await graph.aget_state(config)).values)
        assert len(state['tool_messages']) == 4
        assert state['trace'][-1] == {'stage': 'persist', 'status': 'completed'}
        # Persist only appends its own trace stage and current history. Restore
        # the actual settled pre-persist shape, then corrupt one immutable field.
        state.update(status='pending', history=[], offers=[])
        state['trace'] = state['trace'][:-1]
        if conflict == 'empty_tools':
            state['tool_messages'] = []
        elif conflict == 'prefix_tools':
            state['tool_messages'] = state['tool_messages'][:2]
        elif conflict == 'node_path':
            state['trace'][0]['stage'] = 'unexpected_stage'
        elif conflict == 'missing_answer':
            state['answer'] = None
        else:
            tool_trace = next(item for item in state['trace'] if item['stage'] == 'tool')
            tool_trace['attempts'] += 1
        await graph.aupdate_state(config, state, as_node='agent')
        before = await graph.aget_state(config)
        gold = await repo.get_turn(completed.ref, 'demo')
        audit_before = await repo.audit(cid, 'demo')
        model_calls, tool_calls = recorder.model_calls, recorder.tool_calls
        assert before.next == ('persist',) and gold.status == 'completed'

        with pytest.raises(ServiceError) as error:
            async with service.prepare('新问题不得开始', cid):
                pass

        assert error.value.code == 'WORKFLOW_RECOVERY_CONFLICT'
        after = await graph.aget_state(config)
        assert after.config == before.config and after.values == before.values
        assert await repo.get_turn(completed.ref, 'demo') == gold
        assert await repo.audit(cid, 'demo') == audit_before
        assert recorder.model_calls == model_calls and recorder.tool_calls == tool_calls
