"""Real isolated MySQL coverage: dropped deduplication/order/ownership must fail."""
import asyncio
import importlib.util
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from sqlalchemy import select, func, update

from app.db.contracts import TurnRef
from app.db.models import Conversation, Message, Ticket
from app.errors import ServiceError
from app.tools.schemas import TicketInput

pytestmark = pytest.mark.asyncio


def call(step):
    return AIMessage(content='', tool_calls=[{'name':'query_order','args':{'order_id':'1001'},'id':f'call-{step}','type':'tool_call'}])


async def test_two_steps_replay_conflict_and_restart(repos, new_turn, mysql_db):
    conversations, _, _ = repos
    await conversations.create(new_turn.conversation_id, 'demo')
    await conversations.start_turn(new_turn, 'demo', '先查订单，再查物流')
    await conversations.start_turn(new_turn, 'demo', '先查订单，再查物流')
    with pytest.raises(ServiceError) as err:
        await conversations.start_turn(new_turn, 'demo', '不同问题')
    assert err.value.code == 'EVENT_CONFLICT'
    for step in range(2):
        await conversations.append_call(new_turn, call(step), step=step)
        await conversations.append_call(new_turn, call(step), step=step)
        result = ToolMessage(content='{"status":"ok"}', tool_call_id=f'call-{step}')
        await conversations.append_result(new_turn, result, step=step)
        await conversations.append_result(new_turn, result, step=step)
    metadata = {'route':'agent','budget':{'reserved':200}}
    await conversations.finish_turn(new_turn, '已按查询结果说明。', 'completed', event_data=metadata)
    await conversations.finish_turn(new_turn, '已按查询结果说明。', 'completed', event_data=metadata)
    from app.db.conversations import ConversationRepository
    fresh = ConversationRepository(mysql_db.sessions)
    history = await fresh.history(new_turn.conversation_id, 'demo', 12)
    assert len(history[0].messages) == 6
    assert len(await fresh.audit(new_turn.conversation_id, 'demo')) == 6
    snapshot = await fresh.get_turn(new_turn, 'demo')
    assert snapshot.original_question == '先查订单，再查物流'
    assert snapshot.final_content == '已按查询结果说明。'
    assert snapshot.event_data == metadata
    assert snapshot.status == 'completed'
    assert await fresh.get_turn(new_turn, 'other') is None
    assert await fresh.unfinished_turns(new_turn.conversation_id, 'demo') == []
    with pytest.raises(ServiceError) as err:
        await fresh.finish_turn(new_turn, '已按查询结果说明。', 'completed', event_data={'route':'other'})
    assert err.value.code == 'EVENT_CONFLICT'


async def test_sequence_and_metadata_validation(repos, new_turn):
    repo, _, _ = repos
    await repo.start_turn(new_turn, 'demo', '问题')
    for step in (-1, 1):
        with pytest.raises(ServiceError):
            await repo.append_call(new_turn, call(step), step=step)
    await repo.append_call(new_turn, call(0))
    with pytest.raises(ServiceError):
        await repo.append_call(new_turn, call(1), step=1)
    with pytest.raises(ServiceError):
        await repo.append_result(new_turn, ToolMessage(content='x', tool_call_id='wrong'))
    with pytest.raises(ServiceError):
        await repo.finish_turn(new_turn, '不完整', 'completed')
    await repo.append_result(new_turn, ToolMessage(content='x', tool_call_id='call-0'))
    with pytest.raises(ServiceError):
        await repo.append_call(new_turn, call(0), step=1)
    with pytest.raises(ServiceError):
        await repo.finish_turn(new_turn, '回答', 'completed', event_data={'large':'x'*70000})
    await repo.finish_turn(new_turn, '', 'cancelled')
    await repo.finish_turn(new_turn, '', 'cancelled')
    assert await repo.history(new_turn.conversation_id, 'demo', 12) == []
    assert (await repo.get_turn(new_turn, 'demo')).status == 'cancelled'


async def test_actions_concurrent_confirmable_and_completed_recovery(repos, new_turn, mysql_db):
    assert importlib.util.find_spec('app.db.actions'), 'ActionRepository missing'
    from app.db.actions import ActionRepository
    from app.db.workflow_models import ConversationAction
    repo, _, tickets = repos
    actions = ActionRepository(mysql_db.sessions)
    await repo.start_turn(new_turn, 'demo', '包装破损')
    draft = TicketInput(issue_description='包装破损', ticket_type='complaint')
    offers = await asyncio.gather(*(actions.offer_once(new_turn, 'demo', draft) for _ in range(2)))
    assert offers[0] == offers[1]
    offer = offers[0]
    async with mysql_db.sessions() as session:
        assert await session.scalar(select(func.count()).select_from(Ticket).where(Ticket.conversation_id == new_turn.conversation_id)) == 0
        assert await session.scalar(select(func.count()).select_from(ConversationAction)) == 1
    for cid, uid in ((new_turn.conversation_id,'demo'),(str(uuid4()),'demo'),(new_turn.conversation_id,'other')):
        with pytest.raises(ServiceError):
            await actions.get_confirmable(cid, offer.action_id, uid)
    with pytest.raises(ServiceError):
        await actions.offer_once(new_turn, 'demo', TicketInput(issue_description='changed',ticket_type='other'))
    await repo.finish_turn(new_turn, '建议建工单', 'completed')
    assert await actions.get_confirmable(new_turn.conversation_id, offer.action_id, 'demo') == offer
    with pytest.raises(ServiceError):
        await actions.mark_completed(offer.action_id, offer.ticket_no)
    await tickets.create_once(offer.ticket_no, new_turn.conversation_id, 'demo', draft.issue_description, draft.ticket_type)
    fresh = ActionRepository(mysql_db.sessions)
    completed = await fresh.mark_completed(offer.action_id, offer.ticket_no)
    assert completed.status == 'completed'
    assert await fresh.mark_completed(offer.action_id, offer.ticket_no) == completed
    async with mysql_db.sessions.begin() as session:
        await session.execute(update(Conversation).values(status='closed'))
    assert await fresh.get_confirmable(new_turn.conversation_id, offer.action_id, 'demo') == completed


async def test_offered_closed_and_historical_human_pending(repos, new_turn, mysql_db):
    from app.db.actions import ActionRepository
    repo, _, tickets = repos
    await repo.start_turn(new_turn, 'demo', '问题')
    actions = ActionRepository(mysql_db.sessions)
    offer = await actions.offer_once(new_turn, 'demo', TicketInput(issue_description='问题',ticket_type='other'))
    await repo.finish_turn(new_turn, '回答', 'completed')
    async with mysql_db.sessions.begin() as session:
        await session.execute(update(Conversation).values(status='closed'))
    with pytest.raises(ServiceError):
        await actions.get_confirmable(new_turn.conversation_id, offer.action_id, 'demo')
    async with mysql_db.sessions.begin() as session:
        await session.execute(update(Conversation).values(status='human_pending'))
    for _ in range(2):
        await tickets.create_once('TK-history',new_turn.conversation_id,'demo','问题','other')
    assert (await repo.get(new_turn.conversation_id,'demo'))['status'] == 'human_pending'


class CommitThenRaise:
    def __init__(self, transaction):
        self.transaction = transaction

    async def __aenter__(self):
        return await self.transaction.__aenter__()

    async def __aexit__(self, typ, value, traceback):
        await self.transaction.__aexit__(typ, value, traceback)
        if value is None:
            from sqlalchemy.exc import DBAPIError
            raise DBAPIError(None, None, RuntimeError('simulated lost commit acknowledgement'))


class LoseFirstCommit:
    def __init__(self, sessions):
        self.sessions = sessions
        self.first = True

    def begin(self):
        transaction = self.sessions.begin()
        if self.first:
            self.first = False
            return CommitThenRaise(transaction)
        return transaction


async def test_uncertain_commit_recovers_in_new_transactions(repos, new_turn, mysql_db):
    from app.db.actions import ActionRepository
    from app.db.conversations import ConversationRepository
    from app.db.tickets import TicketRepository
    def restarted():
        return ConversationRepository(LoseFirstCommit(mysql_db.sessions))
    await restarted().create(new_turn.conversation_id, 'demo')
    await restarted().start_turn(new_turn, 'demo', '恢复问题')
    await restarted().append_call(new_turn, call(0))
    await restarted().append_result(new_turn, ToolMessage(content='结果', tool_call_id='call-0'))
    draft = TicketInput(issue_description='恢复问题', ticket_type='other')
    offer = await ActionRepository(LoseFirstCommit(mysql_db.sessions)).offer_once(new_turn, 'demo', draft)
    await restarted().finish_turn(new_turn, '恢复回答', 'completed', event_data={'route':'agent'})
    await TicketRepository(LoseFirstCommit(mysql_db.sessions)).create_once(offer.ticket_no, new_turn.conversation_id, 'demo', '恢复问题', 'other')
    result = await ActionRepository(LoseFirstCommit(mysql_db.sessions)).mark_completed(offer.action_id,offer.ticket_no)
    assert result.status == 'completed'
    assert len((await repos[0].history(new_turn.conversation_id,'demo',12))[0].messages) == 4
    assert (await repos[0].get(new_turn.conversation_id,'demo'))['status'] == 'open'
    async with mysql_db.sessions() as session:
        assert await session.scalar(select(func.count()).select_from(Ticket).where(Ticket.conversation_id == new_turn.conversation_id)) == 1


async def test_every_row_must_be_completed_for_history(repos, new_turn, mysql_db):
    repo, _, _ = repos
    await repo.start_turn(new_turn,'demo','问题')
    await repo.finish_turn(new_turn,'回答','completed')
    async with mysql_db.sessions.begin() as session:
        await session.execute(update(Message).where(Message.conversation_id == new_turn.conversation_id,Message.event_key == 'user').values(turn_status='pending'))
    assert await repo.history(new_turn.conversation_id,'demo',12) == []
    with pytest.raises(ServiceError):
        await repo.get_turn(new_turn,'demo')


async def test_case_different_event_content_and_turn_are_not_replayed(repos, new_turn):
    repo, _, _ = repos
    await repo.start_turn(new_turn,'demo','ABC')
    for ref, content in ((new_turn,'abc'), (TurnRef(new_turn.conversation_id,new_turn.turn_id.upper()),'ABC')):
        with pytest.raises(ServiceError):
            await repo.start_turn(ref,'demo',content)
    pending = await repo.unfinished_turns(new_turn.conversation_id,'demo')
    assert len(pending) == 1
    assert pending[0].original_question == 'ABC'


async def test_owner_checks_are_exact_despite_mysql_collation(repos, new_turn):
    repo, _, _ = repos
    await repo.start_turn(new_turn,'demo','问题')
    await repo.finish_turn(new_turn,'回答','completed')
    assert await repo.get(new_turn.conversation_id, 'DEMO') is None
    assert await repo.history(new_turn.conversation_id, 'DEMO', 12) == []
    assert await repo.audit(new_turn.conversation_id, 'DEMO') == []


async def test_seed_can_be_repeated_without_changing_legacy_rows(mysql_db):
    from app.db.seed import seed_database, DEMO_MESSAGE_ID
    await seed_database(mysql_db)
    await seed_database(mysql_db)
    async with mysql_db.sessions() as session:
        row = await session.get(Message, DEMO_MESSAGE_ID)
        assert row.event_key == 'legacy:9000000001'
        assert row.content == '收到的商品有破损，请帮我联系人工客服。'
        assert row.role == 'user'


async def test_unique_event_constraint_and_new_session_after_conflict(repos, new_turn, mysql_db):
    from sqlalchemy.exc import IntegrityError
    repo, _, _ = repos
    await asyncio.gather(*(repo.start_turn(new_turn,'demo','问题') for _ in range(2)))
    with pytest.raises(IntegrityError):
        async with mysql_db.sessions.begin() as session:
            session.add(Message(conversation_id=new_turn.conversation_id, turn_id=new_turn.turn_id,
                event_key='user', role='user',content='different',turn_status='pending'))
    await repo.start_turn(new_turn,'demo','问题')
    await repo.finish_turn(new_turn,'回答','completed')
    assert len(await repo.audit(new_turn.conversation_id,'demo')) == 2
    with pytest.raises(ServiceError) as err:
        await repo.finish_turn(new_turn,'回答','failed')
    assert err.value.code == 'EVENT_CONFLICT'


async def test_append_cannot_alias_existing_turn_by_case(repos, new_turn):
    repo, _, _ = repos
    await repo.start_turn(new_turn,'demo','问题')
    with pytest.raises(ServiceError):
        await repo.append_call(TurnRef(new_turn.conversation_id,new_turn.turn_id.upper()),call(0))
    assert len(await repo.audit(new_turn.conversation_id,'demo')) == 1


async def test_ticket_owner_and_number_are_exact(repos, new_turn):
    _, _, tickets = repos
    await tickets.create_once('TK-EXACT',new_turn.conversation_id,'demo','问题','other')
    for number, user in (('tk-exact','demo'),('TK-EXACT','DEMO')):
        with pytest.raises(ServiceError):
            await tickets.create_once(number,new_turn.conversation_id,user,'问题','other')


@pytest.mark.parametrize('raw_calls', [
    pytest.param(['malformed'], id='non-dict-element'),
    pytest.param('malformed', id='non-list-container'),
    pytest.param([{'name':'query_order','args':{},'id':'call-0','type':'wrong'}], id='wrong-type'),
    pytest.param([{'name':'query_order','args':{},'id':'call-0'}], id='missing-type'),
    pytest.param([{'args':{},'id':'call-0','type':'tool_call'}], id='missing-name'),
    pytest.param([{'name':'query_order','args':[],'id':'call-0','type':'tool_call'}], id='non-dict-args'),
    pytest.param([{'name':'query_order','args':{},'id':1,'type':'tool_call'}], id='non-string-id'),
])
@pytest.mark.parametrize('reader', ['history', 'snapshot', 'action'])
async def test_raw_corrupt_tool_calls_never_enter_restored_turns(
    repos, new_turn, mysql_db, raw_calls, reader
):
    """Raw JSON validation must precede LangChain normalization for every reader."""
    from app.db.actions import ActionRepository
    repo, _, _ = repos
    actions = ActionRepository(mysql_db.sessions)
    await repo.start_turn(new_turn, 'demo', '问题')
    await repo.append_call(new_turn, call(0))
    await repo.append_result(new_turn, ToolMessage(content='结果', tool_call_id='call-0'))
    offer = await actions.offer_once(new_turn, 'demo', TicketInput(issue_description='问题', ticket_type='other'))
    await repo.finish_turn(new_turn, '回答', 'completed')
    async with mysql_db.sessions.begin() as session:
        await session.execute(update(Message).where(
            Message.conversation_id == new_turn.conversation_id,
            Message.turn_id == new_turn.turn_id,
            Message.event_key == 'call:0',
        ).values(tool_calls=raw_calls))

    if reader == 'history':
        valid = TurnRef(new_turn.conversation_id, str(uuid4()))
        await repo.start_turn(valid, 'demo', '后续合法问题')
        await repo.finish_turn(valid, '后续合法回答', 'completed')
        history = await repo.history(new_turn.conversation_id, 'demo', 12)
        assert [turn.turn_id for turn in history] == [valid.turn_id]
    else:
        with pytest.raises(ServiceError) as err:
            if reader == 'snapshot':
                await repo.get_turn(new_turn, 'demo')
            else:
                await actions.get_confirmable(new_turn.conversation_id, offer.action_id, 'demo')
        assert err.value.code == 'TURN_INVALID'
        assert err.value.status_code == 409

    # Invalid input remains queryable as audit data; rejection never rewrites it.
    audit = await repo.audit(new_turn.conversation_id, 'demo')
    assert next(row for row in audit if row['role'] == 'assistant' and row['tool_call_id'])['tool_calls'] == raw_calls
