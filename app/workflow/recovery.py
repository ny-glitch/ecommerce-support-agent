"""Reconcile committed audit with public graph state; never replay external work."""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from app.db.contracts import StoredTurn, TurnRef
from app.errors import ServiceError
from app.workflow.state import dump_turns, fresh_state, load_tool_messages, load_turns


_METADATA_FIELDS = ('intent', 'route', 'score', 'band', 'assessment', 'sources',
                    'used_citations', 'suggestions', 'offers', 'budget')


def recovery_conflict():
    return ServiceError('WORKFLOW_RECOVERY_CONFLICT', '会话保存状态不一致，请稍后重试', 409)


def canonical_history(history):
    """Validate pairing first, then compare only fields actually persisted in MySQL.

    Tool name/status are optional LC reconstruction defaults, not business data.
    Call name/arguments/ID and exact result content/ID remain authoritative.
    """
    result = dump_turns(load_turns(history))
    for turn in result:
        for message in turn['messages']:
            if message['role'] == 'tool':
                message.pop('name', None)
                message.pop('status', None)
    return result


def _audit_history(turn):
    return dump_turns([StoredTurn(turn.ref.turn_id, turn.messages)])


def verify_completed_turn(snapshot, turn_snapshot) -> dict:
    try:
        state, turn = snapshot.values, turn_snapshot
        if (snapshot.next or getattr(snapshot, 'tasks', ()) or not turn or state['status'] != 'completed'
                or turn.status != 'completed' or not turn.event_data
                or state['conversation_id'] != turn.ref.conversation_id
                or state['turn_id'] != turn.ref.turn_id
                or state['original_question'] != turn.original_question
                or state['answer'] != turn.final_content):
            raise ValueError()
        metadata = turn.event_data
        if (metadata['session_id'] != turn.ref.conversation_id
                or metadata['turn_id'] != turn.ref.turn_id
                or metadata['status'] != 'completed'
                or any(state[key] != metadata[key] for key in _METADATA_FIELDS)):
            raise ValueError()
        trace = state['trace']
        tools = [{key: item.get(key) for key in ('name', 'tool_call_id', 'status', 'attempts')}
                 for item in trace if item.get('stage') == 'tool']
        if ([item.get('stage') for item in trace] != metadata['node_path']
                or tools != metadata['tools']
                or type(metadata['elapsed_ms']) is not int or metadata['elapsed_ms'] < 0):
            raise ValueError()
        history = canonical_history(state['history'])
        if not history or history[-1] != canonical_history(_audit_history(turn))[0]:
            raise ValueError()
        return {'session_id': turn.ref.conversation_id, 'turn_id': turn.ref.turn_id,
                'status': 'completed',
                'refused': bool(state.get('knowledge_target') == 'fallback' or (
                    state.get('budget_exhausted') and state.get('agent_mode') is not None)),
                'citations': deepcopy(metadata['used_citations'])}
    except (KeyError, ValueError, TypeError, AttributeError, StopIteration):
        raise recovery_conflict() from None


async def recover_conversation(graph, conversations, conversation_id, user_id) -> list[dict]:
    """Caller holds the process-local conversation guard throughout recovery."""
    if await conversations.get(conversation_id, user_id) is None:
        raise ServiceError('CONVERSATION_NOT_FOUND', '会话不存在', 404)
    config = {'configurable': {'thread_id': conversation_id}, 'recursion_limit': 64}
    snapshot = await graph.aget_state(config)
    rows = await conversations.audit(conversation_id, user_id)
    turns = {}
    for turn_id in dict.fromkeys(row['turn_id'] for row in rows):
        turn = await conversations.get_turn(TurnRef(conversation_id, turn_id), user_id)
        if turn is None:
            raise recovery_conflict()
        turns[turn_id] = turn
    complete = [t for t in turns.values() if t.status == 'completed']
    try:
        mysql_history = dump_turns([StoredTurn(t.ref.turn_id, t.messages) for t in complete])
        state = snapshot.values
        pending_cancelled = False
        if state:
            if state['conversation_id'] != conversation_id or state['user_id'] != user_id:
                raise recovery_conflict()
            history = deepcopy(state['history'])
            canonical = canonical_history(history)
            authoritative = {t['turn_id']: t for t in canonical_history(mysql_history)}
            for item in canonical:
                if authoritative.get(item['turn_id']) != item:
                    raise recovery_conflict()
            # A graph can retain a suffix of old history, never silently omit a
            # newly committed turn unrelated to its current in-flight turn.
            expected_ids = [t['turn_id'] for t in mysql_history]
            prior_ids = [t['turn_id'] for t in canonical]
            current = turns.get(state['turn_id'])
            if current is None:
                raise recovery_conflict()
            permitted = prior_ids + ([current.ref.turn_id] if current.status == 'completed'
                and current.ref.turn_id not in prior_ids else [])
            if permitted and expected_ids[-len(permitted):] != permitted:
                raise recovery_conflict()
            if not permitted and expected_ids:
                raise recovery_conflict()
            if current.status == 'completed':
                if state['status'] == 'completed' and not snapshot.tasks:
                    verify_completed_turn(snapshot, current)
                else:
                    if state['status'] == 'completed':
                        verify_completed_turn(SimpleNamespace(values=state, next=()), current)
                    # Check durable pre-persist business output before repairing
                    # checkpoint acknowledgement loss from committed audit.
                    if (state['original_question'] != current.original_question
                            or (state.get('answer') is not None and state['answer'] != current.final_content)):
                        raise recovery_conflict()
                    wire = _audit_history(current)[0]['messages']
                    partial = state.get('tool_messages', [])
                    load_tool_messages(partial)
                    expected = wire[1:-1]
                    def business(messages):
                        return [{k: v for k, v in m.items() if k not in ('name', 'status')}
                                for m in messages]
                    if business(partial) != business(expected[:len(partial)]):
                        raise recovery_conflict()
                    metadata = current.event_data or {}
                    if (any(key not in metadata for key in _METADATA_FIELDS)
                            or any(state[key] != metadata[key] for key in _METADATA_FIELDS
                                   if key not in {'offers', 'budget'})):
                        raise recovery_conflict()
                    tool_trace = iter(metadata['tools'])
                    repaired_trace = [{'stage': stage, **(next(tool_trace) if stage == 'tool' else {})}
                                      for stage in metadata['node_path']]
                    repaired = {**state, 'trace': repaired_trace, **{k: deepcopy(metadata[k]) for k in _METADATA_FIELDS},
                        'answer': current.final_content, 'status': 'completed',
                        'history': history if current.ref.turn_id in prior_ids else history + _audit_history(current)}
                    verify_completed_turn(SimpleNamespace(values=repaired, next=()), current)
                    await graph.aupdate_state(config, repaired, as_node='persist')
                    snapshot = await graph.aget_state(config)
                    verify_completed_turn(snapshot, current)
                    history = deepcopy(snapshot.values['history'])
            elif state['status'] == 'completed':
                raise recovery_conflict()
            else:
                reset = fresh_state(current.ref, user_id, '', None, history, 49152)
                reset['status'] = 'cancelled'
                # All cancellation audits settle before clearing checkpoint work.
                for turn in turns.values():
                    if turn.status == 'pending':
                        await conversations.finish_turn(turn.ref, '', 'cancelled',
                            event_data={'status': 'cancelled', 'error_code': 'TURN_INTERRUPTED'})
                pending_cancelled = True
                if state != reset or snapshot.next or snapshot.tasks:
                    await graph.aupdate_state(config, reset, as_node='persist')
                    cleared = await graph.aget_state(config)
                    if cleared.next or cleared.tasks:
                        raise recovery_conflict()
        else:
            history = mysql_history
        for turn in turns.values():
            if turn.status == 'pending' and not pending_cancelled:
                # Repository finish is idempotent with this stable content.
                await conversations.finish_turn(turn.ref, '', 'cancelled',
                    event_data={'status': 'cancelled', 'error_code': 'TURN_INTERRUPTED'})
        return history
    except (KeyError, ValueError, TypeError, AttributeError, StopIteration):
        raise recovery_conflict() from None
