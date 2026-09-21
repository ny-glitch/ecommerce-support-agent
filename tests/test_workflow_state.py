import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from app.db.contracts import StoredTurn, TurnRef
from app.tools.schemas import TicketInput
from app.workflow.contracts import ActionOffer, FinalControl, IntentResult
from app.workflow.routing import knowledge_band, route_intent
from app.workflow.state import dump_turns, fresh_state, load_turns, validate_turn_messages


@pytest.mark.parametrize('score,expected', [(0, 'low'), (.6999, 'low'), (.7, 'middle'), (.8, 'middle'), (.8001, 'high'), (1, 'high')])
def test_knowledge_boundaries(score, expected):
    assert knowledge_band(score) == expected


@pytest.mark.parametrize('score', [True, False, -.1, 1.1, float('inf'), float('nan'), '0.8', None])
def test_invalid_scores_fail_closed(score):
    with pytest.raises(ValueError):
        knowledge_band(score)


@pytest.mark.parametrize('lower,upper', [(.8, .7), (.7, .7), (-.1, .8), (.7, 1.1), (float('nan'), .8), (False, .8)])
def test_invalid_thresholds_fail_closed(lower, upper):
    with pytest.raises(ValueError):
        knowledge_band(.75, lower=lower, upper=upper)


@pytest.mark.parametrize('intent,route', [('logistics', 'business'), ('order', 'business'), ('after_sales', 'business'), ('product', 'knowledge'), ('return_refund', 'knowledge'), ('complaint', 'complaint'), ('chitchat', 'chitchat')])
@pytest.mark.parametrize('needs_business_data', [True, False])
def test_all_routes_are_fixed_and_business_flag_cannot_bypass_policy(intent, route, needs_business_data):
    assert route_intent(IntentResult(intent=intent, needs_business_data=needs_business_data)) == route


@pytest.mark.parametrize('payload', [dict(intent='other', needs_business_data=False), dict(intent='order', needs_business_data='true'), dict(intent='order', needs_business_data=1), dict(intent='order', needs_business_data=True, route='knowledge')])
def test_intent_rejects_coercions_unknown_labels_and_model_generated_routes(payload):
    with pytest.raises(ValidationError):
        IntentResult.model_validate(payload)


def test_control_and_offer_roundtrip_reuse_ticket_input():
    ticket = TicketInput(issue_description='包裹损坏', ticket_type='repair')
    control = FinalControl(kind='respond', actions=['handoff', 'create_ticket'], ticket=ticket)
    offer = ActionOffer(action_id='a1', conversation_id='c1', turn_id='t1', ticket_no='TK-1', draft=ticket, status='offered')
    assert isinstance(control.ticket, TicketInput)
    assert ActionOffer.model_validate_json(offer.model_dump_json()) == offer
    assert json.loads(control.model_dump_json())['actions'] == ['handoff', 'create_ticket']


@pytest.mark.parametrize('payload', [dict(kind='respond', actions=['create_ticket']), dict(kind='respond', actions=['handoff', 'handoff']), dict(kind='clarify', actions=['delete']), dict(kind='respond', actions=[], ticket={'issue_description': '问题', 'ticket_type': 'other'}), dict(kind='answer', actions=[])])
def test_final_control_rejects_invalid_actions_or_mismatched_ticket(payload):
    with pytest.raises(ValidationError):
        FinalControl.model_validate(payload)


def complete_turn():
    return StoredTurn('old', (
        HumanMessage('先查订单，再查物流'),
        AIMessage('', tool_calls=[{'name': 'query_order', 'args': {'order_id': 'O1'}, 'id': 'call1'}], additional_kwargs={'reasoning_content': 'private'}),
        ToolMessage('{"status":"shipped"}', tool_call_id='call1', name='query_order'),
        AIMessage('', tool_calls=[{'name': 'query_logistics', 'args': {'order_id': 'O1'}, 'id': 'call2'}]),
        ToolMessage('明天送达', tool_call_id='call2', name='query_logistics'),
        AIMessage('预计明天送达', response_metadata={'private': 'hidden'}),
    ))


def test_multiple_tool_steps_roundtrip_without_provider_private_metadata():
    wire = dump_turns([complete_turn()])
    encoded = json.dumps(wire, ensure_ascii=False, allow_nan=False)
    assert 'private' not in encoded and 'hidden' not in encoded
    restored = load_turns(json.loads(encoded))
    assert [m.type for m in restored[0].messages] == ['human', 'ai', 'tool', 'ai', 'tool', 'ai']
    assert restored[0].messages[3].tool_calls[0]['id'] == 'call2'
    assert restored[0].messages[4].tool_call_id == 'call2'
    assert dump_turns(restored) == wire


@pytest.mark.parametrize('messages', [
    (HumanMessage('q'),),
    (HumanMessage('q'), AIMessage('')),
    (HumanMessage('q'), ToolMessage('orphan', tool_call_id='x'), AIMessage('a')),
    (HumanMessage('q'), AIMessage('', tool_calls=[{'id':'x','name':'tool','args':{}}]), AIMessage('a')),
    (HumanMessage('q'), AIMessage('', tool_calls=[{'id':'x','name':'tool','args':{}}]), ToolMessage('result',tool_call_id='y'), AIMessage('a')),
    (HumanMessage('q'), AIMessage('a'), AIMessage('extra')),
])
def test_history_rejects_incomplete_or_unpaired_turns(messages):
    with pytest.raises(ValueError):
        validate_turn_messages(messages)
    with pytest.raises(ValueError):
        dump_turns([StoredTurn('bad', messages)])


def test_duplicate_call_ids_results_and_untrusted_wire_metadata_are_rejected():
    original = dump_turns([complete_turn()])
    for mutate in (
        lambda msgs: msgs[3]['tool_calls'][0].update(id='call1'),
        lambda msgs: msgs.insert(3, dict(msgs[2])),
        lambda msgs: msgs[0].update(response_metadata={'secret': True}),
        lambda msgs: msgs[1]['tool_calls'][0]['args'].update(bad=float('nan')),
    ):
        wire = json.loads(json.dumps(original))
        mutate(wire[0]['messages'])
        with pytest.raises(ValueError):
            load_turns(wire)
    with pytest.raises(ValueError):
        load_turns(original + original)


def test_fresh_state_overwrites_all_old_turn_fields_and_detaches_history():
    history = dump_turns([complete_turn()])
    first = fresh_state(TurnRef('c', 't1'), 'u', '旧问题', '旧分类', history, 49152)
    first.update(sources=[{'number': 1}], suggestions=['handoff'], offers=[{'action_id':'old'}], answer='旧答案', score=.99, budget_exhausted=True)
    second = fresh_state(TurnRef('c', 't2'), 'u', '它能退吗', None, history, 49152)
    checkpoint_update = first | second
    assert checkpoint_update == second
    assert second['category'] is None and second['score'] is None
    assert second['sources'] == second['suggestions'] == second['offers'] == []
    assert second['original_question'] == second['question'] == '它能退吗'
    assert second['schema_version'] == 1 and second['status'] == 'pending'
    assert second['budget_exhausted'] is False
    assert 'deadline' not in second and 'started_at' not in second
    assert json.loads(json.dumps(second, allow_nan=False)) == second
    history[0]['messages'][0]['content'] = 'changed'
    assert second['history'][0]['messages'][0]['content'] == '先查订单，再查物流'
