import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from app.config import Settings
from app.context import estimate_tokens
from app.errors import ServiceError
from app.workflow.budget import RequestBudget


def test_reserve_counts_input_schema_and_output_before_request_without_usage_refund():
    budget = RequestBudget(100)
    # Existing estimator: 3 + 12 overhead + 1 ASCII byte; empty schemas: 2.
    assert budget.reserve('intent', [HumanMessage('q')], output_tokens=10) == 28
    budget.record_usage('intent', None)
    assert budget.reserve('answer', [HumanMessage('q')], output_tokens=54) == 72
    issued = []
    with pytest.raises(ServiceError) as caught:
        budget.reserve('extra', [], output_tokens=1)
        issued.append('external call')
    assert caught.value.code == 'TURN_BUDGET_EXHAUSTED'
    assert issued == []
    assert budget.snapshot()['reserved'] == 100
    assert budget.snapshot()['remaining'] == 0


def test_reserve_counts_tool_metadata_once_and_canonical_schema_utf8_bytes():
    budget = RequestBudget(10000)
    messages = [AIMessage('', tool_calls=[{'name':'query_order','args':{'order_id':'O1'},'id':'c'}]), ToolMessage('已发货', tool_call_id='c', name='query_order')]
    schema = [{'name':'工具','parameters':{'type':'object'}}]
    schema_bytes = len(json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode())
    assert budget.reserve('agent', messages, output_tokens=32, tool_schemas=schema) == estimate_tokens(messages) + schema_bytes + 32


def test_usage_is_separate_sanitized_and_snapshot_is_detached():
    budget = RequestBudget(100)
    budget.reserve('agent', [HumanMessage('q')], output_tokens=10)
    budget.record_usage('agent', {'input_tokens': 3, 'output_tokens': 2, 'total_tokens': 5, 'private': 'secret'})
    budget.reserve('agent', [HumanMessage('q')], output_tokens=10)
    budget.record_usage('agent', None)
    snapshot = budget.snapshot()
    assert snapshot['reserved'] == 56 and snapshot['remaining'] == 44
    assert snapshot['reservations'][0]['usage'] == {'input_tokens': 3, 'output_tokens': 2, 'total_tokens': 5}
    assert snapshot['reservations'][1]['usage'] is None
    snapshot['reservations'].clear()
    assert len(budget.snapshot()['reservations']) == 2
    assert 'secret' not in json.dumps(budget.snapshot())


@pytest.mark.parametrize('limit', [0, -1, True, 1.2])
def test_invalid_limit_rejected(limit):
    with pytest.raises(ValueError):
        RequestBudget(limit)


@pytest.mark.parametrize('output', [0, -1, True, 1.2])
def test_invalid_reservation_does_not_consume_budget(output):
    budget = RequestBudget(100)
    with pytest.raises(ValueError):
        budget.reserve('answer', [], output_tokens=output)
    assert budget.snapshot()['reserved'] == 0


@pytest.mark.parametrize('field,invalid', [('agent_max_tool_calls',0), ('agent_max_tool_calls',5), ('agent_max_decisions',0), ('agent_max_decisions',6), ('turn_model_budget',0), ('turn_model_budget',49153)])
def test_settings_limits_cannot_exceed_approved_bounds(field, invalid):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_base_url='http://localhost:11434/v1', llm_model='test', llm_api_key='unused', **{field: invalid})
