"""JSON checkpoint values and explicit, paired completed-turn history.

History wire format: [{"turn_id": str, "messages": [{"role": ..., ...}]}].
Only textual content, normalized tool calls and tool-result identity survive.
Provider metadata, reasoning fields, tasks and runtime clocks never persist.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
from typing import Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from pydantic import JsonValue

from app.db.contracts import StoredTurn, TurnRef
from app.services.turn_operations import TurnOperations
from app.workflow.budget import RequestBudget


class WorkflowState(TypedDict):
    schema_version: int
    conversation_id: str
    user_id: str
    turn_id: str
    original_question: str
    question: str
    category: str | None
    history: list[dict]
    intent: dict | None
    route: str | None
    query: str | None
    retrieval: dict[str, JsonValue] | None
    score: float | None
    band: str | None
    sources: list[dict]
    assessment: dict | None
    knowledge_status: str | None
    knowledge_target: str | None
    refusal_reason: str | None
    agent_mode: str | None
    tool_messages: list[dict]
    pending_call: dict | None
    decision_count: int
    tool_count: int
    control: dict | None
    suggestions: list[str]
    offers: list[dict]
    answer: str | None
    used_citations: list[int]
    budget: dict
    budget_exhausted: bool
    trace: list[dict]
    status: Literal['pending', 'completed', 'failed', 'cancelled']


@dataclass
class TurnRuntime:
    """Graph context only. The deadline may be extended after knowledge routing."""
    ref: TurnRef
    user_id: str
    started_at: float
    deadline: float
    budget: RequestBudget
    operations: TurnOperations


def fresh_state(
    ref: TurnRef, user_id: str, question: str, category: str | None,
    history: list[dict], budget_limit: int,
) -> WorkflowState:
    # Returning every key prevents shallow checkpoint updates inheriting any
    # previous turn's evidence, category, counters, offers or budget.
    return WorkflowState(
        schema_version=1, conversation_id=ref.conversation_id,
        user_id=user_id, turn_id=ref.turn_id, original_question=question,
        question=question, category=category, history=dump_turns(load_turns(history)),
        intent=None, route=None, query=None, retrieval=None, score=None, band=None,
        sources=[], assessment=None, knowledge_status=None, knowledge_target=None,
        refusal_reason=None, agent_mode=None, tool_messages=[], pending_call=None,
        decision_count=0, tool_count=0, control=None, suggestions=[], offers=[],
        answer=None, used_citations=[], budget=RequestBudget(budget_limit).snapshot(),
        budget_exhausted=False, trace=[], status='pending',
    )


def _json_copy(value):
    def validate(item):
        if item is None or type(item) in (str, int, bool):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                validate(child)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                validate(child)
            return
        raise ValueError('history must contain only finite JSON values')
    validate(value)
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _dump_message(message: BaseMessage) -> dict:
    if not isinstance(message.content, str):
        raise ValueError('history requires text content')
    if type(message) is HumanMessage:
        return {'role': 'user', 'content': message.content}
    if type(message) is AIMessage:
        if message.invalid_tool_calls or (message.additional_kwargs.get('tool_calls') and not message.tool_calls):
            raise ValueError('history contains invalid tool calls')
        return {'role': 'assistant', 'content': message.content,
                'tool_calls': _json_copy(message.tool_calls)}
    if type(message) is ToolMessage:
        return {'role': 'tool', 'content': message.content,
                'tool_call_id': message.tool_call_id, 'name': message.name,
                'status': message.status}
    raise ValueError('unsupported history message')


def _load_messages(values: list[dict]) -> tuple[BaseMessage, ...]:
    if not isinstance(values, list) or len(values) < 2:
        raise ValueError('history requires a complete turn')
    messages: list[BaseMessage] = []
    pending: dict[str, str] = {}
    seen: set[str] = set()
    for index, value in enumerate(values):
        if not isinstance(value, dict) or not isinstance(value.get('content'), str):
            raise ValueError('invalid history message')
        role = value.get('role')
        content = value['content']
        if role == 'user':
            if index != 0 or set(value) != {'role', 'content'}:
                raise ValueError('history must begin with exactly one user message')
            messages.append(HumanMessage(content))
        elif role == 'assistant':
            if index == 0 or pending or set(value) - {'role', 'content', 'tool_calls'}:
                raise ValueError('unpaired assistant message')
            calls = value.get('tool_calls', [])
            if not isinstance(calls, list):
                raise ValueError('invalid tool calls')
            if not calls:
                if index != len(values) - 1 or not content.strip():
                    raise ValueError('history requires a final nonempty assistant answer')
            else:
                if index == len(values) - 1:
                    raise ValueError('unfinished tool step')
                for call in calls:
                    if (
                        not isinstance(call, dict)
                        or set(call) - {'id', 'name', 'args', 'type'}
                        or not isinstance(call.get('id'), str) or not call['id']
                        or not isinstance(call.get('name'), str) or not call['name']
                        or not isinstance(call.get('args'), dict)
                        or call.get('type', 'tool_call') != 'tool_call'
                        or call['id'] in seen
                    ):
                        raise ValueError('invalid or duplicate tool call')
                    seen.add(call['id'])
                    pending[call['id']] = call['name']
            messages.append(AIMessage(content, tool_calls=calls))
        elif role == 'tool':
            call_id = value.get('tool_call_id')
            name = value.get('name')
            status = value.get('status', 'success')
            if (
                set(value) - {'role', 'content', 'tool_call_id', 'name', 'status'}
                or not isinstance(call_id, str) or call_id not in pending
                or (name is not None and name != pending[call_id])
                or status not in ('success', 'error')
            ):
                raise ValueError('orphaned or mismatched tool result')
            del pending[call_id]
            messages.append(ToolMessage(content, tool_call_id=call_id, name=name, status=status))
        else:
            raise ValueError('unsupported history role')
    if pending or type(messages[0]) is not HumanMessage or type(messages[-1]) is not AIMessage or messages[-1].tool_calls:
        raise ValueError('incomplete history turn')
    return tuple(messages)


def validate_turn_messages(messages: Sequence[BaseMessage]) -> None:
    """Reusable repository boundary: raise ValueError for incomplete/pairing errors.

    Accept multiple sequential steps, including multiple calls in one step,
    each with exactly one matching result before the next assistant message.
    Call IDs are unique within a turn; the final assistant answer is nonempty.
    This validates structure only: the repository must also require completed
    audit status for *every* row before adding a turn to model history.
    """
    _load_messages([_dump_message(message) for message in messages])


def dump_turns(turns: Sequence[StoredTurn]) -> list[dict]:
    values = [{'turn_id': turn.turn_id,
               'messages': [_dump_message(message) for message in turn.messages]}
              for turn in turns]
    load_turns(values)
    return values


def load_turns(values: list[dict]) -> list[StoredTurn]:
    values = _json_copy(values)
    if not isinstance(values, list):
        raise ValueError('history must be a list')
    turns = []
    seen = set()
    for value in values:
        if (
            not isinstance(value, dict) or set(value) != {'turn_id', 'messages'}
            or not isinstance(value['turn_id'], str) or not value['turn_id']
            or value['turn_id'] in seen
        ):
            raise ValueError('invalid or duplicate history turn')
        seen.add(value['turn_id'])
        turns.append(StoredTurn(value['turn_id'], _load_messages(value['messages'])))
    return turns


def load_tool_messages(values: list[dict]) -> list[BaseMessage]:
    """Validate settled tool pairs from the current, unfinished turn.

    This protocol is deliberately separate from completed-history validation:
    it contains neither a user message nor a fabricated final answer.
    """
    values = _json_copy(values)
    if not isinstance(values, list) or len(values) % 2:
        raise ValueError('current tool messages require settled pairs')
    messages: list[BaseMessage] = []
    seen: set[str] = set()
    for index in range(0, len(values), 2):
        call, result = values[index:index + 2]
        if (not isinstance(call, dict) or set(call) != {'role', 'content', 'tool_calls'}
                or call['role'] != 'assistant' or not isinstance(call['content'], str)
                or not isinstance(call['tool_calls'], list) or len(call['tool_calls']) != 1):
            raise ValueError('invalid current tool call')
        item = call['tool_calls'][0]
        if (not isinstance(item, dict) or set(item) - {'id', 'name', 'args', 'type'}
                or not isinstance(item.get('id'), str) or not item['id'] or item['id'] in seen
                or not isinstance(item.get('name'), str) or not item['name']
                or not isinstance(item.get('args'), dict) or item.get('type', 'tool_call') != 'tool_call'):
            raise ValueError('invalid or duplicate current tool call')
        seen.add(item['id'])
        if (not isinstance(result, dict)
                or set(result) - {'role', 'content', 'tool_call_id', 'name', 'status'}
                or result.get('role') != 'tool' or not isinstance(result.get('content'), str)
                or result.get('tool_call_id') != item['id']
                or result.get('name') not in (None, item['name'])
                or result.get('status', 'success') not in ('success', 'error')):
            raise ValueError('orphaned or mismatched current tool result')
        messages.extend([
            AIMessage(call['content'], tool_calls=[item]),
            ToolMessage(result['content'], tool_call_id=item['id'], name=result.get('name'),
                        status=result.get('status', 'success')),
        ])
    return messages


def dump_tool_messages(messages: Sequence[BaseMessage]) -> list[dict]:
    values = [_dump_message(message) for message in messages]
    load_tool_messages(values)
    return values
