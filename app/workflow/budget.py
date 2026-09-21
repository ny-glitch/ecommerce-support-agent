"""Conservative cumulative reservations, distinct from provider billing usage."""
from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
import json

from langchain_core.messages import BaseMessage

from app.context import estimate_tokens
from app.errors import ServiceError


def _positive_integer(value: int) -> bool:
    return type(value) is int and value > 0


class RequestBudget:
    def __init__(self, limit: int) -> None:
        if not _positive_integer(limit):
            raise ValueError('budget limit must be a positive integer')
        self._limit = limit
        self._reserved = 0
        self._reservations: list[dict] = []
        self._recorded: set[int] = set()

    def reserve(
        self, stage: str, messages: Sequence[BaseMessage], *,
        output_tokens: int, tool_schemas: Sequence[dict] = (),
    ) -> int:
        """Call before dispatch; failures/cancellation/missing usage never refund."""
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError('stage must be nonempty')
        if not _positive_integer(output_tokens):
            raise ValueError('output budget must be a positive integer')
        schema_bytes = len(json.dumps(
            tool_schemas, ensure_ascii=False, sort_keys=True,
            separators=(',', ':'), allow_nan=False,
        ).encode('utf-8'))
        estimated_input = estimate_tokens(messages) + schema_bytes
        reserved = estimated_input + output_tokens
        if self._reserved + reserved > self._limit:
            raise ServiceError('TURN_BUDGET_EXHAUSTED', '本轮模型预算已用尽', 429)
        self._reserved += reserved
        self._reservations.append({
            'stage': stage, 'estimated_input': estimated_input,
            'output_tokens': output_tokens, 'reserved': reserved, 'usage': None,
        })
        return reserved

    def record_usage(self, stage: str, usage: dict | None) -> None:
        """Record once for the latest unrecorded reservation for this stage.

        Stages are sequential within a turn. Unknown/untrusted usage remains
        unknown; only nonnegative token counters from known formats are stored.
        """
        index = next((i for i in reversed(range(len(self._reservations)))
                      if self._reservations[i]['stage'] == stage and i not in self._recorded), None)
        if index is None:
            raise ValueError('no unrecorded reservation for stage')
        normalized = None
        if isinstance(usage, dict):
            normalized = {}
            for key, alias in (('input_tokens', 'prompt_tokens'), ('output_tokens', 'completion_tokens'), ('total_tokens', 'total_tokens')):
                value = usage.get(key, usage.get(alias))
                if type(value) is int and value >= 0:
                    normalized[key] = value
            normalized = normalized or None
        self._reservations[index]['usage'] = normalized
        self._recorded.add(index)

    def snapshot(self) -> dict:
        return {
            'limit': self._limit, 'reserved': self._reserved,
            'remaining': self._limit - self._reserved,
            'reservations': deepcopy(self._reservations),
        }
