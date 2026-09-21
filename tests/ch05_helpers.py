"""Strict scripted model and audit boundaries; never a database acceptance test."""
from collections import deque
from copy import deepcopy

from langchain_core.utils.function_calling import convert_to_openai_tool

from app.workflow.contracts import ActionOffer
from app.workflow.prompts import build_workflow_messages
from app.tools.executor import ToolExecutor


class ScriptedWorkflowGateway:
    def __init__(self, *, intents=(), decisions=(), tokens=(), assessments=()):
        self.intents = deque(intents)
        self.decisions = deque(decisions)
        self.tokens = deque(tokens)
        self.assessments = deque(assessments)
        self.calls = []
        self.closed = False
        self._tool_calls = 0

    @property
    def model_calls(self):
        return len(self.calls)

    @property
    def tool_calls(self):
        return self._tool_calls

    def bind(self, runtime, settings):
        self.runtime, self.settings = runtime, settings
        return self

    def _request(self, stage, messages, schemas=()):
        # Same hook boundary as production: reserve before a physical request.
        self.runtime.budget.reserve(stage, messages,
            output_tokens=self.settings.max_output_tokens, tool_schemas=schemas)
        self.calls.append({'stage': stage, 'messages': deepcopy(messages),
                           'schemas': deepcopy(list(schemas))})

    @staticmethod
    def _pop(queue):
        assert queue, 'missing scripted gateway result'
        value = queue.popleft()
        if isinstance(value, BaseException):
            raise value
        return value

    async def classify(self, messages):
        self._request('intent', messages)
        return self._pop(self.intents)

    async def decide(self, messages, tools):
        self._request('agent', messages, [convert_to_openai_tool(t) for t in tools])
        value = self._pop(self.decisions)
        self.runtime.budget.record_usage('agent', None)
        return value

    async def assess(self, question, sources, *, normalized_question, intent):
        messages = build_workflow_messages('evidence', settings=self.settings,
            question=question, sources=sources, normalized_question=normalized_question,
            intent=intent).messages
        self._request('evidence', messages)
        return self._pop(self.assessments)

    async def stream_final(self, messages):
        self._request('answer', messages)
        assert self.tokens, 'missing scripted gateway result'
        try:
            while self.tokens:
                value = self._pop(self.tokens)
                if callable(value):
                    value = await value()
                yield value
            self.runtime.budget.record_usage('answer', None)
        finally:
            self.closed = True


class RecordingConversations:
    def __init__(self, trace=None):
        self.trace = trace if trace is not None else []
        self.calls = {}
        self.results = {}
        self.finished = []

    async def start_turn(self, ref, user_id, content):
        self.trace.append(('start', ref.turn_id))

    async def append_call(self, ref, message, *, step=0):
        key = (ref, step)
        if key in self.calls:
            assert self.calls[key] == message
        self.calls[key] = message
        self.trace.append(('call', step))

    async def append_result(self, ref, message, *, step=0):
        key = (ref, step)
        assert self.calls[key].tool_calls[0]['id'] == message.tool_call_id
        if key in self.results:
            assert self.results[key] == message
        self.results[key] = message
        self.trace.append(('result', step))

    async def finish_turn(self, ref, content, status, *, event_data=None):
        self.finished.append((ref, content, status, event_data))
        self.trace.append(('finish', status))


class RecordingActions:
    def __init__(self, trace=None):
        self.trace = trace if trace is not None else []
        self.offered = []

    async def offer_once(self, ref, user_id, draft):
        self.offered.append((ref, user_id, draft))
        self.trace.append(('offer', draft.issue_description))
        return ActionOffer(
            action_id=f'action-{ref.turn_id}',
            conversation_id=ref.conversation_id,
            turn_id=ref.turn_id,
            ticket_no=f'TK-{ref.turn_id}',
            draft=draft,
            status='offered',
        )


class RecordingLowConfidence:
    def __init__(self, trace=None):
        self.trace = trace if trace is not None else []
        self.records = []

    async def record_once(self, ref, question, reason_code, reason, entry_point='chat'):
        self.records.append((ref, question, reason_code, reason, entry_point))
        self.trace.append(('low_confidence', reason_code))
        return len(self.records)


class RecordingToolExecutor(ToolExecutor):
    """Count real executor entries while retaining its complete physical lifetime."""
    def __init__(self, recorder):
        super().__init__()
        self.recorder = recorder

    async def run(self, *args, **kwargs):
        self.recorder._tool_calls += 1
        async for event in super().run(*args, **kwargs):
            yield event
