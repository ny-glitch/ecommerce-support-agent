"""HTTP-compatible workflow adapter and sole owner of terminal SSE events."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field
import logging
from uuid import uuid4

from anyio import CancelScope

from app.db.contracts import TurnRef
from app.errors import ServiceError
from app.knowledge.contracts import Citation
from app.services.events import ChatEvent
from app.services.turn_operations import TurnOperations
from app.workflow.budget import RequestBudget
from app.workflow.contracts import ActionOffer
from app.workflow.prompts import build_workflow_messages
from app.workflow.recovery import recover_conversation, verify_completed_turn
from app.workflow.state import TurnRuntime, dump_turns, fresh_state, load_turns

logger = logging.getLogger(__name__)
_FIELDS = {
    'token': {'content'}, 'message': {'content', 'kind'},
    'refusal': {'content', 'reason_code'}, 'sources': {'sources'},
    'workflow_status': {'session_id', 'turn_id', 'node', 'stage', 'message', 'intent', 'route', 'band'},
    'tool_status': {'name', 'tool_call_id', 'status', 'attempt', 'message'},
    'retrieval_status': {'tool_call_id', 'stage', 'message'},
}
_NODES = {'resolve', 'classify', 'retrieve', 'evidence_gate', 'agent', 'workflow_answer',
          'fallback', 'complaint', 'chitchat', 'budget_reply', 'persist'}
_STAGES = _NODES | {'normalizing', 'retrieving', 'reranking', 'checking_evidence'}


def checked_chat_event(value: dict) -> ChatEvent:
    """Reject unrecognized fields, including nested source/provider metadata."""
    try:
        if not isinstance(value, dict) or set(value) != {'name', 'data'}:
            raise ValueError()
        name, data = value['name'], value['data']
        if name not in _FIELDS or not isinstance(data, dict) or set(data) != _FIELDS[name]:
            raise ValueError()
        if name == 'sources':
            if not isinstance(data['sources'], list) or len(data['sources']) > 10:
                raise ValueError()
            data = {'sources': [Citation.model_validate(s, strict=True).model_dump(mode='json')
                                for s in data['sources']]}
        else:
            nullable = {'intent', 'route', 'band'} if name == 'workflow_status' else set()
            for key, item in data.items():
                if key == 'attempt':
                    if type(item) is not int or not 0 <= item <= 2:
                        raise ValueError()
                elif not isinstance(item, str) and not (key in nullable and item is None):
                    raise ValueError()
            if name == 'message' and data['kind'] not in {'workflow', 'complaint', 'chitchat', 'budget'}:
                raise ValueError()
            if name == 'workflow_status' and (data['node'] not in _NODES or data['stage'] not in _STAGES):
                raise ValueError()
            if name == 'tool_status' and (data['name'] not in {'query_order','query_product','query_logistics'}
                    or data['status'] not in {'running','retrying','succeeded','failed','not_found'}):
                raise ValueError()
        return ChatEvent(name, deepcopy(data))
    except (KeyError, TypeError, ValueError):
        raise ServiceError('WORKFLOW_EVENT_INVALID', '工作流事件无效，请重试', 502) from None


def _safe_error(error):
    if isinstance(error, ServiceError):
        return error
    if isinstance(error, TimeoutError):
        return ServiceError('UPSTREAM_TIMEOUT', '本轮处理超时，请重试', 504)
    return ServiceError('WORKFLOW_UNAVAILABLE', '本轮处理未能完成，请重试', 503)


async def _settled(task):
    """A cancelled waiter never abandons physical cleanup or terminal audit."""
    cancelled = False
    with CancelScope(shield=True):
        while True:
            try:
                result = await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                cancelled = True
                if task.done():
                    result = task.result()
                    break
    if cancelled:
        raise asyncio.CancelledError
    return result


@dataclass
class PreparedWorkflowTurn:
    ref: TurnRef
    runtime: TurnRuntime
    initial_state: dict
    _parts: list[str] = field(default_factory=list, repr=False)
    _trace: list[dict] = field(default_factory=list, repr=False)
    _started: bool = False
    _stream_started: bool = False
    _error_reported: bool = False
    _cleanup: asyncio.Task | None = field(default=None, repr=False)
    _done: dict | None = field(default=None, repr=False)
    _offers: list[dict] = field(default_factory=list, repr=False)
    _meta: dict = field(default_factory=dict, repr=False)

    @property
    def deadline(self):
        return self.runtime.deadline

    @deadline.setter
    def deadline(self, value):
        self.runtime.deadline = value


class WorkflowChatService:
    def __init__(self, settings, graph, conversations, guard):
        self.settings, self.graph = settings, graph
        self.conversations, self.guard = conversations, guard
        self._cleanup_tasks = set()

    @asynccontextmanager
    async def prepare(self, message, session_id, user_id='demo', *, category=None):
        if not isinstance(message, str) or not message.strip():
            raise ServiceError('INVALID_REQUEST', '请求参数无效', 422)
        # Mandatory intent input must fit before any durable conversation write.
        window = build_workflow_messages('intent', settings=self.settings, question=message)
        started = asyncio.get_running_loop().time()
        ref = TurnRef(session_id or str(uuid4()), str(uuid4()))
        runtime = TurnRuntime(ref, user_id, started, started + self.settings.request_timeout_seconds,
            RequestBudget(self.settings.turn_model_budget), TurnOperations())
        prepared = PreparedWorkflowTurn(ref, runtime, {})
        self.guard.acquire(ref.conversation_id)
        failure_status, error_code = 'cancelled', 'TURN_CANCELLED'
        try:
            if session_id is not None:
                history = await runtime.operations.run(
                    lambda: recover_conversation(self.graph, self.conversations, session_id, user_id), runtime.deadline,
                    mutation=True)
                window = build_workflow_messages('intent', settings=self.settings,
                    question=message, history=load_turns(history[-self.settings.max_history_turns:]))
            else:
                await runtime.operations.run(lambda: self.conversations.create(ref.conversation_id, user_id),
                    runtime.deadline, mutation=True)
            prepared.initial_state = fresh_state(ref, user_id, message, category,
                dump_turns(window.retained_turns), self.settings.turn_model_budget)
            prepared._meta = {'session_id': ref.conversation_id, 'turn_id': ref.turn_id,
                'estimated_input_tokens': window.estimated_input_tokens,
                'token_count_is_estimate': True, 'dropped_turns': window.dropped_turns}
            prepared._started = True  # Cancellation may lose a successful commit acknowledgement.
            await runtime.operations.run(lambda: self.conversations.start_turn(ref, user_id, message),
                runtime.deadline, mutation=True)
            yield prepared
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except Exception as error:
            failure_status, error_code = 'failed', _safe_error(error).code
            raise _safe_error(error) from None
        finally:
            try:
                await self._finish(prepared, failure_status, error_code)
            finally:
                # _finish cannot return until physical work AND audit settle.
                self.guard.release(ref.conversation_id)

    async def _finish(self, prepared, status, error_code=None, *, completed=False):
        if prepared._cleanup is None:
            async def cleanup():
                drain_error = None
                try:
                    await prepared.runtime.operations.drain()
                except Exception as error:
                    drain_error = error
                # Post-drain audit owns a separate task: the operations owner is sealed.
                if prepared._started:
                    audit = await self.conversations.get_turn(prepared.ref, prepared.runtime.user_id)
                    if audit is not None and audit.status == 'pending':
                        known = next((v for v in reversed(prepared._trace) if 'node' in v), {})
                        metadata = {'intent': known.get('intent'), 'route': known.get('route'),
                            'band': known.get('band'), 'session_id': prepared.ref.conversation_id, 'turn_id': prepared.ref.turn_id,
                            'status': status, 'error_code': error_code,
                            'node_path': [v['node'] for v in prepared._trace if 'node' in v],
                            'tools': [v for v in prepared._trace if 'tool_call_id' in v],
                            'budget': prepared.runtime.budget.snapshot(),
                            'elapsed_ms': max(0, int((asyncio.get_running_loop().time() - prepared.runtime.started_at) * 1000))}
                        await self.conversations.finish_turn(prepared.ref, ''.join(prepared._parts), status,
                            event_data=metadata)
                        logger.info('workflow turn terminated', extra={'workflow': metadata})
                    if completed and drain_error is None:
                        config = {'configurable': {'thread_id': prepared.ref.conversation_id}, 'recursion_limit': 64}
                        snapshot = await self.graph.aget_state(config)
                        prepared._done = verify_completed_turn(snapshot, audit)
                        offers = snapshot.values['offers']
                        for offer in offers:
                            if offer == {'type': 'handoff'}:
                                continue
                            if not isinstance(offer, dict) or offer.get('type') != 'create_ticket':
                                raise ValueError('invalid action offer')
                            dto = ActionOffer.model_validate({k:v for k,v in offer.items() if k != 'type'})
                            if dto.conversation_id != prepared.ref.conversation_id or dto.turn_id != prepared.ref.turn_id:
                                raise ValueError('invalid action identity')
                        prepared._offers = deepcopy(offers)
                if drain_error is not None:
                    raise drain_error
            prepared._cleanup = asyncio.create_task(cleanup())
            self._cleanup_tasks.add(prepared._cleanup)
            prepared._cleanup.add_done_callback(self._cleanup_tasks.discard)
        try:
            await _settled(prepared._cleanup)
        except Exception as error:
            if not prepared._error_reported:
                raise _safe_error(error) from None

    async def stream(self, prepared):
        if prepared._stream_started or prepared._cleanup is not None:
            raise ServiceError('TURN_INVALID', '本轮不能重复执行', 409)
        prepared._stream_started = True
        try:
            yield ChatEvent('meta', prepared._meta)
            config = {'configurable': {'thread_id': prepared.ref.conversation_id}, 'recursion_limit': 64}
            iterator = prepared.runtime.operations.track_iterator(self.graph.astream(
                prepared.initial_state, config=config, context=prepared.runtime,
                stream_mode='custom', subgraphs=True, version='v2', durability='sync'))
            while True:
                try:
                    part = await prepared.runtime.operations.run(lambda: anext(iterator), prepared.deadline)
                except StopAsyncIteration:
                    break
                if part['type'] != 'custom':
                    continue
                event = checked_chat_event(part['data'])
                if event.name == 'workflow_status':
                    if (event.data['session_id'] != prepared.ref.conversation_id
                            or event.data['turn_id'] != prepared.ref.turn_id):
                        raise ServiceError('WORKFLOW_EVENT_INVALID', '工作流事件无效，请重试', 502)
                    prepared._trace.append(event.data)
                elif event.name == 'tool_status':
                    prepared._trace.append(event.data)
                if event.name == 'token':
                    prepared._parts.append(event.data['content'])
                elif event.name in {'message', 'refusal'}:
                    prepared._parts[:] = [event.data['content']]
                yield event
            await self._finish(prepared, 'failed', 'WORKFLOW_INCOMPLETE', completed=True)
            if prepared._offers:
                yield ChatEvent('actions', {'session_id': prepared.ref.conversation_id,
                    'turn_id': prepared.ref.turn_id, 'actions': prepared._offers})
            yield ChatEvent('done', prepared._done)
        except (asyncio.CancelledError, GeneratorExit):
            await self._finish(prepared, 'cancelled', 'TURN_CANCELLED')
            raise
        except Exception as error:
            safe = _safe_error(error)
            try:
                await self._finish(prepared, 'failed', safe.code)
            except Exception as cleanup_error:
                safe = _safe_error(cleanup_error)
            prepared._error_reported = True
            yield ChatEvent('error', {'code': safe.code, 'message': safe.message})
        finally:
            await self._finish(prepared, 'cancelled', 'TURN_CANCELLED')

    async def aclose(self):
        first_error: Exception | None = None
        cancellation: asyncio.CancelledError | None = None
        for task in tuple(self._cleanup_tasks):
            try:
                await _settled(task)
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
            except Exception as error:
                if first_error is None:
                    first_error = error
        if cancellation is not None:
            if first_error is not None:
                raise cancellation from first_error
            raise cancellation
        if first_error is not None:
            raise first_error
