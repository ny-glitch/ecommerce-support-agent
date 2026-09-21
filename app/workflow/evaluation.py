"""Label-blind graph execution and honest, resumable Chapter 5 evaluation.

In-memory audit adapters isolate evaluation side effects. They are deliberately
not a checkpoint/MySQL persistence acceptance test. Read-only business tools
retain the application's original simulated data behavior.
"""
from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import time
from uuid import uuid4

from app.evaluation_io import atomic_json, atomic_text, strict_json_dumps
from app.errors import ServiceError
from app.workflow.contracts import INTENT_LABELS, ActionOffer, IntentResult
from app.workflow.routing import knowledge_band, route_intent


_CONFIG_FIELDS = {
    'scope', 'limit', 'expected_count', 'execution_kind', 'cases_sha256',
    'corpus_fingerprint', 'settings_sha256', 'prompt_bundle_sha256',
    'model_manifest_sha256', 'code_sha256', 'dependency_versions_sha256',
    'workflow_knowledge_lower_threshold', 'workflow_knowledge_upper_threshold',
}
_DIGEST = re.compile(r'[0-9a-f]{64}')


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def _configuration(configuration):
    if not isinstance(configuration, dict) or set(configuration) - _CONFIG_FIELDS:
        raise ValueError('unsafe evaluation configuration')
    value = deepcopy(configuration)
    if value.get('scope', 'workflow') not in {'workflow', 'assessor'}:
        raise ValueError('invalid evaluation configuration scope')
    value.setdefault('scope', 'workflow')
    for key, item in value.items():
        if key.endswith('_sha256') or key == 'corpus_fingerprint':
            if not isinstance(item, str) or not _DIGEST.fullmatch(item):
                raise ValueError('invalid configuration fingerprint')
        elif key in {'limit', 'expected_count'}:
            if item is not None and (type(item) is not int or item <= 0):
                raise ValueError('invalid configuration count')
        elif key == 'execution_kind' and item not in {'actual_model', 'controlled_fixture'}:
            raise ValueError('invalid configuration execution kind')
    lower = value.get('workflow_knowledge_lower_threshold', .7)
    upper = value.get('workflow_knowledge_upper_threshold', .8)
    if type(lower) not in (int, float) or type(upper) not in (int, float) or not 0 <= lower < upper <= 1:
        raise ValueError('invalid configuration thresholds')
    if value['scope'] == 'workflow':
        value.update(workflow_knowledge_lower_threshold=lower, workflow_knowledge_upper_threshold=upper)
    return value


def _inputs(case, scope):
    if not isinstance(case, dict) or not isinstance(case.get('id'), str) or not case['id']:
        raise ValueError('case requires an id')
    if scope == 'assessor':
        if case.get('evaluation_scope') != 'assessor_fixture' or set(case.get('input', {})) != {'question','intent','sources'}:
            raise ValueError('assessor scope requires explicit fixture inputs')
        value = case['input']
        expected = case.get('expected', {})
        if type(expected.get('sufficient')) is not bool or not isinstance(expected.get('supporting_chunk_ids'), list):
            raise ValueError('invalid assessor labels')
        IntentResult.model_validate(value['intent'])
        from app.knowledge.contracts import Citation
        for source in value['sources']:
            Citation.model_validate(source)
        result = {key: value[key] for key in ('question', 'intent', 'sources')}
    else:
        if 'input' in case or 'evaluation_scope' in case:
            raise ValueError('assessor fixtures cannot enter workflow scope')
        if case.get('expected_intent') not in INTENT_LABELS or type(case.get('needs_business_data')) is not bool:
            raise ValueError('invalid workflow labels')
        if 'expected_sufficient' in case and (type(case['expected_sufficient']) is not bool or
                not isinstance(case.get('supporting_chunk_ids'), list)):
            raise ValueError('invalid evidence labels')
        from app.workflow.state import load_turns
        load_turns(case.get('history', []))
        result = {'question': case.get('question'), 'history': case.get('history', []), 'category': case.get('category')}
        if result['category'] is not None and not isinstance(result['category'], str):
            raise ValueError('invalid category')
    if not isinstance(result['question'], str) or not result['question'].strip():
        raise ValueError('case requires a question')
    return deepcopy(result)


def _ratio(correct, denominator):
    return {'correct': correct, 'denominator': denominator,
            'rate': correct / denominator if denominator else None}


def _grade(case, observation, scope):
    ok = observation.get('status') == 'completed' and not observation.get('error_code')
    expected = case['expected'] if scope == 'assessor' else case
    sufficient = (observation.get('assessment') or {}).get('sufficient')
    if sufficient is None and observation.get('refused') and observation.get('route') == 'knowledge':
        sufficient = False  # Actual no-hit fallback has no assessment model call.
    wanted = expected.get('sufficient') if scope == 'assessor' else expected.get('expected_sufficient')
    ids = set((observation.get('assessment') or {}).get('supporting_chunk_ids', []))
    gold_ids = set(expected.get('supporting_chunk_ids', []))
    grades, failures = {}, []
    if wanted is not None:
        grades['support'] = ok and sufficient is wanted
        grades['supporting_ids'] = ok and sufficient is wanted and ids == gold_ids
        if not grades['support']: failures.append('sufficiency_incorrect')
        if not grades['supporting_ids']: failures.append('supporting_ids_incorrect')
    if scope == 'workflow':
        intent = case['expected_intent']
        route = route_intent(IntentResult(intent=intent, needs_business_data=case['needs_business_data']))
        grades['classification'] = ok and observation.get('intent') == intent
        grades['needs_business_data'] = ok and observation.get('needs_business_data') is case['needs_business_data']
        grades['route'] = ok and observation.get('route') == route
        if not grades['classification'] or not grades['route']: failures.append('intent_misroute')
        if wanted is False:
            grades['refusal'] = ok and observation.get('refused') is True
            if ok and not observation.get('refused'): failures.append('unsupported_answer')
        if wanted is True:
            sources = {source['number']: source['chunk_id'] for source in observation.get('sources', [])}
            used = observation.get('used_citations', [])
            cited = {sources.get(number) for number in used}
            grades['citations'] = ok and sufficient is True and bool(used) and cited == gold_ids
            if not grades['citations']: failures.append('citation_incorrect')
        if 'create_ticket' in observation.get('tools', []) or observation.get('tickets_created', 0):
            failures.append('unconfirmed_ticket')
    if not ok: failures.append('technical_failure')
    return grades, failures


def _summary(rows, configuration):
    names = ('support','supporting_ids') if configuration['scope'] == 'assessor' else (
        'classification','needs_business_data','route','support','supporting_ids','refusal','citations')
    result = {name: _ratio(sum(row['grades'].get(name) is True for row in rows),
        sum(name in row['grades'] for row in rows)) for name in names}
    result['errors'] = dict(Counter(row['observation']['error_code'] for row in rows if row['observation'].get('error_code')))
    result['failed_cases'] = [{'id': row['id'], 'failures': row['failures']} for row in rows if row['failures']]
    if configuration['scope'] == 'assessor': return result
    confusion = defaultdict(Counter)
    bands = dict(low=0, middle=0, high=0, unknown=0)
    business, misrouted = 0, 0
    calls = []
    usages = []
    for row in rows:
        obs, case = row['observation'], row['case']
        actual = obs.get('intent') if obs.get('status') == 'completed' and not obs.get('error_code') else 'failed'
        confusion[case['expected_intent']][actual or 'unknown'] += 1
        if case['expected_intent'] in {'logistics','order','after_sales'}:
            business += 1
            misrouted += obs.get('route') == 'knowledge'
        score = obs.get('score')
        band = 'unknown' if score is None else knowledge_band(score,
            lower=configuration['workflow_knowledge_lower_threshold'],
            upper=configuration['workflow_knowledge_upper_threshold'])
        bands[band] += 1
        calls.append(obs.get('model_calls'))
        usages.append(obs.get('usage'))
    known_tokens = [usage['total_tokens'] for usage in usages if isinstance(usage, dict) and type(usage.get('total_tokens')) is int]
    result.update(confusion={key: dict(value) for key,value in confusion.items()},
        knowledge_misroute={'count': misrouted, 'denominator': business, 'rate': misrouted/business if business else None},
        score_bands=bands,
        model_calls={'known_total': sum(value for value in calls if type(value) is int),
                     'unknown_cases': sum(value is None for value in calls)},
        usage={'total_tokens': sum(known_tokens) if len(known_tokens) == len(rows) else None,
               'known_total_tokens': sum(known_tokens) if known_tokens else None,
               'unknown_cases': len(rows)-len(known_tokens)})
    return result


async def evaluate_workflow(cases, runner, *, output_dir, configuration) -> dict:
    configuration = _configuration(configuration)
    cases = deepcopy(list(cases))
    if not cases or len({case.get('id') for case in cases}) != len(cases):
        raise ValueError('evaluation requires unique cases')
    inputs = [_inputs(case, configuration['scope']) for case in cases]
    identity = digest({'configuration': configuration, 'cases': cases, 'inputs': inputs})
    output_dir = Path(output_dir)
    manifest_path = output_dir/'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get('status') == 'invalid' or previous.get('configuration_sha256') != identity:
            raise ValueError('run configuration mismatch or invalid run; use a fresh output directory')
    manifest = {'schema_version': 1, 'scope': configuration['scope'], 'status': 'running',
        'configuration': configuration, 'configuration_sha256': identity,
        'planned_cases': len(cases), 'completed_cases': 0}
    atomic_json(manifest_path, manifest)
    rows = []
    for case, value in zip(cases, inputs):
        path = output_dir/'.results'/f'{digest(case["id"])}.json'
        if path.exists():
            row = json.loads(path.read_text())
            if row.get('id') != case['id'] or row.get('case') != case or row.get('input') != value or row.get('run_sha256') != identity:
                raise ValueError('cached row configuration mismatch')
        else:
            try:
                observation = await runner(*deepcopy(tuple(value.values())))
                if not isinstance(observation, dict) or observation.get('status') not in {'completed','failed','cancelled'}:
                    raise ValueError('invalid observation')
                strict_json_dumps(observation)
                observation = deepcopy(observation)
                if observation['status'] != 'completed' and not observation.get('error_code'):
                    observation['error_code'] = 'WORKFLOW_INCOMPLETE'
            except Exception as error:
                observation = {'status': 'failed', 'error_code': _error_code(error)}
            grades, failures = _grade(case, observation, configuration['scope'])
            row = {'id': case['id'], 'run_sha256': identity, 'case': case,
                'input': value, 'observation': observation, 'grades': grades, 'failures': failures}
            atomic_json(path, row)
        rows.append(row)
        manifest['completed_cases'] = len(rows)
        atomic_json(manifest_path, manifest)
    summary = _summary(rows, configuration)
    status = ('incomplete' if summary['errors'] else 'smoke' if configuration.get('limit') is not None
        else 'complete' if len(cases) == configuration.get('expected_count') else 'partial')
    manifest.update(status=status, summary=summary)
    atomic_text(output_dir/'results.jsonl', ''.join(strict_json_dumps(row)+'\n' for row in rows))
    note = ('Assessor fixture only: no retrieval, routing, score bands or end-to-end quality claims.'
        if configuration['scope'] == 'assessor' else
        'Actual observed scores determine bands. Failed cases remain in labeled denominators. '
        'Citation correctness checks labeled IDs, not semantic faithfulness. '
        'Model calls count successful pre-dispatch reservations; these are attempts, not completed responses. '
        'Reserved units are not billed tokens. Unknown usage stays null. '
        'Isolated audit/pool adapters are not persistent database acceptance.')
    atomic_text(output_dir/'report.md', f'# Chapter 5 {configuration["scope"]} evaluation\n\nStatus: {status}\n\n'
        f'{note}\n\nComplete means all planned cases executed, not that quality passed.\n\n'
        f'```json\n{strict_json_dumps(summary, indent=2)}\n```\n')
    atomic_json(manifest_path, manifest)
    return manifest


def _error_code(error):
    if isinstance(error, ServiceError) and re.fullmatch(r'[A-Z][A-Z0-9_]{0,63}', error.code):
        return error.code
    return 'EVALUATION_TIMEOUT' if isinstance(error, TimeoutError) else 'EVALUATION_ERROR'


class _IsolatedAudit:
    """Evaluation-owned write sink; never forwards mutations to SQL repositories."""
    def __init__(self):
        self.events, self.pool, self.offers = [], [], []

    async def start_turn(self, ref, user_id, content):
        self.events.append({'event': 'start', 'turn_id': ref.turn_id})

    async def append_call(self, ref, message, *, step=0):
        self.events.append({'event': 'call', 'step': step, 'calls': message.tool_calls})

    async def append_result(self, ref, message, *, step=0):
        self.events.append({'event': 'result', 'step': step, 'name': message.name,
                            'tool_call_id': message.tool_call_id, 'content': message.content})

    async def finish_turn(self, ref, content, status, *, event_data=None):
        self.events.append({'event': 'finish', 'status': status})

    async def record_once(self, ref, question, reason_code, reason, entry_point='chat'):
        self.pool.append({'turn_id': ref.turn_id, 'question': question,
            'reason_code': reason_code, 'entry_point': entry_point})
        return len(self.pool)

    async def offer_once(self, ref, user_id, draft):
        offer = ActionOffer(action_id=str(uuid4()), conversation_id=ref.conversation_id,
            turn_id=ref.turn_id, ticket_no=f'EVAL-{ref.turn_id}', draft=draft, status='offered')
        self.offers.append(offer.model_dump(mode='json'))
        return offer

    async def create_once(self, *args, **kwargs):
        raise RuntimeError('evaluation cannot create tickets')


def _runtime(settings):
    from app.db.contracts import TurnRef
    from app.services.turn_operations import TurnOperations
    from app.workflow.budget import RequestBudget
    from app.workflow.state import TurnRuntime
    started = time.monotonic()
    return TurnRuntime(TurnRef(str(uuid4()),str(uuid4())), 'evaluation', started,
        started+settings.request_timeout_seconds, RequestBudget(settings.turn_model_budget), TurnOperations())


def _budget_observation(runtime):
    budget = runtime.budget.snapshot()
    reservations = budget['reservations']
    usage = {}
    for key in ('input_tokens','output_tokens','total_tokens'):
        values = [(item['usage'] or {}).get(key) for item in reservations]
        if all(type(value) is int for value in values): usage[key] = sum(values)
    return {'budget': budget, 'model_calls': len(reservations), 'usage': usage or None}


class GraphRunner:
    def __init__(self, dependencies):
        self.dependencies = dependencies

    async def __call__(self, question, history, category):
        from app.workflow.graph import build_workflow
        from app.workflow.state import fresh_state
        deps = self.dependencies
        runtime = _runtime(deps.settings)
        audit = _IsolatedAudit()
        isolated = replace(deps, conversations=audit, actions=audit, low_confidence=audit,
            agent_dependencies=replace(deps.agent_dependencies, conversations=audit, faq=audit, tickets=audit))
        graph = build_workflow(isolated, None)
        state = fresh_state(runtime.ref, runtime.user_id, question, category, history, deps.settings.turn_model_budget)
        events, error_code = [], None
        await audit.start_turn(runtime.ref, runtime.user_id, question)
        try:
            iterator = runtime.operations.track_iterator(graph.astream(state,
                config={'recursion_limit':64}, context=runtime,
                stream_mode=['custom','values'], subgraphs=True, version='v2'))
            while True:
                try: part = await runtime.operations.run(lambda: anext(iterator), runtime.deadline)
                except StopAsyncIteration: break
                if part['type'] == 'custom': events.append(part['data'])
                elif part['type'] == 'values' and not part['ns']: state = part['data']
        except Exception as error:
            error_code = _error_code(error)
        finally:
            await runtime.operations.drain()
        intent = state.get('intent') or {}
        trace = state.get('trace', [])
        if not error_code and state.get('budget_exhausted'):
            error_code = 'TURN_BUDGET_EXHAUSTED'
        score = state['score']
        if score is None and state.get('retrieval'):
            score = max((item['score'] for item in state['retrieval']['ranked']), default=None)
        band = None if score is None else knowledge_band(score,
            lower=deps.settings.workflow_knowledge_lower_threshold,
            upper=deps.settings.workflow_knowledge_upper_threshold)
        return {'status': 'failed' if error_code else state['status'], 'graph_status': state['status'],
            'error_code': error_code,
            'intent': intent.get('intent'), 'needs_business_data': intent.get('needs_business_data'),
            'route': state['route'], 'score': score, 'band': band,
            'assessment': state['assessment'], 'sources': state['sources'],
            'used_citations': state['used_citations'], 'refused': state['knowledge_target'] == 'fallback',
            'answer': state['answer'], 'events': events, 'trace': trace,
            'node_path': [item['stage'] for item in trace],
            'tools': [item['name'] for item in trace if item['stage'] == 'tool'],
            'isolated_audit': {'events': audit.events, 'pool': audit.pool, 'offers': audit.offers},
            **_budget_observation(runtime)}


class AssessorRunner:
    """Same gateway/prompt/protocol/budget, explicit fixture sources, no retrieval."""
    def __init__(self, settings, gateway_factory):
        self.settings, self.gateway_factory = settings, gateway_factory

    async def __call__(self, question, intent, sources):
        from app.knowledge.contracts import Citation
        runtime = _runtime(self.settings)
        runtime.deadline = runtime.started_at+self.settings.knowledge_request_timeout_seconds
        result = {'status': 'failed'}
        try:
            assessment = await runtime.operations.run(lambda: self.gateway_factory(runtime).assess(
                question, [Citation.model_validate(value) for value in sources],
                normalized_question=question, intent=IntentResult.model_validate(intent)), runtime.deadline)
            result = {'status': 'completed', 'assessment': assessment.model_dump(mode='json')}
        except Exception as error:
            result['error_code'] = _error_code(error)
        finally:
            await runtime.operations.drain()
        return {**result, **_budget_observation(runtime)}
