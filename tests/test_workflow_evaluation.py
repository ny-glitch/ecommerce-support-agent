"""Evaluation contracts, using controlled observations rather than quality claims."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path

import pytest

from app.errors import ServiceError
from app.workflow.evaluation import evaluate_workflow, GraphRunner, AssessorRunner


CONFIG = {'scope': 'workflow', 'limit': None, 'expected_count': 3,
          'workflow_knowledge_lower_threshold': .7,
          'workflow_knowledge_upper_threshold': .8}


def case(identifier='one', intent='product', sufficient=True):
    return {'id': identifier, 'question': identifier, 'history': [], 'category': None,
            'expected_intent': intent, 'needs_business_data': False,
            'expected_sufficient': sufficient,
            'supporting_chunk_ids': [910001] if sufficient else []}


def observation(**changes):
    return {'intent': 'product', 'needs_business_data': False, 'route': 'knowledge',
            'status': 'completed', 'model_calls': 2, 'tools': [], 'score': .69,
            'assessment': {'sufficient': True, 'supporting_chunk_ids': [910001]},
            'sources': [{'number': 1, 'chunk_id': 910001}], 'used_citations': [1],
            'refused': False, 'usage': None, **changes}


async def test_failures_stay_in_denominators_and_labels_never_reach_runner(tmp_path):
    seen = []
    async def runner(question, history, category):
        seen.append((question, history, category))
        if question == 'two':
            raise ServiceError('UPSTREAM_ERROR', 'private response', 502)
        return observation(intent='logistics', route='knowledge') if question == 'three' else observation()
    cases = [case(), case('two'), case('three', 'logistics', False)]
    report = await evaluate_workflow(cases, runner, output_dir=tmp_path, configuration=CONFIG)
    summary = report['summary']
    assert seen == [('one', [], None), ('two', [], None), ('three', [], None)]
    assert summary['classification'] == {'correct': 2, 'denominator': 3, 'rate': 2/3}
    assert summary['confusion']['product'] == {'product': 1, 'failed': 1}
    assert summary['knowledge_misroute'] == {'count': 1, 'denominator': 1, 'rate': 1.0}
    assert summary['support']['denominator'] == 3
    assert summary['support']['correct'] == 1
    assert summary['refusal']['denominator'] == 1 and summary['refusal']['correct'] == 0
    assert summary['citations']['correct'] == 1 and summary['citations']['denominator'] == 2
    assert summary['errors'] == {'UPSTREAM_ERROR': 1}
    assert summary['score_bands'] == {'low': 2, 'middle': 0, 'high': 0, 'unknown': 1}
    assert summary['usage']['total_tokens'] is None
    assert report['status'] == 'incomplete'
    rows = [json.loads(line) for line in (tmp_path/'results.jsonl').read_text().splitlines()]
    assert rows[1]['observation']['error_code'] == 'UPSTREAM_ERROR'
    assert 'private response' not in (tmp_path/'results.jsonl').read_text()
    assert any('unsupported_answer' in row['failures'] for row in rows)


@pytest.mark.parametrize('score,want', [(0.7,'middle'),(.8,'middle'),(.80001,'high'),(.6999,'low')])
async def test_actual_score_determines_band_without_expected_label_override(tmp_path, score, want):
    async def runner(*args): return observation(score=score, band='fictional')
    report = await evaluate_workflow([case()], runner, output_dir=tmp_path,
        configuration={**CONFIG, 'expected_count': 1})
    assert report['summary']['score_bands'][want] == 1


async def test_resume_uses_atomic_rows_and_rejects_config_or_labels_change(tmp_path):
    calls = []
    async def runner(question, *args):
        calls.append(question)
        if question == 'two': raise asyncio.CancelledError()
        return observation()
    with pytest.raises(asyncio.CancelledError):
        await evaluate_workflow([case(), case('two')], runner, output_dir=tmp_path, configuration=CONFIG)
    assert json.loads((tmp_path/'manifest.json').read_text())['status'] == 'running'
    async def resume(question, *args):
        calls.append(question)
        return observation()
    report = await evaluate_workflow([case(), case('two')], resume, output_dir=tmp_path, configuration=CONFIG)
    assert calls == ['one', 'two', 'two'] and report['status'] == 'partial'
    with pytest.raises(ValueError, match='configuration'):
        await evaluate_workflow([case(), case('two')], resume, output_dir=tmp_path,
            configuration={**CONFIG, 'workflow_knowledge_upper_threshold': .85})
    with pytest.raises(ValueError, match='configuration'):
        await evaluate_workflow([case(sufficient=False), case('two')], resume,
            output_dir=tmp_path, configuration=CONFIG)


async def test_smoke_is_not_complete_and_unknown_config_cannot_leak(tmp_path):
    async def runner(*args): return observation()
    result = await evaluate_workflow([case()], runner, output_dir=tmp_path,
        configuration={**CONFIG, 'limit': 1})
    assert result['status'] == 'smoke'
    with pytest.raises(ValueError, match='configuration'):
        await evaluate_workflow([case()], runner, output_dir=tmp_path/'unsafe',
            configuration={**CONFIG, 'api_key': 'DO_NOT_WRITE'})
    assert not (tmp_path/'unsafe').exists()


async def test_assessor_scope_has_only_assessment_metrics_and_inputs(tmp_path):
    cases = [json.loads(line) for line in Path('evals/ch05/evidence_adversarial.jsonl').read_text().splitlines()]
    seen = []
    async def runner(question, intent, sources):
        seen.append((question, intent, sources))
        return {'status': 'completed', 'assessment': {'sufficient': False,
            'supporting_chunk_ids': [], 'reason_code': 'insufficient_evidence'}}
    result = await evaluate_workflow(cases, runner, output_dir=tmp_path,
        configuration={'scope': 'assessor', 'expected_count': 2, 'limit': None})
    assert len(seen) == 2 and seen[0] == tuple(cases[0]['input'][k] for k in ('question','intent','sources'))
    assert set(result['summary']) == {'support', 'supporting_ids', 'errors', 'failed_cases'}
    assert result['summary']['support']['correct'] == 2
    assert result['scope'] == 'assessor'
    with pytest.raises(ValueError):
        await evaluate_workflow(cases, runner, output_dir=tmp_path/'wrong', configuration=CONFIG)


async def test_graph_runner_executes_actual_graph_with_isolated_audit_and_pool():
    from tests.test_workflow_graph import setup_workflow, ScriptedKnowledgeStage, knowledge_result
    from app.workflow.contracts import IntentResult
    from app.workflow.nodes import WorkflowDependencies
    # Existing test boundaries supply deterministic model/retrieval only.
    existing = setup_workflow(intent=IntentResult(intent='product', needs_business_data=False),
        knowledge=ScriptedKnowledgeStage(knowledge_result('fallback', sufficient=False,
            reason_code='insufficient_evidence')))
    from tests.test_workflow_graph import settings
    from app.workflow.agent import AgentDependencies
    from app.tools.executor import ToolExecutor
    gateway = existing[3]
    factory = lambda runtime: gateway.bind(runtime, settings())
    agent = AgentDependencies(settings(), factory, existing[5], None, None, ToolExecutor())
    deps = WorkflowDependencies(settings(), factory, existing[4], agent, existing[5], existing[6], existing[7])
    result = await GraphRunner(deps)('未知问题', [], None)
    assert result['status'] == 'completed' and result['refused'] is True
    assert result['node_path'] == ['resolve','classify','retrieve','evidence_gate','fallback','persist']
    assert len(result['isolated_audit']['pool']) == 1
    assert result['model_calls'] == 1 and result['usage'] is None
    assert existing[5].finished == [] and existing[7].records == []


async def test_assessor_runner_reuses_gateway_budget_and_unknown_usage():
    from tests.test_workflow_graph import settings
    from tests.ch05_helpers import ScriptedWorkflowGateway
    from app.knowledge.contracts import EvidenceAssessment
    gateway = ScriptedWorkflowGateway(assessments=[EvidenceAssessment(sufficient=False,
        supporting_chunk_ids=[], reason='资料不足', reason_code='insufficient_evidence')])
    runner = AssessorRunner(settings(), lambda runtime: gateway.bind(runtime, settings()))
    row = json.loads(Path('evals/ch05/evidence_adversarial.jsonl').read_text().splitlines()[0])['input']
    result = await runner(row['question'], row['intent'], row['sources'])
    assert result['assessment']['sufficient'] is False
    assert result['model_calls'] == 1 and result['usage'] is None
    assert result['budget']['reservations'][0]['stage'] == 'evidence'


async def test_graph_runner_keeps_partial_failure_model_count_and_business_tool_path():
    from tests.test_workflow_graph import setup_workflow, settings
    from app.workflow.contracts import FinalControl, IntentResult
    from app.workflow.nodes import WorkflowDependencies
    from app.workflow.agent import AgentDependencies
    from app.tools.executor import ToolExecutor
    from langchain_core.messages import AIMessage
    selected = settings()
    existing = setup_workflow(intent=IntentResult(intent='order',needs_business_data=True),
        decisions=[AIMessage('',tool_calls=[{'name':'query_order','id':'call-one','args':{'order_id':'1001'}}]),
            FinalControl(kind='respond')], tokens=['订单查询结果。'])
    factory=lambda runtime: existing[3].bind(runtime,selected)
    agent=AgentDependencies(selected,factory,existing[5],None,None,ToolExecutor())
    deps=WorkflowDependencies(selected,factory,existing[4],agent,existing[5],existing[6],existing[7])
    result=await GraphRunner(deps)('查订单1001',[],None)
    assert result['status']=='completed' and result['tools']==['query_order']
    assert result['model_calls']==4
    assert [event['event'] for event in result['isolated_audit']['events']]==['start','call','result','finish']
    assert existing[5].calls=={}
    existing[3].intents.append(ServiceError('UPSTREAM_ERROR','secret',502))
    result=await GraphRunner(deps)('再查订单',[],None)
    assert result['status']=='failed' and result['model_calls']==1
    assert result['error_code']=='UPSTREAM_ERROR'
    assert 'secret' not in json.dumps(result)


def test_cli_help_has_no_model_or_settings_side_effects(monkeypatch,capsys):
    from scripts.evaluate_workflow import main
    import app.config
    monkeypatch.setattr(app.config,'load_settings',lambda: pytest.fail('help must not load settings'))
    with pytest.raises(SystemExit) as error:
        main(['--help'])
    assert error.value.code==0
    assert '--scope' in capsys.readouterr().out


def test_cli_validates_formal_case_counts_and_safe_effective_settings():
    from scripts.evaluate_workflow import validate_case_set, build_configuration
    from tests.test_workflow_graph import settings
    cases=[json.loads(line) for line in Path('evals/ch05/intents.jsonl').read_text().splitlines()]
    assert validate_case_set(cases,'workflow',None)==35
    with pytest.raises(ValueError): validate_case_set(cases[:3],'workflow',None)
    with pytest.raises(ValueError): validate_case_set(cases,'workflow',36)
    selected=settings(llm_chat_extra_body={'thinking':{'type':'disabled'}})
    args=dict(scope='workflow',limit=None,expected_count=35,cases_sha256='a'*64,
              corpus_fingerprint='b'*64,model_manifest_sha256='c'*64)
    config=build_configuration(selected,**args)
    assert config['workflow_knowledge_lower_threshold']==.7
    assert config['workflow_knowledge_upper_threshold']==.8
    assert 'localhost' not in json.dumps(config) and 'thinking' not in json.dumps(config)
    changed=build_configuration(selected.model_copy(update={'tool_max_attempts':1}),**args)
    assert changed['settings_sha256'] != config['settings_sha256']


async def test_graph_failure_preserves_retrieved_score_and_budget_exhaustion_is_incomplete(tmp_path):
    from tests.test_workflow_graph import setup_workflow,settings,ScriptedKnowledgeStage,knowledge_result
    from app.workflow.contracts import IntentResult
    from app.workflow.nodes import WorkflowDependencies
    from app.workflow.agent import AgentDependencies
    from app.tools.executor import ToolExecutor
    stage=ScriptedKnowledgeStage(knowledge_result('workflow_answer',score=.93),
        assess_error=ServiceError('EVIDENCE_ASSESSMENT_ERROR','secret',502))
    existing=setup_workflow(intent=IntentResult(intent='product',needs_business_data=False),knowledge=stage)
    selected=settings()
    factory=lambda runtime: existing[3].bind(runtime,selected)
    agent=AgentDependencies(selected,factory,existing[5],None,None,ToolExecutor())
    deps=WorkflowDependencies(selected,factory,stage,agent,existing[5],existing[6],existing[7])
    result=await GraphRunner(deps)('问题',[],None)
    assert result['status']=='failed' and result['score']==.93 and result['band']=='high'
    exhausted=replace(deps,settings=selected.model_copy(update={'turn_model_budget':1}))
    report=await evaluate_workflow([case()],GraphRunner(exhausted),output_dir=tmp_path,
        configuration={**CONFIG,'expected_count':1})
    assert report['status']=='incomplete'
    assert report['summary']['errors']=={'TURN_BUDGET_EXHAUSTED':1}


async def test_shared_atomic_writer_preserves_old_file_and_removes_temporary_on_failure(tmp_path,monkeypatch):
    import app.evaluation_io as shared
    from app.knowledge.evaluation_artifacts import _atomic_json,_atomic_text
    from app.knowledge.evaluation import strict_json_dumps
    assert _atomic_json is shared.atomic_json and _atomic_text is shared.atomic_text
    path=tmp_path/'artifact.json'
    shared.atomic_json(path,{'值':1})
    prior=path.read_text()
    with pytest.raises(ValueError): shared.atomic_json(path,{'bad':float('nan')})
    assert path.read_text()==prior
    monkeypatch.setattr(shared.os,'replace',lambda *args: (_ for _ in ()).throw(OSError('controlled')))
    with pytest.raises(OSError): shared.atomic_text(path,'replacement')
    assert path.read_text()==prior
    assert list(tmp_path.iterdir())==[path]
    assert strict_json_dumps({'值':1})=='{"值":1}'


@pytest.mark.parametrize('ending',['unchanged','changed','unavailable'])
async def test_cli_real_graph_assembly_uses_readonly_corpus_and_invalidates_unverifiable_run(tmp_path,monkeypatch,ending):
    from argparse import Namespace
    from types import SimpleNamespace
    import httpx
    from scripts.evaluate_workflow import run_command
    from tests.test_workflow_gateway import _gateway,_completion,_settings
    from app.knowledge.corpus import load_corpus
    import app.config,app.model,app.db.database,app.knowledge.bootstrap,app.knowledge.calibration
    cases=[json.loads(line) for line in Path('evals/ch05/intents.jsonl').read_text().splitlines()]
    cases.sort(key=lambda row: row['expected_intent']!='chitchat')
    case_path=tmp_path/'cases.jsonl'
    case_path.write_text('\n'.join(json.dumps(row) for row in cases))
    selected=_settings()
    from pydantic import SecretStr
    selected=selected.model_copy(update={'database_url':SecretStr('unused-local-fixture')})
    monkeypatch.setattr(app.config,'load_settings',lambda:selected)
    clients=[]
    requests=[]
    closed=[]
    class Owner:
        def __init__(self,settings): pass
        def create_workflow_gateway(self,before,usage):
            def handler(request):
                requests.append(json.loads(request.content))
                return httpx.Response(200,json=_completion(json.dumps({'intent':'chitchat','needs_business_data':False})))
            gateway,client=_gateway(handler,before_request=before,record_usage=usage,settings=selected)
            clients.append(client)
            return gateway
        async def aclose(self):
            for client in clients: await client.aclose()
            closed.append('model')
    class Database:
        sessions=object()
        def __init__(self,url): pass
        async def check(self): pass
        async def aclose(self): closed.append('database')
    chunks=load_corpus(Path('data/knowledge/ch04/chunks.json'))
    class Repository:
        calls=0
        async def list_all(self):
            self.calls+=1
            if self.calls==1 or ending=='unchanged': return chunks
            if ending=='changed': return chunks[:-1]
            raise RuntimeError('secret SQL connection text')
    async def components(*args):
        return SimpleNamespace(repository=Repository(),retriever=None,low_confidence=object(),
            knowledge_gateway_factory=lambda **kwargs:pytest.fail('chitchat cannot normalize'))
    monkeypatch.setattr(app.model,'OpenAIModelGateway',Owner)
    monkeypatch.setattr(app.db.database,'Database',Database)
    monkeypatch.setattr(app.knowledge.bootstrap,'build_knowledge_components',components)
    monkeypatch.setattr(app.knowledge.calibration,'model_manifest_fingerprint',lambda settings:'c'*64)
    args=Namespace(cases=case_path,output_dir=tmp_path/'out',scope='workflow',limit=1)
    code=await run_command(args)
    report=json.loads((args.output_dir/'manifest.json').read_text())
    assert code==(0 if ending=='unchanged' else 1)
    assert report['status']==('smoke' if ending=='unchanged' else 'invalid')
    assert closed==['database','model'] and len(requests)==1
    assert 'expected_intent' not in json.dumps(requests)
    assert 'secret SQL' not in (args.output_dir/'manifest.json').read_text()
    row=json.loads((args.output_dir/'results.jsonl').read_text())
    assert row['observation']['model_calls']==1
    assert row['observation']['usage']['total_tokens']==25
