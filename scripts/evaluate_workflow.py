#!/usr/bin/env python3
"""Run Chapter 5 evaluation after the operator has authorized external transfer.

--help is offline. Workflow scope reads the configured knowledge corpus but all
turn audits, action offers and low-confidence entries are isolated in artifacts.
Assessor scope calls the same gateway with explicit fixture sources only.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from importlib import metadata, resources
import json
from pathlib import Path
import sys

# Direct execution must resolve this checkout, including a shared editable venv.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evaluation_io import atomic_json, atomic_text, strict_json_dumps
from app.workflow.evaluation import AssessorRunner, GraphRunner, digest, evaluate_workflow, _inputs



def _positive(value):
    parsed=int(value)
    if parsed<=0: raise argparse.ArgumentTypeError('limit must be positive')
    return parsed


def build_parser():
    parser=argparse.ArgumentParser(description='Evaluate actual Workflow or independent assessor fixtures. External model transfer requires prior authorization.')
    parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--limit',type=_positive)
    parser.add_argument('--scope',choices=('workflow','assessor'),default='workflow')
    return parser


def validate_case_set(cases,scope,limit):
    from app.workflow.contracts import INTENT_LABELS
    for case in cases: _inputs(case,scope)
    if not cases or len({case['id'] for case in cases})!=len(cases):
        raise ValueError('cases must have unique ids')
    if scope=='assessor':
        expected=2
    elif all('expected_sufficient' in case for case in cases):
        expected=12
    elif all('expected_sufficient' not in case for case in cases):
        expected=35
        if Counter(case['expected_intent'] for case in cases)!={intent:5 for intent in INTENT_LABELS}:
            raise ValueError('intent evaluation requires five cases in each of seven classes')
    else: raise ValueError('mixed case sets are not supported')
    if len(cases)!=expected or (limit is not None and limit>expected):
        raise ValueError('full case set count or limit is invalid')
    return expected


def build_configuration(settings,*,scope,limit,expected_count,cases_sha256,
                        corpus_fingerprint=None,model_manifest_sha256=None):
    # Never serialize credentials, DSNs, endpoint URLs or arbitrary settings to artifacts.
    # All noncredential controls (including request/tool deadlines, history, budgets,
    # extra-body, revisions and endpoint) participate in the opaque identity.
    effective=settings.model_dump(mode='json',exclude={
        'llm_api_key','database_url','checkpoint_database_url','milvus_token'})
    prompt_root=resources.files('app').joinpath('prompts')
    prompts={entry.name:entry.read_text(encoding='utf-8') for entry in prompt_root.iterdir() if entry.name.endswith('.txt')}
    code={str(path.relative_to(ROOT)):path.read_text(encoding='utf-8')
        for path in sorted((ROOT/'app').rglob('*.py'))}
    code['scripts/evaluate_workflow.py']=Path(__file__).read_text(encoding='utf-8')
    versions={name:metadata.version(name) for name in (
        'langgraph','langchain-core','langchain-openai','pydantic','SQLAlchemy',
        'pymilvus','FlagEmbedding','torch','transformers','openai')}
    config={'scope':scope,'limit':limit,'expected_count':expected_count,'execution_kind':'actual_model',
        'cases_sha256':cases_sha256,'settings_sha256':digest(effective),
        'prompt_bundle_sha256':digest(prompts),'code_sha256':digest(code),
        'dependency_versions_sha256':digest(versions)}
    if scope=='workflow':
        config.update(corpus_fingerprint=corpus_fingerprint,model_manifest_sha256=model_manifest_sha256,
            workflow_knowledge_lower_threshold=settings.workflow_knowledge_lower_threshold,
            workflow_knowledge_upper_threshold=settings.workflow_knowledge_upper_threshold)
    return config


async def run_command(args):
    # Validate the complete input before constructing owners or dispatching calls.
    from app.config import load_settings
    from app.model import OpenAIModelGateway
    from app.resource_lifecycle import close_resources
    from app.knowledge.calibration import file_sha256, model_manifest_fingerprint
    from app.knowledge.corpus import corpus_fingerprint
    from app.workflow.bootstrap import build_workflow_dependencies, _request_hooks
    cases=[json.loads(line) for line in args.cases.read_text(encoding='utf-8').splitlines() if line.strip()]
    expected=validate_case_set(cases,args.scope,args.limit)
    settings=load_settings()
    owned=[]
    cancellation=None
    repository=None
    starting=None
    try:
        gateway=OpenAIModelGateway(settings)
        owned.append(gateway)
        if args.scope=='workflow':
            from app.db.database import Database
            from app.knowledge.bootstrap import build_knowledge_components
            if settings.database_url is None: raise ValueError('knowledge database is required')
            database=Database(settings.database_url.get_secret_value())
            owned.append(database)
            await database.check()
            components=await build_knowledge_components(settings,database,gateway,owned)
            repository=components.repository
            starting=corpus_fingerprint(await repository.list_all())
            # Only the graph runner receives these deps; it substitutes all writes.
            runner=GraphRunner(build_workflow_dependencies(settings,database,gateway,components))
            manifest_hash=model_manifest_fingerprint(settings)
        else:
            def factory(runtime):
                before,usage=_request_hooks(runtime,settings)
                return gateway.create_workflow_gateway(before,usage)
            runner=AssessorRunner(settings,factory)
            manifest_hash=None
        config=build_configuration(settings,scope=args.scope,limit=args.limit,expected_count=expected,
            cases_sha256=file_sha256(args.cases),corpus_fingerprint=starting,model_manifest_sha256=manifest_hash)
        manifest=await evaluate_workflow(cases[:args.limit] if args.limit else cases,runner,
            output_dir=args.output_dir,configuration=config)
        invalid_reason=None
        if repository is not None:
            try:
                if corpus_fingerprint(await repository.list_all())!=starting:
                    invalid_reason='corpus_changed_during_run'
            except Exception:
                invalid_reason='corpus_verification_failed'
        if invalid_reason:
            manifest['status']='invalid'
            manifest['invalid_reason']=invalid_reason
            atomic_json(args.output_dir/'manifest.json',manifest)
            report_path=args.output_dir/'report.md'
            atomic_text(report_path,f'INVALID: {invalid_reason}.\n\n'+report_path.read_text())
        print(strict_json_dumps({'scope':args.scope,'status':manifest['status'],
            'completed_cases':manifest['completed_cases'],'output_dir':str(args.output_dir)}))
        return 0 if manifest['status'] in {'complete','smoke'} else 1
    except asyncio.CancelledError as error:
        cancellation=error
        raise
    finally:
        await close_resources(owned,cancellation=cancellation)


def main(argv=None):
    args=build_parser().parse_args(argv)
    try: return asyncio.run(run_command(args))
    except Exception:
        print('Workflow evaluation failed; inspect safe artifacts. Raw errors and connection details are withheld.',file=sys.stderr)
        return 2


if __name__=='__main__': raise SystemExit(main())
