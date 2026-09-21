import json
from uuid import UUID

import httpx
import pytest

from scripts.demo_workflow import run_demo, DemoError

SESSION = '00000000-0000-0000-0000-000000000001'
TURN = '00000000-0000-0000-0000-000000000002'
ACTION = '00000000-0000-0000-0000-000000000003'
HASH = 'a' * 64


def stream(*events):
    return ''.join(f'event: {name}\r\ndata: {json.dumps(data)}\r\n\r\n' for name,data in events)


def body(done=True):
    events = [('meta', {'session_id': SESSION, 'turn_id': TURN}),
        ('message', {'content':'建议','kind':'complaint'}),
        ('actions', {'session_id': SESSION, 'turn_id': TURN, 'actions': [
            {'type':'handoff'}, {'type':'create_ticket','action_id':ACTION,
             'conversation_id':SESSION,'turn_id':TURN,'ticket_no':'TK-actual','status':'offered',
             'draft':{'issue_description':'投诉','ticket_type':'complaint'}}]})]
    if done: events.append(('done', {'session_id':SESSION,'turn_id':TURN,'status':'completed','refused':False,'citations':[]}))
    return stream(*events)


@pytest.mark.parametrize('confirm,count', [(False,0),(True,2)])
async def test_complaint_requires_explicit_flag_and_posts_empty_object_twice(confirm,count):
    requests=[]
    def handle(request):
        requests.append(request)
        if request.url.path == '/api/chat':
            return httpx.Response(200, headers={'content-type':'text/event-stream'}, text=body())
        assert request.url.path == f'/api/conversations/{SESSION}/actions/{ACTION}/confirm'
        assert json.loads(request.content) == {}
        return httpx.Response(200,json={'ticket_no':'TK-actual','status':'completed','conversation_id':SESSION,'action_id':ACTION})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        result = await run_demo(client,'http://localhost:8002','complaint',confirm_ticket=confirm)
    assert len(requests) == 1+count
    assert result['ticket_numbers'] == (['TK-actual','TK-actual'] if confirm else [])


@pytest.mark.parametrize('text', [body(False), stream(('error', {'code':'UPSTREAM_ERROR','message':'SECRET'})),
    'event: done\ndata: {}', stream(('done', {}))])
async def test_eof_error_or_invalid_done_cannot_succeed(text):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req:httpx.Response(200,
        headers={'content-type':'text/event-stream'},text=text))) as client:
        with pytest.raises(DemoError) as error:
            await run_demo(client,'http://localhost:8002','policy')
    assert 'SECRET' not in str(error.value)


async def test_sources_ignore_model_url_and_only_use_fixed_same_base_route():
    requests=[]
    def handle(request):
        requests.append(str(request.url))
        if request.url.path == '/api/chat':
            return httpx.Response(200,headers={'content-type':'text/event-stream'},text=stream(
                ('meta',{'session_id':SESSION,'turn_id':TURN}),
                ('sources',{'sources':[{'chunk_id':910001,'content_hash':HASH,'url':'https://evil.invalid/steal'}]}),
                ('done',{'session_id':SESSION,'turn_id':TURN,'status':'completed','refused':False,'citations':[1]})))
        return httpx.Response(200,json={'id':910001,'content_hash':HASH,'answer':'原文'})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        result=await run_demo(client,'http://localhost:8002','policy')
    assert requests == ['http://localhost:8002/api/chat',
        'http://localhost:8002/api/knowledge/chunks/910001?expected_hash='+HASH]
    assert len(result['source_checks'])==1


async def test_redirects_never_follow_and_invalid_source_id_fails():
    seen=[]
    def handle(request):
        seen.append(str(request.url))
        return httpx.Response(307,headers={'location':'https://evil.invalid'})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle),follow_redirects=True) as client:
        with pytest.raises(DemoError): await run_demo(client,'http://localhost:8002','policy')
    assert len(seen)==1


@pytest.mark.parametrize('source',[{'chunk_id':True,'content_hash':HASH},
    {'chunk_id':2**63,'content_hash':HASH}, {'chunk_id':1,'content_hash':'../bad'}])
async def test_invalid_source_identity_is_rejected_before_get(source):
    requests=[]
    def handle(request):
        requests.append(request)
        return httpx.Response(200,headers={'content-type':'text/event-stream'},text=stream(
            ('meta',{'session_id':SESSION,'turn_id':TURN}),('sources',{'sources':[source]}),
            ('done',{'session_id':SESSION,'turn_id':TURN,'status':'completed'})))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(DemoError): await run_demo(client,'http://localhost:8002','policy')
    assert len(requests)==1


async def test_action_scope_mismatch_cannot_confirm():
    requests=[]
    def handle(request):
        requests.append(request)
        return httpx.Response(200,headers={'content-type':'text/event-stream'},text=body().replace(
            '"conversation_id": "'+SESSION+'"','"conversation_id": "'+TURN+'"'))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(DemoError): await run_demo(client,'http://localhost:8002','complaint',confirm_ticket=True)
    assert len(requests)==1


async def test_confirmation_replay_changed_number_fails():
    confirmations=0
    def handle(request):
        nonlocal confirmations
        if request.url.path=='/api/chat':
            return httpx.Response(200,headers={'content-type':'text/event-stream'},text=body())
        confirmations+=1
        return httpx.Response(200,json={'status':'completed','conversation_id':SESSION,
            'action_id':ACTION,'ticket_no':'TK-actual' if confirmations==1 else 'TK-new'})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(DemoError): await run_demo(client,'http://localhost:8002','complaint',confirm_ticket=True)
    assert confirmations==2


@pytest.mark.parametrize('script',['demo_workflow.py','evaluate_workflow.py'])
def test_script_help_loads_this_worktree_from_another_directory(tmp_path,script):
    import subprocess
    import sys
    from pathlib import Path
    target=Path(__file__).resolve().parents[1]/'scripts'/script
    result=subprocess.run([sys.executable,str(target),'--help'],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0, 'CLI help must import its own checkout without constructing service owners'
    assert '--help' in result.stdout
