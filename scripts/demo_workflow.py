#!/usr/bin/env python3
"""Observe a real SSE turn; only an explicit flag confirms a ticket."""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID

import httpx

# Direct execution must resolve this checkout, including a shared editable venv.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation_io import strict_json_dumps


SCENARIOS = {
    'policy': 'C65-Pro 签收第五天，只验货且无使用痕迹，符合七天无理由退货政策吗？',
    'logistics': '订单 1001 的物流到哪了？',
    'complaint': '我要投诉，商品包装破损且客服没有解决。',
    'chitchat': '你好！',
    'multi_step': '请先查订单 1001 的状态，再查询订单 1001 的物流轨迹；如果可以查物流，请把两个查询结果分别告诉我。',
    'unknown': 'Z99-Pro 耳机可以戴着游泳吗？',
}


class DemoError(ValueError):
    pass


def _uuid(value):
    if not isinstance(value, str): raise DemoError('invalid response identity')
    try: return str(UUID(value))
    except ValueError: raise DemoError('invalid response identity') from None


def _success(response):
    if not 200 <= response.status_code < 300:
        raise DemoError(f'HTTP status {response.status_code}')


async def _events(response):
    name, data = 'message', []
    async for line in response.aiter_lines():
        if line == '':
            if data:
                try: value = json.loads('\n'.join(data))
                except ValueError: raise DemoError('invalid SSE JSON') from None
                if not isinstance(value, dict): raise DemoError('invalid SSE object')
                yield name, value
            name, data = 'message', []
        elif line.startswith(':'): continue
        else:
            field, _, value = line.partition(':')
            value = value[1:] if value.startswith(' ') else value
            if field == 'event': name = value
            elif field == 'data': data.append(value)
    # An unterminated event is not dispatched; EOF is never a terminal done.


async def run_demo(client, base_url, scenario, *, confirm_ticket=False):
    parsed = urlsplit(base_url)
    if (parsed.scheme not in {'http','https'} or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment or parsed.path not in {'','/'}):
        raise DemoError('base URL must be an HTTP origin')
    base = base_url.rstrip('/')
    if scenario not in SCENARIOS or (confirm_ticket and scenario != 'complaint'):
        raise DemoError('ticket confirmation requires the complaint scenario')
    events, sources, actions = [], [], []
    session, turn, done = None, None, False
    async with client.stream('POST',base+'/api/chat',json={'message':SCENARIOS[scenario]},
            follow_redirects=False) as response:
        _success(response)
        if response.headers.get('content-type','').split(';')[0].strip() != 'text/event-stream':
            raise DemoError('expected SSE response')
        async for name, value in _events(response):
            if done: raise DemoError('event after terminal done')
            if name == 'error': raise DemoError('server reported a stream error')
            if name == 'meta':
                if session is not None: raise DemoError('duplicate meta event')
                session, turn = _uuid(value.get('session_id')), _uuid(value.get('turn_id'))
            if name == 'sources': sources.extend(value.get('sources', []))
            if name == 'actions':
                if session is None or value.get('session_id') != session or value.get('turn_id') != turn:
                    raise DemoError('action event identity mismatch')
                actions.extend(value.get('actions', []))
            if name == 'done':
                if (session is None or value.get('session_id') != session or value.get('turn_id') != turn
                        or value.get('status') != 'completed'):
                    raise DemoError('invalid terminal done')
                done = True
            events.append({'name':name,'data':value})
    if not done: raise DemoError('stream ended without terminal done')
    source_checks=[]
    for source in sources:
        identifier, content_hash = source.get('chunk_id'), source.get('content_hash')
        if (type(identifier) is not int or not 1 <= identifier <= 2**63-1 or
                not isinstance(content_hash,str) or not re.fullmatch('[0-9a-f]{64}',content_hash)):
            raise DemoError('invalid source identity')
        # The model-supplied URL is never followed, even if it looks same-origin.
        url = f'{base}/api/knowledge/chunks/{identifier}'
        response=await client.get(url,params={'expected_hash':content_hash},follow_redirects=False)
        _success(response)
        value=response.json()
        if value.get('id') != identifier or value.get('content_hash') != content_hash:
            raise DemoError('source response identity mismatch')
        source_checks.append({'id':identifier,'content_hash':content_hash,'source':value})
    ticket_numbers=[]
    if confirm_ticket:
        offers=[value for value in actions if value.get('type') == 'create_ticket']
        if len(offers)!=1: raise DemoError('expected one ticket suggestion')
        offer=offers[0]
        action_id=_uuid(offer.get('action_id'))
        if offer.get('conversation_id') != session or offer.get('turn_id') != turn:
            raise DemoError('ticket suggestion identity mismatch')
        for _ in range(2):
            response=await client.post(f'{base}/api/conversations/{session}/actions/{action_id}/confirm',
                json={},follow_redirects=False)
            _success(response)
            value=response.json()
            if (value.get('status') != 'completed' or value.get('conversation_id') != session or
                    value.get('action_id') != action_id or not isinstance(value.get('ticket_no'),str)
                    or not value['ticket_no'] or value['ticket_no'] != offer.get('ticket_no')):
                raise DemoError('ticket confirmation identity mismatch')
            ticket_numbers.append(value['ticket_no'])
        if ticket_numbers[0] != ticket_numbers[1]: raise DemoError('ticket replay changed identity')
    return {'scenario':scenario,'session_id':session,'turn_id':turn,'terminal_done':done,
        'events':events,'source_checks':source_checks,'ticket_numbers':ticket_numbers}


def build_parser():
    parser=argparse.ArgumentParser(description='Observe actual Workflow SSE and optional explicit ticket confirmation.')
    parser.add_argument('--base-url',required=True)
    parser.add_argument('--scenario',choices=tuple(SCENARIOS),required=True)
    parser.add_argument('--confirm-ticket',action='store_true')
    return parser


async def _run(args):
    async with httpx.AsyncClient(timeout=httpx.Timeout(260,connect=10),follow_redirects=False) as client:
        value=await run_demo(client,args.base_url,args.scenario,confirm_ticket=args.confirm_ticket)
    print(strict_json_dumps(value,indent=2))
    return 0


def main(argv=None):
    args=build_parser().parse_args(argv)
    try: return asyncio.run(_run(args))
    except Exception:
        print('Workflow demo failed; no successful terminal verification. Raw errors are withheld.',file=sys.stderr)
        return 1


if __name__ == '__main__': raise SystemExit(main())
