import asyncio

import pytest

from app.errors import ServiceError
from app.services.turn_operations import TurnOperations


async def next_tick():
    done = asyncio.get_running_loop().create_future()
    asyncio.get_running_loop().call_soon(done.set_result, None)
    await done


@pytest.mark.asyncio
async def test_expired_deadline_never_starts_factory():
    operations = TurnOperations()
    calls = []
    async def request():
        calls.append('call')
    with pytest.raises(TimeoutError):
        await operations.run(request, asyncio.get_running_loop().time() - 1)
    assert calls == []
    await operations.drain()


@pytest.mark.asyncio
async def test_cancelled_write_can_commit_late_and_double_cancel_does_not_release_owner():
    operations = TurnOperations()
    entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    events = []
    async def write():
        events.append('request')
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
            events.append('commit')
    async def owner():
        try:
            await operations.run(write, asyncio.get_running_loop().time() + 10, mutation=True)
        finally:
            try:
                await operations.drain()
            finally:
                events.append('guard released')
    task = asyncio.create_task(owner())
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    await asyncio.wait_for(cancelled.wait(), 1)
    await next_tick()
    task.cancel()
    await next_tick()
    assert not task.done() and events == ['request']
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    assert events == ['request', 'commit', 'guard released']
    await operations.drain()


@pytest.mark.asyncio
async def test_drain_waits_active_anext_unwind_before_aclose_and_survives_repeated_cancel():
    operations = TurnOperations()
    entered, unwinding, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    events = []
    class Iterator:
        active = False
        async def __anext__(self):
            self.active = True
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                unwinding.set()
                await release.wait()
                self.active = False
                events.append('anext ended')
        async def aclose(self):
            assert not self.active
            events.append('closed')
    iterator = Iterator()
    operations.track_iterator(iterator)
    operations.track_iterator(iterator)
    call = asyncio.create_task(operations.run(lambda: anext(iterator), asyncio.get_running_loop().time() + 10))
    await asyncio.wait_for(entered.wait(), 1)
    drain = asyncio.create_task(operations.drain())
    await asyncio.wait_for(unwinding.wait(), 1)
    drain.cancel()
    await next_tick()
    drain.cancel()
    await next_tick()
    assert not drain.done() and events == []
    with pytest.raises(RuntimeError):
        await operations.run(lambda: anext(iterator), asyncio.get_running_loop().time() + 10)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(drain, 1)
    with pytest.raises(asyncio.CancelledError):
        await call
    assert events == ['anext ended', 'closed']
    await operations.drain()
    assert events == ['anext ended', 'closed']


@pytest.mark.asyncio
async def test_timeout_remains_owned_until_delayed_write_settles():
    operations = TurnOperations()
    entered, unwinding, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    commits = []
    async def write():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            unwinding.set()
            await release.wait()
            commits.append('committed')
    call = asyncio.create_task(operations.run(write, asyncio.get_running_loop().time() + .03, mutation=True))
    await asyncio.wait_for(entered.wait(), 1)
    with pytest.raises(TimeoutError):
        await call
    await asyncio.wait_for(unwinding.wait(), 1)
    drain = asyncio.create_task(operations.drain())
    await next_tick()
    assert not drain.done() and commits == []
    release.set()
    await asyncio.wait_for(drain, 1)
    assert commits == ['committed']


@pytest.mark.asyncio
async def test_close_failure_is_safe_and_not_reported_as_successful_drain():
    operations = TurnOperations()
    closed = []
    class Broken:
        async def aclose(self):
            raise RuntimeError('private connection data')
    class Other:
        async def aclose(self):
            closed.append(True)
    operations.track_iterator(Broken())
    operations.track_iterator(Other())
    with pytest.raises(ServiceError) as caught:
        await operations.drain()
    assert caught.value.code == 'TURN_CLEANUP_FAILED'
    assert 'private' not in str(caught.value)
    assert closed == [True]


@pytest.mark.asyncio
async def test_consumer_cancel_racing_drain_does_not_cancel_physical_write_twice():
    operations = TurnOperations()
    entered, unwinding, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    commits = []
    async def write():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            unwinding.set()
            await release.wait()
            commits.append('committed')
    call = asyncio.create_task(operations.run(write, asyncio.get_running_loop().time() + 10, mutation=True))
    await asyncio.wait_for(entered.wait(), 1)
    drain = asyncio.create_task(operations.drain())
    await asyncio.wait_for(unwinding.wait(), 1)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    await next_tick()
    try:
        assert not drain.done() and commits == []
    finally:
        release.set()
        await asyncio.wait_for(drain, 1)
    assert commits == ['committed']
