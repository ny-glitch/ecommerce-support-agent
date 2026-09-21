from __future__ import annotations

from typing import TypedDict
from uuid import uuid4

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph

from app.workflow.checkpoints import CheckpointStore


class CounterState(TypedDict):
    count: int


def build_counter_graph(checkpointer: AsyncPostgresSaver):
    async def increment(state: CounterState) -> CounterState:
        return {"count": state["count"] + 1}

    builder = StateGraph(CounterState)
    builder.add_node("increment", increment)
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)
    return builder.compile(checkpointer=checkpointer)


async def test_open_never_creates_schema(monkeypatch, checkpoint_settings):
    calls = []

    async def forbidden_setup(self):
        calls.append("setup")
        raise AssertionError("setup during ordinary startup")

    monkeypatch.setattr(AsyncPostgresSaver, "setup", forbidden_setup)
    store = CheckpointStore(checkpoint_settings)
    try:
        await store.open()
        assert calls == []
    finally:
        await store.aclose()


async def test_checkpoint_persists_across_saver_instances(checkpoint_settings):
    test_prefix = f"ch05_test_{uuid4().hex}"
    config = {"configurable": {"thread_id": f"{test_prefix}_persisted"}}
    other_config = {"configurable": {"thread_id": f"{test_prefix}_other"}}

    first_store = CheckpointStore(checkpoint_settings, test_mode=True)
    try:
        first_graph = build_counter_graph(await first_store.open())
        result = await first_graph.ainvoke({"count": 0}, config)
        assert result == {"count": 1}
    finally:
        await first_store.aclose()

    second_store = CheckpointStore(checkpoint_settings, test_mode=True)
    try:
        second_graph = build_counter_graph(await second_store.open())
        persisted = await second_graph.aget_state(config)
        isolated = await second_graph.aget_state(other_config)
        assert persisted.values == {"count": 1}
        assert isolated.values == {}
    finally:
        await second_store.aclose()
