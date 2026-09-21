from __future__ import annotations

import asyncio

from app.config import load_settings
from app.workflow.checkpoints import CheckpointStore


async def initialize_workflow_checkpoints() -> None:
    store = CheckpointStore(load_settings())
    try:
        await store.setup()
    finally:
        await store.aclose()


if __name__ == "__main__":
    asyncio.run(initialize_workflow_checkpoints())
