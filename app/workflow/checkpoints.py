from __future__ import annotations

from contextlib import AbstractAsyncContextManager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.config import Settings


_HEALTHCHECK_CONFIG = {"configurable": {"thread_id": "healthcheck-only"}}


class CheckpointStore:
    """Own the official PostgreSQL checkpointer connection lifecycle."""

    def __init__(self, settings: Settings, *, test_mode: bool = False) -> None:
        self._url = (
            settings.checkpoint_database_url.get_secret_value()
            if settings.checkpoint_database_url is not None
            else None
        )
        self._test_mode = test_mode
        self._manager: AbstractAsyncContextManager[AsyncPostgresSaver] | None = None
        self._saver: AsyncPostgresSaver | None = None

    async def open(self) -> AsyncPostgresSaver:
        if self._saver is not None:
            return self._saver
        if self._url is None:
            raise RuntimeError(
                "CHECKPOINT_CONFIG_MISSING: CHECKPOINT_DATABASE_URL is required"
            )

        manager = AsyncPostgresSaver.from_conn_string(self._url)
        try:
            saver = await manager.__aenter__()
        except Exception:
            raise RuntimeError(
                "CHECKPOINT_CONNECTION_FAILED: PostgreSQL checkpoint connection failed"
            ) from None
        self._manager = manager
        self._saver = saver
        return saver

    async def setup(self) -> None:
        saver = await self.open()
        try:
            await saver.setup()
        except Exception:
            raise RuntimeError(
                "CHECKPOINT_SETUP_FAILED: PostgreSQL checkpoint schema "
                "initialization failed"
            ) from None

    async def check(self) -> None:
        saver = await self.open()
        try:
            await saver.aget_tuple(_HEALTHCHECK_CONFIG)
        except Exception:
            raise RuntimeError(
                "CHECKPOINT_SCHEMA_UNAVAILABLE: PostgreSQL checkpoint schema "
                "is unavailable; "
                "run scripts/init_workflow_checkpoints.py"
            ) from None

    async def aclose(self) -> None:
        manager = self._manager
        self._manager = None
        self._saver = None
        if manager is not None:
            try:
                await manager.__aexit__(None, None, None)
            except Exception:
                raise RuntimeError(
                    "CHECKPOINT_CLOSE_FAILED: PostgreSQL checkpoint connection "
                    "could not close cleanly"
                ) from None
