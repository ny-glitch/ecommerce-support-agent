from __future__ import annotations

import fcntl
import logging
import time
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from app.knowledge.contracts import KnowledgeChunk
from app.knowledge.local_models import InputTooLongError
from app.knowledge.text import embedding_text, source_hash


_TEXT_MAX_BYTES = 65_535
_INT64_MAX = 2**63 - 1
_DEFAULT_TIMEOUT_SECONDS = 240
logger = logging.getLogger(__name__)


class IndexerAlreadyRunningError(RuntimeError):
    pass


class ChunkValidationError(ValueError):
    pass


class _Repository(Protocol):
    async def list_all(self) -> list[KnowledgeChunk]: ...

    async def mark_pending(self, ids: Iterable[int]) -> None: ...

    async def mark_done_if_current(
        self, chunk_id: int, expected_hash: str
    ) -> bool: ...


class _Store(Protocol):
    async def ensure_schema(self) -> None: ...

    async def fingerprints(self, ids: list[int]) -> dict[int, str]: ...

    async def upsert(
        self,
        chunks: list[KnowledgeChunk],
        vectors: list[list[float]],
        *,
        deadline: float,
    ) -> None: ...


class _Models(Protocol):
    async def embed(
        self, texts: list[str], *, deadline: float
    ) -> list[list[float]]: ...


class KnowledgeIndexer:
    def __init__(
        self,
        repo: _Repository,
        store: _Store,
        models: _Models,
        *,
        lock_path: Path,
        timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._repo = repo
        self._store = store
        self._models = models
        self._lock_path = lock_path
        self._timeout_seconds = timeout_seconds

    async def run(self, *, repair: bool = False) -> dict[str, int]:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a+", encoding="utf-8") as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise IndexerAlreadyRunningError(
                    "knowledge indexing is already running on this host"
                ) from exc
            try:
                return await self._run_locked(repair=repair)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    async def _run_locked(self, *, repair: bool) -> dict[str, int]:
        await self._store.ensure_schema()
        chunks = await self._repo.list_all()
        if repair:
            chunks = await self._reset_stale_done(chunks)

        counts = {"indexed": 0, "skipped": 0, "failed": 0}
        pending: list[KnowledgeChunk] = []
        texts: list[str] = []
        for chunk in chunks:
            if chunk.vectorize_status == "done":
                counts["skipped"] += 1
                continue
            try:
                texts.append(_validated_text(chunk))
                pending.append(chunk)
            except Exception as exc:
                counts["failed"] += 1
                _log_failure(chunk.id, exc)

        if not pending:
            return counts

        deadline = time.monotonic() + self._timeout_seconds
        try:
            while pending:
                try:
                    vectors = await self._models.embed(texts, deadline=deadline)
                    break
                except InputTooLongError as exc:
                    index = exc.input_index
                    if (
                        exc.input_kind != "embedding"
                        or index is None
                        or not 0 <= index < len(pending)
                    ):
                        raise
                    invalid = pending.pop(index)
                    texts.pop(index)
                    counts["failed"] += 1
                    _log_failure(invalid.id, exc)
            else:
                return counts
        except Exception as exc:
            counts["failed"] += len(pending)
            for chunk in pending:
                _log_failure(chunk.id, exc)
            return counts

        try:
            await self._store.upsert(pending, vectors, deadline=deadline)
        except Exception as exc:
            counts["failed"] += len(pending)
            for chunk in pending:
                _log_failure(chunk.id, exc)
            return counts

        for chunk in pending:
            try:
                if not await self._repo.mark_done_if_current(
                    chunk.id, source_hash(chunk)
                ):
                    raise RuntimeError("source changed before SQL confirmation")
            except Exception as exc:
                counts["failed"] += 1
                _log_failure(chunk.id, exc)
            else:
                counts["indexed"] += 1
        return counts

    async def _reset_stale_done(
        self, chunks: list[KnowledgeChunk]
    ) -> list[KnowledgeChunk]:
        done = [chunk for chunk in chunks if chunk.vectorize_status == "done"]
        actual = await self._store.fingerprints([chunk.id for chunk in done])
        stale_ids = [
            chunk.id
            for chunk in done
            if actual.get(chunk.id) != source_hash(chunk)
        ]
        if stale_ids:
            await self._repo.mark_pending(stale_ids)
            stale = set(stale_ids)
            chunks = [
                replace(chunk, vector_id=None, vectorize_status="pending")
                if chunk.id in stale
                else chunk
                for chunk in chunks
            ]
        return chunks


def _validated_text(chunk: KnowledgeChunk) -> str:
    if not 1 <= chunk.id <= _INT64_MAX:
        raise ChunkValidationError(
            f"chunk {chunk.id} is outside signed INT64 range"
        )
    _validate_utf8_field(chunk, "category", chunk.category, 1_020)
    _validate_utf8_field(chunk, "section_path", chunk.section_path, 2_048)
    _validate_utf8_field(chunk, "content_type", chunk.content_type, 128)
    text = embedding_text(chunk)
    byte_count = len(text.encode("utf-8"))
    if byte_count > _TEXT_MAX_BYTES:
        raise ChunkValidationError(
            f"chunk {chunk.id} text has {byte_count} UTF-8 bytes; "
            f"maximum is {_TEXT_MAX_BYTES}"
        )
    return text


def _validate_utf8_field(
    chunk: KnowledgeChunk,
    name: str,
    value: str | None,
    maximum: int,
) -> None:
    if value is None:
        return
    byte_count = len(value.encode("utf-8"))
    if byte_count > maximum:
        raise ChunkValidationError(
            f"chunk {chunk.id} {name} has {byte_count} UTF-8 bytes; "
            f"maximum is {maximum}"
        )


def _log_failure(chunk_id: int, exc: Exception) -> None:
    if isinstance(exc, (ChunkValidationError, InputTooLongError)):
        logger.error("knowledge chunk %s validation failed: %s", chunk_id, exc)
        return
    logger.error(
        "knowledge chunk %s indexing failed: %s",
        chunk_id,
        type(exc).__name__,
    )
