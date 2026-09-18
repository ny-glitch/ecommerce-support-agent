from __future__ import annotations

import asyncio
import json
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
)

from app.config import Settings
from app.knowledge.contracts import KnowledgeChunk, SearchHit
from app.knowledge.local_models import DENSE_DIMENSION
from app.knowledge.text import embedding_text, source_hash


_TEST_COLLECTION = re.compile(r"^ch04_test_[0-9a-f]{32}$")
_TEXT_MAX_BYTES = 65_535
_ABANDONED = object()
_FIELD_SPECS = {
    "id": (DataType.INT64, None),
    "text": (DataType.VARCHAR, _TEXT_MAX_BYTES),
    "dense_vector": (DataType.FLOAT_VECTOR, DENSE_DIMENSION),
    "sparse_vector": (DataType.SPARSE_FLOAT_VECTOR, None),
    "category": (DataType.VARCHAR, 1_020),
    "section_path": (DataType.VARCHAR, 2_048),
    "content_type": (DataType.VARCHAR, 128),
    "is_key_clause": (DataType.BOOL, None),
    "source_hash": (DataType.VARCHAR, 64),
}


class IncompatibleMilvusSchemaError(RuntimeError):
    pass


class MilvusDeadlineExceeded(TimeoutError):
    pass


class MilvusQueueFullError(RuntimeError):
    pass


def validate_test_collection(name: str) -> None:
    if _TEST_COLLECTION.fullmatch(name) is None:
        raise ValueError("refusing to operate on a non-test Milvus collection")


class MilvusStore:
    def __init__(
        self,
        settings: Settings,
        *,
        collection_name: str | None = None,
    ) -> None:
        self._settings = settings
        self._collection_name = collection_name or settings.milvus_collection
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="knowledge-milvus",
        )
        self._client: MilvusClient | None = None
        self._capacity = 2
        self._state_lock = threading.Lock()
        self._inflight = 0
        self._close_lock = asyncio.Lock()
        self._closed = False
        self._shutdown_task: asyncio.Task[None] | None = None

    async def ensure_schema(self) -> None:
        deadline = self._deadline()
        exists = await self._call(
            "has_collection", self._collection_name, deadline=deadline
        )
        if not exists:
            schema = _build_schema()
            indexes = _build_indexes()
            await self._call(
                "create_collection",
                self._collection_name,
                schema=schema,
                index_params=indexes,
                consistency_level="Strong",
                deadline=deadline,
            )
        else:
            await self._validate_schema(deadline)
        await self._call("load_collection", self._collection_name, deadline=deadline)

    async def prepare_existing_collection(self) -> None:
        """Validate and load an existing collection without creating or mutating it."""
        await self.check()
        deadline = self._deadline()
        exists = await self._call(
            "has_collection", self._collection_name, deadline=deadline
        )
        if not exists:
            raise IncompatibleMilvusSchemaError(
                f"Milvus collection {self._collection_name!r} does not exist; "
                "run scripts/index_knowledge.py"
            )
        await self._validate_schema(deadline)
        await self._call("load_collection", self._collection_name, deadline=deadline)

    async def upsert(
        self,
        chunks: list[KnowledgeChunk],
        vectors: list[list[float]],
        *,
        deadline: float,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunk and vector counts differ")
        rows = [_upsert_row(chunk, vector) for chunk, vector in zip(chunks, vectors)]
        if not rows:
            return
        await self._call(
            "upsert",
            self._collection_name,
            data=rows,
            deadline=deadline,
        )
        await self._call("flush", self._collection_name, deadline=deadline)
        actual = await self._fingerprints([chunk.id for chunk in chunks], deadline)
        expected = {chunk.id: source_hash(chunk) for chunk in chunks}
        if actual != expected:
            raise RuntimeError(
                "Milvus upsert confirmation did not match source fingerprints"
            )

    async def fingerprints(self, ids: list[int]) -> dict[int, str]:
        return await self._fingerprints(ids, self._deadline())

    async def search_dense(
        self,
        vector: list[float],
        category: str | None,
        *,
        deadline: float,
    ) -> list[SearchHit]:
        _validate_vector(vector)
        result = await self._call(
            "search",
            self._collection_name,
            data=[vector],
            anns_field="dense_vector",
            search_params={"metric_type": "COSINE", "params": {"ef": 100}},
            filter=_category_filter(category),
            limit=50,
            output_fields=["source_hash"],
            consistency_level="Strong",
            deadline=deadline,
        )
        return _search_hits(result)

    async def search_bm25(
        self,
        text: str,
        category: str | None,
        *,
        deadline: float,
    ) -> list[SearchHit]:
        result = await self._call(
            "search",
            self._collection_name,
            data=[text],
            anns_field="sparse_vector",
            search_params={"metric_type": "BM25", "params": {}},
            filter=_category_filter(category),
            limit=50,
            output_fields=["source_hash"],
            consistency_level="Strong",
            deadline=deadline,
        )
        return _search_hits(result)

    async def search_hybrid(
        self,
        vector: list[float],
        text: str,
        category: str | None,
        *,
        deadline: float,
    ) -> list[SearchHit]:
        _validate_vector(vector)
        expression = _category_filter(category)
        requests = [
            AnnSearchRequest(
                [vector],
                "dense_vector",
                {"metric_type": "COSINE", "params": {"ef": 100}},
                50,
                expr=expression,
            ),
            AnnSearchRequest(
                [text],
                "sparse_vector",
                {"metric_type": "BM25", "params": {}},
                50,
                expr=expression,
            ),
        ]
        result = await self._call(
            "hybrid_search",
            self._collection_name,
            requests,
            RRFRanker(k=60),
            limit=50,
            output_fields=["source_hash"],
            consistency_level="Strong",
            deadline=deadline,
        )
        return _search_hits(result)

    async def check(self) -> None:
        version = await self._call("get_server_version", deadline=self._deadline())
        if version != "2.6.23":
            raise RuntimeError(f"Milvus server version must be 2.6.23, got {version!r}")

    async def aclose(self) -> None:
        async with self._close_lock:
            if self._shutdown_task is None:
                with self._state_lock:
                    self._closed = True
                self._shutdown_task = asyncio.create_task(
                    asyncio.to_thread(self._shutdown_sync)
                )
            shutdown_task = self._shutdown_task
        await asyncio.shield(shutdown_task)

    async def _fingerprints(
        self, ids: list[int], deadline: float
    ) -> dict[int, str]:
        if not ids:
            return {}
        rows = await self._call(
            "query",
            self._collection_name,
            ids=list(dict.fromkeys(ids)),
            output_fields=["id", "source_hash"],
            consistency_level="Strong",
            deadline=deadline,
        )
        return {int(row["id"]): str(row["source_hash"]) for row in rows}

    async def _validate_schema(self, deadline: float) -> None:
        description = await self._call(
            "describe_collection", self._collection_name, deadline=deadline
        )
        problems = _schema_problems(description)
        index_names = await self._call(
            "list_indexes", self._collection_name, deadline=deadline
        )
        indexes = {
            name: await self._call(
                "describe_index",
                self._collection_name,
                name,
                deadline=deadline,
            )
            for name in index_names
        }
        problems.extend(_index_problems(indexes))
        if problems:
            raise IncompatibleMilvusSchemaError(
                "existing Milvus collection is incompatible: " + "; ".join(problems)
            )

    async def _call(
        self,
        method_name: str,
        *args: Any,
        deadline: float,
        **kwargs: Any,
    ) -> Any:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise MilvusDeadlineExceeded("Milvus deadline exceeded")
        abandoned = threading.Event()
        with self._state_lock:
            if self._closed:
                raise RuntimeError("Milvus store is closed")
            if self._inflight >= self._capacity:
                raise MilvusQueueFullError("Milvus RPC queue is full")
            self._inflight += 1
            try:
                future = self._executor.submit(
                    self._invoke,
                    method_name,
                    args,
                    kwargs,
                    deadline,
                    abandoned,
                )
            except BaseException:
                self._inflight -= 1
                raise
        future.add_done_callback(lambda _future: self._release_capacity())
        loop = asyncio.get_running_loop()
        wrapped = asyncio.wrap_future(future, loop=loop)
        try:
            result = await asyncio.wait_for(
                asyncio.shield(wrapped), timeout=remaining
            )
            if result is _ABANDONED:
                raise MilvusDeadlineExceeded("Milvus deadline exceeded")
            return result
        except asyncio.CancelledError:
            abandoned.set()
            await _drain_future(wrapped)
            raise
        except TimeoutError as exc:
            if future.done() and not future.cancelled():
                underlying = future.exception()
                if isinstance(underlying, TimeoutError):
                    raise underlying
            abandoned.set()
            cancelled_during_drain = await _drain_future(wrapped)
            if cancelled_during_drain:
                raise asyncio.CancelledError
            raise MilvusDeadlineExceeded("Milvus deadline exceeded") from exc

    def _invoke(
        self,
        method_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        deadline: float,
        abandoned: threading.Event,
    ) -> Any:
        if abandoned.is_set():
            return _ABANDONED
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise MilvusDeadlineExceeded("Milvus deadline exceeded")
        if self._client is None:
            token = self._settings.milvus_token
            client_kwargs: dict[str, Any] = {"uri": self._settings.milvus_uri}
            if token is not None and token.get_secret_value():
                client_kwargs["token"] = token.get_secret_value()
            client_kwargs["timeout"] = remaining
            self._client = MilvusClient(**client_kwargs)
        if abandoned.is_set():
            return _ABANDONED
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise MilvusDeadlineExceeded("Milvus deadline exceeded")
        call_kwargs = dict(kwargs)
        call_kwargs["timeout"] = remaining
        return getattr(self._client, method_name)(*args, **call_kwargs)

    def _release_capacity(self) -> None:
        with self._state_lock:
            self._inflight -= 1

    def _shutdown_sync(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
        client = self._client
        if client is not None:
            client.close()

    def _deadline(self) -> float:
        return time.monotonic() + self._settings.knowledge_request_timeout_seconds


async def _drain_future(future: asyncio.Future[Any]) -> bool:
    cancelled = False
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError:
            cancelled = True
            continue
        except BaseException:
            return cancelled
    if not future.cancelled():
        try:
            future.exception()
        except BaseException:
            pass
    return cancelled


def _build_schema():
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(
        field_name="id",
        datatype=DataType.INT64,
        is_primary=True,
        auto_id=False,
    )
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=_TEXT_MAX_BYTES,
        enable_analyzer=True,
        analyzer_params={"type": "chinese"},
    )
    schema.add_field(
        field_name="dense_vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=DENSE_DIMENSION,
    )
    schema.add_field(
        field_name="sparse_vector",
        datatype=DataType.SPARSE_FLOAT_VECTOR,
    )
    schema.add_field(
        field_name="category", datatype=DataType.VARCHAR, max_length=1_020
    )
    schema.add_field(
        field_name="section_path", datatype=DataType.VARCHAR, max_length=2_048
    )
    schema.add_field(
        field_name="content_type", datatype=DataType.VARCHAR, max_length=128
    )
    schema.add_field(field_name="is_key_clause", datatype=DataType.BOOL)
    schema.add_field(
        field_name="source_hash", datatype=DataType.VARCHAR, max_length=64
    )
    schema.add_function(
        Function(
            name="text_bm25",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse_vector"],
        )
    )
    return schema


def _build_indexes():
    indexes = MilvusClient.prepare_index_params()
    indexes.add_index(
        "dense_vector",
        index_name="dense_hnsw",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    indexes.add_index(
        "sparse_vector",
        index_name="sparse_bm25",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={},
    )
    return indexes


def _upsert_row(chunk: KnowledgeChunk, vector: list[float]) -> dict[str, Any]:
    if not 1 <= chunk.id <= 2**63 - 1:
        raise ValueError(f"chunk {chunk.id} is outside signed INT64 range")
    text = embedding_text(chunk)
    if len(text.encode("utf-8")) > _TEXT_MAX_BYTES:
        raise ValueError(f"chunk {chunk.id} text exceeds Milvus VARCHAR byte limit")
    for name, value, maximum in (
        ("category", chunk.category, 1_020),
        ("section_path", chunk.section_path or "", 2_048),
        ("content_type", chunk.content_type or "", 128),
    ):
        if len(value.encode("utf-8")) > maximum:
            raise ValueError(
                f"chunk {chunk.id} {name} exceeds Milvus VARCHAR byte limit"
            )
    _validate_vector(vector)
    return {
        "id": chunk.id,
        "text": text,
        "dense_vector": vector,
        "category": chunk.category,
        "section_path": chunk.section_path or "",
        "content_type": chunk.content_type or "",
        "is_key_clause": chunk.is_key_clause,
        "source_hash": source_hash(chunk),
    }


def _validate_vector(vector: list[float]) -> None:
    if len(vector) != DENSE_DIMENSION:
        raise ValueError(f"dense vector dimension must be {DENSE_DIMENSION}")
    if not all(
        isinstance(value, (int, float)) and math.isfinite(value)
        for value in vector
    ):
        raise ValueError("dense vector values must be finite numbers")


def _category_filter(category: str | None) -> str:
    if category is None:
        return ""
    return "category == " + json.dumps(category, ensure_ascii=False)


def _search_hits(result: Any) -> list[SearchHit]:
    rows = result[0] if result else []
    hits: list[SearchHit] = []
    for row in rows:
        entity = row.get("entity") or {}
        hits.append(
            SearchHit(
                id=int(row["id"]),
                score=float(row.get("distance", row.get("score", 0.0))),
                source_hash=str(entity["source_hash"]),
            )
        )
    return hits


def _schema_problems(description: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    fields = {field["name"]: field for field in description.get("fields", [])}
    for name, (expected_type, expected_size) in _FIELD_SPECS.items():
        field = fields.get(name)
        if field is None:
            problems.append(f"missing field {name}")
            continue
        actual_type = field.get("type")
        if int(actual_type) != int(expected_type):
            problems.append(f"field {name} has wrong type")
        params = field.get("params") or {}
        if expected_size is not None:
            size_key = "dim" if name == "dense_vector" else "max_length"
            if int(params.get(size_key, -1)) != expected_size:
                problems.append(f"field {name} has wrong {size_key}")
    id_field = fields.get("id") or {}
    if not id_field.get("is_primary") or id_field.get("auto_id", False):
        problems.append("id must be a manual primary key")
    text_params = (fields.get("text") or {}).get("params") or {}
    analyzer = text_params.get("analyzer_params")
    if isinstance(analyzer, str):
        try:
            analyzer = json.loads(analyzer)
        except json.JSONDecodeError:
            pass
    analyzer_enabled = str(text_params.get("enable_analyzer", "")).casefold() == "true"
    if not analyzer_enabled or analyzer != {"type": "chinese"}:
        problems.append("text must use the chinese analyzer")
    functions = description.get("functions") or []
    valid_function = any(
        function.get("name") == "text_bm25"
        and int(function.get("type", -1)) == int(FunctionType.BM25)
        and function.get("input_field_names") == ["text"]
        and function.get("output_field_names") == ["sparse_vector"]
        for function in functions
    )
    if not valid_function:
        problems.append("native BM25 function is missing or incompatible")
    return problems


def _index_problems(indexes: dict[str, dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    dense = indexes.get("dense_hnsw")
    sparse = indexes.get("sparse_bm25")
    if not _index_matches(
        dense,
        field_name="dense_vector",
        index_type="HNSW",
        metric_type="COSINE",
        M=16,
        efConstruction=200,
    ):
        problems.append("dense HNSW index is missing or incompatible")
    if not _index_matches(
        sparse,
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
    ):
        problems.append("sparse BM25 index is missing or incompatible")
    return problems


def _index_matches(index: dict[str, Any] | None, **expected: Any) -> bool:
    if index is None:
        return False
    flat = dict(index)
    flat.update(index.get("params") or {})
    return all(str(flat.get(key)) == str(value) for key, value in expected.items())
