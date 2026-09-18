from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from app.config import Settings
from app.knowledge.worker import InferenceDeadlineExceeded, InferenceWorker


EMBED_MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
RERANK_MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
MODEL_MAX_LENGTH = 8192
DENSE_DIMENSION = 1024
_MODEL_FILES = {
    "bge-m3": (
        "config.json", "tokenizer.json", "pytorch_model.bin",
        "colbert_linear.pt", "sparse_linear.pt",
    ),
    "bge-reranker-v2-m3": (
        "config.json", "tokenizer.json", "model.safetensors",
    ),
}


class _Tokenizer(Protocol):
    model_max_length: int

    def __call__(self, text: Any, text_pair: Any = None, **kwargs: Any) -> Any: ...


class _Embedder(Protocol):
    tokenizer: _Tokenizer

    def encode(self, texts: list[str], **kwargs: Any) -> Any: ...


class _Reranker(Protocol):
    tokenizer: _Tokenizer

    def compute_score(self, pairs: list[list[str]], **kwargs: Any) -> Any: ...


class InputTooLongError(ValueError):
    def __init__(
        self,
        *,
        input_kind: Literal["embedding", "reranker_query", "reranker_pair"],
        input_index: int | None,
        token_count: int,
        token_limit: int,
    ) -> None:
        self.input_kind = input_kind
        self.input_index = input_index
        self.token_count = token_count
        self.token_limit = token_limit
        label = {
            "embedding": "embedding input",
            "reranker_query": "reranker query",
            "reranker_pair": "reranker pair",
        }[input_kind]
        index = "" if input_index is None else f" {input_index}"
        super().__init__(
            f"{label}{index} has {token_count} tokens; "
            f"maximum is {token_limit}"
        )


class ModelOutputError(RuntimeError):
    pass


class LocalModels:
    def __init__(
        self,
        settings: Settings,
        *,
        embedder_factory: Callable[..., _Embedder] | None = None,
        reranker_factory: Callable[..., _Reranker] | None = None,
    ) -> None:
        self._settings = settings
        self._embedder_factory = embedder_factory or _default_embedder_factory
        self._reranker_factory = reranker_factory or _default_reranker_factory
        self._embedder: _Embedder | None = None
        self._reranker: _Reranker | None = None
        self._worker = InferenceWorker(settings.knowledge_worker_queue_size)

    def warmup(self) -> None:
        if self._embedder is not None and self._reranker is not None:
            return
        root = self._settings.knowledge_models_dir
        _require_local_artifacts(root)
        self._embedder = self._embedder_factory(
            str(root / "bge-m3"),
            devices=["cpu"],
            use_fp16=False,
            batch_size=self._settings.knowledge_batch_size,
            query_max_length=MODEL_MAX_LENGTH,
            passage_max_length=MODEL_MAX_LENGTH,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        self._reranker = self._reranker_factory(
            str(root / "bge-reranker-v2-m3"),
            devices=["cpu"],
            use_fp16=False,
            batch_size=self._settings.knowledge_batch_size,
            query_max_length=MODEL_MAX_LENGTH,
            max_length=MODEL_MAX_LENGTH,
            normalize=True,
        )

    async def embed(
        self, texts: list[str], *, deadline: float
    ) -> list[list[float]]:
        embedder, _ = self._loaded_models()
        if not texts:
            return []
        _check_deadline(deadline)
        _validate_embedding_lengths(embedder.tokenizer, texts)

        vectors: list[list[float]] = []
        batch_size = self._settings.knowledge_batch_size
        for start in range(0, len(texts), batch_size):
            _check_deadline(deadline)
            batch = texts[start : start + batch_size]

            def encode_batch(batch: list[str] = batch) -> Any:
                return embedder.encode(
                    batch,
                    batch_size=batch_size,
                    max_length=MODEL_MAX_LENGTH,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )

            output = await self._worker.run(encode_batch, deadline=deadline)
            dense = output.get("dense_vecs") if isinstance(output, dict) else None
            if hasattr(dense, "tolist"):
                dense = dense.tolist()
            if not isinstance(dense, list) or len(dense) != len(batch):
                raise ModelOutputError("embedding output count does not match input")
            vectors.extend(_validate_vectors(dense))
        if len(vectors) != len(texts):
            raise ModelOutputError("embedding output count does not match input")
        return vectors

    async def score(
        self, query: str, texts: list[str], *, deadline: float
    ) -> list[float]:
        _, reranker = self._loaded_models()
        if not texts:
            return []
        _check_deadline(deadline)
        _validate_reranker_lengths(reranker.tokenizer, query, texts)

        scores: list[float] = []
        batch_size = self._settings.knowledge_batch_size
        for start in range(0, len(texts), batch_size):
            _check_deadline(deadline)
            batch = texts[start : start + batch_size]
            pairs = [[query, text] for text in batch]

            def score_batch(pairs: list[list[str]] = pairs) -> Any:
                return reranker.compute_score(
                    pairs,
                    batch_size=batch_size,
                    query_max_length=MODEL_MAX_LENGTH,
                    max_length=MODEL_MAX_LENGTH,
                    normalize=True,
                )

            output = await self._worker.run(score_batch, deadline=deadline)
            if isinstance(output, (int, float)) and len(batch) == 1:
                output = [output]
            if hasattr(output, "tolist"):
                output = output.tolist()
            if not isinstance(output, list) or len(output) != len(batch):
                raise ModelOutputError("reranker output count does not match input")
            for value in output:
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ModelOutputError("reranker output must contain finite scores")
                scores.append(float(value))
        return scores

    async def aclose(self) -> None:
        await self._worker.aclose()
        self._embedder = None
        self._reranker = None

    def _loaded_models(self) -> tuple[_Embedder, _Reranker]:
        if self._embedder is None or self._reranker is None:
            raise RuntimeError("local models are not warmed up")
        return self._embedder, self._reranker


def _default_embedder_factory(*args: Any, **kwargs: Any) -> _Embedder:
    from FlagEmbedding import BGEM3FlagModel

    return cast(_Embedder, BGEM3FlagModel(*args, **kwargs))


def _default_reranker_factory(*args: Any, **kwargs: Any) -> _Reranker:
    from FlagEmbedding import FlagReranker

    return cast(_Reranker, FlagReranker(*args, **kwargs))


def _require_local_artifacts(root: Path) -> None:
    missing = [
        str(root / model / filename)
        for model, filenames in _MODEL_FILES.items()
        for filename in filenames
        if not (root / model / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "local model artifacts are missing; run scripts/prepare_knowledge_models.py: "
            + ", ".join(missing)
        )


def _token_limit(tokenizer: _Tokenizer) -> int:
    configured = getattr(tokenizer, "model_max_length", MODEL_MAX_LENGTH)
    if not isinstance(configured, int) or configured <= 0:
        return MODEL_MAX_LENGTH
    return min(configured, MODEL_MAX_LENGTH)


def _input_ids(tokenizer: _Tokenizer, text: str, **kwargs: Any) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=kwargs.pop("add_special_tokens", True),
        truncation=False,
        **kwargs,
    )
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def _validate_embedding_lengths(tokenizer: _Tokenizer, texts: list[str]) -> None:
    limit = _token_limit(tokenizer)
    for index, text in enumerate(texts):
        count = len(_input_ids(tokenizer, text))
        if count > limit:
            raise InputTooLongError(
                input_kind="embedding",
                input_index=index,
                token_count=count,
                token_limit=limit,
            )


def _validate_reranker_lengths(
    tokenizer: _Tokenizer, query: str, texts: list[str]
) -> None:
    limit = _token_limit(tokenizer)
    query_count = len(_input_ids(tokenizer, query, add_special_tokens=False))
    if query_count > limit:
        raise InputTooLongError(
            input_kind="reranker_query",
            input_index=None,
            token_count=query_count,
            token_limit=limit,
        )
    for index, text in enumerate(texts):
        pair_count = len(_input_ids(tokenizer, query, text_pair=text))
        if pair_count > limit:
            raise InputTooLongError(
                input_kind="reranker_pair",
                input_index=index,
                token_count=pair_count,
                token_limit=limit,
            )


def _validate_vectors(values: list[Any]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for value in values:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, list) or len(value) != DENSE_DIMENSION:
            raise ModelOutputError(f"embedding dimension must be {DENSE_DIMENSION}")
        vector = [float(item) for item in value]
        if not all(math.isfinite(item) for item in vector):
            raise ModelOutputError("embedding values must be finite")
        vectors.append(vector)
    return vectors


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise InferenceDeadlineExceeded("inference deadline exceeded")
