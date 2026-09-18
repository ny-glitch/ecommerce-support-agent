from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.knowledge.local_models import InputTooLongError, LocalModels, ModelOutputError
from scripts.prepare_knowledge_models import MODEL_SPECS, prepare_models


class FakeTokenizer:
    model_max_length = 8

    def __call__(
        self,
        text: str | list[str],
        text_pair: str | None = None,
        *,
        add_special_tokens: bool = True,
        truncation: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        assert truncation is False
        if isinstance(text, list):
            return {
                "input_ids": [
                    self(item, add_special_tokens=add_special_tokens)["input_ids"]
                    for item in text
                ]
            }
        ids = list(range(len(text.split())))
        if text_pair is not None:
            ids += list(range(len(text_pair.split())))
        if add_special_tokens:
            ids += [100, 101] if text_pair is None else [100, 101, 102, 103]
        return {"input_ids": ids}


class FakeEmbedder:
    def __init__(self, vectors: list[list[float]] | None = None) -> None:
        self.tokenizer = FakeTokenizer()
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.vectors = vectors

    def encode(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
        self.calls.append((list(texts), kwargs))
        vectors = (
            self.vectors
            if self.vectors is not None
            else [[float(i)] * 1024 for i, _ in enumerate(texts, 1)]
        )
        return {"dense_vecs": vectors[: len(texts)]}


class FakeReranker:
    def __init__(self, scores: float | list[float] = 0.75) -> None:
        self.tokenizer = FakeTokenizer()
        self.calls: list[tuple[list[list[str]], dict[str, Any]]] = []
        self.scores = scores

    def compute_score(self, pairs: list[list[str]], **kwargs: Any) -> Any:
        self.calls.append((pairs, kwargs))
        if isinstance(self.scores, list):
            return self.scores[: len(pairs)]
        return self.scores if len(pairs) == 1 else [self.scores] * len(pairs)


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "_env_file": None,
        "llm_base_url": "https://api.example.com/v1",
        "llm_model": "test-model",
        "llm_api_key": "test-key",
        "knowledge_models_dir": tmp_path,
        "knowledge_batch_size": 2,
        "knowledge_worker_queue_size": 2,
    }
    values.update(overrides)
    for name in ("bge-m3", "bge-reranker-v2-m3"):
        model_dir = tmp_path / name
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "bge-m3" / "pytorch_model.bin").touch()
    (tmp_path / "bge-m3" / "tokenizer.json").touch()
    (tmp_path / "bge-m3" / "colbert_linear.pt").touch()
    (tmp_path / "bge-m3" / "sparse_linear.pt").touch()
    (tmp_path / "bge-reranker-v2-m3" / "model.safetensors").touch()
    (tmp_path / "bge-reranker-v2-m3" / "tokenizer.json").touch()
    return Settings(**values)


def make_models(
    tmp_path: Path,
    *,
    embedder: FakeEmbedder | None = None,
    reranker: FakeReranker | None = None,
) -> tuple[LocalModels, FakeEmbedder, FakeReranker]:
    embedder = embedder or FakeEmbedder()
    reranker = reranker or FakeReranker()
    models = LocalModels(
        make_settings(tmp_path),
        embedder_factory=lambda *_args, **_kwargs: embedder,
        reranker_factory=lambda *_args, **_kwargs: reranker,
    )
    models.warmup()
    return models, embedder, reranker


async def test_embed_batches_dense_only_and_returns_plain_finite_vectors(
    tmp_path: Path,
) -> None:
    models, embedder, _ = make_models(tmp_path)
    try:
        result = await models.embed(
            ["one", "two", "three"], deadline=time.monotonic() + 2
        )
    finally:
        await models.aclose()

    assert len(result) == 3
    assert all(len(vector) == 1024 for vector in result)
    assert all(math.isfinite(value) for vector in result for value in vector)
    assert [call[0] for call in embedder.calls] == [["one", "two"], ["three"]]
    assert all(
        call[1]
        == {
            "batch_size": 2,
            "max_length": 8192,
            "return_dense": True,
            "return_sparse": False,
            "return_colbert_vecs": False,
        }
        for call in embedder.calls
    )


async def test_embed_rejects_oversized_input_before_model_call(tmp_path: Path) -> None:
    models, embedder, _ = make_models(tmp_path)
    try:
        with pytest.raises(InputTooLongError, match="embedding input") as exc_info:
            await models.embed(
                ["one two three four five six seven"],
                deadline=time.monotonic() + 2,
            )
        assert exc_info.value.input_index == 0
        assert exc_info.value.token_count == 9
        assert exc_info.value.token_limit == 8
        assert embedder.calls == []
    finally:
        await models.aclose()


async def test_score_checks_query_and_combined_pair_without_truncation(
    tmp_path: Path,
) -> None:
    models, _, reranker = make_models(tmp_path)
    try:
        with pytest.raises(InputTooLongError, match="reranker query"):
            await models.score(
                "one two three four five six seven eight nine",
                ["short"],
                deadline=time.monotonic() + 2,
            )
        with pytest.raises(InputTooLongError, match="reranker pair"):
            await models.score(
                "one two",
                ["three four five"],
                deadline=time.monotonic() + 2,
            )
        assert reranker.calls == []
    finally:
        await models.aclose()


async def test_score_normalizes_scalar_and_passes_full_length_limits(
    tmp_path: Path,
) -> None:
    models, _, reranker = make_models(tmp_path, reranker=FakeReranker(0.25))
    try:
        assert await models.score(
            "query", ["passage"], deadline=time.monotonic() + 2
        ) == [0.25]
    finally:
        await models.aclose()

    assert reranker.calls == [
        (
            [["query", "passage"]],
            {
                "batch_size": 2,
                "query_max_length": 8192,
                "max_length": 8192,
                "normalize": True,
            },
        )
    ]


@pytest.mark.parametrize(
    ("embedder", "message"),
    [
        (FakeEmbedder([[1.0] * 3]), "dimension"),
        (FakeEmbedder([[float("nan")] * 1024]), "finite"),
        (FakeEmbedder([]), "count"),
    ],
)
async def test_embed_rejects_invalid_model_output(
    tmp_path: Path, embedder: FakeEmbedder, message: str
) -> None:
    models, _, _ = make_models(tmp_path, embedder=embedder)
    try:
        with pytest.raises(ModelOutputError, match=message):
            await models.embed(["valid"], deadline=time.monotonic() + 2)
    finally:
        await models.aclose()


def test_warmup_requires_local_artifacts_and_never_calls_factory(tmp_path: Path) -> None:
    called = False

    def factory(*_args: Any, **_kwargs: Any) -> FakeEmbedder:
        nonlocal called
        called = True
        return FakeEmbedder()

    models = LocalModels(
        make_settings(tmp_path / "settings").model_copy(
            update={"knowledge_models_dir": tmp_path / "missing"}
        ),
        embedder_factory=factory,
        reranker_factory=lambda *_args, **_kwargs: FakeReranker(),
    )

    with pytest.raises(FileNotFoundError, match="prepare_knowledge_models"):
        models.warmup()
    assert called is False


def test_knowledge_settings_defaults_and_secret_errors_are_safe() -> None:
    settings = Settings(
        _env_file=None,
        llm_base_url="https://api.example.com/v1",
        llm_model="test-model",
        llm_api_key="test-key",
    )
    assert settings.milvus_uri == "http://127.0.0.1:19530"
    assert settings.milvus_collection == "knowledge"
    assert settings.milvus_token is None
    assert settings.knowledge_models_dir == Path(".cache/ch04/models")
    assert settings.knowledge_calibration_path == Path(".cache/ch04/calibration.json")
    assert settings.knowledge_request_timeout_seconds == 240
    assert settings.knowledge_batch_size == 4
    assert settings.knowledge_worker_queue_size == 4

    marker = "never-print-this-milvus-token"
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            _env_file=None,
            llm_base_url="https://api.example.com/v1",
            llm_model="test-model",
            llm_api_key="test-key",
            milvus_token=marker,
            knowledge_batch_size=0,
        )
    assert marker not in str(exc_info.value)


def _write_snapshot(root: Path, name: str, revision: str) -> None:
    spec = MODEL_SPECS[name]
    model_dir = root / name
    metadata_dir = model_dir / ".cache" / "huggingface" / "download"
    metadata_dir.mkdir(parents=True)
    for filename in spec.required_files:
        (model_dir / filename).write_bytes(f"{name}:{filename}".encode())
        (metadata_dir / f"{filename}.metadata").write_text(
            f"{revision}\netag\n0\n", encoding="utf-8"
        )


def test_prepare_models_validates_existing_revisions_and_writes_manifest(
    tmp_path: Path,
) -> None:
    for name, spec in MODEL_SPECS.items():
        _write_snapshot(tmp_path, name, spec.revision)

    def forbidden_download(**_kwargs: Any) -> str:
        raise AssertionError("valid local snapshots must not be downloaded again")

    manifest_path = prepare_models(
        tmp_path,
        cache_dir=tmp_path / "hub-cache",
        downloader=forbidden_download,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest_path == tmp_path / "manifest.json"
    assert {item["name"] for item in manifest["models"]} == set(MODEL_SPECS)
    assert {
        item["revision"] for item in manifest["models"]
    } == {spec.revision for spec in MODEL_SPECS.values()}
    assert all(
        file["sha256"] and file["bytes"] > 0
        for item in manifest["models"]
        for file in item["files"]
    )


def test_prepare_models_downloads_only_fixed_root_artifacts(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def local_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        repo_name = Path(kwargs["local_dir"]).name
        spec = MODEL_SPECS[repo_name]
        _write_snapshot(tmp_path, repo_name, spec.revision)
        return str(kwargs["local_dir"])

    prepare_models(
        tmp_path,
        cache_dir=tmp_path / "hub-cache",
        downloader=local_download,
    )

    assert len(calls) == 2
    for call in calls:
        spec = MODEL_SPECS[Path(call["local_dir"]).name]
        assert call["repo_id"] == spec.repo_id
        assert call["revision"] == spec.revision
        assert call["allow_patterns"] == list(spec.required_files)
        assert "onnx/**" in call["ignore_patterns"]
        assert call["token"] is False
