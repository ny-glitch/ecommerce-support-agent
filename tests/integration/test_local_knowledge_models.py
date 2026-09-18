from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import pytest
import pytest_asyncio

from app.config import Settings
from app.knowledge.corpus import load_corpus
from app.knowledge.local_models import EMBED_MODEL_REVISION, RERANK_MODEL_REVISION, LocalModels
from app.knowledge.text import embedding_text


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def local_models(request: pytest.FixtureRequest) -> LocalModels:
    root = Path(".cache/ch04/models")
    required = [
        root / "bge-m3" / "pytorch_model.bin",
        root / "bge-reranker-v2-m3" / "model.safetensors",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        message = f"local knowledge models are missing: {', '.join(missing)}"
        if request.config.getoption("--require-local-models"):
            pytest.fail(message)
        pytest.skip(message)

    settings = Settings(
        _env_file=None,
        llm_base_url="https://api.example.com/v1",
        llm_model="test-model",
        llm_api_key="test-key",
        knowledge_models_dir=root,
        knowledge_batch_size=4,
    )
    models = LocalModels(settings)
    models.warmup()
    yield models
    await models.aclose()


@pytest.mark.asyncio(loop_scope="session")
async def test_real_models_embed_rerank_and_time_fifty_corpus_candidates(
    local_models: LocalModels,
) -> None:
    chunks = load_corpus(Path("data/knowledge/ch04/chunks.json"))
    candidates = [embedding_text(chunk) for chunk in chunks[:50]]
    deadline = time.monotonic() + 240

    dense_started = time.monotonic()
    vectors = await local_models.embed(candidates[:2], deadline=deadline)
    dense_seconds = time.monotonic() - dense_started

    query = chunks[0].questions
    comparison_scores = await local_models.score(
        query,
        [candidates[0], "火星上的天气和本地充电器售后政策无关。"],
        deadline=deadline,
    )
    rerank_started = time.monotonic()
    scores = await local_models.score(query, candidates, deadline=deadline)
    rerank_seconds = time.monotonic() - rerank_started

    assert len(vectors) == 2
    assert all(len(vector) == 1024 for vector in vectors)
    assert len(scores) == 50
    assert comparison_scores[0] > comparison_scores[1]
    assert scores[0] > scores[-1]

    print(
        json.dumps(
            {
                "dense_revision": EMBED_MODEL_REVISION,
                "rerank_revision": RERANK_MODEL_REVISION,
                "dense_dimension": len(vectors[0]),
                "dense_two_seconds": dense_seconds,
                "related_score": comparison_scores[0],
                "unrelated_score": comparison_scores[1],
                "rerank_50_seconds": rerank_seconds,
                "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            },
            sort_keys=True,
        )
    )
