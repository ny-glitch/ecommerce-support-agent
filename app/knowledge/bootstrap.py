from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sqlalchemy import inspect

from app.config import Settings
from app.db.database import Database
from app.db.knowledge import KnowledgeRepository
from app.db.low_confidence import LowConfidenceRepository
from app.knowledge.calibration import model_manifest_fingerprint
from app.knowledge.corpus import corpus_fingerprint
from app.knowledge.local_models import LocalModels
from app.knowledge.milvus_store import MilvusStore
from app.knowledge.retrieval import KnowledgeRetriever
from app.resource_lifecycle import warmup_local_models


_KNOWLEDGE_TABLES = frozenset(
    {"knowledge_chunks", "qa_extraction_staging", "low_confidence_questions"}
)


@dataclass(frozen=True)
class KnowledgeComponents:
    repository: KnowledgeRepository
    store: MilvusStore
    local_models: LocalModels
    retriever: KnowledgeRetriever
    low_confidence: LowConfidenceRepository
    knowledge_gateway_factory: Callable[..., Any]


async def build_knowledge_components(
    settings: Settings,
    database: Database,
    model_gateway: Any,
    owned_resources: list[Any],
) -> KnowledgeComponents:
    """Validate and warm existing knowledge resources without creating them."""
    store = MilvusStore(settings)
    local_models = LocalModels(settings)
    owned_resources.extend((store, local_models))

    await check_knowledge_tables(database)
    await store.prepare_existing_collection()

    repository = KnowledgeRepository(database.sessions)
    chunks = await repository.list_all()
    if not chunks:
        raise RuntimeError("knowledge corpus is empty; run scripts/init_knowledge.py")

    # The workflow uses the approved 0.7/0.8 thresholds. Validate the fixed
    # local model snapshot directly, without making the Chapter 4 calibration
    # artifact a new workflow startup dependency.
    corpus_fingerprint(chunks)
    model_manifest_fingerprint(settings)
    await warmup_local_models(local_models)

    factory = getattr(model_gateway, "create_knowledge_gateway", None)
    if factory is None or not callable(factory):
        raise RuntimeError("model gateway cannot create the knowledge gateway")

    retriever = KnowledgeRetriever(repository, store, local_models)
    return KnowledgeComponents(
        repository=repository,
        store=store,
        local_models=local_models,
        retriever=retriever,
        low_confidence=LowConfidenceRepository(database.sessions),
        knowledge_gateway_factory=factory,
    )


async def check_knowledge_tables(database: Database) -> None:
    async with database.engine.connect() as connection:
        names = await connection.run_sync(
            lambda sync_connection: set(
                inspect(sync_connection).get_table_names()
            )
        )
    missing = sorted(_KNOWLEDGE_TABLES - names)
    if missing:
        raise RuntimeError(
            "knowledge tables are missing: "
            + ", ".join(missing)
            + "; run schema setup"
        )
