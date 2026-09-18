from __future__ import annotations

import argparse
import asyncio
import hashlib
import tempfile
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy.engine import make_url

from app.config import load_settings
from app.db.database import Database
from app.db.knowledge import KnowledgeRepository
from app.knowledge.indexer import KnowledgeIndexer
from app.knowledge.local_models import LocalModels
from app.knowledge.milvus_store import MilvusStore


def index_lock_path(database_url: str, collection_name: str) -> Path:
    url = make_url(database_url)
    identity = "\0".join(
        (
            url.host or "",
            str(url.port or 3306),
            url.database or "",
            collection_name,
        )
    )
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return Path(tempfile.gettempdir()) / f"ch04-knowledge-index-{suffix}.lock"


async def run_index(*, repair: bool) -> dict[str, int]:
    settings = load_settings()
    if settings.database_url is None:
        raise RuntimeError("DATABASE_URL is required to index knowledge")
    database_url = settings.database_url.get_secret_value()
    db = Database(database_url)
    store = MilvusStore(settings)
    models = LocalModels(settings)
    try:
        await db.check()
        await store.check()
        await asyncio.to_thread(models.warmup)
        indexer = KnowledgeIndexer(
            KnowledgeRepository(db.sessions),
            store,
            models,
            lock_path=index_lock_path(database_url, settings.milvus_collection),
            timeout_seconds=settings.knowledge_request_timeout_seconds,
        )
        return await indexer.run(repair=repair)
    finally:
        await models.aclose()
        await store.aclose()
        await db.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Index MySQL knowledge chunks into Milvus."
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="reconcile completed SQL rows against Milvus before indexing",
    )
    args = parser.parse_args(argv)
    counts = asyncio.run(run_index(repair=args.repair))
    print(
        "knowledge index: "
        f"indexed={counts['indexed']}; skipped={counts['skipped']}; "
        f"failed={counts['failed']}"
    )
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
