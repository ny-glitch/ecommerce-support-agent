from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from app.config import load_settings
from app.db.database import Database
from app.db.knowledge import KnowledgeRepository
from app.knowledge.corpus import corpus_fingerprint, load_corpus


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data/knowledge/ch04/chunks.json"


async def initialize_knowledge(corpus_path: Path) -> None:
    chunks = load_corpus(corpus_path)
    settings = load_settings()
    if settings.database_url is None:
        raise SystemExit("DATABASE_URL is required to initialize knowledge data")

    db = Database(settings.database_url.get_secret_value())
    try:
        await db.check()
        await db.create_schema()
        repository = KnowledgeRepository(db.sessions)
        before = await repository.list_all()
        await repository.insert_seed(chunks)
        after = await repository.list_all()
        seed_ids = {chunk.id for chunk in chunks}
        imported = sum(chunk.id in seed_ids for chunk in after)
        print(
            f"knowledge import: corpus={len(chunks)}; matched={imported}; "
            f"total_before={len(before)}; total_after={len(after)}"
        )
        print(f"corpus_fingerprint: {corpus_fingerprint(chunks)}")
    finally:
        await db.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Import traced chapter 4 knowledge data")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    args = parser.parse_args()
    asyncio.run(initialize_knowledge(args.corpus))


if __name__ == "__main__":
    main()
