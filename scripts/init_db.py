from __future__ import annotations

import asyncio

from app.config import load_settings
from app.db.database import Database
from app.db.seed import seed_database


async def initialize_database() -> None:
    settings = load_settings()
    if settings.database_url is None:
        raise SystemExit("DATABASE_URL is required to initialize MySQL")

    db = Database(settings.database_url.get_secret_value())
    try:
        await db.check()
        await db.create_schema()
        await seed_database(db)
    finally:
        await db.aclose()


if __name__ == "__main__":
    asyncio.run(initialize_database())
