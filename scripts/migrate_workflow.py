"""Explicit migration command; stop all old writers before applying to a database."""
import argparse
import asyncio
import json

from app.config import Settings
from app.db.database import Database
from app.db.workflow_migrations import migrate_workflow


async def run(check_only: bool) -> int:
    database = None
    try:
        settings = Settings()
        database = Database(settings.database_url.get_secret_value())
        result = await migrate_workflow(database, check_only=check_only)
        print(json.dumps(result))
        return 0 if result['ready'] else 1
    except Exception:
        print('工作流迁移未完成；请核对数据库连接及现有结构。')
        return 1
    finally:
        if database is not None:
            await database.aclose()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args.check_only)))
