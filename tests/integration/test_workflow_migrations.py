"""Only the prevalidated isolated test DB is ever changed by these fixtures."""
import importlib.util
import pytest
from sqlalchemy import text

pytestmark = pytest.mark.asyncio


async def old_schema(db):
    # mysql_db has already cleared only the isolated database.
    async with db.engine.begin() as conn:
        cols = set((await conn.execute(text('SHOW COLUMNS FROM messages'))).scalars())
        if 'event_key' in cols:
            await conn.execute(text('ALTER TABLE messages DROP INDEX uq_messages_event'))
            await conn.execute(text('ALTER TABLE messages DROP COLUMN event_key, DROP COLUMN event_data'))
        await conn.execute(text('DROP TABLE IF EXISTS conversation_actions'))
        await conn.execute(text("INSERT INTO conversations(id,user_id,status) VALUES ('legacy','demo','human_pending')"))
        for role, body, call_id in [('user','原问题',None),('assistant','', 'old-call'),('tool','原工具结果','old-call'),('assistant','原回答',None)]:
            await conn.execute(text("INSERT INTO messages(conversation_id,turn_id,role,content,tool_call_id,turn_status) VALUES ('legacy','old-turn',:role,:body,:call_id,'completed')"),dict(role=role,body=body,call_id=call_id))
        await conn.execute(text("UPDATE messages SET tool_calls=JSON_ARRAY(JSON_OBJECT('name','query_order','args',JSON_OBJECT('order_id','1001'),'id','old-call','type','tool_call')) WHERE role='assistant' AND tool_call_id='old-call'"))
        await conn.execute(text("INSERT INTO tickets(ticket_no,conversation_id,issue_description,ticket_type,status) VALUES ('legacy-ticket','legacy','原问题','other','pending')"))


async def test_non_destructive_resumable_migration(mysql_db):
    assert importlib.util.find_spec('app.db.workflow_migrations'), 'workflow migration missing'
    from app.db.workflow_migrations import migrate_workflow
    await old_schema(mysql_db)
    async with mysql_db.engine.connect() as conn:
        before = (await conn.execute(text('SELECT id,role,content,tool_call_id,tool_calls FROM messages ORDER BY id'))).all()
    check = await migrate_workflow(mysql_db, check_only=True)
    assert check['ready'] == 0
    async with mysql_db.engine.connect() as conn:
        assert 'event_key' not in set((await conn.execute(text('SHOW COLUMNS FROM messages'))).scalars())
    first = await migrate_workflow(mysql_db)
    second = await migrate_workflow(mysql_db)
    assert first['backfilled'] == 4
    assert second['backfilled'] == 0
    async with mysql_db.engine.connect() as conn:
        assert (await conn.execute(text('SELECT id,role,content,tool_call_id,tool_calls FROM messages ORDER BY id'))).all() == before
        assert await conn.scalar(text('SELECT COUNT(*) FROM tickets')) == 1
        assert await conn.scalar(text("SELECT status FROM conversations WHERE id='legacy'")) == 'human_pending'
        assert (await conn.execute(text('SELECT event_key FROM messages ORDER BY id'))).scalars().all() == [f'legacy:{row.id}' for row in before]
    assert (await migrate_workflow(mysql_db, check_only=True))['ready'] == 1


async def test_migration_rejects_conflicting_column_before_other_ddl(mysql_db):
    assert importlib.util.find_spec('app.db.workflow_migrations'), 'workflow migration missing'
    from app.db.workflow_migrations import migrate_workflow
    from app.errors import ServiceError
    await old_schema(mysql_db)
    async with mysql_db.engine.begin() as conn:
        await conn.execute(text('ALTER TABLE messages ADD COLUMN event_key VARCHAR(12) NULL'))
    try:
        with pytest.raises(ServiceError) as err:
            await migrate_workflow(mysql_db)
        assert err.value.code == 'MIGRATION_CONFLICT'
        async with mysql_db.engine.connect() as conn:
            assert 'event_data' not in set((await conn.execute(text('SHOW COLUMNS FROM messages'))).scalars())
    finally:
        async with mysql_db.engine.begin() as conn:
            await conn.execute(text('ALTER TABLE messages DROP COLUMN event_key'))
        await migrate_workflow(mysql_db)


async def test_partial_backfill_resumes_and_wrong_index_fails_closed(mysql_db):
    from app.db.workflow_migrations import migrate_workflow
    from app.errors import ServiceError
    await old_schema(mysql_db)
    async with mysql_db.engine.begin() as conn:
        await conn.execute(text('ALTER TABLE messages ADD COLUMN event_key VARCHAR(128) NULL, ADD COLUMN event_data JSON NULL'))
        await conn.execute(text("UPDATE messages SET event_key=CONCAT('legacy:',id) ORDER BY id LIMIT 2"))
        await conn.execute(text('CREATE INDEX uq_messages_event ON messages(event_key)'))
    try:
        with pytest.raises(ServiceError):
            await migrate_workflow(mysql_db)
        async with mysql_db.engine.connect() as conn:
            assert await conn.scalar(text('SELECT COUNT(*) FROM messages WHERE event_key IS NULL')) == 2
    finally:
        async with mysql_db.engine.begin() as conn:
            await conn.execute(text('DROP INDEX uq_messages_event ON messages'))
        assert (await migrate_workflow(mysql_db))['backfilled'] == 2
    async with mysql_db.engine.connect() as conn:
        assert await conn.scalar(text('SELECT COUNT(*) FROM messages')) == 4


async def test_action_schema_conflict_is_detected(mysql_db):
    from app.db.workflow_migrations import migrate_workflow
    from app.errors import ServiceError
    async with mysql_db.engine.begin() as conn:
        await conn.execute(text('ALTER TABLE conversation_actions MODIFY COLUMN ticket_no VARCHAR(63) NOT NULL'))
    try:
        for check_only in (True, False):
            with pytest.raises(ServiceError):
                await migrate_workflow(mysql_db, check_only=check_only)
    finally:
        async with mysql_db.engine.begin() as conn:
            await conn.execute(text('ALTER TABLE conversation_actions MODIFY COLUMN ticket_no VARCHAR(64) NOT NULL'))


async def test_action_extra_unique_constraint_is_a_conflict(mysql_db):
    from app.db.workflow_migrations import migrate_workflow
    from app.errors import ServiceError
    async with mysql_db.engine.begin() as conn:
        await conn.execute(text('CREATE UNIQUE INDEX unexpected_unique ON conversation_actions(conversation_id)'))
    try:
        with pytest.raises(ServiceError):
            await migrate_workflow(mysql_db, check_only=True)
    finally:
        async with mysql_db.engine.begin() as conn:
            await conn.execute(text('DROP INDEX unexpected_unique ON conversation_actions'))
