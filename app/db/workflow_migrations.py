"""Resumable MySQL DDL. MySQL commits DDL steps independently; no atomicity claim.

Run with writes stopped. Validate existing definitions before making any change.
A crash can leave nullable event_key or partial backfill; rerunning resumes safely.
"""
from sqlalchemy import JSON, String, inspect, text

from app.db.database import Database
from app.db.workflow_models import ConversationAction
from app.errors import ServiceError

EVENT_INDEX = 'uq_messages_event'
EVENT_COLUMNS = ['conversation_id', 'turn_id', 'event_key']


def _conflict():
    return ServiceError('MIGRATION_CONFLICT', '已有工作流结构或数据与迁移定义冲突', 409)


def _inspect_schema(conn):
    inspector = inspect(conn)
    if not inspector.has_table('messages') or not inspector.has_table('conversations'):
        raise _conflict()
    columns = {c['name']: c for c in inspector.get_columns('messages')}
    key, data = columns.get('event_key'), columns.get('event_data')
    if key is not None and (not isinstance(key['type'], String) or key['type'].length != 128 or key.get('default') is not None):
        raise _conflict()
    if data is not None and (not isinstance(data['type'], JSON) or not data['nullable']):
        raise _conflict()
    indexes = inspector.get_indexes('messages')
    event_index = next((i for i in indexes if i['name'] == EVENT_INDEX), None)
    if event_index and (not event_index['unique'] or event_index['column_names'] != EVENT_COLUMNS
                        or event_index.get('dialect_options', {}).get('mysql_length')):
        raise _conflict()
    # A differently named UNIQUE key must not silently prohibit multi-step events.
    for index in indexes:
        if index['unique'] and index['name'] != EVENT_INDEX:
            if index['column_names'] != ['id']:
                raise _conflict()
    has_actions = inspector.has_table('conversation_actions')
    if has_actions:
        table = ConversationAction.__table__
        actual = {c['name']: c for c in inspector.get_columns(table.name)}
        if set(actual) != set(table.columns.keys()):
            raise _conflict()
        for expected in table.columns:
            found = actual[expected.name]
            expected_type = expected.type.compile(dialect=conn.dialect)
            actual_type = found['type'].compile(dialect=conn.dialect)
            if expected_type != actual_type or expected.nullable != found['nullable']:
                raise _conflict()
        if inspector.get_pk_constraint(table.name)['constrained_columns'] != ['action_id']:
            raise _conflict()
        indexes = {i['name']:i for i in inspector.get_indexes(table.name)}
        if {i['name'] for i in indexes.values() if i['unique']} != {i.name for i in table.indexes if i.unique}:
            raise _conflict()
        for expected in table.indexes:
            actual_index = indexes.get(expected.name)
            if (not actual_index or not actual_index['unique']
                    or actual_index['column_names'] != [c.name for c in expected.columns]
                    or actual_index.get('dialect_options', {}).get('mysql_length')):
                raise _conflict()
        foreign = inspector.get_foreign_keys(table.name)
        if not any(f['constrained_columns'] == ['conversation_id'] and f['referred_table'] == 'conversations'
                   and f['referred_columns'] == ['id'] and not f.get('options') for f in foreign):
            raise _conflict()
        # Compare normalized reflected CHECK SQL with what this dialect emits.
        from sqlalchemy.schema import CheckConstraint
        def normalized(value):
            import re
            value = re.sub(r"_(?:utf8mb4|utf8mb3|utf8|latin1)", '', value.lower())
            return re.sub(r'[\s`()]+', '', value)
        checks = {c['name']:normalized(c['sqltext']) for c in inspector.get_check_constraints(table.name)}
        for constraint in table.constraints:
            if isinstance(constraint, CheckConstraint) and checks.get(constraint.name) != normalized(str(constraint.sqltext)):
                raise _conflict()
    return key, data, event_index, has_actions


async def migrate_workflow(database: Database, *, check_only: bool = False) -> dict[str, int]:
    """Return ready/backfilled; check_only validates and never executes any DDL."""
    async with database.engine.connect() as conn:
        if conn.dialect.name != 'mysql':
            raise _conflict()
        key, data, index, actions = await conn.run_sync(_inspect_schema)
        if key is not None:
            # Reject non-null bad keys before ALTER; compare with MySQL uniqueness semantics.
            bad = await conn.scalar(text("SELECT COUNT(*) FROM messages WHERE event_key = ''"))
            duplicates = await conn.scalar(text('SELECT COUNT(*) FROM (SELECT 1 FROM messages WHERE event_key IS NOT NULL GROUP BY conversation_id,turn_id,event_key HAVING COUNT(*) > 1) d'))
            collision = await conn.scalar(text("SELECT COUNT(*) FROM messages a JOIN messages b ON a.conversation_id=b.conversation_id AND a.turn_id=b.turn_id AND b.event_key=CONCAT('legacy:',a.id) WHERE a.event_key IS NULL"))
            if bad or duplicates or collision:
                raise _conflict()
        ready = key is not None and not key['nullable'] and data is not None and index is not None and actions
        if check_only:
            return {'ready':int(ready), 'backfilled':0}
    if key is None:
        async with database.engine.begin() as conn:
            await conn.execute(text('ALTER TABLE messages ADD COLUMN event_key VARCHAR(128) NULL'))
    if data is None:
        async with database.engine.begin() as conn:
            await conn.execute(text('ALTER TABLE messages ADD COLUMN event_data JSON NULL'))
    backfilled = 0
    while True:
        async with database.engine.begin() as conn:
            ids = list((await conn.execute(text('SELECT id FROM messages WHERE event_key IS NULL ORDER BY id LIMIT 500'))).scalars())
            if not ids:
                break
            result = await conn.execute(text("UPDATE messages SET event_key=CONCAT('legacy:',id) WHERE event_key IS NULL AND id >= :lo AND id <= :hi"), {'lo':ids[0],'hi':ids[-1]})
            backfilled += result.rowcount
    async with database.engine.begin() as conn:
        if await conn.scalar(text('SELECT COUNT(*) FROM messages WHERE event_key IS NULL')):
            raise _conflict()
        if key is None or key['nullable']:
            await conn.execute(text('ALTER TABLE messages MODIFY COLUMN event_key VARCHAR(128) NOT NULL'))
        if index is None:
            await conn.execute(text('CREATE UNIQUE INDEX uq_messages_event ON messages(conversation_id,turn_id,event_key)'))
        if not actions:
            await conn.run_sync(lambda sync: ConversationAction.__table__.create(sync))
    return {'ready':1, 'backfilled':backfilled}
