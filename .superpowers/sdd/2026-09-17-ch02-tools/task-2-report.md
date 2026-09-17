# Task 2 implementation report

## Status

DONE

## What changed

- Added immutable `TurnRef` and `StoredTurn` repository contracts.
- Added `ConversationRepository` with user-scoped conversation lookup, conversation creation, pending turn writes, paired tool call/result writes, atomic turn completion, strict completed-turn history restoration, and full ordered audit output.
- Added `FaqRepository.search()` with literal escaped `LIKE` matching across question, answer, and category, ordered and limited to five results.
- Added `TicketRepository.create_once()` with user ownership checks, field-consistent idempotency, atomic ticket/conversation status updates, unique-key race recovery, and post-error verification using a fresh session.
- Added process-local `SessionGuard` while preserving the existing `SessionStore` for the not-yet-migrated runtime.
- Added reusable `repos` and `new_turn` integration fixtures in `tests/integration/conftest.py` for downstream task tests.
- Added real MySQL repository coverage for tool-pair restoration, literal FAQ misses, ticket idempotency/conflicts/races/rollback, history filtering/limits/restart recovery, user isolation, audit retention, and malformed completed groups.

## TDD evidence

### RED

Guard command:

```text
.venv/bin/python -m pytest tests/test_session_guard.py -q
```

Relevant result before implementation:

```text
FAILED test_guard_rejects_a_second_request_for_the_same_conversation
Failed: DID NOT RAISE ServiceError
FAILED test_guard_limits_active_requests_and_release_restores_capacity
Failed: DID NOT RAISE ServiceError
2 failed
```

Core real-MySQL command, repeated while advancing the behavior one method at a time:

```text
.venv/bin/python -m pytest tests/integration/test_repositories.py::test_completed_turn_preserves_tool_pair --require-mysql -q
```

The test successively failed at `start_turn`, `append_call`, `append_result`, `history`, and `finish_turn` with `NotImplementedError`. These were expected behavioral gaps after the minimal importable interfaces were established. After each failure, only the reached behavior was implemented before rerunning.

Literal FAQ command:

```text
.venv/bin/python -m pytest tests/integration/test_repositories.py::test_literal_faq_search_escapes_wildcards --require-mysql -q
```

Relevant result before implementation:

```text
FAILED test_literal_faq_search_escapes_wildcards
NotImplementedError: FaqRepository.search
1 failed
```

Ticket command:

```text
.venv/bin/python -m pytest tests/integration/test_repositories.py::test_ticket_retry_returns_one_ticket_and_marks_conversation --require-mysql -q
```

Relevant result before implementation:

```text
FAILED test_ticket_retry_returns_one_ticket_and_marks_conversation
NotImplementedError: TicketRepository.create_once
1 failed
```

Malformed-history regression command:

```text
.venv/bin/python -m pytest tests/integration/test_repositories.py::test_malformed_completed_tool_group_is_audited_but_not_returned --require-mysql -q
```

Relevant result before the restoration guard:

```text
FAILED test_malformed_completed_tool_group_is_audited_but_not_returned
TypeError: tool_call() missing 2 required keyword-only arguments: 'name' and 'args'
1 failed
```

This proved that paired IDs alone were insufficient and malformed stored JSON could escape as a LangChain construction error instead of being excluded from model history.

### GREEN

Focused Task 2 command:

```text
.venv/bin/python -m pytest tests/test_session_guard.py tests/integration/test_repositories.py --require-mysql -q
```

Result:

```text
............ [100%]
12 passed in 0.93s
```

Full-suite command:

```text
.venv/bin/python -m pytest --require-mysql -q
```

Result:

```text
112 passed, 1 warning in 12.92s
```

The warning is the pre-existing Starlette reference to the deprecated AnyIO `BlockingPortal` alias recorded by Task 1; no Task 2 test or implementation introduced a warning.

Additional checks:

```text
.venv/bin/python -m compileall -q app tests
git diff --check
```

Both completed successfully with no output.

## Files changed

- `app/db/contracts.py`
- `app/db/conversations.py`
- `app/db/faq.py`
- `app/db/tickets.py`
- `app/sessions.py`
- `tests/integration/conftest.py`
- `tests/integration/test_repositories.py`
- `tests/test_session_guard.py`
- `.superpowers/sdd/2026-09-17-ch02-tools/task-2-report.md`

## Self-review

- Confirmed every repository operation uses its own `async_sessionmaker.begin()` context and returns DTOs/messages after materializing required data inside the session.
- Confirmed a failed ticket transaction is never reused for verification; the code opens a fresh session before checking the durable ticket outcome.
- Confirmed ticket insertion and `human_pending` conversation status update share one transaction, including the rollback test that forces a real MySQL write error.
- Confirmed history limits whole valid turns, returns them chronologically, excludes all non-completed or malformed groups, and retains those rows in audit.
- Confirmed wildcard characters are escaped by SQLAlchemy `contains(..., autoescape=True)` and the intentional `邮费` miss remains.
- Confirmed the existing `SessionStore` and unrelated controller-owned notes/plan edits were not changed or staged as part of this task.

## Concerns

No Task 2 correctness concerns. The full suite retains one known dependency deprecation warning from Starlette/AnyIO.
