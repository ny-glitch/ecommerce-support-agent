# Task 10 implementation report

Status: DONE. Production assembly, startup validation, explicit legacy injection compatibility, documentation, package validation, and focused isolated-database verification are complete. No merge, push, server cutover, dependency change, model download, or external model call was performed.

## Implementation

- Added `app/knowledge/bootstrap.py` with the public frozen `KnowledgeComponents(repository, store, local_models, retriever, low_confidence, knowledge_gateway_factory)` and `build_knowledge_components(settings, database, model_gateway, owned_resources)`. It owns the Milvus/local-model resources before checking them, verifies the existing knowledge tables and Milvus collection/schema, rejects an empty corpus, validates the fixed local model manifest, warms existing local models with the established cancellation drain, and returns the shared retriever/repositories/factory. It does not create schema/indexes and does not add a Chapter 5 runtime evaluation-artifact gate.
- The old Chapter 4 calibration/evaluation code is unchanged. The new workflow bootstrap validates the model manifest directly and consumes the approved Settings thresholds 0.7/0.8 through `KnowledgeStage`; it does not use the old calibrated threshold as Chapter 5 evidence.
- Added `app/workflow/bootstrap.py` with `build_workflow_dependencies(settings, database, model_gateway, components)`. It creates repositories, action ownership, bounded tool executor, runtime-bound workflow and normalization gateway factories, request-budget reservation/usage hooks, `KnowledgeStage`, and the exact `WorkflowDependencies` graph input. Installed concrete gateway signatures were inspected before use.
- Replaced the normal no-injection branch in `app.main.create_app` with the required order: database connection/check, `migrate_workflow(..., check_only=True)`, knowledge component checks/warmup, `CheckpointStore.open()`, `CheckpointStore.check()`, workflow dependency build, graph compile, `WorkflowChatService`, and `ActionService`. The branch never calls MySQL `create_schema`, migration DDL, checkpoint `setup`, or model inference.
- Normal startup no longer falls back to legacy `ChatService` based on gateway capabilities. Explicit `chat_service` and explicit old `knowledge_dependencies` remain supported; `workflow_dependencies` is now an explicit injection. A chat-service-only injection with no gateway creates no production model/database/checkpoint/knowledge resource. Explicitly supplied gateways retain the previous owner/close behavior.
- Ownership append order makes reverse shutdown close `WorkflowChatService` first, then checkpoint, local models, Milvus, MySQL, and shared model/HTTP. Existing repeat-cancellation cleanup remains in `close_resources`; no lifecycle helper change was needed.
- Registered the actions router alongside chat and the existing knowledge/source router, while retaining the original after-sales extraction endpoint.
- `.env.example` now exposes the already validated workflow call/decision/model-budget defaults. README documents Docker dependencies -> MySQL migration/check -> explicit checkpoint setup -> local corpus/model/index preparation -> single-worker port 8002 startup. It states ordinary startup is validation-only, preserves 8001 until real acceptance, and labels the old two-call/single-tool/LIKE discussion as historical Chapter 2 behavior.

## TDD evidence

### Initial production assembly RED

Command:

```sh
.venv/bin/python -m pytest tests/integration/test_workflow_startup.py -q --tb=short
```

Result before implementation: exit 1, `1 failed, 5 errors`. Every case stopped at the missing `app.main.CheckpointStore` assembly boundary. This was the expected failure because production workflow/checkpoint startup did not exist.

After the first minimal assembly, the same command produced `1 failed, 5 passed`: the remaining failure showed workflow dependencies were built before checkpoint `open/check`, contrary to the approved exact sequence. Moving dependency construction after `check()` produced `6 passed in 1.49s`.

### Explicit-service isolation RED/GREEN

Command:

```sh
.venv/bin/python -m pytest tests/integration/test_workflow_startup.py::test_explicit_service_without_gateway_does_not_create_model_owner -q --tb=short
```

RED: exit 1; the forbidden `OpenAIModelGateway` constructor was reached. Root cause was model-owner construction preceding the explicit-service branch. The minimal condition change made the exact command pass: `1 passed in 1.31s`.

### Existing startup compatibility evidence

The first compatibility probe was:

```sh
.venv/bin/python -m pytest tests/integration/test_knowledge_startup.py tests/test_tool_api.py tests/test_api.py tests/test_knowledge_api.py -q --tb=short
```

It reported `7 failed, 39 passed, 1 warning, 1 error`. Five failures were stale `app.main` monkeypatch targets after extracting knowledge bootstrap. Two unit failures were the `assembly` and successful parameter rows of `test_lifespan_closes_partial_and_successful_resources`, which had relied on the removed fake-gateway implicit fallback. The one error was the real-MySQL test blocked by sandbox network policy, not a functional result.

The exact three old default-startup behaviors requiring explicit legacy injection were:

1. `test_lifespan_closes_partial_and_successful_resources[assembly]`
2. `test_lifespan_closes_partial_and_successful_resources[None]`
3. `test_production_http_assembly_persists_tool_turn_across_restart`

The first two now explicitly inject old `KnowledgeDependencies` and still test partial/success cleanup. The third explicitly injects the same old dependencies and retains real MySQL cross-restart persistence. Production fallback was not restored. `tests/integration/test_knowledge_startup.py` now exercises the extracted public builder directly, including empty corpus, bad model manifest, physical warmup drain, and repeated cancellation.

The non-database affected compatibility command then reported `52 passed, 2 deselected, 1 known warning`.

## Isolated database evidence

Only the configured isolated test services were used: MySQL at 127.0.0.1:13307 and PostgreSQL at 127.0.0.1:15433. No demo 3307 database was touched.

Focused command:

```sh
.venv/bin/python -m pytest tests/integration/test_workflow_startup.py::test_real_checkpoint_missing_table_schema_fails_startup tests/test_tool_api.py::test_production_http_assembly_persists_tool_turn_across_restart -q --require-postgres --require-mysql --tb=short
```

First result: `1 failed, 1 passed`. The MySQL cross-restart case passed. The PG test failed before connection because its test-only `Settings.model_copy` inserted an unvalidated string where production Settings carries `SecretStr`. This was a fixture defect, not a startup failure.

After using `SecretStr`, the corrected real missing-table test command reported `1 passed in 1.35s`. It connects to the isolated PostgreSQL database with a deliberately nonexistent `search_path`, reaches the real `CheckpointStore.open/check`, receives `CHECKPOINT_SCHEMA_UNAVAILABLE`, never compiles the graph, and closes the partial database/gateway owners.

Final affected command from the final implementation tree:

```sh
.venv/bin/python -m pytest tests/integration/test_knowledge_startup.py tests/integration/test_workflow_startup.py tests/test_tool_api.py tests/test_api.py tests/test_knowledge_api.py tests/test_workflow_graph.py tests/test_workflow_knowledge.py tests/test_workflow_chat.py tests/test_model.py -q --require-mysql --require-postgres --tb=short
```

Result: exit 0, `147 passed, 1 warning in 3.29s`. The warning is the known pre-existing Starlette `anyio.abc.BlockingPortal` deprecation warning.

## Installed signatures and package/import evidence

Installed signatures checked before final assembly:

- `CheckpointStore.open(self) -> AsyncPostgresSaver`
- `CheckpointStore.check(self) -> None`
- `OpenAIModelGateway.create_workflow_gateway(self, before_request, record_usage)`
- `OpenAIModelGateway.create_knowledge_gateway(self, *, before_request=None, record_usage=None)`
- `LocalModels.warmup(self) -> None`
- `MilvusStore.prepare_existing_collection(self) -> None`

The environment did not include the optional `build` frontend (`python -m build` returned `No module named build`), so the wheel was built without dependency changes using the installed PEP 517 backend:

```sh
.venv/bin/python -m pip wheel --no-deps --no-build-isolation . --wheel-dir "$final_wheel_dir"
```

Final result: wheel build exit 0. ZIP inspection found exactly 11 `app/prompts/*.txt` files and all four `workflow_intent`, `workflow_agent`, `workflow_evidence`, and `workflow_answer` prompts.

The first import probe incorrectly removed every `sys.path` entry containing the Chinese workspace name, which also removed the symlinked `.venv` site-packages and caused a false `ModuleNotFoundError: langgraph`. The corrected and final probe placed the wheel first on `sys.path`, asserted the imported bootstrap path came from the wheel, and replaced `socket.socket.connect` with a forbidden function. Result: `workflow_import=ok; network_connects=0`. Importing workflow bootstrap/graph/prompts did not initialize models, download assets, or make a model request.

Final package checks:

- `.venv/bin/python -m pip check`: `No broken requirements found` (the pip-cache ownership warning is environmental and does not indicate a broken requirement).
- `compileall` over all changed Python files: exit 0, no output.
- `git diff --check`: exit 0, no output.

## Files changed

- Created: `app/knowledge/bootstrap.py`, `app/workflow/bootstrap.py`, `tests/integration/test_workflow_startup.py`.
- Modified: `app/main.py`, `.env.example`, `README.md`, `tests/integration/test_knowledge_startup.py`, `tests/test_tool_api.py`, `dev-notes/ch05.md`.
- `app/model.py` and `app/resource_lifecycle.py` were inspected but not changed because their accepted factory and repeated-cancellation contracts already met Task 10.
- The controller-owned plan-file modification present in the shared worktree is deliberately excluded from this task's commit.

## Self-review and concerns

- Verified line by line: normal default is workflow-only; explicit old injections remain explicit; MySQL and checkpoint startup are check-only; no runtime evaluation file gate exists; 0.7/0.8 remain the workflow thresholds; service drains before shared owners; partial startup and repeat cancellation close once in reverse order; actions/source/extract routes are registered; package data has 11 prompts; import has no network side effect.
- No real DeepSeek/model request was made. External data-send authorization remains pending, so this task proves the local startup/lifecycle/package contract only. It does not claim real prompt quality, real customer-service acceptance, or permission to switch 8001.
- The known Starlette deprecation warning remains unchanged. No dependency pin was changed to suppress it.
