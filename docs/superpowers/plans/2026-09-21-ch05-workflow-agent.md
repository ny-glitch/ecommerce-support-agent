# Chapter 05 Workflow + Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有客服聊天中实现确定性四出口、知识专用三级分流、有界 Agent 子图、PostgreSQL checkpoint，以及用户确认后才执行的工单操作。

**Architecture:** 外层 LangGraph 固定指代透传、一次分类、分流、检索、证据闸及审计；内层 LangGraph 执行受预算约束的工具循环或仅生成。MySQL 保存业务与消息审计，PostgreSQL 保存图状态，SSE 仅转发允许的状态与最终答案。

**Tech Stack:** 现有 FastAPI 0.141.1、SQLAlchemy 2.0.54、LangChain core 1.6.3/openai 1.6.2、MySQL、Milvus 2.6.23、bge-m3/bge-reranker-v2-m3；新增 LangGraph、官方 PostgreSQL checkpointer、psycopg；原生 HTML/CSS/JavaScript。

**Spec:** `docs/superpowers/specs/2026-09-21-ch05-workflow-agent-design.md`，用户于 2026-09-21 回复“规格通过”。

**Status:** 用户已回复“计划通过，选 1”，采用子代理逐任务实施与独立评审；本文件中尚未勾选的命令和测试是执行步骤，不是已执行证据。

## Global Constraints

- 三级分流只用于商品咨询、退款退货；物流、订单、售后直接进入 Agent；投诉、闲聊不进入 Agent。
- 意图识别每轮最多一次；中档复用该结果。指代消解原样透传。
- 高档 `s > 0.8`；中档 `0.7 <= s <= 0.8`；低档 `s < 0.7`；使用有效候选最高归一化重排分。
- 三档都在答案生成前核验证据；无证据固定兜底并入池，低分有充分证据才生成；技术错误不伪装知识缺口。
- 高档纯知识展示支持片段完整答案；需要具体订单事实时先过政策闸再进入 Agent。
- Agent 最多 4 次工具执行、5 次决策、1 次最终生成；累计模型预算默认 49152 个保守估算单位；业务 60 秒、知识从本轮开始计时 240 秒。
- 工具白名单仅 query_order/query_product/query_logistics；query_faq 的能力由固定检索复用；create_ticket 仅用户点击并确认后的专用接口可执行。
- 转人工仅前端模拟，两个按钮独立；建工单不再修改 conversations.status 为 human_pending。
- 单应用 worker；保留现有上下文裁剪、模型配置、关闭思考模式及来源/反馈 UI；不实现新业务工具、登录体系、自动恢复外部调用或断线续传。
- 所有库/API 实现前先 Context7、再核对锁定版本；发现固定技术走不通先询问用户。
- 不打印 `.env`、密钥、连接串、HTTP 错误原文或模型内部推理；生产/测试数据严格隔离。
- 每个完成阶段立即向 `dev-notes/ch05.md` 追加四项；第 4 章依赖修复还须同步 `dev-notes/ch04.md`，不得收尾批量补记。
- 前端按既有 Vibe Coding 例外，不做页面单测/独立 code review；后端和 API 正常 TDD/评审；Prompt/标注数据以评估集验收。
- 第 4 章真实外部评估仍待明确数据发送授权；不得借新 CLI、图调用或新进程绕过原自动审核拒绝。mock/本机数据库/离线本地模型验证可继续，真实质量门槛必须保留。
- 保留当前 8001 服务直到新版本独立验收；禁止碰其他项目的 8000；不清库、不重下既有模型、不擅自合并分支。

## Review Focus

1. 政策加订单的复合问题：知识闸只核实政策是否齐全，缺少实时订单事实应交 Agent 查询，而非永远拒答；测试归 Task 5/7。
2. MySQL 已提交、PostgreSQL 确认失败：恢复不得重放随机工具或让旧证据串入下一轮；测试归 Task 9。
3. 并发点击与提交确认丢失：同一个工单建议只能形成一个 ticket，转人工状态不受影响；测试归 Task 3/8。
4. 在持续 token 输出、工具重试、提交取消之间耗尽预算：不能越过总时限、重复调用或提前 done；测试归 Task 2/6/9。
5. 近似型号、Unicode 引用与资料中伪指令：不能用错来源、生成无法点击的引用，或把中间控制 JSON 当用户答案；测试归 Task 4/5/7/11。

## 基线、版本核对与执行次序

文档基线提交为 `c631ce3`，产品实现基线为第 4 章 `a1b6956`。当前未提交文件：`app/knowledge/evaluation.py`、`evaluation_artifacts.py`、`gateway.py`、`tests/test_knowledge_evaluation.py` 属于待完成修复；`app/web/index.html`、`dev-notes/ch04.md`、第 4 章 plan 是先前控制器改动。不能一起随意提交或覆盖。

执行前按 using-git-worktrees 核对已有隔离及用户偏好。先在原工作区收敛 Task 0 的已有改动，再从明确提交创建 `codex/ch05-workflow-agent`；未纳入基线的第 4 章 UI 先单独保存、复核已有浏览器记录并提交，不能在 checkout 时丢失。工作区创建不等于复制/清空业务数据库。

2026-09-21 已先查 Context7，再核对官方发布页：候选固定版本为 LangGraph 1.2.11、langgraph-checkpoint-postgres 3.1.2、psycopg[binary,pool] 3.3.6、PostgreSQL 17.11。现环境尚未安装前三者；这些是 Task 1 的兼容性验证输入，不是安装成功声明。依赖解析或镜像架构失败必须报告，不自行更换固定栈。

依赖顺序：`0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12`。按任务依次实现；配置、脚手架和文档随相关功能提交。每个后端任务有自身 RED/GREEN 和验收点；是否逐任务独立评审由用户选择的执行方式决定。

## 文件责任与接口总表

| 文件 | 单一职责 | 任务 |
| --- | --- | --- |
| app/workflow/checkpoints.py | PostgreSQL saver 生命周期和显式 setup | 1 |
| app/workflow/contracts.py、state.py | 严格输入/控制 DTO，纯数据 State 与轮次重置 | 2 |
| app/workflow/routing.py、budget.py | 固定路由、三级分档、调用预算 | 2 |
| app/services/turn_operations.py | 有界异步操作、取消和写入排空 | 2 |
| app/db/workflow_migrations.py、workflow_models.py | 非破坏迁移、工单建议表 | 3 |
| app/db/conversations.py、actions.py | 多步审计及工单建议幂等仓储 | 3 |
| app/workflow/gateway.py、prompts.py | 分类、证据/Agent 控制协议、最终流 | 4 |
| app/workflow/knowledge.py | 固定检索、独立证据闸、知识档位响应 | 5 |
| app/workflow/agent.py | 有界模型/工具循环子图 | 6 |
| app/workflow/graph.py、nodes.py | 外层固定图和固定回复节点 | 7 |
| app/services/actions.py、api/actions.py | 用户确认后复用工单工具 | 8 |
| app/workflow/recovery.py、services/workflow_chat.py | 两库恢复、聊天准备、SSE 适配和终止 | 9 |
| app/workflow/bootstrap.py、app/main.py | 生产资源组装与启动验证 | 10 |
| app/web/index.html | 原生页面独立按钮、固定消息和状态 | 11 |
| app/workflow/evaluation.py、scripts/evaluate_workflow.py、scripts/demo_workflow.py | 标注评估、真实演示、报告 | 12 |

共享类型在 Task 2 定义，后续任务只扩展其明确拥有的字段。对既有大文件只做必要接口改动；共享行为提取后保持旧测试，不复制一套“近似相同”的取消、证据选择或安全校验逻辑。

---

### Task 0: 收敛第 4 章依赖修复，建立可追溯基线

**Files:** Modify `app/knowledge/evaluation.py`、`app/knowledge/evaluation_artifacts.py`、`app/knowledge/gateway.py`、`tests/test_knowledge_evaluation.py`、`dev-notes/ch04.md`、`dev-notes/ch05.md`；读取 `.superpowers/sdd/2026-09-18-ch04-hybrid-rag/task-9-findings.md`、`task-9-report.md`、`task-9-review.md`、`external-evaluation-authorization.md`。第 4 章 `app/web/index.html` 单独归档提交，不混入后端评审。

**Interfaces:** 保持 `KnowledgeGateway.assess`、`QueryNormalizer.prepare`、`KnowledgeRetriever.retrieve`、`CalibrationArtifact` 的既有公共合同；产出复审通过的具体提交和剩余真实验收清单。

- [x] 读取已有部分修改，恢复原 fix round；对照四个 Important：invalid run 不复活、有效请求控制进 fingerprint、保留失败生成/judge 诊断、生成/judge 受总 deadline 约束。
- [x] 运行现有失败回归，追加缺少的持续流超时用例；先记录实际 RED，不将先前 441 passed 套到当前未完成修复。

```bash
.venv/bin/python -m pytest tests/test_knowledge_evaluation.py tests/test_knowledge_evidence.py -q
```

- [x] 修复时使 invalid 为终态；在收集流时先保存收到的正文再校验引用，超时关闭流且不调用 judge；保留安全异常合同。修复核心顺序如下，沿用已有记录字段名：

```python
if manifest["status"] == "invalid":
    raise EvaluationDataError("invalid runs require a fresh output directory")
async with asyncio.timeout_at(deadline):
    async with aclosing(upstream) as tokens:
        async for token in tokens:
            parts.append(token)
```

- [x] 重跑覆盖测试；共享网关有改动时执行证据评估测试，生命周期未改不重复跑本地权重全套。记录准确命令、结果和修复提交。
- [x] 由与修复者不同的评审者对原四组问题及新改动破坏做 scoped rereview；Minor 继续进入原最终评审清单。恢复原 UI/文档所有权，确定第 5 章代码基线；真实外部验证保持 pending，不制造 calibration 文件。
- [ ] 精确暂存该任务文件并提交；立即写两个章节依赖阶段记录。确认基线后建立第 5 章分支，读取原开发记录中的剩余章节验收，不标第 4 章 finish。

### Task 1: 官方 PostgreSQL checkpointer 与版本/启动合同

**Files:** Create `app/workflow/__init__.py`、`app/workflow/checkpoints.py`、`scripts/init_workflow_checkpoints.py`、`tests/test_workflow_checkpoints.py`、`tests/integration/test_workflow_checkpoints.py`；Modify `pyproject.toml`、`compose.yaml`、`.env.example`、`app/config.py`、`tests/conftest.py`、`tests/integration/conftest.py`。

**Interfaces:** `CheckpointStore(settings: Settings, *, test_mode: bool=False)` 提供 `async open() -> AsyncPostgresSaver`、`async setup() -> None`、`async check() -> None`、`async aclose() -> None`；open/check 不建表，只有 setup 显式创建官方表。Settings 新增 `checkpoint_database_url: SecretStr | None`，测试配置单独使用 `TEST_CHECKPOINT_DATABASE_URL`。

- [ ] 先写 RED：正常 open 不调用 setup、缺表安全报错且无连接串、重复关闭只关一次；真实 PostgreSQL saver 重新创建后能读取相同 thread_id 的值。下面的测试放在 integration 文件，checkpoint_settings 来自该目录 fixture；其余生命周期单测使用受控 saver/context manager，不访问数据库。

```python
async def test_open_never_creates_schema(monkeypatch, checkpoint_settings):
    calls = []
    async def forbidden_setup(self):
        calls.append("setup")
        raise AssertionError("setup during ordinary startup")
    monkeypatch.setattr(AsyncPostgresSaver, "setup", forbidden_setup)
    store = CheckpointStore(checkpoint_settings)
    try:
        await store.open()
        assert calls == []
    finally:
        await store.aclose()
```

- [ ] 用 `.venv/bin/python -m pytest tests/test_workflow_checkpoints.py -q` 确认缺模块/行为 RED。先做依赖 dry-run，核对当前 Python 3.13 与现有 LangChain pin，再安装精确版本并运行 `pip check`：

```bash
.venv/bin/python -m pip install --dry-run 'langgraph==1.2.11' 'langgraph-checkpoint-postgres==3.1.2' 'psycopg[binary,pool]==3.3.6' 'langchain-core==1.6.3' 'langchain-openai==1.6.2'
```

- [ ] Context7 + 安装源码确认 `AsyncPostgresSaver.from_conn_string`、`setup`、`aget_tuple`、关闭语义；采用异步上下文管理器由 store 持有。初始化脚本调用 setup，普通启动调用 check；空测试 thread_id 的 `aget_tuple` 用于只读探测表可用性。

```python
manager = AsyncPostgresSaver.from_conn_string(url)
saver = await manager.__aenter__()
await saver.aget_tuple({"configurable": {"thread_id": "healthcheck-only"}})
```

- [ ] 添加 `workflow-db`/`test-workflow-db`，候选镜像 `postgres:17.11`，端口 `127.0.0.1:5433/15433 → 5432`，库/用户 `support_graph` 与 `support_graph_test`，独立卷，测试服务使用 test profile。镜像拉取前核对 arm64 清单；只新增所需本地密码，保留原 `.env`，不输出内容。测试 DSN 必须匹配 127.0.0.1:15433、测试库/用户，cleanup 只删除测试 thread 前缀，不 truncate 演示库。
- [ ] 增加 `--require-postgres`，运行 focused 与真实持久化测试；测试示例的 `checkpoint_settings` fixture 指向已显式 setup 的独立测试库。用 state 为 `{"count": 0}` 的单节点图写成 1，关闭连接，再用新 saver/graph `aget_state` 验证仍为 1；验证第二个 thread 不可见该值。
- [ ] 记录版本、镜像 digest、API 实测、RED/GREEN；精确提交本任务文件与开发记录。

### Task 2: State、固定路由、预算与取消操作

**Files:** Create `app/workflow/contracts.py`、`app/workflow/state.py`、`app/workflow/routing.py`、`app/workflow/budget.py`、`app/services/turn_operations.py`、`tests/test_workflow_state.py`、`tests/test_workflow_budget.py`、`tests/test_turn_operations.py`；Modify `app/config.py`、`app/services/chat.py`、`app/services/knowledge_turn.py`（仅共享 bounded 操作的必要提取）。

**Interfaces:**
- `IntentResult(intent, needs_business_data)` 为 strict Pydantic DTO；intent 值固定 `logistics/order/product/return_refund/after_sales/complaint/chitchat`，分别映射规格中的七类中文标签。
- `FinalControl(kind, actions, ticket)`：kind 为 respond/clarify；actions 为无重复的 handoff/create_ticket，最多两个；ticket 使用原 `TicketInput`，且仅 create_ticket 时必填。
- `ActionOffer(action_id, conversation_id, turn_id, ticket_no, draft, status)` 为 JSON 可序列化 DTO；draft 为 `TicketInput`，status 为 offered/completed。
- `route_intent(intent: IntentResult) -> str` 返回 knowledge/business/complaint/chitchat；`knowledge_band(score: float, *, lower=.7, upper=.8) -> str` 返回 high/middle/low 并拒绝非有限、超范围数值。
- `fresh_state(ref: TurnRef, user_id: str, question: str, category: str|None, history: list[dict], budget_limit: int) -> WorkflowState`；`dump_turns`/`load_turns` 在 `StoredTurn` 与 JSON 消息之间互转并验证配对。
- `RequestBudget(limit: int)`：`reserve(stage, messages, *, output_tokens, tool_schemas=()) -> int` 返回本次保守额度、超限抛 `ServiceError('TURN_BUDGET_EXHAUSTED', ...)`；`record_usage(stage, usage: dict|None)`、`snapshot() -> dict`。预算按预留额累计，不因缺 usage 退款。
- `TurnOperations.run(factory, deadline, *, mutation=False)`、`track_iterator(iterator)`、`async drain()`；`TurnRuntime(ref, user_id, started_at, deadline, budget, operations)` 只传 graph runtime context，不进 checkpoint。

- [ ] RED 精确边界、知识与业务路由、上一轮 sources/category/actions 不继承、JSON roundtrip、预算在请求前拒绝；Task 2 测试同时验证四类路由返回值是代码映射而非模型生成。

```python
@pytest.mark.parametrize("score,expected", [(0.6999,"low"),(.7,"middle"),(.8,"middle"),(.8001,"high")])
def test_knowledge_boundaries(score, expected):
    assert knowledge_band(score) == expected

def test_business_route_does_not_depend_on_knowledge_score():
    intent = IntentResult(intent="logistics", needs_business_data=True)
    assert route_intent(intent) == "business"
```

- [ ] Run `.venv/bin/python -m pytest tests/test_workflow_state.py tests/test_workflow_budget.py tests/test_turn_operations.py -q`；先确认 RED。
- [ ] 实现纯分档/映射与 strict DTO。State 字段固定为：schema_version、conversation_id、user_id、turn_id、original_question、question、category、history、intent、route、query、retrieval、score、band、sources、assessment、knowledge_status、knowledge_target、refusal_reason、agent_mode、tool_messages、pending_call、decision_count、tool_count、control、suggestions、offers、answer、used_citations、budget、budget_exhausted、trace、status。初始可选值为空，列表为空，budget_exhausted=False，schema_version=1，status=pending；单轮字段全覆盖，history 仅来自已完成轮次。retrieval 是有界 JSON 快照，不能把运行中的模型/迭代器塞入其中。

```python
def knowledge_band(score, *, lower=.7, upper=.8):
    if isinstance(score, bool) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("invalid reranker score")
    if not 0 <= lower < upper <= 1:
        raise ValueError("invalid knowledge thresholds")
    return "high" if score > upper else "middle" if score >= lower else "low"
```

- [ ] reserve 使用现有 `estimate_tokens` 与工具 Schema 的规范 JSON 字节数，加最大输出额；记录 actual usage 与估算分开。新增配置上下限、默认 4/5/49152；运行时 start/deadline 为单调时钟，只放 TurnRuntime，不把不可跨进程的时钟值写进 State。
- [ ] 从旧 chat 提取 bounded 行为至公共模块，保持 shield、取消排空、先等 mutation 结束再做审计终止。新增 Event 控制的取消测试：两次 cancel、操作在取消后延迟提交、迭代器正在 anext 时不得 aclose；真实外部调用次数始终有界。
- [ ] GREEN 后同时跑 `tests/test_tool_streaming.py tests/test_knowledge_chat.py tests/test_streaming.py`，防止提取破坏旧行为；记录并提交。

### Task 3: MySQL 幂等多步流水与工单建议仓储

**Files:** Create `app/db/workflow_models.py`、`app/db/workflow_migrations.py`、`app/db/actions.py`、`scripts/migrate_workflow.py`、`tests/integration/test_workflow_migrations.py`、`tests/integration/test_workflow_repositories.py`；Modify `app/db/models.py`、`app/db/database.py`、`app/db/contracts.py`、`app/db/conversations.py`、`app/db/tickets.py`、`tests/integration/conftest.py`、`tests/integration/test_repositories.py`、`tests/integration/test_tools_mysql.py`（仅被本章改变的单调用/人工作业状态断言）。

**Interfaces:**
- `async migrate_workflow(database: Database) -> dict[str,int]` 非破坏、重复可运行；check-only 不执行 ALTER。
- 原 `start_turn`/`create` 增加相同请求幂等；`append_call(ref, message, *, step:int=0)`、`append_result(ref, message, *, step:int=0)`；`finish_turn(ref, content, status, *, event_data:dict|None=None)`。
- `TurnSnapshot(ref, original_question, status, final_content, event_data, messages)` 存审计恢复所需字段；`get_turn(ref,user_id) -> TurnSnapshot|None`、`unfinished_turns(conversation_id,user_id) -> list[TurnSnapshot]`。
- `ActionRepository(sessions)` 提供 `offer_once(ref,user_id,draft:TicketInput)->ActionOffer`、`get_confirmable(conversation_id,action_id,user_id)->ActionOffer`、`mark_completed(action_id,ticket_no)->ActionOffer`。

- [ ] 先用测试库旧 schema fixture 写 RED：带已有 user/assistant/tool 数据的迁移、二次迁移、两组工具往返完整历史、重复事件不插入、同键不同内容冲突；测试库清理顺序先 conversation_actions 再 tickets/messages/conversations。

```python
async def test_two_tool_steps_are_kept_in_history(repos, new_turn):
    conversations, _, _ = repos
    await conversations.start_turn(new_turn, "demo", "先查订单，再查物流")
    for step, name in enumerate(("query_order", "query_logistics")):
        call_id = f"call-{step}"
        call = AIMessage(content="", tool_calls=[{"name":name,"args":{"order_id":"1001"},"id":call_id,"type":"tool_call"}])
        await conversations.append_call(new_turn, call, step=step)
        await conversations.append_result(new_turn, ToolMessage(content='{"status":"ok"}',tool_call_id=call_id), step=step)
    await conversations.finish_turn(new_turn, "已按查询结果说明。", "completed")
    history = await conversations.history(new_turn.conversation_id, "demo", 12)
    assert len(history[-1].messages) == 6
```

- [ ] Run `.venv/bin/python -m pytest tests/integration/test_workflow_migrations.py tests/integration/test_workflow_repositories.py --require-mysql -q` 并记录 RED。迁移先核对已存在字段/索引定义；MySQL DDL 非跨步骤原子事务，采用“新增可空列 → 分批按 id 回填 → 验证 → NOT NULL/唯一索引”的可续跑过程，不用 `create_all` 冒充旧表迁移。
- [ ] event_key 固定为 user、call:0、result:0、final 等；旧行 legacy:{id}。event_data 保存来源/路由/建议和 budget 等 JSON，有限大小；新调用必须接在完整上一组结果之后，同一轮 tool_call_id 不重复；只接受匹配申请的结果。

```python
existing = await session.scalar(select(Message).where(
    Message.conversation_id == ref.conversation_id,
    Message.turn_id == ref.turn_id,
    Message.event_key == event_key,
))
if existing is not None:
    if not same_event(existing, candidate):
        raise ServiceError("EVENT_CONFLICT", "该事件与已保存内容不一致", 409)
    return
```

`same_event(existing: Message,candidate:dict)->bool` 在仓储内定义，比较 role/content/tool_calls/tool_call_id/event_data，使用 Python 精确比较避免 MySQL 宽松 collation；唯一键竞态后用新短事务读取并核对。
- [ ] conversation_actions 使用规格字段、外键/唯一索引；action_id 使用服务端稳定 UUID5（conversation_id+turn_id+create_ticket），ticket_no 为 TK-加 UUID hex；校验原 TicketInput。get_confirmable 校验用户归属与轮次 completed；closed 会话不能新执行，但已完成建议可查询原结果。行锁只用于短事务，不能跨工具执行。
- [ ] TicketRepository.create_once 保持原 ticket 幂等核对，移除新建/重试时自动改 human_pending 的行为。明确更新旧测试的预期，同时验证历史 human_pending 不被清空。
- [ ] GREEN 测试包括两个同时 offer、唯一键冲突、新 session 恢复提交确认丢失、非法跨会话/未完成建议拒绝；迁移前后消息条数/正文与现有 ticket 数保持。提交并即时记账。

### Task 4: 一次意图识别与 Agent 控制协议

**Files:** Create `app/workflow/gateway.py`、`app/workflow/prompts.py`、`app/prompts/workflow_intent.txt`、`app/prompts/workflow_agent.txt`、`app/prompts/workflow_answer.txt`、`app/prompts/workflow_evidence.txt`、`evals/ch05/intents.jsonl`、`tests/test_workflow_gateway.py`、`dev-notes/ch05-evaluation.md`；Modify `app/model.py`（共享模型工厂）、`app/knowledge/gateway.py`（共用 assessment 校验与请求钩子）、`app/knowledge/query.py`（预算错误不能降级吞掉）。

**Interfaces:** `WorkflowGateway(model,settings,*,chat_extra_body,before_request,record_usage)`；`classify(messages)->IntentResult`；`decide(messages,tools)->AIMessage|FinalControl`；`assess(question,sources,*,normalized_question,intent:IntentResult)->EvidenceAssessment`；`stream_final(messages)->AsyncIterator[str]`。before_request(stage,messages,tool_schemas) 在真实调用前 reserve；record_usage(stage,usage) 只接受非敏感计数字段。`OpenAIModelGateway.create_workflow_gateway(before_request,record_usage)` 共享已有 ChatOpenAI/HTTP owner，不新建连接或私自改 `.env`。

`build_workflow_messages(purpose, *, settings,question,history=(),sources=(),tool_messages=(),intent=None,control=None,normalized_question=None,tool_schemas=())->ToolContextWindow` 在本任务的 prompts.py 定义，purpose 仅 intent/agent/evidence/answer；后续任务共用该实际请求构造，evidence 不读历史。before_request 绑定到 `runtime.budget.reserve(stage,messages,output_tokens=settings.max_output_tokens,tool_schemas=tool_schemas)`，只能在真正发请求时扣额，预算 fits 计算不扣额。

- [ ] MockTransport RED：intent JSON 限定七类，缺字段/额外字段拒绝；最多一次请求；有效 native tool_call 返回 AIMessage；无工具控制 JSON 返回 FinalControl；未知工具、并行调用、finish_reason=length、损坏动作草案均安全报错。

```python
def test_control_cannot_request_ticket_without_draft():
    with pytest.raises(ValidationError):
        FinalControl(kind="respond", actions=["create_ticket"], ticket=None)

def test_control_actions_are_not_executable_tools():
    value = FinalControl(kind="clarify", actions=["handoff"], ticket=None)
    assert value.actions == ["handoff"]
```

- [ ] Run `.venv/bin/python -m pytest tests/test_workflow_gateway.py -q` RED；用真实安装的 ChatOpenAI + MockTransport 捕获请求体，不能只 mock `.bind_tools`。
- [ ] classify/assess 使用 strict JSON Schema 对应的 json_mode/include_raw 校验；请求控制在构造 structured runnable 时绑定，避免已知外层 ainvoke 丢 extra_body 的问题。decide 使用 native bind_tools(auto, parallel_tool_calls=False)，有 tool_calls 时只接受白名单的一条；无 calls 时只接受 FinalControl JSON，绝不执行 JSON 文本中的工具名。

```python
reply = await model.bind_tools(tools, tool_choice="auto", parallel_tool_calls=False,
                               extra_body=chat_extra_body).ainvoke(messages)
if reply.invalid_tool_calls or len(reply.tool_calls) > 1:
    raise ServiceError("INVALID_TOOL_CALL", "工具调用格式无效", 502)
if reply.tool_calls:
    return reply
return FinalControl.model_validate_json(str(reply.content))
```

- [ ] 四份模板用 PromptTemplate：intent 定义七类与复合问题；agent 规定缺参追问、事实只来自证据/工具、允许动作只是建议；answer 禁止到账/送达/已执行承诺并要求知识引用；evidence 在 needs_business_data=true 时只核实相关政策/静态知识是否足以指导后续查询，不要求知识库包含订单实时事实。仍禁止政策缺失时交 Agent 猜测。
- [ ] 将 assessment 原有结构/支持 ID 验证提为 `validate_assessment(value:EvidenceAssessment, sources:Sequence[Citation])->EvidenceAssessment`，原网关和新网关共用，不改旧 JSON 合同。KnowledgeGateway 和 `create_knowledge_gateway` 新增可选 before_request/record_usage 关键字钩子，默认空；normalize 构造好完整请求后才调用钩子。预算超限由 QueryNormalizer 显式传播，不能当普通改写失败吞掉；可用 usage 记录到本轮预算，没有 usage 保留预留额。
- [ ] 标注 35 条 JSONL（每类 5 条），字段 id/question/history/expected_intent/needs_business_data；包含“你好，订单1001物流到哪”“我要投诉”“订单1001未拆封能否退货”“退货政策是什么”“你叫什么”。逐条核对标注与规则，记录评估待真实授权，mock 只证明协议不证明 Prompt 准确率。
- [ ] GREEN 加现有 `tests/test_tool_model.py tests/test_knowledge_query.py tests/test_knowledge_evidence.py tests/test_knowledge_evaluation.py`，验证 thinking-disabled/body limits 与旧网关不退化；即时记录、提交。

### Task 5: 固定检索、三级分档与独立证据闸

**Files:** Create `app/workflow/knowledge.py`、`tests/test_workflow_knowledge.py`、`evals/ch05/evidence.jsonl`；Modify `app/knowledge/evidence.py`、`app/knowledge/pipeline.py`（共享选择协议）、`app/workflow/contracts.py`、`app/workflow/prompts.py`、`tests/test_knowledge_evidence.py`。

**Interfaces:**
- `EvidenceSelector(fits)` 提供 `select(ranked:tuple[RankedChunk,...],query:QueryPlan)->EvidencePlan`；原 EvidenceBudget 与新的 `WorkflowEvidenceBudget(settings,history,question,intent,tool_schemas)` 共用选择/编号/edge_order，不再为 Workflow 造 AIMessage 工具申请。
- `WorkflowKnowledgeResult(decision:KnowledgeDecision, score:float|None, band:str|None, target:str)`；target 为 workflow_answer/agent_tools/agent_generate/fallback。
- `KnowledgeStage(normalizer_factory,retriever,gateway_factory,settings)`：`retrieve(question,category,*,runtime:TurnRuntime,emit)->RetrievalResult`；`assess(retrieval:RetrievalResult,intent,history,tool_schemas,*,runtime:TurnRuntime,emit)->WorkflowKnowledgeResult`；`run(question,category,intent,history,tool_schemas,*,runtime:TurnRuntime,emit)->WorkflowKnowledgeResult` 是前两者的顺序组合，仅供独立测试/评估，不写数据库。
- `serialize_retrieval(result:RetrievalResult)->dict` 与 `deserialize_retrieval(value:dict)->RetrievalResult` 在 knowledge.py 实现；保存 query、最多 50 个完整 RankedChunk 快照、raw_count/stale_count/strategy，校验字段/条数/分数，不写资源对象。外层 retrieve 节点将这个 JSON 放入 State.retrieval，evidence_gate 读取，不做第二次检索。
- 消费 Task 4 的 `build_workflow_messages`。`knowledge_target(intent:IntentResult, band:str|None, sufficient:bool)->str` 为本任务产出的纯路由函数。

- [ ] 先写 RED：0.65 且充分可以生成、0.95 但证据不充分必须 fallback；两阈值边界、零命中/过期、类别过滤不继承、预算删除整块后才自评；不得把旧 calibration 下限应用到新三级路由。

```python
@pytest.mark.parametrize("score,sufficient,target",[(.65,True,"agent_generate"),(.95,False,"fallback"),(.9,True,"workflow_answer"),(.75,True,"agent_tools")])
def test_knowledge_target_is_not_score_only(score, sufficient, target):
    intent = IntentResult(intent="product",needs_business_data=False)
    assert knowledge_target(intent, knowledge_band(score), sufficient) == target
```

- [ ] `.venv/bin/python -m pytest tests/test_workflow_knowledge.py -q` 确认 RED；另设复合退款案例：来源包含完整未拆封退货政策而没有订单 1001 的状态，闸应允许后续查询订单，最终资格必须等工具结果。
- [ ] 把原 EvidenceBudget 的 Top-10、稳定编号、首尾排序循环提入 EvidenceSelector；旧 EvidenceBudget 仍使用原 fits 合同。WorkflowEvidenceBudget 用真实 evidence/agent/answer 请求构造验证预算，不伪造 query_faq tool_call；不能只算答案正文而漏掉 Schema、原话、模板或元数据。

```python
def knowledge_target(intent, band, sufficient):
    if not sufficient:
        return "fallback"
    if intent.needs_business_data or band == "middle":
        return "agent_tools"
    return "workflow_answer" if band == "high" else "agent_generate"
```

- [ ] KnowledgeStage 只运行一次 normalize/retrieve(hybrid_rerank)，取有效 ranked 的最高分；没有候选不伪造分数。复用 `decide_evidence(..., threshold=None)` 与公共充分性校验，用一个绑定当前 IntentResult 的 assessor 适配新 workflow evidence Prompt；低分仍进入自评，真实技术异常向外传播。
- [ ] 高档模板按 supporting_chunk_ids 选择实际来源，固定格式为 `来源章节路径 + 完整 answer + [number]`；多条逐块展示，无新生成调用。模型生成需验证 ASCII `[n]`，拒绝把 `[１]`、`[١]` 视为合法链接；修正公共 citation 校验并测试旧合同，不能仅修改前端正则掩盖后端问题。
- [ ] 准备 12 条证据标注：expected_sufficient、supporting_chunk_ids、needs_business_data、reason；不伪标真实分数。至少含型号错配、无答案、政策条件不足、跨块支持、低相关但事实完整、资料伪指令；标注不送入待评模型 Prompt。记录真实评估 pending。
- [ ] GREEN 加原 `test_knowledge_evidence.py/test_knowledge_pipeline.py/test_knowledge_query.py`；确认没有修改 corpus 与原评估四策略定义。记录并提交。

### Task 6: 有界 ReAct Agent 子图

**Files:** Create `app/workflow/agent.py`、`tests/test_workflow_agent.py`、`tests/ch05_helpers.py`；Modify `app/tools/business.py`（只新增原工具的白名单组装函数）、`app/workflow/state.py`。

**Interfaces:** `build_readonly_registry(context:ToolContext,faq,tickets)->ToolRegistry` 只筛选既有三个查询工具。`AgentDependencies(settings,gateway_factory,conversations,faq,tickets,executor)`；`build_agent_graph(deps)->CompiledStateGraph` 使用 WorkflowState 和 TurnRuntime context，compile() 不建立独立跨轮记忆。输入 agent_mode=tools/generate_only，输出 answer/used_citations/suggestions、工具与预算轨迹；不输出 done、不建工单。

测试 helper 定义 `ScriptedWorkflowGateway`：构造参数 intents/decisions/tokens/assessments 为队列，方法与 Task 4 同名，每次调用记录 stage/messages；缺脚本结果抛 AssertionError，不自动给成功值。`RecordingConversations` 实现 Task 3 的 append/finish 方法并保存 trace；真实持久化另测，不能用它代替 MySQL 验收。

- [ ] RED 使用 ScriptedWorkflowGateway 返回 order call、logistics call、FinalControl 三个决策；测试实际执行顺序、每个 ToolMessage ID 对应、再决策输入含上一工具结果；简单物流只执行一次工具，缺订单号控制为 clarify 时零工具执行。

```python
assert [item["name"] for item in result["tool_messages"] if item["role"] == "tool"] == ["query_order", "query_logistics"]
assert result["tool_count"] == 2
assert all(event["name"] != "token" for event in events_before_final)
```

- [ ] Run `.venv/bin/python -m pytest tests/test_workflow_agent.py -q` RED。先实现 registry 筛选，未知名称与 create_ticket/query_faq 即使模型返回也不能执行；服务端仍检查白名单，不能只依赖提供给模型的 Schema。
- [ ] 建立 decide → tools → decide、decide → final、任意预算耗尽 → budget_reply；generate_only 入口直接 final。每步实际调用前 reserve、检查总 deadline；tool_count 以模型申请计一次，原 executor 的物理重试次数独立记录但同受总时限。

```python
def after_decision(state):
    if state["pending_call"] is None:
        return "final"
    if state["tool_count"] >= 4:
        return "budget_reply"
    return "tools"
```

实际实现使用已验证配置值而非散落常量；decision_count 达 5 不允许第 6 次请求。若无法预留下一调用预算，用固定 budget_reply；若还有最终回答预算但不能继续工具，仅允许依据已有结果收敛。固定话术不得声称查到了未执行的工具。
- [ ] tools 节点先幂等 append_call(step)，再执行，结果 settle 后 append_result(step)，最后才发 succeeded；延续旧服务“实际完成优先于状态帧”的保证。最终流使用 runtime.operations 管理 anext/取消，不在 yield 期间悬挂 timeout/CancelScope。
- [ ] 测试持续 token 流越过总时限会关闭、工具两次 transient 重试不重开 60 秒、第五个工具申请被拒、非法多工具调用不执行、失败结果可正常回灌、超预算零额外上游调用、最终引用失败不给成功状态。保留已经流出的部分 answer 供失败审计，不计 completed。
- [ ] GREEN 加现有 `test_tool_executor.py/test_tools.py/test_tool_context.py`；记录调用次数/取消测试及提交。

### Task 7: 外层固定 Workflow 图与状态事件

**Files:** Create `app/workflow/graph.py`、`app/workflow/nodes.py`、`tests/test_workflow_graph.py`；Modify `tests/ch05_helpers.py`。

**Interfaces:** `WorkflowDependencies(settings,gateway_factory,knowledge_stage,agent_dependencies,conversations,actions,low_confidence)`；`build_workflow(deps,checkpointer)->CompiledStateGraph`；`build_nodes(deps)->dict[str,Callable]` 返回下面的固定节点名。节点通过 `StreamWriter` 发 `{"name": str,"data":dict}`，不放 ORM/异常/资源对象。子图作为 agent 节点，继承父 saver 的每次调用命名空间。

- [ ] RED 七类出口矩阵：每轮 classify 恰一次；业务节点不碰知识，投诉/闲聊不碰 Agent；高档纯知识无最终模型生成；低分证据弱时 first token 从未发生；低置信度写入在拒答事件前完成。

```python
def test_outer_graph_has_fixed_business_and_knowledge_edges(workflow):
    edges = {(edge.source, edge.target) for edge in workflow.get_graph().edges}
    assert ("retrieve", "evidence_gate") in edges
    assert ("resolve", "classify") in edges
    assert ("persist", "__end__") in edges
```

该结构测试之外，必须用脚本模型执行实际各分支，不以节点存在替代行为验证。
- [ ] Run `.venv/bin/python -m pytest tests/test_workflow_graph.py -q` RED。实现节点：resolve 原样返回；classify 更新 intent/route 和 runtime.deadline；retrieve 执行固定检索；evidence_gate 核验并决定知识 target；workflow_answer/fallback/complaint/chitchat/budget_reply 为固定回复；persist 写来源元数据、建议与最终消息。分类/归一化/自评 reserve 超限时节点只捕获 TURN_BUDGET_EXHAUSTED 并置 budget_exhausted，不发起请求、不记知识缺口；后续条件边转 budget_reply。其他异常保持技术错误。

```python
graph = StateGraph(WorkflowState, context_schema=TurnRuntime)
for name, node in build_nodes(deps).items():
    graph.add_node(name, node)
graph.add_node("agent", build_agent_graph(deps.agent_dependencies))
graph.add_edge(START, "resolve")
graph.add_edge("resolve", "classify")
graph.add_conditional_edges("classify", lambda s:"budget_reply" if s["budget_exhausted"] else s["route"],
    {"knowledge":"retrieve","business":"agent","complaint":"complaint","chitchat":"chitchat","budget_reply":"budget_reply"})
graph.add_conditional_edges("retrieve", lambda s:"budget_reply" if s["budget_exhausted"] else "evidence_gate",
    {"budget_reply":"budget_reply","evidence_gate":"evidence_gate"})
graph.add_conditional_edges("evidence_gate", lambda s:"budget_reply" if s["budget_exhausted"] else s["knowledge_target"],
    {"workflow_answer":"workflow_answer","agent_tools":"agent","agent_generate":"agent","fallback":"fallback","budget_reply":"budget_reply"})
for name in ("agent","workflow_answer","fallback","complaint","chitchat","budget_reply"):
    graph.add_edge(name,"persist")
graph.add_edge("persist",END)
return graph.compile(checkpointer=checkpointer)
```

State.knowledge_target 已在 Task 2 声明，初始 None；本任务填入 Task 5 的路由函数结果。retrieve 与 evidence_gate 分别调用 Task 5 的 retrieve/assess，通过序列化快照衔接；不调用组合 run 后再重复自评。
- [ ] fallback 用 REFUSALS 受控模板，entry_point=workflow；complaint 固定安抚加 handoff/create_ticket 草案；chitchat 固定“您好，我是客服助手，可以帮您查询商品、订单、物流和售后问题。”；persist 先 offer_once，再 finish_turn(event_data)，不调用 create_ticket。完成后将本轮 user/实际工具流水/最终 assistant 加入已完成 history，按原最大轮次裁剪；控制 JSON 不加入对话正文。返回 status=completed、budget.snapshot 和可展示 offers，供 Task 9 验证。
- [ ] workflow_status 记录节点、意图、档位；固定检索的 normalizing/retrieving/reranking/checking_evidence 也用 workflow_status，不伪造 tool_call_id，页面继续兼容旧 retrieval_status。sources 在回答前发；模板使用 message、低证据用 refusal，只有 final 节点实际文本用 token。persist 写规格所列的安全结构化日志，失败日志由服务清理补齐。未完成轮次不发 actions/done；这两个终端事件归 Task 9 适配器。
- [ ] GREEN 执行中档分类计数=1、两轮 category/sources 不串、型号错配不答、证据伪指令不能触发工具、控制 JSON 不进入气泡；补测分类前/归一化前/自评前三处预算耗尽都只给固定预算话术，不发额外 LLM 请求。记录并提交。

### Task 8: 用户确认建工单接口与独立副作用

**Files:** Create `app/services/actions.py`、`app/api/actions.py`、`tests/test_workflow_actions.py`、`tests/integration/test_workflow_actions.py`；Modify `app/schemas.py`（确认请求/结果）。

**Interfaces:** `ActionService(actions:ActionRepository,faq,tickets,executor,settings)`；`async confirm(conversation_id:str,action_id:str,user_id:str='demo')->dict` 返回 ticket_no/conversation_id/status/action_id。路由 `POST /api/conversations/{conversation_id}/actions/{action_id}/confirm` 使用 UUID path 校验，body 为严格空对象；数据取自持久化建议，不能提交新 description 或 tool 名称。API 挂载归 Task 10。

- [ ] RED：确认前 tickets 数不变；两次确认同一编号；另一会话/用户、pending/failed 轮次、伪造 action_id、额外 body 字段均拒绝；不提供任何后端 handoff endpoint。

```python
async def test_repeated_confirmation_returns_same_ticket(action_service, offered_action):
    first = await action_service.confirm(offered_action.conversation_id,offered_action.action_id)
    second = await action_service.confirm(offered_action.conversation_id,offered_action.action_id)
    assert first["ticket_no"] == second["ticket_no"]
```

- [ ] Run `.venv/bin/python -m pytest tests/test_workflow_actions.py -q` RED；用测试 FastAPI 挂载 router 并注入 service，捕获真实 HTTP 错误/422 格式。
- [ ] confirm 加载 get_confirmable；以 offer 的 ref、原问题、稳定 ticket_no 构造 ToolContext；用原 build_registry 取 create_ticket，通过 ToolExecutor 执行合法 TicketInput；只接受 status=ok、ticket_no 与 offer 匹配，之后 mark_completed。

```python
offer = await actions.get_confirmable(conversation_id,action_id,user_id)
call = {"name":"create_ticket","id":f"action-{offer.action_id}",
        "args":offer.draft.model_dump(),"type":"tool_call"}
```

service 收集 executor 的 ToolOutcome（不把工具状态伪造成聊天新轮次）；业务失败原安全码返回，不用模型生成成功话术。已经 completed 也核对已有 ticket 的归属和内容后返回，不能只信客户端已反馈状态。
- [ ] 真实 MySQL 并发两个 confirm；注入 ticket 已 commit 但 mark_completed 抛异常，重试必须仍一条 ticket 且会话 status 不变。body 超长、草案类型非法、closed 会话中新操作拒绝；已成功操作重复查询仍返回原编号。
- [ ] GREEN + `test_tools_mysql.py`，记录真实 ticket 计数和提交。

### Task 9: 重启恢复、聊天适配与终端持久化屏障

**Files:** Create `app/workflow/recovery.py`、`app/services/workflow_chat.py`、`tests/test_workflow_chat.py`、`tests/integration/test_workflow_recovery.py`；Modify `app/api/chat.py`（保持请求合同的 service 适配）、`tests/ch05_helpers.py`。

**Interfaces:** `recover_conversation(graph,conversations,conversation_id,user_id)->list[dict]` 返回安全已完成历史；`PreparedWorkflowTurn(ref,runtime,initial_state)` 暴露可变 deadline 属性代理 runtime.deadline。`WorkflowChatService(settings,graph,conversations,guard).prepare(message,session_id,user_id='demo',*,category=None)` 为异步上下文管理器；`.stream(prepared)->AsyncIterator[ChatEvent]`、`.aclose()` 与旧 API 生命周期兼容。

- [ ] RED：同会话并发 409、首次迁移只导入 completed 历史、MySQL completed/checkpoint 未完成时修复而无模型调用、pending 崩溃轮次标中断、两库内容冲突拒绝新请求、第二轮证据/category 清零。

```python
assert recorder.model_calls == 0
assert recorder.tool_calls == 0
assert recovered_history[-1]["turn_id"] == completed_ref.turn_id
assert (await graph.aget_state(config)).next == ()
```

以上断言用于实际故障注入后的恢复测试；completed_ref/config/recorder 由用例建立，不能预先直接设成预期结果。
- [ ] Run focused RED；实现恢复：先核对会话归属，在 guard 内读取 saver state 和审计；必要时用公开 `aupdate_state(..., as_node='persist')` 根据已提交审计修复终态，并验证 next 为空。pending 只结束为 cancelled、清除未完成上下文，不调用 `ainvoke(None)` 自动重放。
- [ ] prepare 在创建持久会话前校验最小意图请求的上下文预算，随后创建 ref/start_turn，fresh_state 导入恢复后的历史；持有 guard 到 operations.drain 完成才释放，异常依然尝试结束审计。graph 不持有 HTTP request、不把资源对象放进 State。
- [ ] stream 发送 meta 后消费 graph 的 custom 输出，严格 allowlist 事件/字段；透传 runtime 更新后的 deadline，采用同步 checkpoint durability。使用 LangGraph 1.2.11 实测的 v2 event 格式，不把整份 State 更新发给用户。

```python
config = {"configurable":{"thread_id":prepared.ref.conversation_id},"recursion_limit":64}
async for part in graph.astream(prepared.initial_state, config=config,
        context=prepared.runtime, stream_mode="custom", subgraphs=True,
        version="v2", durability="sync"):
    if part["type"] == "custom":
        yield checked_chat_event(part["data"])
snapshot = await graph.aget_state(config)
```

`checked_chat_event(value:dict)->ChatEvent` 在 workflow_chat.py 定义，只接受规格事件；拒绝节点伪造 done/error/actions。确认 graph next 为空、状态 completed、MySQL 同轮 completed 且最终内容/元数据一致，才发 actions（如有）和 done。实现该检查为 `verify_completed_turn(snapshot,turn_snapshot)->dict`，返回已经验证的 done payload。
- [ ] 在最终 token 后注入 PG commit 失败，确保没有 done/可用 actions；断连中关闭模型流、等待在途写入、只记一次 failed/cancelled。必要清理按既有有限宽限完成，不能用释放 guard 伪装已排空。
- [ ] 真实两库集成验证重建 service 后续聊、两个线程隔离、恢复重复执行幂等；验证旧 SSE、非法请求、上下文预算回归；提交并记账。

### Task 10: 生产依赖组装、迁移检查与包验证

**Files:** Create `app/workflow/bootstrap.py`、`app/knowledge/bootstrap.py`、`tests/integration/test_workflow_startup.py`；Modify `app/main.py`、`app/model.py`、`app/resource_lifecycle.py`（仅实际必要变更）、`.env.example`、`README.md`、已有 startup 测试。

**Interfaces:** `KnowledgeComponents(repository,store,local_models,retriever,low_confidence,knowledge_gateway_factory)`；`build_knowledge_components(settings,database,model_gateway,owned_resources)->KnowledgeComponents` 公开复用现有非创建校验/预热。`build_workflow_dependencies(settings,database,model_gateway,components)->WorkflowDependencies`。`create_app` 保留显式 chat_service/knowledge_dependencies 测试注入，增加显式 workflow_dependencies；正常无注入启动必须使用 WorkflowChatService，不在失败时回落旧聊天实现。

- [ ] RED 缺 PG 配置/表、缺 MySQL 迁移列、无来源数据、模型 manifest 错误均启动失败并正确逆序关闭；已注入服务不隐式创建生产资源；初始化任一步取消也要一次性排空关闭。
- [ ] `.venv/bin/python -m pytest tests/integration/test_workflow_startup.py -q` RED；从原 `_production_knowledge_dependencies` 提取共享 KnowledgeComponents。保留旧流水线的 calibration 读取与校验供旧四策略评估使用；新图使用已批准的 0.7/0.8，不伪造或把旧 calibration 当新策略凭证。新图仍验证 corpus、固定模型 manifest、Milvus schema 和本章评估状态。

```python
components = await build_knowledge_components(settings,database,gateway,owned_resources)
saver = await checkpoint_store.open()
await checkpoint_store.check()
deps = build_workflow_dependencies(settings,database,gateway,components)
graph = build_workflow(deps,saver)
```

- [ ] 正常 lifespan 只检查 schema，不调用 MySQL create_all 或 saver.setup。注册 actions router、source router 和原 extract API；先关闭新 chat_service 的在途操作，再关 saver/本地模型/数据库/共享 HTTP，保持原重复取消行为。
- [ ] README 添加显式顺序：Docker 数据库 → MySQL 迁移 → checkpoint setup → 数据/索引检查 → `uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8002 --workers 1`。旧章节“单次调用/LIKE 漏召回”的说明移到历史行为，不作为本章承诺。
- [ ] GREEN lifecycle、旧 API 注入测试；`pip check`；构建 wheel 并检查新增 4 份 Prompt 全部在包中（原 7 份 + 新 4 份 = 11 份），从 wheel 导入 workflow 模块不触发下载/初始化/模型请求。
- [ ] 仅在本章真实验收通过后切换 8001；此任务可完成本地启动合同，不以 mock 启动宣布真实客服可用。记录并提交。

### Task 11: 页面 Vibe Coding 与交互验证

**Files:** Modify `app/web/index.html`、`dev-notes/ch05.md`；不新增前端单测或页面独立 code review。

**Interfaces:** 消费 workflow_status/message/actions 与既有 SSE；actions 在 done 后启用；create_ticket 调 Task 8 接口，handoff 无网络请求。

- [ ] 读取现有原生页面与已完成第 4 章 UI；直接增量实现固定 message 渲染、流程/多工具徽章、两个独立按钮和建单确认卡。DOM 使用 textContent，沿用 source 安全映射；不把 action_id 拼成任意外域 URL。

```javascript
handoffButton.addEventListener('click', () => {
  handoffButton.disabled = true;
  appendLocalText('已转接人工客服');
  appendLocalText('您好，我是客服小猫，请问有什么可以帮您的');
});
```

`appendLocalText(text)` 是本任务定义的纯 DOM 气泡 helper，不发后端聊天请求；新问候不伪装成新的模型回复或未保存的后台人工会话。
- [ ] 建单按钮打开确认卡，取消不请求，确认时固定同源 path 和空 body；保存当前 action_id，处理中禁用、失败可重试、成功展示同一 ticket_no。只锁本按钮，不锁转人工；未点击可直接继续输入。
- [ ] 用明确标注的本地 SSE fixture 浏览器验证：消息任意切帧、控制 JSON 不展示、actions 早到但 done 未到不可操作、失败半段无成功按钮、转人工后仍可建单、先建单后仍能转人工、取消建单零请求、断网重试同 action_id、两轮按钮互不串。
- [ ] 检查引用弹窗/ASCII 编号、旧反馈锁定、新会话/category reset 没有退化；保存浏览器实际结果至开发记录。fixture 验证与真实服务验收分开标注；不发送真实模型请求直至授权。
- [ ] 单独提交 HTML/前端记录，不把 HTML 放入后端 code review 包。

### Task 12: 标注评估、真实演示与完成交付

**Files:** Create `app/workflow/evaluation.py`、`app/evaluation_io.py`、`scripts/evaluate_workflow.py`、`scripts/demo_workflow.py`、`tests/test_workflow_evaluation.py`、`tests/test_workflow_demo.py`；Modify `app/knowledge/evaluation.py`、`app/knowledge/evaluation_artifacts.py`（仅通用文件 I/O 提取）、`evals/ch05/intents.jsonl`、`evals/ch05/evidence.jsonl`、`dev-notes/ch05-evaluation.md`、`dev-notes/ch05.md`、`README.md`。

**Interfaces:** `evaluate_workflow(cases,runner,*,output_dir,configuration)->dict`；runner(question,history,category) 返回实际事件/路由/模型次数/工具名/证据与完成状态，不接收 expected_*。CLI：`evaluate_workflow.py --cases PATH --output-dir PATH [--limit N]`；`demo_workflow.py --base-url URL --scenario policy|logistics|complaint|chitchat|multi_step|unknown [--confirm-ticket]`。

- [ ] RED 评估统计 fixture：分类分母含失败、每类混淆、知识误入率、支持/拒答/引用正确性、错误码独立、score 档位按实际值；禁止标注泄漏给 runner；partial/smoke 不写 complete，配置指纹不匹配拒绝续跑。

```python
seen = []
async def runner(question, history, category):
    seen.append((question,history,category))
    return {"intent":"logistics","route":"business","status":"completed","model_calls":2,"tools":["query_logistics"]}
await evaluate_workflow(cases,runner,output_dir=tmp_path,configuration=config)
assert all("expected_intent" not in str(item) for item in seen)
```

- [ ] `.venv/bin/python -m pytest tests/test_workflow_evaluation.py tests/test_workflow_demo.py -q` RED；实现逐条原子产物/安全 fingerprint/失败保留。将通用 `strict_json_dumps(value,*,indent=None)->str`、`atomic_json(path:Path,value)->None`、`atomic_text(path:Path,value:str)->None` 提至 app/evaluation_io.py，原模块保留旧导入名，两个评估器共用；不复制第二套知识缓存/校准逻辑。该提取加跑 `tests/test_knowledge_evaluation.py`，确认原 invalid 终态和诊断保留不回退。
- [ ] demo 解析真实 SSE，必须见到 terminal done；error/EOF 无 done 返回非零；从 meta/actions 提取真实 ID，不硬编码。默认 complaint 只展示建议、不确认工单；`--confirm-ticket` 是明确的演示执行开关，确认同一 action 两次并输出同一编号。source 只访问同一 base 的固定数字 ID+摘要路径，不跟随任意模型 URL。
- [ ] 在外部数据发送许可获得前，只完成 CLI help、评估器 fixture、受控模型图测试与真实本机数据库/离线模型检查；报告清楚写未执行真实 Prompt 评估。授权到达后记录目的地与数据范围，再执行：

```bash
.venv/bin/python scripts/evaluate_workflow.py --cases evals/ch05/intents.jsonl --output-dir evals/reports/ch05/intents
.venv/bin/python scripts/evaluate_workflow.py --cases evals/ch05/evidence.jsonl --output-dir evals/reports/ch05/evidence
.venv/bin/python scripts/demo_workflow.py --base-url http://127.0.0.1:8002 --scenario logistics
.venv/bin/python scripts/demo_workflow.py --base-url http://127.0.0.1:8002 --scenario complaint --confirm-ticket
.venv/bin/python scripts/demo_workflow.py --base-url http://127.0.0.1:8002 --scenario multi_step
```

评估运行器复用真实 Workflow 图，离线评估的审计/问题池用隔离适配器，不能污染演示数据库；真实浏览器验收另验证 MySQL/PG 实际行。不得用内存 saver 的评估结果替代持久化集成测试。
- [ ] 执行规格七项真实浏览器验收及 policy/chitchat/unknown CLI，用实际日志和库行证明路径。对随机订单导致的合理提前结束如实记录；用明确需求和可继续查物流的真实演示结果展示多步，不篡改返回。
- [ ] 跑最终 required suite 与包校验，首次执行后只为新改动或失败重跑覆盖集：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest --require-mysql --require-postgres --require-milvus --require-local-models -q
.venv/bin/python -m pip check
```

- [ ] 按用户选定执行方式完成最强可用模型的整分支后端评审，带上原第 4 章未清 Minor 清单与本章 Review Focus；修复 Important 再复审，不把前端 Vibe 页面加入正式评审。
- [ ] 所有真实门槛通过后才停临时 8002 并排空，核对旧 8001 PID、切到新单 worker、再验健康与真实聊天；不承诺零停机。README 给完整 demo 命令、测试数字、评估报告和 dev-notes 路径。
- [ ] 提交本任务文件，写 code review 结论与 finish 阶段记录；调用 finishing-a-development-branch 让用户选择本章集成方式。任何前置授权/真实验证未完成时不标 finish、不复用第 2 章“本地合并”作为本章授权。

## 规格覆盖与计划自审

| 规格章节 | 实施任务 |
| --- | --- |
| 1–3 目标/范围/总图 | 0、2、7、10 |
| 4 一次分类与四出口 | 2、4、7 |
| 5 分档/证据/问题池 | 5、7、12 |
| 6 Agent | 4、6 |
| 7 预算/时限/流式 | 2、6、9 |
| 8 State/两库审计/恢复 | 1、2、3、9 |
| 9 两个独立动作 | 3、8、11 |
| 10 API/SSE/日志 | 7、8、9、11 |
| 11 资源与部署边界 | 0、1、10、12 |
| 12 评估与交付 | 4、5、11、12 |

执行前引用资料：Context7 查询记录见本章 dev-notes；官方 [LangGraph graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)、[子图](https://docs.langchain.com/oss/python/langgraph/use-subgraphs)、[checkpoint 包](https://pypi.org/project/langgraph-checkpoint-postgres/)、[LangGraph 包](https://pypi.org/project/langgraph/)、[psycopg](https://pypi.org/project/psycopg/)、[PostgreSQL 支持版本](https://www.postgresql.org/support/versioning/)。每个任务进入具体 API 实现前再次以 Context7 和实际安装源码核对，计划示例不能压过实际版本合同。

计划审核交接：用户已批准计划，选择子代理逐任务实施并独立评审；先完成 Task 0 的已有修复和基线整理，再依次实施 Task 1–12。执行证据随各任务追加，不把计划批准当作测试通过。
