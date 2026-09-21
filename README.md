# 中文电商客服示例

这是一个基于 FastAPI、LangGraph、MySQL、PostgreSQL、Milvus 和 OpenAI 兼容 Chat Completions 接口的本地客服示例。当前工作流使用固定路由、知识检索与证据核验、有界 Agent 工具循环，并将会话审计和图检查点分别持久化。售后信息提取接口仍可独立使用。

订单、商品和物流结果均由本地工具随机生成，只用于演示；知识原文与工单保存在本地 MySQL。本项目没有生产鉴权、真实电商接口或退款操作。

## 安装与配置

需要 Python 3.11 或更高版本、Docker 和 Docker Compose。`requirements.lock` 锁定开发及运行依赖；安装锁文件后再安装仓库本身：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps -e .
[ -f .env ] || cp .env.example .env
```

最后一条命令会保留已有 `.env`。至少填写真实可用的模型、密钥和本地数据库密码；不要提交 `.env`：

```dotenv
LLM_BASE_URL=https://api.example.com/v1
LLM_MODEL=填写账户实际可用且支持 tools 和流式输出的模型名
LLM_API_KEY=填写密钥
DATABASE_URL=mysql+asyncmy://support:本地密码@127.0.0.1:3307/support
MYSQL_TEST_PASSWORD=replace-support-test-password
LLM_TOKEN_LIMIT_PARAM=max_completion_tokens
LLM_CHAT_EXTRA_BODY={}
CONTEXT_WINDOW_TOKENS=8192
MAX_OUTPUT_TOKENS=1024
TOKEN_SAFETY_MARGIN=512
MAX_HISTORY_TURNS=12
MAX_SESSIONS=100
REQUEST_TIMEOUT_SECONDS=60
TOOL_TIMEOUT_SECONDS=5
TOOL_MAX_ATTEMPTS=2
```

DeepSeek 工具聊天的两个模型阶段需关闭思考模式，可配置 `LLM_CHAT_EXTRA_BODY={"thinking":{"type":"disabled"}}`。这是 DeepSeek 专用字段；其他上游默认使用 `{}`，除非其官方接口明确支持同名字段。OpenAI 兼容并不保证模型支持本示例依赖的工具选择、流式输出和 JSON 行为，需按实际模型验证。`LLM_TOKEN_LIMIT_PARAM` 对现代 OpenAI 模型通常是 `max_completion_tokens`，DeepSeek 及 Ollama 通常使用 `max_tokens`。

若 Docker CLI 未加入当前 shell 的 `PATH`，macOS Docker Desktop 可临时执行：

```bash
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
```

启动数据库并初始化四张业务表及 FAQ 种子数据：

```bash
docker compose up -d --wait db
.venv/bin/python scripts/init_db.py
```

初始化可重复执行，不会覆盖已有密钥或删除业务数据。

## 工作流启动顺序

生产 lifespan 只检查已有 schema、语料、索引和固定模型 manifest，不会自动建表、执行 `ALTER`、调用 checkpoint `setup()` 或修复索引。首次部署按以下顺序显式准备：

1. 启动 MySQL、PostgreSQL 和 Milvus 依赖：

   ```bash
   docker compose up -d --wait db workflow-db knowledge-etcd knowledge-minio knowledge-milvus
   ```

2. 在旧写入者已停止的维护窗口执行 MySQL 工作流迁移，然后用只读模式确认就绪：

   ```bash
   .venv/bin/python scripts/migrate_workflow.py
   .venv/bin/python scripts/migrate_workflow.py --check-only
   ```

3. 显式初始化 PostgreSQL checkpoint 表：

   ```bash
   .venv/bin/python scripts/init_workflow_checkpoints.py
   ```

4. 确认第四章语料已导入、固定模型 manifest 已准备，并运行现有索引校验/构建命令。该命令只使用本地模型和本地数据：

   ```bash
   .venv/bin/python scripts/init_knowledge.py
   .venv/bin/python scripts/index_knowledge.py
   ```

5. 在独立的 8002 端口以单 worker 启动并验收：

```bash
.venv/bin/python -m uvicorn app.main:create_app --factory \
  --host 127.0.0.1 --port 8002 --workers 1
```

打开 `http://127.0.0.1:8002/` 可使用网页聊天、停止回复、新对话及售后提取。聊天的并发 guard 在进程内，因此必须使用单 worker。现有 8001 预览只在本章真实验收通过后切换；不要占用其他项目的 8000。

会话、完整消息审计和工单持久化到 MySQL，服务重启后仍可恢复。只有 `completed` 的完整轮次会回灌给模型；失败、取消或结构不完整的轮次保留作审计，但不进入后续模型上下文。`MAX_HISTORY_TURNS` 限制回灌的最近完整轮次数。进程内 guard 只负责同会话互斥和活动请求容量，所以多 worker 会绕过这项约束。

## 历史行为：第二章 API 与两阶段工具调用

以下“每轮两次模型调用”、单工具和 FAQ `LIKE` 漏召回说明是第二章历史演示合同，不是当前工作流的行为承诺。当前工作流的路由、次数上限和知识证据规则由第五章配置与图约束。

发送物流问题：

```bash
curl --noproxy '*' -N --fail-with-body http://127.0.0.1:8001/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"订单 1001 的物流到哪了"}'
```

从 `meta` 复制真实 `session_id` 可继续同一会话：

```bash
curl --noproxy '*' -N --fail-with-body http://127.0.0.1:8001/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"复制 meta 中的 session_id","message":"我刚才问的是哪个订单的物流？"}'
```

每轮聊天固定采用两个模型阶段：第一阶段决定是否调用工具；若有调用，服务最多执行一个逻辑工具（暂时性错误最多尝试两次）；第二阶段不绑定工具，并生成最终文本。即使问题无需工具，也会进入第二阶段，第一阶段的普通文本不会直接展示。因此每轮通常产生两次模型调用及相应费用。

`POST /api/chat` 返回 `text/event-stream`：

- `meta`：含 `session_id`、`turn_id`、输入 token 估算和历史裁剪数。
- `tool_status`：工具开始、重试和终态；`not_found` 是有效业务结果，不是传输失败。
- `token`：第二阶段的真实文本增量，可能出现多次。
- `done`：正常完成并包含同一个 `session_id`。
- `error`：流开始后的安全错误；出现后不会发送 `done`。

客户端必须以 `done` 判断成功。HTTP 200、部分 token 或工具完成都不代表整轮完成。

售后提取仍是独立的单次 JSON 接口：

```bash
curl --noproxy '*' --fail-with-body http://127.0.0.1:8001/api/after-sales/extract \
  -H 'Content-Type: application/json' \
  -d '{"description":"订单 A123 到货破损，希望换一个新的"}'
```

## 演示与标注评估

三条独立工具流演示会依次运行物流、退货政策和邮费问题，并校验每条流都以 `done` 结束。邮费的 `not_found` 是预期结果：FAQ 使用当前问题中的字面关键词查询，种子数据故意没有“邮费”，脚本不会把该业务结果当成失败。

```bash
BASE_URL=http://127.0.0.1:8001 bash scripts/demo_tools.sh
```

第二章评估器读取 `evals/ch02-cases.json` 的九组标注（共十轮），通过真实 HTTP SSE 发问，再用同一 `.env` 的 `DATABASE_URL` 读取 `demo` 用户的会话审计。它按 `meta.turn_id` 隔离每轮，自动检查 `done`、轮次完成状态、最多一次模型工具申请、调用与结果配对、标注参数和结果状态。多轮样例复用首轮 `session_id`，但分别判定每一轮。

```bash
.venv/bin/python scripts/evaluate_tools.py \
  --base-url http://127.0.0.1:8001 \
  --output evals/reports/ch02-latest.json
```

协议或工具证据失败会令命令返回非零。自动通过只表示协议及标注结构匹配；每条 `semantic_review` 始终为 `pending_manual_review`，必须按标注集中的 rubric 人工核对最终回答。报告仅保存问题、会话/轮次标识、安全 SSE 事件、实际工具调用与结果及最终文本，不读取或输出密钥。第一章的提取与普通多轮回归仍可用 `scripts/evaluate.py` 和 `scripts/demo.sh` 独立运行。

## 测试

离线测试不产生真实模型费用：

```bash
.venv/bin/python -m pytest -q
```

MySQL 集成测试使用隔离的 `support_test` 数据库，不能指向开发库。Docker Compose 从 `.env` 读取 `MYSQL_TEST_PASSWORD`，pytest 只从进程环境或被 Git 忽略的 `.env.test` 读取 `TEST_DATABASE_URL`。创建 `.env.test`，并让 URL 密码与 `.env` 中的 `MYSQL_TEST_PASSWORD` 一致；不要填写真实生产凭据：

```dotenv
# .env.test
TEST_DATABASE_URL=mysql+asyncmy://support_test:replace-support-test-password@127.0.0.1:13307/support_test
```

随后启动测试库并强制运行集成测试：

```bash
docker compose --profile test up -d --wait test-db
.venv/bin/python -m pytest tests/integration --require-mysql -q
```

离线和 MySQL 测试验证代码、持久化及协议失败处理，不能替代真实模型的工具选择、语义 rubric 或浏览器验收。

## 错误与限制

在 SSE 开始前，HTTP 错误为 `{"error":{"code":"...","message":"..."}}`。常见错误包括：

| HTTP | `code` | 含义 |
| --- | --- | --- |
| 422 | `INVALID_REQUEST` | 请求字段、类型或长度不符合要求 |
| 413 | `INPUT_TOO_LONG` | 必需提示、工具 schema 与当前输入无法放入预算 |
| 404 | `SESSION_NOT_FOUND` | 会话不存在或不属于当前固定演示用户 |
| 409 | `SESSION_BUSY` | 同一会话已有进行中的请求 |
| 503 | `SESSION_CAPACITY` | 当前进程的活动会话数达到上限 |
| 503 | `DB_ERROR` | 会话审计暂时无法保存 |
| 504 | `UPSTREAM_TIMEOUT` | 整轮请求超过共享时限 |
| 502 | `INVALID_TOOL_CALL` | 模型返回多个、缺少标识或格式无效的工具调用 |
| 502 | `UPSTREAM_ERROR` | 上游连接或调用失败 |
| 502 | `UPSTREAM_INCOMPLETE` | 最终文本为空或未正常完成 |
| 502 | `STRUCTURED_OUTPUT_ERROR` | 售后提取结果未通过 JSON/Pydantic 校验 |

单条 `message` 或 `description` 最多 32,000 个字符。上下文预算包含系统提示、五个工具 schema、完整历史轮次、本轮工具调用及结果。`estimated_input_tokens` 基于 UTF-8 字节数保守估算，并非供应商计费 tokenizer 的精确值；应用还为输出和安全余量预留空间。
