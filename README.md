# 中文电商客服示例

这是一个基于 FastAPI、LangChain 和 OpenAI 兼容 Chat Completions 接口的本地示例。它提供流式客服聊天和售后信息提取，不包含订单查询、真实退款、工具调用、Agent、RAG、数据库或生产鉴权。

## 安装与配置

需要 Python 3.11 或更高版本。`requirements.lock` 锁定了开发和运行时的完整依赖；它不包含本仓库自身，所以安装锁文件后再以无依赖模式安装项目。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps -e .
[ -f .env ] || cp .env.example .env
```

最后一条命令会保留已有 `.env`，不会覆盖已配置的密钥。编辑 `.env` 时必须填写供应商实际可用的模型名：

```dotenv
LLM_BASE_URL=https://api.example.com/v1
LLM_MODEL=填写账户实际可用的模型名
LLM_API_KEY=填写密钥
LLM_TOKEN_LIMIT_PARAM=max_completion_tokens
CONTEXT_WINDOW_TOKENS=8192
MAX_OUTPUT_TOKENS=1024
TOKEN_SAFETY_MARGIN=512
MAX_HISTORY_TURNS=12
SESSION_TTL_SECONDS=3600
MAX_SESSIONS=100
REQUEST_TIMEOUT_SECONDS=60
```

常见供应商的关键设置如下。模型名会随账户和服务变化，因此这里不硬编码具体型号。

| 供应商 | `LLM_BASE_URL` | `LLM_API_KEY` | `LLM_TOKEN_LIMIT_PARAM` | 说明 |
| --- | --- | --- | --- | --- |
| OpenAI | `https://api.openai.com/v1` | OpenAI API key | `max_completion_tokens` | 现代 GPT 模型使用该输出 token 字段 |
| DeepSeek | `https://api.deepseek.com` | DeepSeek API key | `max_tokens` | 使用 OpenAI 兼容接口 |
| Ollama | `http://127.0.0.1:11434/v1/` | 客户端要求非空值，可填 `ollama` | `max_tokens` | 先在本机安装并启动用户选择的模型 |
| Claude OpenAI 兼容端点 | `https://api.anthropic.com/v1/` | Anthropic API key | `max_tokens` | 兼容层会忽略 `response_format`；本应用仍用提示词和 Pydantic 校验提取结果，校验失败会报错 |

## 启动

只启动一个 worker，并默认监听本机：

```bash
.venv/bin/python -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

会话只保存在当前进程内存中。重启会清空全部会话；多 worker 之间也不会共享会话，因此本示例固定使用单 worker。

## API 调用

第一轮聊天：

```bash
curl --noproxy '*' -N --fail-with-body -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"我叫小林，刚买的耳机有杂音"}'
```

从上一条 `meta` 事件复制真实 `session_id`，再发第二轮：

```bash
curl --noproxy '*' -N --fail-with-body -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"复制上一条 meta 的 session_id","message":"我叫什么，商品出了什么问题？"}'
```

售后提取：

```bash
curl --noproxy '*' --fail-with-body -X POST http://127.0.0.1:8000/api/after-sales/extract \
  -H 'Content-Type: application/json' \
  -d '{"description":"订单 A123 到货破损，希望换一个新的"}'
```

也可以运行完整演示。脚本会从首轮 `meta` 读取并校验 UUID，第二轮复用该会话，再调用提取接口；HTTP 失败、SSE `error` 或流中缺少 `done` 都会让脚本返回非零。

```bash
bash scripts/demo.sh
```

### SSE 协议

`POST /api/chat` 返回 `text/event-stream`，事件顺序为：

- `meta`：`session_id`、`estimated_input_tokens`、`token_count_is_estimate: true` 和 `dropped_turns`。
- `token`：每个真实上游增量放在 `content` 字段，可能出现多次。
- `done`：正常完成，包含同一个 `session_id`。
- `error`：流已开始后失败，包含安全的 `code` 和 `message`；出现 `error` 时不会再发送 `done`。

只有完整且正常结束的轮次才写入会话历史。客户端应以 `done` 作为成功依据，不能把 HTTP 200 或收到部分 token 当成完成。

## 错误与限制

在开始 SSE 响应前，HTTP 错误使用 `{"error":{"code":"...","message":"..."}}`。可能的错误码包括：

| HTTP | `code` | 含义 |
| --- | --- | --- |
| 422 | `INVALID_REQUEST` | 请求字段、类型或长度不符合要求 |
| 413 | `INPUT_TOO_LONG` | 系统提示和当前输入在预留输出空间后仍超出上下文预算 |
| 404 | `SESSION_NOT_FOUND` | 会话不存在、已过期或服务已重启 |
| 409 | `SESSION_BUSY` | 同一会话已有进行中的请求 |
| 503 | `SESSION_CAPACITY` | 进程内会话数达到上限 |
| 504 | `UPSTREAM_TIMEOUT` | 上游超过配置的请求时限 |
| 502 | `UPSTREAM_ERROR` | 上游连接或调用失败 |
| 502 | `UPSTREAM_INCOMPLETE` | 聊天未以正常停止原因完整结束 |
| 502 | `STRUCTURED_OUTPUT_ERROR` | 提取结果为空、不完整或未通过 JSON/Pydantic 校验 |

单条 `message` 或 `description` 最多 32,000 个字符。上下文会保留系统消息和当前输入，并按完整历史轮次从最旧处裁剪；还受 `MAX_HISTORY_TURNS` 限制。`estimated_input_tokens` 是基于 UTF-8 字节数的保守估算，不是供应商计费 tokenizer 的精确值。应用为输出预留 `MAX_OUTPUT_TOKENS`，并额外保留 `TOKEN_SAFETY_MARGIN`，因此配置的 `CONTEXT_WINDOW_TOKENS` 必须大于两者之和。

## 标注评估

启动服务并确认它连接到真实模型后运行：

```bash
.venv/bin/python scripts/evaluate.py \
  --base-url http://127.0.0.1:8000 \
  --output evals/reports/latest.json
```

评估器对 `evals/cases.json` 中的提取样例做三字段自动比对，并记录每条实际输出。网络、HTTP、协议或结构化匹配失败都会返回非零。普通客服回复只记录实际多轮输出和人工 rubric，状态明确为 `pending_manual_review`；脚本不会用关键词命中冒充完整语义正确。报告不会读取或写入 API key，`evals/reports/` 默认不提交版本控制。

真实模型和具体模型名必须由运行者配置。本仓库的离线单元测试只验证代码、HTTP 协议和失败处理，不能替代真实模型的 Prompt 与语义验收。
