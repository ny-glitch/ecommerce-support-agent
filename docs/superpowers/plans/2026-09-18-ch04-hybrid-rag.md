# Chapter 04 Hybrid RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 SSE 客服聊天中完成可引用、可拒答的 Milvus 混合检索与重排，并提供四策略真实评估报告。

**Architecture:** 扩展现有 query_faq，原文以 MySQL knowledge_chunks 为准，Milvus knowledge 保存原生 BM25 与本地 bge-m3 dense 索引。问题归一化、双路召回、重排、证据预算及一次自评组成固定流程；保留其他业务工具与会话审计。

**Tech Stack:** Python 3.13、FastAPI 0.141.1、SQLAlchemy 2.0.54、Docker MySQL 8.4.11、LangChain core/openai 1.6.3/1.6.2、Milvus 2.6.23 / pymilvus 2.6.17、FlagEmbedding、BAAI/bge-m3、BAAI/bge-reranker-v2-m3、现有 DeepSeek。

**Spec:** [已批准的书面规格](/Users/ny/Documents/ChatGPT/智能客服/docs/superpowers/specs/2026-09-18-ch04-hybrid-rag-design.md)，用户回复“规格通过”。

## Global Constraints

- MySQL knowledge_chunks 是原文权威源，用户原始 DDL 的字段、默认值、注释、索引与外键不变；新增 qa_extraction_staging 和 low_confidence_questions。
- 只将 category/questions/answer 拼成检索文本；其他字段不进入 dense/BM25 文本，同义词不回写入库。
- Milvus 集合名为 knowledge；INT64 主键 auto_id=False，索引 ID 支持 1..2^63-1。
- 内置 chinese analyzer、原生 BM25、dense/BM25 各 Top-50、hybrid_search、RRFRanker(k=60)、融合 Top-50、bge-reranker-v2-m3 Top-10。
- 查询归一化与证据自评只读本轮问题，不作指代消解、多轮改写或 Agent Loop。
- 模型相关性分数不是正确率；正式集不调阈值，拒答的 Faithfulness 为 N/A。
- 原业务工具结果上限 4096 UTF-8 字节；query_faq 专门结果上限 48000 字节，裁剪整块后只自评一次。
- 后端 TDD/评审；Prompt、资料与标注用评估验证；前端 Vibe Coding，不做页面 brainstorm/TDD/code review。
- 每完成任务或评审立刻追加 dev-notes/ch04.md 四项记录，再提交；不在 finish 集中补记。
- 所有库/API 先查 Context7，再核对锁定版本；固定选型走不通时停止相关实现并询问，不换栈。
- 不输出 .env、DSN、密钥、展开的 Compose 配置；演示 MySQL 3307 与测试 MySQL 13307 隔离。
- 保留现有 8001 预览直到新版本验证可启动；不占用其他项目的 8000。

## 执行准备与验证依据

当前普通 checkout，基线 3087e93；本轮执行 `.venv/bin/python -m pytest --require-mysql -q --tb=short` 得到 244 passed、零跳过、1 条既有 Starlette/AnyIO 弃用警告，24.67 秒。实现前按 using-git-worktrees 检查状态；另建 worktree 需要用户明确选择，未选择时在当前目录建立 codex/ch04-hybrid-rag 分支，不改写既有历史。

2026-09-18 的 PyPI dry-run 在现有 requirements.lock 约束下成功解析以下新顶层版本，没有安装：FlagEmbedding==1.4.2、pymilvus==2.6.17、torch==2.14.0、transformers==4.57.6、sentence-transformers==5.1.2、peft==0.18.1。Torch 提供 CPython 3.13/macOS arm64 wheel。完整解析报告暂存 `/private/tmp/ch04-dependencies.json`；Task 3 安装后执行 import smoke/pip check 再锁定全部依赖，解析成功不代表运行验证通过。

模型 revision 已从官方元数据核对：bge-m3 为 `5617a9f61b028005a4858fdac845db406aefb181`；bge-reranker-v2-m3 为 `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`。下载时记录 revision 和本地清单，不依赖漂移的 main。

Context7 本轮已查 FlagEmbedding dense/score 接口、PyMilvus schema/AnnSearchRequest/Strong/upsert、SQLAlchemy unsigned BIGINT/ON UPDATE/短事务。并只读核对 PyMilvus v2.6.17 源码：MilvusClient.hybrid_search 的参数名为 ranker，AnnSearchRequest 支持 expr/expr_params；expr 与 filter 不同时传。

Milvus v2.6.23 仓库 Compose 模板仍写 milvus:v2.6.22，实施显式改为 v2.6.23；etcd v3.5.25 和 MinIO RELEASE.2024-05-28T17-19-04Z 取自该模板。执行预检已成功拉取三个 ARM64 镜像；MinIO 的 Docker Hub 仓库不可用，按同版本官方 README 使用 quay.io/minio/minio:RELEASE.2024-05-28T17-19-04Z（组件与版本不变）。实际运行仍是 Task 4 的验收项。

## 文件和任务依赖

| Task | 交付与主要文件 | 依赖 |
| --- | --- | --- |
| 1 | 三表、原文/问题池仓储、文本指纹：app/db/knowledge_models.py、knowledge.py、low_confidence.py，app/knowledge/contracts.py、text.py | 既有 DB |
| 2 | 120 条知识、60 正式/30 校准标注、幂等导入：data/knowledge/ch04、evals/ch04、app/knowledge/corpus.py、scripts/init_knowledge.py | 1 |
| 3 | 可取消的本地模型适配、依赖和下载：app/knowledge/local_models.py、worker.py、scripts/prepare_knowledge_models.py | 1 |
| 4 | Milvus 原生 BM25、索引恢复：app/knowledge/milvus_store.py、indexer.py、scripts/index_knowledge.py、compose.yaml | 1–3 |
| 5 | 当前问题理解及四策略排名：app/knowledge/query.py、retrieval.py、gateway.py | 1、3、4 |
| 6 | 证据预算、自评、引用、拒答：app/knowledge/evidence.py、pipeline.py、prompts 与 gateway | 5 |
| 7 | 既有工具与持久化聊天编排：app/tools、app/services/chat.py、knowledge_turn.py、app/api/streaming.py | 6 |
| 8 | 生命周期、品类/原文 API、设置与打包：app/main.py、app/api/knowledge.py、schemas.py | 7 |
| 9 | 评估 CLI、阈值、指标和真实四策略运行：app/knowledge/evaluation.py、scripts/evaluate_knowledge.py | 2、5、6 |
| 10 | 页面引用、状态和本地反馈：app/web/index.html | 8；Vibe Coding |
| 11 | 真实浏览器/curl/MySQL 验收、回归、最终评审与 finish | 全部 |

本计划是一条端到端功能链，不拆成互不相干的子项目。实现默认按任务顺序；Task 9 的正式评估可在页面改造前完成，Task 8 的测试注入校准配置，真实启动须等 Task 9 生成生产校准产物。

## Task 1：用户 DDL 与可恢复的知识仓储

**Files:** Create `app/db/knowledge_models.py`、`app/db/knowledge.py`、`app/db/low_confidence.py`、`app/knowledge/__init__.py`、`app/knowledge/contracts.py`、`app/knowledge/text.py`、`tests/ch04_helpers.py`、`tests/test_knowledge_text.py`、`tests/integration/test_knowledge_repository.py`；Modify `app/db/__init__.py`、`app/db/database.py`、`tests/test_db_schema.py`、`tests/integration/conftest.py`、`tests/integration/test_db_init.py`。

**Interfaces:** `KnowledgeRepository(sessions)` 提供 async get(id)->KnowledgeChunk|None、get_many(ids)->dict[int,KnowledgeChunk]、list_all()->list[KnowledgeChunk]、categories()->list[str]、insert_seed(chunks)->None、mark_pending(ids)->None、mark_done_if_current(id,expected_hash)->bool。最后一个方法锁行、重算指纹，一致才填 vector_id/置 done。`LowConfidenceRepository(sessions).record_once(ref:TurnRef,question:str,reason_code:str,reason:str,entry_point:str='chat')->int` 返回稳定主键。

在 contracts.py 定义以下不可变数据对象；后续任务扩展自己的对象，不给已有对象改字段名：

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class KnowledgeChunk:
    id: int
    category: str
    questions: str
    answer: str
    section_path: str | None = None
    content_type: str | None = None
    is_key_clause: bool = False
    prev_chunk_id: int | None = None
    next_chunk_id: int | None = None
    vector_id: str | None = None
    vectorize_status: Literal['pending', 'done'] = 'pending'

@dataclass(frozen=True)
class SearchHit:
    id: int
    score: float
    source_hash: str

@dataclass(frozen=True)
class RankedChunk:
    chunk: KnowledgeChunk
    score: float
```

- [x] **Step 1：写原文边界与真实数据库 RED。** `tests/ch04_helpers.py` 的 `make_chunk(**changes)` 返回 `dataclasses.replace(KnowledgeChunk(910001,'数码配件/充电器','C65-Pro 支持什么协议？','支持 PD 3.0。','商品手册/C65-Pro/协议','manual'),**changes)`。mysql_db fixture 清理顺序增加 low_confidence_questions、qa_extraction_staging，并先将 knowledge_chunks 的 prev/next 置 NULL 再 DELETE，最后才清理原四表。更新“只有四表”的旧断言为七表，同时保留原四表字段约束检查。

```python
from dataclasses import replace
from app.knowledge.text import embedding_text, source_hash
from tests.ch04_helpers import make_chunk

def test_metadata_does_not_enter_embedding_but_changes_source_revision():
    a = make_chunk()
    b = replace(a, section_path='商品手册/修订版/协议', vector_id='910001', vectorize_status='done')
    assert embedding_text(a) == embedding_text(b)
    assert source_hash(a) != source_hash(b)
    assert source_hash(a) == source_hash(replace(a, vector_id='910001', vectorize_status='done'))

async def test_seed_and_conditional_done(mysql_db):
    from app.db.knowledge import KnowledgeRepository
    repo = KnowledgeRepository(mysql_db.sessions)
    chunk = make_chunk()
    await repo.insert_seed([chunk])
    await repo.insert_seed([chunk])
    assert len(await repo.list_all()) == 1
    assert not await repo.mark_done_if_current(chunk.id, 'outdated')
    assert (await repo.get(chunk.id)).vectorize_status == 'pending'
    assert await repo.mark_done_if_current(chunk.id, source_hash(chunk))
    assert (await repo.get(chunk.id)).vector_id == str(chunk.id)
```

Run: `.venv/bin/python -m pytest tests/test_knowledge_text.py tests/integration/test_knowledge_repository.py --require-mysql -q`。预期先因缺少新模块/方法而失败，保留 RED 输出。

- [x] **Step 2：最小实现七表与仓储。** KnowledgeChunkRecord/QAExtractionStaging 的字段与规格附录逐项对齐，中文 comment 完整保留；updated_at 用 `server_default=text('CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP')`，不是仅设置 Python onupdate。app/db/__init__.py 先导入现有 models 再导入 knowledge_models，保证任何入口导入 Base 都已注册七表，knowledge_models 只向 models 单向导入 Base。源指纹使用排序 JSON 的 UTF-8 SHA256，字段集合显式列出；不 hash ORM 对象或状态列。

```python
import hashlib
import json

def embedding_text(chunk):
    return f'分类：{chunk.category}\n问法：{chunk.questions}\n答案：{chunk.answer}'

def source_hash(chunk):
    fields = ('category','questions','answer','section_path','content_type',
              'is_key_clause','prev_chunk_id','next_chunk_id')
    value = {name: getattr(chunk, name) for name in fields}
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()
```

insert_seed 使用单次短事务做全批冲突核验、插入原文、设置指针；已有 ID 内容不同则整批回滚，不覆盖。record_once 校验 ref 会话存在，唯一键竞争后使用新 Session 回读同一记录；不得用会话存在代替轮次原话匹配。数据库异常对外脱敏。

- [x] **Step 3：GREEN 与 DDL 实证。** 重跑 RED 命令及现有数据库测试；真实 Inspector/SHOW CREATE TABLE 核对 unsigned、自引用 ON DELETE SET NULL、ENUM、ON UPDATE、utf8mb4 和中文注释。测试删除前块导致相邻指针置空、批量冲突不部分导入、同轮重复入池只一行、已变原文不能误标 done。
- [x] **Step 4：评审、留痕、提交。** 完成规格/代码质量评审后记录实际命令、结果和返工，提交 `feat: add authoritative knowledge storage and refusal pool`。

## Task 2：可溯源的演示资料与独立标注集

**Files:** Create `data/knowledge/ch04/chunks.json`、`data/knowledge/ch04/manifest.json`、`evals/ch04/calibration.jsonl`、`evals/ch04/test.jsonl`、`app/knowledge/corpus.py`、`scripts/init_knowledge.py`、`scripts/validate_knowledge_data.py`、`dev-notes/ch04-data-evaluation.md`、`tests/test_knowledge_corpus.py`。

**Interfaces:** `load_corpus(path:Path)->list[KnowledgeChunk]`；`validate_corpus(chunks)->list[str]` 返回错误列表；`load_cases(path:Path)->list[dict]`，case 字段为 query_id/query/category/relevant_chunk_ids/reference_answer/answerable/query_type/difficulty/rationale。`corpus_fingerprint(chunks)->str` 将 ID 与 source_hash 按 ID 排序后 hash。

- [x] **Step 1：先建标注样例与数据评审表。** 本任务纯资料的 TDD 替代为标注验证：四商品类各24+通用24，ID 为 910001..910120，manifest 将 source_key/source_document/ID 映射固定。每类资料独立成章；不把同义词扩展成新条目。正式六桶各十条；校准五个可回答桶各四条+无答案十条；两个集合 query_id 和原话不重合。

```json
{"query_id":"test-model-01","query":"C65-Pro 能用 PD 3.0 吗？","category":"数码配件/充电器","relevant_chunk_ids":[910001],"reference_answer":"C65-Pro 支持 PD 3.0。","answerable":true,"query_type":"model","difficulty":"easy","rationale":"型号和协议均在 910001 原文中明确出现。"}
```

该条必须对应实际原文；其余条目逐条标注，不用脚本把一个问句替换序号生成60条。型号、近似型号、售后条件、跨块证据和未知内容都需人工阅读验证。记录原文内容冲突、缺少条件及重写过程。

- [x] **Step 2：为导入校验写 RED 后实现。** 只对可单测的加载/冲突校验写测试，资料答案质量不写字符串包含式单测。

```python
from dataclasses import replace
from app.knowledge.corpus import validate_corpus
from tests.ch04_helpers import make_chunk

def test_missing_neighbor_is_rejected():
    errors = validate_corpus([replace(make_chunk(), next_chunk_id=999999)])
    assert any('999999' in error for error in errors)
```

Run: `.venv/bin/python -m pytest tests/test_knowledge_corpus.py -q`。实现解析时使用显式字段校验、唯一 ID、完整章节路径、合法相邻指针和 source_key；正式问题目标 ID 必须存在且符合 category，可回答项 reference_answer 非空，无答案项 relevant IDs 为空。

- [x] **Step 3：执行标注验证并冻结。** `scripts/validate_knowledge_data.py` 调用校验函数，报告语料/集合数量、桶/难度分布、跨集合重复、标注引用、内容哈希；任一错误非零退出。`scripts/init_knowledge.py` 显式 create_schema 后只调用知识种子仓储，不调用旧 FAQ seed、不清空数据；默认读取仓库文件，可通过 --corpus 指定文件。运行两次核对计数不变。

```bash
.venv/bin/python scripts/validate_knowledge_data.py
.venv/bin/python scripts/init_knowledge.py
.venv/bin/python scripts/init_knowledge.py
```

- [x] **Step 4：评估记录与提交。** dev-notes/ch04-data-evaluation.md 逐类记录答案来源、条件保留、未知项无误标、资料无真实品牌承诺；记阶段日志并提交 `data: add traced ecommerce corpus and held-out evaluation cases`。

## Task 3：本地 dense/reranker 与有界推理

**Files:** Create `app/knowledge/worker.py`、`app/knowledge/local_models.py`、`scripts/prepare_knowledge_models.py`、`tests/test_knowledge_worker.py`、`tests/test_local_knowledge_models.py`、`tests/integration/test_local_knowledge_models.py`；Modify `app/config.py`、`.env.example`、`.gitignore`、`pyproject.toml`、`requirements.lock`、`tests/conftest.py`。

**Interfaces:** `LocalModels(settings).warmup()->None`、`async embed(texts:list[str],*,deadline:float)->list[list[float]]`、`async score(query:str,texts:list[str],*,deadline:float)->list[float]`、`async aclose()->None`。warmup 由 CLI 直接调用，应用通过工作线程调用。`InferenceWorker(queue_size:int).run(fn:Callable[[],T],*,deadline:float)->T` 为 async 串行有界任务入口，另有 async aclose()->None；队列满返回服务繁忙，取消后底层任务真正结束前不释放容量。

新增设置：`milvus_uri='http://127.0.0.1:19530'`、`milvus_collection='knowledge'`、`milvus_token:SecretStr|None=None`、`knowledge_models_dir='.cache/ch04/models'`、`knowledge_calibration_path='.cache/ch04/calibration.json'`、`knowledge_request_timeout_seconds=240`、`knowledge_batch_size=4`、`knowledge_worker_queue_size=4`，型号和 revision 使用上述固定值。将 `.cache/` 加入 gitignore；配置错误不得回显 secret。

- [x] **Step 1：安装、导入检查与 CPU 基线准备。** 用已解析的版本安装，保留现有锁约束；必要的源构建使用正常包构建流程，不禁 TLS、不跳依赖安装。certifi 路径只作为本次 PIP_CERT，不修改系统 CA。

```bash
PIP_CERT="$PWD/.venv/lib/python3.13/site-packages/certifi/cacert.pem" .venv/bin/python -m pip install -c requirements.lock 'FlagEmbedding==1.4.2' 'pymilvus==2.6.17' 'torch==2.14.0' 'transformers==4.57.6' 'sentence-transformers==5.1.2' 'peft==0.18.1'
.venv/bin/python -c 'from FlagEmbedding import BGEM3FlagModel, FlagReranker; from pymilvus import MilvusClient, Function, FunctionType; import torch; print(torch.__version__)'
.venv/bin/python -m pip check
```

导入不通过先按 systematic-debugging 找到具体依赖，不把 dry-run 当通过。新增上述顶层固定依赖，刷新完整 lock 时排除本项目 editable 路径；保留原业务依赖版本。

- [x] **Step 2：写 worker 取消与输入长度 RED。** 用 threading.Event 在测试内构造受控阻塞函数，观察取消后第二个函数在第一函数实际结束前不能运行；finally 必须释放测试 Event，避免悬挂。长度测试注入可计数 tokenizer/model 替身，超出上限时模型调用数为0，不默默 truncation。

```python
import asyncio
import threading
import time
import pytest
from app.knowledge.worker import InferenceWorker

async def test_cancelled_running_job_keeps_slot_until_finished():
    worker = InferenceWorker(queue_size=4)
    entered, release, second = threading.Event(), threading.Event(), threading.Event()
    def first():
        entered.set()
        release.wait(2)
    task = asyncio.create_task(worker.run(first, deadline=time.monotonic()+5))
    try:
        await asyncio.to_thread(entered.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        next_task = asyncio.create_task(worker.run(second.set, deadline=time.monotonic()+5))
        await asyncio.sleep(0.02)
        assert not second.is_set()
        release.set()
        await next_task
    finally:
        release.set()
        await worker.aclose()
```

Run: `.venv/bin/python -m pytest tests/test_knowledge_worker.py tests/test_local_knowledge_models.py -q`。

- [x] **Step 3：实现 adapter 和显式下载。** `prepare_knowledge_models.py` 使用官方 snapshot_download，固定 revision/local_dir，只下载模型、tokenizer、config、M3 所需的 linear.pt 等文件，排除 onnx/及其他导出副本。不要在 HTTP 请求中下载。加载本地路径，CPU/use_fp16=False；embed 指定 return_dense=True、return_sparse=False、return_colbert_vecs=False；score 使用 normalize=True。验证 1024 维、有限数值、输入/输出条数一致。逐批执行，批次之间检查截止时间/取消，评分完整文本，超长输入显式报错。

```python
vectors = embedder.encode(texts, batch_size=4, max_length=8192,
    return_dense=True, return_sparse=False, return_colbert_vecs=False)['dense_vecs']
scores = reranker.compute_score([[query, text] for text in texts],
    batch_size=4, normalize=True)
```

max_length 与 query/passage 联合长度以已下载 tokenizer 和模型限制核对，不能让示例参数触发隐式截断。将单条 compute_score 的 scalar 归一为一元素列表。Worker 的执行容量在实际线程 Future 的 done 回调里释放，不在等待方取消时释放。

- [x] **Step 4：GREEN、真实模型与提交。** 新增 --require-local-models，缺模型时普通测试可跳过，要求真实模型时必须失败。下载、预热并跑两条 dense、一对相关/不相关评分及50候选计时，记录模型 revision、维度、耗时、峰值内存；不保证未测的低延迟。评审后即时记日志并提交 `feat: add bounded local embedding and reranking runtime`。

## Task 4：Milvus 原生 BM25 与双写索引

**Files:** Create `app/knowledge/milvus_store.py`、`app/knowledge/indexer.py`、`scripts/index_knowledge.py`、`tests/test_knowledge_indexer.py`、`tests/integration/test_knowledge_milvus.py`；Modify `compose.yaml`、`.env.example`、`tests/conftest.py`、`tests/integration/conftest.py`。

**Interfaces:** `MilvusStore(settings,*,collection_name:str|None=None)` 的 async 方法：ensure_schema()->None、upsert(chunks:list[KnowledgeChunk],vectors:list[list[float]],*,deadline:float)->None、fingerprints(ids:list[int])->dict[int,str]、search_dense(vector:list[float],category:str|None,*,deadline:float)->list[SearchHit]、search_bm25(text:str,category:str|None,*,deadline:float)->list[SearchHit]、search_hybrid(vector:list[float],text:str,category:str|None,*,deadline:float)->list[SearchHit]、check()->None、aclose()->None。`validate_test_collection(name:str)->None` 只接受 ch04_test_ 加32位十六进制UUID。`KnowledgeIndexer(repo,store,models,*,lock_path:Path).run(*,repair:bool=False)->dict[str,int]` 为 async，返回 indexed/skipped/failed 计数，任一失败 CLI 非零退出。

- [x] **Step 1：实现环境与测试隔离约束的 RED。** --require-milvus 配置必须存在服务，否则失败；测试集合为 ch04_test_<uuid>，清理函数拒绝不匹配此前缀的集合，尤其 knowledge。先写 indexer 的失败恢复测试：注入 upsert 成功而 mark_done 返回False/抛异常的仓储，再次运行只产生相同 ID 的 upsert，不能 done/新增不同主键。测试 store 的 schema 不兼容不 drop。

```python
import pytest
from app.knowledge.milvus_store import validate_test_collection

@pytest.mark.parametrize('name', ['knowledge', 'support', 'ch04_test_'])
def test_cleanup_rejects_non_test_collection(name):
    with pytest.raises(ValueError):
        validate_test_collection(name)
```

恢复的可运行最小用例（同文件使用临时锁，禁止实际访问演示库）：

```python
from dataclasses import replace
from app.knowledge.indexer import KnowledgeIndexer
from tests.ch04_helpers import make_chunk

async def test_repeat_upsert_uses_same_id_after_unconfirmed_sql(tmp_path):
    chunk = make_chunk()
    class Repo:
        def __init__(self):
            self.value, self.attempts = chunk, 0
        async def list_all(self):
            return [self.value]
        async def mark_done_if_current(self, chunk_id, expected_hash):
            self.attempts += 1
            if self.attempts == 1:
                return False
            self.value = replace(self.value, vectorize_status='done', vector_id=str(chunk_id))
            return True
    class Store:
        def __init__(self):
            self.ids = []
        async def ensure_schema(self):
            return None
        async def upsert(self, chunks, vectors, *, deadline):
            self.ids.extend(row.id for row in chunks)
    class Models:
        async def embed(self, texts, *, deadline):
            return [[1.0] + [0.0]*1023 for text in texts]
    repo, store = Repo(), Store()
    indexer = KnowledgeIndexer(repo, store, Models(), lock_path=tmp_path/'index.lock')
    first = await indexer.run()
    assert first['failed'] == 1
    assert repo.value.vectorize_status == 'pending'
    second = await indexer.run()
    assert second['indexed'] == 1
    assert store.ids == [chunk.id, chunk.id]
```

Run: `.venv/bin/python -m pytest tests/test_knowledge_indexer.py tests/integration/test_knowledge_milvus.py --require-milvus -q`。环境缺失时先记录真实原因，不将 skip 算 GREEN。

- [x] **Step 2：增加 standalone 容器并验证版本。** 添加 knowledge-etcd/knowledge-minio/knowledge-milvus 服务、独立卷，内部地址用服务名，无全局 container_name；etcd/MinIO 不暴露宿主端口，Milvus 19530/9091 仅127.0.0.1。MinIO 凭据放忽略的 .env，通过环境项传给两个服务，不打印 Compose 渲染。采用对应官方 healthcheck、依赖健康等待，镜像版本如准备段；实际 docker image inspect 记录架构/digest，client.get_server_version 核对2.6.23。

```bash
/Applications/Docker.app/Contents/Resources/bin/docker compose up -d --wait knowledge-milvus
```

网络故障按具体 registry/DNS/代理错误排查；不从未知镜像源拉同名替代品。模型或镜像暂不可用时继续不依赖它的任务，但 Task4 不标完成。

- [x] **Step 3：实现 schema、预过滤与真实 hybrid API。** 底层使用同步 MilvusClient，通过有界线程调用并设置每次 RPC timeout；所有检索 Strong consistency。text max_length=65535，dense dim=1024，category/section字段按规格，BM25输出不由客户端填写。HNSW 设置 M=16/efConstruction=200，搜索 ef=100；SPARSE_INVERTED_INDEX 使用 BM25。collection已存在时比对字段/Function/analyzer/index，任何不兼容受控退出。

```python
from pymilvus import AnnSearchRequest, RRFRanker
import json

expr = '' if category is None else 'category == ' + json.dumps(category, ensure_ascii=False)
reqs = [
    AnnSearchRequest([vector], 'dense_vector', {'metric_type':'COSINE','params':{'ef':100}}, 50, expr=expr),
    AnnSearchRequest([bm25_text], 'sparse_vector', {'metric_type':'BM25','params':{}}, 50, expr=expr),
]
hits = client.hybrid_search(collection_name, reqs, RRFRanker(k=60), limit=50,
    output_fields=['source_hash'], consistency_level='Strong', timeout=remaining)
```

表达式只允许固定 category 字段，使用 JSON 字符串编码；控制字符、引号和反斜杠有真实过滤测试，失败则改用SDK已核对的 expr_params，不允许直接 f-string 插原值。

- [x] **Step 4：实现索引器与恢复。** 本机索引 CLI 使用 `fcntl.flock(LOCK_EX|LOCK_NB)`，锁名在 tempfile.gettempdir() 下，以数据库host/port/database及Milvus集合名的SHA256作后缀，不含凭据；同一库从不同worktree运行也共享同一把锁。lock_path由CLI计算并注入，测试使用tmp_path。锁覆盖整个索引过程，退出释放；仅声明单机互斥。依次校验 ID/字节/模型 token 长度、upsert 确认、短事务 mark_done_if_current；普通模式扫描 pending，--repair 同时核查 done 行缺失/指纹失配后重置 pending。记录失效 ID，不泄露原文与密钥；取消后未确认行仍 pending。

```bash
.venv/bin/python scripts/index_knowledge.py
.venv/bin/python scripts/index_knowledge.py --repair
.venv/bin/python -m pytest tests/test_knowledge_indexer.py tests/integration/test_knowledge_milvus.py --require-milvus --require-mysql -q
```

- [x] **Step 5：实证、评审、提交。** 在真实 Milvus 上 run_analyzer 检查型号/中英数字；BM25 查询C65-Pro命中目标；两路相同品类过滤；第二次索引不重复；构造过期原文验证恢复。写日志并提交 `feat: add native milvus bm25 and recoverable indexing`。

## Task 5：当前问题归一化与四策略检索

**Files:** Create `app/knowledge/query.py`、`app/knowledge/retrieval.py`、`app/knowledge/gateway.py`、`app/prompts/query_normalization.txt`、`tests/test_knowledge_query.py`、`tests/test_knowledge_retrieval.py`；Modify `app/knowledge/contracts.py`。

**Interfaces:** contracts.py 新增 `QueryPlan(original:str,normalized:str,synonyms:tuple[str,...],category:str|None,fallback:bool=False)`、`RetrievalResult(query:QueryPlan,strategy:str,ranked:tuple[RankedChunk,...],raw_count:int,stale_count:int)`。`KnowledgeGateway(model:ChatOpenAI,*,chat_extra_body:dict,settings:Settings).normalize(question:str)->NormalizationOutput` 为 async；NormalizationOutput 为 Pydantic 字段 normalized(max512)、synonyms(最多3项每项32字)。`QueryNormalizer(gateway).prepare(question:str,category:str|None,*,deadline:float)->QueryPlan` 为 async；`protected_terms_preserved(original:str,normalized:str)->bool` 为纯函数。`KnowledgeRetriever(repo,store,models).retrieve(plan:QueryPlan,strategy:Literal['dense','bm25','hybrid','hybrid_rerank'],*,deadline:float,emit:Callable[[str],Awaitable[None]]|None=None)->RetrievalResult` 为 async，在真实查询/重排开始前分别回调 retrieving/reranking，供pipeline转发真实进度。

- [x] **Step 1：写型号保护和策略纯度 RED。** 归一化替身只实现 normalize(question)，保存收到的 question；测试历史不在输入合同里、负词/数字/型号丢失时 fallback原文、同义词去重不写repo。检索替身在 tests/test_knowledge_retrieval.py 内定义同名 store/models async 方法，记录调用；dense 不调用 score/BM25，BM25 不调用 embed/score，hybrid 不调用 score。

```python
from app.knowledge.query import protected_terms_preserved

def test_rewrite_must_keep_model_and_negation():
    assert not protected_terms_preserved('C65-Pro 不支持哪些协议？', 'C65 支持哪些协议？')
    assert protected_terms_preserved('C65-Pro 不支持啥协议？', 'C65-Pro 不支持哪些协议？')
```

Run: `.venv/bin/python -m pytest tests/test_knowledge_query.py tests/test_knowledge_retrieval.py -q`。

- [x] **Step 2：实现只读本轮的改写模型。** 通过已有 ChatOpenAI 的 with_structured_output(method='json_mode',include_raw=True)，独立 PromptTemplate 明确 JSON 和禁止增加实体，只发 system+当前问题；不绑定业务 tools。Schema、finish_reason及原始JSON复核，失败仅回退一次。受保护项包含英数字型号、数字数值以及“不/不能/不支持/未/无”等否定类别；原文包含而归一化丢失则回退，不能依靠字符串子串判断让“不支持”变“支持”。

BM25 输入为原文/标准问法/去重同义词拼成单条有界文本；密钥与额外模型参数复用现有受保护配置，不覆盖工具/输出限制。 构造器显式接收同一Settings；发归一化请求前按真实system+当前问题、输出预留和安全余量检查预算，不依赖工具选择阶段的另一份prompt预算。词典或规则修订通过独立校准样例评估，不针对正式题硬编码。

- [x] **Step 3：实现排名与原文验证。** store 返回 SearchHit，repo.get_many 后仅保留 done、同 hash 的 KnowledgeChunk，记录 stale_count；命中 ID 去重，固定同分按ID。hybrid_rerank 对与hybrid相同的50个候选score并完整返回排名；Top10截取留给Task6，使Recall@50可直接比较。

```python
valid = [hit for hit in hits if hit.id in originals
         and originals[hit.id].vectorize_status == 'done'
         and source_hash(originals[hit.id]) == hit.source_hash]
ranked = [RankedChunk(originals[hit.id], hit.score) for hit in valid]
if strategy == 'hybrid_rerank' and ranked:
    scores = await models.score(plan.normalized,
        [embedding_text(item.chunk) for item in ranked], deadline=deadline)
    ranked = [RankedChunk(item.chunk, score) for item, score in zip(ranked, scores, strict=True)]
ranked.sort(key=lambda item: (-item.score, item.chunk.id))
```

- [ ] **Step 4：GREEN、标注改写评估与提交。** 运行单测、真实四策略各一条检索及校准集改写标识保留检查；记录回退率和错误改写。评审后提交 `feat: add current-query understanding and four retrieval strategies`。


执行状态：Task5代码及本地验证已独立评审通过；Step4中的真实30题DeepSeek改写评估因自动审批外发授权要求而等待用户回复，不算通过。后续任务可继续本地代码与替身测试，各自真实模型/Prompt/浏览器质量门槛保留为待完成；不得绕过外发拒绝，最终交付不得声称finish。

## Task 6：证据预算、充分性自评与受控拒答

**Files:** Create `app/knowledge/evidence.py`、`app/knowledge/pipeline.py`、`app/prompts/knowledge_answer.txt`、`app/prompts/evidence_assessment.txt`、`tests/test_knowledge_evidence.py`、`tests/test_knowledge_pipeline.py`；Modify `app/knowledge/contracts.py`、`app/knowledge/gateway.py`、`app/context.py`、`app/prompts.py`。

**Interfaces:** `Citation` 为 Pydantic 模型，字段 number(int1..10)/chunk_id/category/section_path/questions/answer/content_hash/url/score；来自 Task1 原文，不由 LLM 填写。`EvidenceAssessment` 为 Pydantic 模型：sufficient(bool)、reason_code(Literal supported/insufficient_evidence/ambiguous_question)、reason(str最长600字符)、supporting_chunk_ids(list[int]最多10)。`EvidencePlan(sources:tuple[Citation,...],dropped_ids:tuple[int,...])` 不持有 DB Session。`KnowledgeDecision(query:QueryPlan,status:Literal['ok','not_found'],sources:tuple[Citation,...],assessment:EvidenceAssessment|None,reason_code:str|None,refusal:str|None)` 提供 to_payload()->dict，序列化后校验不超过48000字节。

`KnowledgeGateway.assess(question:str,sources:tuple[Citation,...],*,normalized_question:str)->EvidenceAssessment`；`EvidenceBudget(settings,history:list[StoredTurn],question:str,call:AIMessage).select(ranked:tuple[RankedChunk,...],query:QueryPlan)->EvidencePlan`；`KnowledgePipeline(normalizer,retriever,gateway,threshold:float).run(question:str,category:str|None,budget:EvidenceBudget,*,deadline:float,emit:Callable[[str],Awaitable[None]])->KnowledgeDecision`。该流程默认 hybrid_rerank；评估通过同模块 `decide_evidence(query,retrieval,budget,*,gateway:KnowledgeGateway,threshold:float|None,deadline)->KnowledgeDecision` 复用预算/自评，前三策略 threshold=None。

- [x] **Step 1：写首尾布局、真实证据预算和拒答 RED。** 测试 sources 编号稳定、1与2位于首尾、预算只减整块、唯一一次 assess 实际收到裁剪后的集合；零命中/全 stale/低分提前拒答不调用 assess；无效支持 ID 受控错误，不能默认放行。测试 input 很长导致连一条都放不下时返回 context_budget。

```python
from app.knowledge.evidence import edge_order, validate_citation_numbers
import pytest

def test_best_evidence_is_at_both_edges():
    assert edge_order(list(range(1,11))) == [1,3,5,7,9,10,8,6,4,2]
    assert edge_order([1,2,3]) == [1,3,2]

def test_unknown_citation_is_generation_error():
    with pytest.raises(ValueError):
        validate_citation_numbers('支持该协议[99]。', {1,2})
    assert validate_citation_numbers('支持该协议[1]。', {1,2}) == {1}
```

Run: `.venv/bin/python -m pytest tests/test_knowledge_evidence.py tests/test_knowledge_pipeline.py -q`。

- [x] **Step 2：实现预算与一次自评。** 生成 sources 时先按相关性编号，再做 edge_order。EvidenceBudget 构造真实最终 system/user/assistant-tool/ToolMessage，预留受限 assessment 等字段最多4096 UTF-8字节；使用已有 build_tool_context 估算并先裁剪旧完整轮次，不够才去掉排名末尾整块。工具结果JSON和模型输入分别校验，不把48000字节当成token预算。 模型输入同时检查实际自评请求（含system/schema、原话、标准问法、所选原文）和最终生成请求；两者都能容纳才调用自评，否则继续删最低排名整块，无法容纳则context_budget。

```python
def edge_order(items):
    return items[::2] + list(reversed(items[1::2]))
```

候选选择结束后才调用一次 assess。评估和在线同用 PromptTemplate：充分必须引用已提供的 ID，不把历史助手回答或常识当本店证据；指定型号、否定条件、证据冲突、保证到账等样例进入校准评估。结构化响应复核与normalize一致，但 assess失败返回技术错误，不能降级为猜测答案。

- [x] **Step 3：实现固定 pipeline 与拒答消息。** emit阶段为 normalizing/retrieving/reranking/checking_evidence，仅传阶段标记。QueryNormalizer一次、Retriever一次、自评最多一次，不循环。零结果区分 raw_count=0 和全stale；最高rerank分低于阈值reason=low_relevance；其余由assess。受控拒答文本不包含内部堆栈、不承诺建单，所有原话由Task7入池，pipeline本身无DB副作用。

```python
REFUSALS = {
    'no_hits': '现有知识不足以确认这个问题，请补充具体商品或政策信息，也可以联系人工客服。',
    'low_relevance': '现有知识不足以确认这个问题，相关证据不足，请补充具体信息。',
    'insufficient_evidence': '现有知识不足以确认您询问的事项，建议补充信息或联系人工客服核实。',
    'ambiguous_question': '现有知识不足以确认您指的是哪件商品或哪项政策，请补充具体型号和问题。',
    'stale_evidence': '现有知识不足以确认答案，相关内容正在同步，请稍后重试。',
    'context_budget': '现有知识不足以确认答案，本轮暂时无法完整放入所需证据，请缩小问题范围。',
}
```

validate_citation_numbers 支持 `[1][2]` 形式，只允许本轮1..n；拒绝无有效引用与未知编号。它只验证映射，不声称验证语义。技术错误用 ServiceError 的 KNOWLEDGE_UNAVAILABLE/EVIDENCE_ASSESSMENT_ERROR 等确定错误码。

- [ ] **Step 4：GREEN、Prompt 评估与提交。** 除单测外，用校准样例检验无证据、近似型号、越权资料指令、到账承诺、跨块证据、不够上下文等结果；记录可回答/应拒答标签及真实自评输出。评审后提交 `feat: gate grounded answers with evidence budgets and refusal decisions`。

执行状态：Task6代码/本地门槛已独立评审通过（0590a1b + fc9ded9）；20项聚焦、358项完整本地依赖测试及裁剪变异检查通过。Step4真实DeepSeek Prompt评估待外发授权，Task6整体不标完成。

## Task 7：知识流程进入既有聊天和工具审计

**Files:** Create `app/services/knowledge_turn.py`、`tests/test_knowledge_chat.py`、`tests/test_knowledge_tool_policy.py`、`tests/integration/test_knowledge_chat_persistence.py`；Modify `app/tools/business.py`、`app/tools/schemas.py`、`app/tools/registry.py`、`app/tools/executor.py`、`app/tools/results.py`、`app/services/chat.py`、`app/api/streaming.py`、`app/prompts/tool_chat.txt`、旧工具/聊天测试及 `tests/ch04_helpers.py`。

**Interfaces:** `build_registry(...,knowledge_call:Callable[[],Awaitable[str]]|None=None)` 中 query_faq 改为无参数 @tool，调用注入的knowledge_call；缺依赖返回明确不可用错误，不回退SQL LIKE。Task6 的 KnowledgeDecision JSON 是唯一允许大结果的合同。新增 `RetrievalProgress(tool_call_id:str,stage:str,message:str)`；ToolExecutor.run 接受可选 `progress_queue:asyncio.Queue[RetrievalProgress]`，产出 ToolProgress/RetrievalProgress/ToolOutcome。

`KnowledgeTurnRunner(pipeline,low_confidence,settings).execute(prepared:PreparedTurn,call:AIMessage,emit)->KnowledgeDecision` 组装 EvidenceBudget、调用pipeline，拒答时调用record_once；服务将decision序列化为工具结果，append_result确认后才输出sources/refusal。PreparedTurn增加started_at/category/knowledge_decision字段。`stream_events(...,deadline:float|Callable[[],float])` 每次循环重新读取live deadline。

- [x] **Step 1：写整轮状态与持久化 RED。** 在 tests/ch04_helpers.py 定义可排队的 FakeKnowledgePipeline：run返回预设KnowledgeDecision、记录原话/品类、可等待 Event 和抛 ServiceError；复用ch02会话仓储替身，不修改其审计语义。测试序列必须是meta→tool_status running→检索状态→tool结果持久化→sources→真实token→完成提交→done。拒答没有gateway.stream调用，先问题池/工具结果、再refusal、完成后done。

```python
async def test_refusal_is_persisted_before_visible(rag_chat_case):
    # Task7 fixture由FakeKnowledgePipeline的not_found决定及记录事件的仓储组成。
    service, prepared, trace = rag_chat_case
    events = [event async for event in service.stream(prepared)]
    assert 'pool_saved' in trace
    assert trace.index('pool_saved') < trace.index('tool_saved')
    assert [event.name for event in events].count('refusal') == 1
    assert not any(event.name == 'token' for event in events)
    assert events[-1].name == 'done'
    assert trace.count('model_stream') == 0
```

fixture `rag_chat_case` 在本测试文件创建完成prepare上下文，finally退出，防止只裸造PreparedTurn漏掉资源生命周期。另测 pool写失败无refusal/done、未知引用最终failed、Milvus异常不入池、执行中取消不追加迟到消息。

Run: `.venv/bin/python -m pytest tests/test_knowledge_chat.py tests/test_knowledge_tool_policy.py tests/integration/test_knowledge_chat_persistence.py --require-mysql -q`。

- [ ] **Step 2：最小扩展工具合同和进度泵。** Registry服务端按工具名固定执行策略：query_faq max_bytes=48000/max_attempts=1/遵守知识共享deadline，其余维持4096/现有重试。JSON序列化前后按工具专属Schema验证，知识正文绝不送旧通用截断器。ToolExecutor用独立invoke Task与有界progress_queue同时等待，转发阶段；退出时取消并结算invoke，保留已有审计清理边界。

```python
@tool(args_schema=KnowledgeInput)
async def query_faq() -> str:
    """查询本店政策、商品型号规格和使用说明；自动使用本轮原始问题。"""
    if knowledge_call is None:
        raise ServiceError('KNOWLEDGE_UNAVAILABLE', '知识服务暂时不可用', 503)
    return await knowledge_call()
```

KnowledgeInput 在 app/tools/schemas.py 中为 extra='forbid' 的空 Pydantic模型；历史keyword参数仅审计读取，不重放。归一化中的售后政策问题不能路由随机query_product，更新工具prompt后用校准样例验证选择。

- [x] **Step 3：固定编排、共享截止时间与审计。** prepare记录 started_at；selector仍受现有时限。选中query_faq后只延长一次到started_at+knowledge_request_timeout_seconds，HTTP层通过lambda读取，不改变其他工具时限。构造knowledge_call闭包时明确尚未选中call不可执行；选择后注入call与当前history，不能把模型工具参数作为问题原话。

```python
deadline_value = deadline if callable(deadline) else lambda: deadline
remaining = deadline_value() - asyncio.get_running_loop().time()
```

知识结果保存后：拒答保存最终模板到assistant并发送refusal/done；充分则生成sources事件并用知识prompt和本轮实际证据进行astream，检查引用后提交done。串联 `_bounded` 时所有DB写加入_mutations，取消清理不得与未决写竞争。技术错误直接SSE error，不调用模型自由解释后当成功。

- [x] **Step 4：GREEN、旧行为调整和提交。** 旧query_faq字面漏召回测试只在在线业务测试中更新；FaqRepository原始LIKE仓储测试保留作第二章行为记录，第一章提取/其他四工具/取消/工单幂等回归继续通过。增加真实MySQL断连前后pool和工具审计验证；知识result >4KB可原样保存、普通工具仍受4KB上限。评审后提交 `feat: integrate grounded retrieval into persisted streaming chat`。

执行状态：Task7代码与本地验证已独立评审通过（bb0ce7c + 6415585）；完整374项通过，修正覆盖58项通过。Step2中的真实工具路由Prompt校准仍待外发授权，其他实现已就绪；Task7整体不标完成。生产依赖和HTTP调用点按计划由Task8装配。

## Task 8：应用生命周期、原文 API 与配置交付

**Files:** Create `app/api/knowledge.py`、`app/knowledge/calibration.py`、`tests/test_knowledge_calibration.py`、`tests/test_knowledge_api.py`、`tests/integration/test_knowledge_startup.py`；Modify `app/knowledge/milvus_store.py`、`app/main.py`、`app/model.py`、`app/api/chat.py`、`app/schemas.py`、`tests/test_milvus_store_rpc.py`、`tests/test_model.py`；Verify existing `app/config.py`、`pyproject.toml`、`.env.example`（前序任务已提供配置及prompts通配打包规则，以现有内容和实际wheel检查验收，无需重复改动）。

**Interfaces:** ChatRequest增加category:可选非空字符串最长255；knowledge router提供 `GET /api/knowledge/categories` 和 `GET /api/knowledge/chunks/{id}?expected_hash=...`。`create_app` 增加可注入的knowledge_dependencies用于测试，生产统一构造仓储/本地模型/store/gateway/pipeline/runner，并在lifespan关闭。`OpenAIModelGateway.create_knowledge_gateway()->KnowledgeGateway` 在app/model.py定义，传入它持有的ChatOpenAI及受保护的chat_extra_body；不让main读取私有字段或另建未受控模型客户端。

- [x] **Step 1：写原文版本与品类 API RED。** source返回字段id/category/questions/answer/section_path/content_type/is_key_clause/prev_chunk_id/next_chunk_id/content_hash；只从MySQL读。不暴露向量/DB连接/模型密钥。GET缺行404，expected_hash不符409，非法ID/参数422。API测试为应用注入知识仓储替身，不为普通单测下载模型。

```python
async def test_source_revision_conflict(knowledge_api_client):
    response = await knowledge_api_client.get('/api/knowledge/chunks/910001?expected_hash=' + '0'*64)
    assert response.status_code == 409
    assert 'answer' not in response.json()
```

knowledge_api_client fixture在本文件使用 Task1 make_chunk、注入只读仓储和现有mock gateway创建app并运行lifespan。另测category透传本轮且下轮不继承、sources帧映射/URL与source API吻合。

Run: `.venv/bin/python -m pytest tests/test_knowledge_api.py -q`。

- [x] **Step 2：生命周期只检查，不默默初始化。** 应用需要三表已建立、Milvus schema已兼容、模型文件已下载、校准文件与语料/模型revision相符。缺依赖返回可诊断启动错误；不在startup建库/拉模型。用线程预热两模型后才ready。沿用一个ChatOpenAI实例封装KnowledgeGateway，网关的JSON调用保留原chat额外参数与输出限制，最终只关闭一次HTTP客户端。

```python
from fastapi import APIRouter, Request
router = APIRouter(prefix='/api/knowledge')

@router.get('/chunks/{chunk_id}')
async def read_chunk(chunk_id: int, request: Request, expected_hash: str | None = None):
    chunk = await request.app.state.knowledge_repository.get(chunk_id)
    if chunk is None:
        raise ServiceError('CHUNK_NOT_FOUND', '知识原文不存在', 404)
    digest = source_hash(chunk)
    if expected_hash is not None and expected_hash != digest:
        raise ServiceError('CHUNK_CHANGED', '原文已更新，请查看本轮保存的引用快照', 409)
    return source_response(chunk, digest)
```

source_response 在本模块定义为上述固定白名单字段字典。所有URL由后端固定相对路径生成，哈希作为编码的query参数；业务原文安全渲染。

- [x] **Step 3：GREEN、打包和提交。** 校验启动失败能关闭已创建的模型线程/store/数据库/HTTP客户端，测试应用注入路径不受真实模型要求影响。新增prompt作为package_data包含到wheel；数据/评估CLI使用明确仓库路径，不谎称wheel自动包含演示资料。运行API回归及wheel内容检查后，评审并提交 `feat: expose knowledge sources and assemble rag lifecycle`。

执行状态：Task8代码/本地门槛完成并独立复核通过（56684d7 + d8577f9）；完整410项通过，关闭修正覆盖18项通过，实际wheel资源检查通过。真实生产启动仍按计划等待Task9有效校准和Task11验收；当前不声称服务已切换。

## Task 9：真实评估、校准阈值与分桶报告

**Files:** Create `app/knowledge/evaluation.py`、`app/prompts/faithfulness_judge.txt`、`scripts/evaluate_knowledge.py`、`tests/test_knowledge_evaluation.py`、`dev-notes/ch04-evaluation.md`；Modify `app/knowledge/gateway.py`。

**Interfaces:** `retrieval_metrics(ranked_ids:list[int],relevant_ids:set[int])->dict[str,float|None]` 返回 recall_at_5/10/50 和 mrr_at_50；空relevant返回None。`faithfulness_score(claims:list[dict])->float|None`，claim字段statement/supported/source_ids，空列表None。`calibrate_threshold(samples:list[tuple[float|None,bool]],*,max_false_accept:float=0.1)->float`，None表示没有候选，永远不算通过，不用NaN/Infinity哨兵；`KnowledgeGateway.judge(question:str,answer:str,sources:tuple[Citation,...])->FaithfulnessJudgement` 为 async，Pydantic judgement含claims列表及每条理由，技术失败单列。

CLI子命令：`calibrate`、`compare`；通用参数 `--corpus`、`--cases`、`--output-dir`、`--calibration`，compare增加`--strategies dense,bm25,hybrid,hybrid_rerank`及`--limit`（默认全部，只有冒烟时限制）。退出码区分数据/依赖错误与存在运行失败。正式数据执行只读知识库，不往low_confidence_questions写入。

- [ ] **Step 1：写手工可算指标与校准 RED。** 无答案N/A、没召回0、技术失败计零且独立计数、拒答faithfulness None；禁止JSONNaN。阈值在独立校准分数分界点上选择，float有限，必要时选择max_score上方一个有限数，全拒答也如实保存。

```python
from app.knowledge.evaluation import retrieval_metrics, faithfulness_score

def test_metrics_do_not_reward_absent_ground_truth():
    scored = retrieval_metrics([99,10,11,12,13], {10,12})
    assert scored['recall_at_5'] == 1.0
    assert scored['mrr_at_50'] == 0.5
    assert retrieval_metrics([99], set())['mrr_at_50'] is None
    assert faithfulness_score([]) is None
    assert faithfulness_score([{'statement':'a','supported':True,'source_ids':[10]},
                              {'statement':'b','supported':False,'source_ids':[]}]) == 0.5
```

Run: `.venv/bin/python -m pytest tests/test_knowledge_evaluation.py -q`。

- [ ] **Step 2：实现确定性指标/缓存/报告。** 每题只归一化一次缓存到run目录；四策略同输入/过滤/预算/prompt。先算原排名指标，再执行Task6的证据选择与自评，充分则用同一个最终生成网关收集真实流，拒答记录模板与reason。为offline构造带稳定本地ID的AIMessage工具申请，只用于prompt关联，不写MySQL会话。

```python
def faithfulness_score(claims):
    if not claims:
        return None
    return sum(claim['supported'] is True for claim in claims) / len(claims)
```

校准仅使用30条calibration；冻结阈值、模型revision、corpus/prompt/calibration哈希。compare读取配置，不重新调阈值；hybrid_rerank用阈值，其余不用reranker。技术错误记录stage/error_code，仍计样本数；JSON逐题原子写入，支持按相同配置哈希断点续跑，配置变动不得混入旧run。

输出report.md、results.jsonl、manifest.json、normalizations.json、calibration.json。报告按六主桶和easy/medium/hard出数，包含样本分母、Recall@5/10/50、MRR@50、Faithfulness、有回答率、无答案拒答率、失败数与耗时。hybrid/Rerank Recall@50不同时检查是不是候选集不一致，不能直接宣称重排提高全集召回。

- [ ] **Step 3：真实校准、四策略完整运行及标注抽查。** 先对校准集运行并冻结，再正式60题×4策略；使用现有DeepSeek与真实Milvus/两本地模型。Prompt仍有问题只用校准题修正后重跑冻结，不能根据正式题命中情况塞特判。

```bash
.venv/bin/python scripts/evaluate_knowledge.py calibrate --cases evals/ch04/calibration.jsonl --output-dir evals/reports/ch04/calibration
.venv/bin/python scripts/evaluate_knowledge.py compare --cases evals/ch04/test.jsonl --calibration evals/reports/ch04/calibration/calibration.json --output-dir evals/reports/ch04/comparison
```

将生产校准副本显式写入settings.knowledge_calibration_path，验证其fingerprint。抽查至少12条（每个主桶至少2条），逐条对照原文，保存原子陈述、证据ID、自评/评审结果及复核结论；开发助手的审读标为“助手复核”，不冒充用户人审或人工金标准。

- [ ] **Step 4：评审、留痕、提交。** 记录真实数字、失败案例、所有实际未完成项；data/prompt质量用此次评估结果，指标代码另做规格/质量评审。提交 `feat: evaluate and calibrate four knowledge retrieval strategies`。原始报告位于gitignore下时，将汇总和哈希写进跟踪的dev-notes/ch04-evaluation.md，交付给用户可点击的实际报告路径。

## Task 10：页面引用与一次性满意度反馈（Vibe Coding）

**Files:** Modify `app/web/index.html`。**Interfaces:** 消费既有SSE和Task8新增sources/refusal/retrieval_status，使用meta的session_id/turn_id作为反馈键。

- [ ] **Step 1：直接改现有页面。** 在本轮气泡中呈现检索状态；上游文本逐增量显示，识别合法`[n]`后关联本轮sources快照，点击展示questions/answer/section_path并提供后端source URL。流中不完整`[`暂作文本，结束后再安全解析。文本节点/DOM API构造内容，不把模型或原文作为innerHTML。
- [ ] **Step 2：直接加入反馈与品类选择。** 完成/拒答回答的左下方放👍/👎；首次选择立即锁两项、点亮所选、“已反馈”，写本地存储键`ch04-feedback:<session>:<turn>`和choice/timestamp。try/catch处理存储不可用仍保留页面锁定。提供“全部品类”及只读category API选项，当前请求携带category，新会话默认全部。
- [ ] **Step 3：浏览器验证并按效果修改。** 在可用临时端口（优先8002）启动已装配且已校准的新后端，再打开对应页面验收，不能拿仍运行第二章后端的8001假验收。验证流式状态、引用点击、恶意HTML原文显示为文本、反馈二次点击不改变、刷新后记录仍在存储、拒答气泡和技术错误样式不同。记录实际浏览器操作与用户纠偏，不为页面写单元测试或派代码评审。
- [ ] **Step 4：留痕提交。** 将前端验收结果即刻追加日志，提交 `feat: show knowledge citations and one-shot feedback in chat`。

## Task 11：全链路验收、文档与 finish

**Files:** Create `scripts/demo_knowledge.sh`；Modify `README.md`、`dev-notes/ch04.md`、`dev-notes/ch04-evaluation.md`、本计划进度。

- [ ] **Step 1：完整真实依赖测试。** 带所有required标志运行，不将skip当通过；新测试的真实模型fixture只加载一份并使用明确的session事件循环，不在函数级测试重复下载/加载。验证旧售后提取、多轮审计、工单幂等、取消行为仍通过。

```bash
.venv/bin/python -m pip check
.venv/bin/python -m pytest --require-mysql --require-milvus --require-local-models -q --tb=short
.venv/bin/python -m pip wheel --no-deps --no-build-isolation -w /private/tmp/ch04-wheel .
```

- [ ] **Step 2：准备真实预览。** 在不占用8000的可用临时端口验证新应用可启动、依赖健康和首条问答，再按PID核对安全替换当前项目8001进程，保持单worker。不得杀别的项目、不得用模糊pkill。确认 `http://127.0.0.1:8001/` 可访问。

- [ ] **Step 3：浏览器/curl/数据库联合验收。** 型号问题至少“C65-Pro 支持 PD 3.0 吗？”、同义词“邮费是多少”、知识缺失“你们的月球配送服务如何收费？”；来源ID必须来自实际结果，不在脚本硬编码预期答案。curl保存meta中session_id并进行第二轮，核查归一化输入仍只有该轮原话；订单1001物流及售后提取继续可用。

```bash
curl -N -H 'Content-Type: application/json' http://127.0.0.1:8001/api/chat -d '{"message":"C65-Pro 支持 PD 3.0 吗？","category":"数码配件/充电器"}'
curl -N -H 'Content-Type: application/json' http://127.0.0.1:8001/api/chat -d '{"message":"邮费是多少"}'
curl -N -H 'Content-Type: application/json' http://127.0.0.1:8001/api/chat -d '{"message":"你们的月球配送服务如何收费？"}'
```

逐例保存工具轨迹、实际命中、答案引用、URL/快照、拒答reason、低置信度池对应行和消息审计。重复检索/再次启动索引不造重复chunk，用户反馈没有网络写请求。demo脚本只发上述业务演示请求，不清库。

- [ ] **Step 4：最终代码评审。** 使用 requesting-code-review，评审后端完整变更与已批准spec；前端按用户例外不纳入代码评审。对评审意见使用receiving-code-review先核实再修正；对实质变更补对应RED/GREEN和受影响真实验证。修正后运行所需回归，不重复无关耗时评估。
- [ ] **Step 5：更新交付文档并finish。** README列安装/Compose/模型准备/建表导入/索引/校准/启动/评估命令、240秒知识时限、单worker/CPU延迟实测、低置信度池与前端反馈边界；报告真实指标和失败局限。使用verification-before-completion、finishing-a-development-branch，按用户选择整合；此前第二章“本地合并”不自动等于对本章尚未完成变更的合并指令。每个完成阶段立即记日志，最后交付命令、测试/评估数字及dev-notes路径。

## 计划自审与执行交接

覆盖映射：spec§2→Task1/2；§3→Task3/4；§4→Task1/4；§5→Task5/8；§6→Task4/5/6；§7→Task6/7/9；§8→Task7；§9→Task8/10；§10→Task3/4/7/8；§11→Task2/9；§12→Task11；§13/14贯穿所有任务。

自审结论：规格各节均映射到任务；检查接口名称与调用顺序，修正实际检索进度回调、共享ChatOpenAI工厂、低置信度分母和跨worktree索引锁路径；空结果/技术错误、两种结果预算、动态deadline、校准/正式集、真服务不跳过、前端例外均有落点。逐条复核由开发助手执行时明确标注“助手复核”，不冒称用户人审，规格同步澄清该文字。

计划完成后按writing-plans征询执行方式：推荐逐任务子代理实现和独立评审，也可在当前任务直接执行。若未选择额外worktree，不创建；在当前目录功能分支执行。用户已通过规格，不重新发起设计批准流程。
