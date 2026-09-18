# 第 04 章：混合检索、重排与评估设计

日期：2026-09-18。代码基线：`b33335d`，分支 `codex/ch01-chat`。

用户已选择“演示数据”，并对“扩展现有 query_faq、本地 bge-m3、指定重排模型、120 个 chunk、60 条正式评估”的方案回复“是”。本文落实已确认的设计方向；书面规格等待用户审阅，实现尚未开始。

## 1. 范围与方案

在现有聊天页与 `POST /api/chat` 中升级知识问答，继续由模型选择业务工具。采用现有应用内的独立检索模块，由 `query_faq` 调用；保留订单、实时商品模拟信息、物流、人工工单及第一章售后提取能力。替代方案是单独部署检索服务，部署与调用边界更多，本章不采用。

固定采用：FastAPI、SQLAlchemy、Docker MySQL、LangChain `@tool`；Milvus 原生 BM25、内置 chinese analyzer、dense/BM25 双路召回与 `hybrid_search` 的 RRF；本地 `BAAI/bge-m3` 提供 dense 向量，本地 `BAAI/bge-reranker-v2-m3` 重排。生成、问题归一化和证据自评继续使用现有 OpenAI 兼容网关与 DeepSeek 配置，保持关闭思考模式。

本章补齐缺失的第三章知识表、演示资料导入、向量索引及双写恢复基础。用户提供的两张表 DDL 是结构依据。不实现历史会话 QA 自动抽取/去重流水线、知识编辑后台、指代消解、多轮改写、Agent Loop、向量数据库替换或用户反馈后端。

后端按 Superpowers、TDD 和代码评审执行；Prompt 与演示数据通过标注样例/评估集验证。前端按用户要求用 Vibe Coding 修改，仅约定后端事件合同及浏览器验收，不对页面另做 brainstorm、TDD 或 code review。

## 2. 数据与演示资料

### 2.1 知识原文

MySQL `knowledge_chunks` 是原文权威源，字段、长度、默认值、中文注释、索引及自引用外键保留用户提供的 DDL，见附录。Milvus 集合名为 `knowledge`，不将原文权威源转移到向量库。

每个 chunk 只生成一份检索文本，固定格式为：

```text
分类：{category}
问法：{questions}
答案：{answer}
```

同一文本用于 dense 向量化和 BM25 的 text 字段。章节路径、内容类型、关键条款标记、相邻块指针、向量状态、主键和时间不拼入检索文本。category 同时保留为标量过滤字段。questions 可保存资料已有的多个问法；不得把检索时扩展的同义词回写成新的 chunk 或原文问法。

演示资料共 120 个独立 chunk：数码配件/充电器、智能家电/扫地机器人、个护电器/电动牙刷、家居用品/保温杯四个商品品类各 24 个，另有 24 个通用/售后与配送条目。品类使用上述稳定路径值；资料覆盖型号规格、使用说明、退换货、运费、保修及服务限制，包含相近型号、不同品类的相似条款。示例型号使用虚构的 `C65-Pro`、`C65` 等标识，明确标注为本店演示资料，不冒充真实品牌政策。

每条必须具备稳定的资料键、category、questions、answer、非空 section_path、content_type 和关键条款标记。同一文档内的相邻块设置 prev/next，首尾为空。原始资料键及内容摘要保存在仓库数据清单中，用于重复导入核对，不擅自增加用户 DDL 的字段。

演示种子使用固定且在支持范围内的 chunk ID。首次导入前核查冲突；已有相同 ID 且内容相同则跳过，不同则停止并报告，不覆盖现有资料。先插入全部原文，再建立相邻指针，避免自引用外键失败。现有 faq、会话、消息和工单不清空、不覆盖。

`qa_extraction_staging` 按用户 DDL 建表，本章保持为空。它用于后续离线抽取流程，不作为本章在线检索来源。

### 2.2 低置信度问题池

新建 `low_confidence_questions`：

| 字段 | 合同 |
| --- | --- |
| id | BIGINT UNSIGNED 自增主键 |
| original_question | TEXT 非空，保存本轮用户原话 |
| conversation_id | VARCHAR(36) 非空，关联 conversations.id |
| turn_id | VARCHAR(64) 非空，用于本轮幂等 |
| entry_point | VARCHAR(32) 非空，本章在线入口为 chat |
| reason_code | VARCHAR(64) 非空，可统计的拒答原因 |
| reason | TEXT 非空，说明缺少什么证据，不保存模型内部推理 |
| created_at | DATETIME 非空，UTC 创建时间 |

`(conversation_id, turn_id)` 唯一，建立时间和原因索引。一轮最多写一条；工具内部重试及持久化确认重试不重复入池。原因区分 `no_hits`、`low_relevance`、`insufficient_evidence`、`ambiguous_question`、`stale_evidence`、`context_budget`。

拒答前完成问题池写入和工具审计；写入失败则报告服务错误，不能声称该轮已成功处理。已提交的问题池记录不随断连删除。离线四策略评估将拒答信息写入评估产物，不污染在线问题池；真实聊天验收必须核查数据库实际新增记录。

## 3. Milvus 与本地模型

服务端采用 Milvus 2.6 系列 standalone。初始版本组合选官方发布说明明确列出的 `Milvus 2.6.23 / pymilvus 2.6.17`，不将 Context7 默认的 3.0 示例直接套入 2.6。依赖计划核对该组合、官方镜像架构及实际 API，并固定镜像与 Python 依赖版本；若发生固定技术无法运行的矛盾，停止相关实现并向用户说明。

集合的必要字段：

| 字段 | 用途 |
| --- | --- |
| id | INT64 主键，auto_id=False，对齐 MySQL id |
| text | VARCHAR，启用 analyzer，`analyzer_params={"type":"chinese"}` |
| dense_vector | FLOAT_VECTOR，1024 维，COSINE |
| sparse_vector | SPARSE_FLOAT_VECTOR，由 text 上的 BM25 Function 生成 |
| category | VARCHAR，品类精确预过滤 |
| content_type / is_key_clause | 检索元数据 |
| source_hash | 原文及引用元数据的内容指纹，用于一致性检查 |

dense 使用 HNSW，BM25 使用 SPARSE_INVERTED_INDEX 与 BM25 metric。索引确认、恢复检查及验收检索使用 Strong consistency，避免把尚未可见的写入当成永久缺失。不使用 bge-m3 的 sparse/ColBERT 输出代替指定的原生 BM25。chinese analyzer 对中文、英文、数字及连字符型号的行为以实际 `run_analyzer` 和型号检索样例验证；发现不能满足要求时报告，不私自更换 analyzer。

MySQL 的 unsigned ID 范围大于 Milvus INT64；索引入口只接受 `1..2^63-1`，超界保留 pending 并明确报错，不截断、不改变用户 MySQL DDL。vector_id 写入相同 ID 的十进制字符串。

本地模型通过 FlagEmbedding 适配器加载，先以 CPU、FP32、小批量建立可复现实测基线，不假定 Apple GPU 加速已可用。实现计划锁定实际可安装的运行库及模型 revision。启动前显式下载并预热，聊天请求内不临时下载模型。bge-m3 只取 dense 输出；reranker 的 sigmoid 分值只是排序分值，不称为正确率或置信概率。

本地推理放在有界工作线程中，不能阻塞 FastAPI 事件循环；同一份模型实例串行推理。批次之间检查取消与截止时间。取消无法强行中断正在运行的本地批次，该批次完成后丢弃结果，完成前仍占用推理槽；不允许已取消任务释放槽后继续堆积后台计算。

## 4. 入库与双写恢复

原文先在 MySQL 提交为 pending。索引器读取原文快照，生成固定检索文本和内容指纹，计算 dense 后按稳定主键 upsert Milvus，等待写入确认；再在短事务内锁定 MySQL 行，重新计算指纹，只有仍与快照一致时才回填 vector_id 并置 done。

source_hash 覆盖 category/questions/answer 及 section_path/content_type/is_key_clause/prev/next，排除 vectorize_status、vector_id 和因标记 done 而改变的 updated_at。两边指纹一致才可作为在线证据。

Milvus 成功而 MySQL 确认失败时，原文保持 pending，下一次 upsert 同一主键。向量化期间原文变化则保持 pending，重新索引。索引任务使用互斥执行，避免不同快照的两个写入器交错；本章不实现分布式双写事务。内容更新入口需同时重置 pending，本章不提供编辑 UI。

已有集合先核对维度、主键、字段和 BM25 Function/analyzer；不匹配则停止，禁止默认 drop collection。恢复检查同时比较 done 行的 ID/指纹与 Milvus；索引缺失或不一致的行重新进入 pending，不仅扫描原来 pending 的行。

不得静默截短待索引原文以适配模型或 VARCHAR：超过模型输入上限或 Milvus text 字节上限则返回包含 chunk ID 的校验错误并保持 pending。演示资料在数据评估阶段验证全部能完整编码和重排。

## 5. 当前问题理解与过滤

归一化模型只接收本轮原始问题和静态规则，不接收会话历史、旧工具结果或评估答案。以结构化输出返回标准问法和最多三个检索同义词；保留具体型号、订单标识、数字、否定和限定条件，禁止补出用户没有说的型号/品类。

模型输出须通过 Schema、长度和标识保留检查。解析失败或改坏标识时，回退为原始问题且无同义词，记录降级原因；不再循环请求改写。技术调用超时遵守整体截止时间。对需要指代消解的问题，例如“那它呢”，不从历史猜测对象，证据自评要求用户补充明确对象。

dense 使用归一化问法；BM25 使用原始问题、归一化问法及去重同义词组成的一条有界查询。每条检索通道只发一次请求，不把同义词变为多轮召回。评估时每个问题的归一化结果缓存一次，四种策略共享它。

`POST /api/chat` 增加可选 category，作为本轮独立参数，不从历史继承；模型不能自行改写该过滤条件。category 采用精确匹配，不自动扩展到通用分类或子分类。dense 与 BM25 的 AnnSearchRequest 均携带相同过滤表达式，过滤发生在各路 Top-50 之前。表达式字段固定、值按 SDK 支持的参数化方式或严格字符串编码传入，不拼接可执行的用户表达式。

没有 category 时检索全部知识；指定不存在的 category 得到零命中，按无证据处理。知识库品类列表可通过只读接口读取，便于演示和页面选择。

## 6. 检索、重排与上下文

在线默认使用 hybrid_rerank：dense Top-50 与 BM25 Top-50，由 Milvus `hybrid_search` 和 `RRFRanker(k=60)` 融合为最多 50 个候选。按 ID 去重并回查 MySQL 原文，剔除不存在、非 done 或指纹不一致的候选；不拿 Milvus 中的旧正文直接回答。剔除后不另起自动检索轮次，记录候选损失。

bge-reranker-v2-m3 对归一化问题与候选完整检索文本打分，按分数降序、同分按 chunk ID 排序，取最多 10 个。重排失败或超时返回可诊断的服务错误，不把没有执行重排的结果标为 hybrid_rerank。

为生产重排策略设一个来自独立校准集的相关性下限：最高重排分低于下限，直接作为 low_relevance 拒答。不将 COSINE、BM25、RRF 与 reranker 分数混用。前三种评估基线不调用 reranker，不借它建立隐含的第五条检索路径。

选出的 chunk 按相关性编号 `[1]..[n]`，编号与 chunk 一一对应。放入 prompt 时按 `1,3,5,...,最大偶数,...,4,2` 排列，例如十条为 `1,3,5,7,9,10,8,6,4,2`，让最相关的两条分别处于首尾。编号不随排版重排而改变。

Top-10 指检索结果上限，不强行凑满十条。上下文先裁剪旧的完整历史轮次，再从本轮证据中删除最低排名的整块；不截断政策句、不打散关键条款。证据选择使用当前 token 预算及后述结果字节上限，记录 selected/context/dropped 数量。证据充分性自评必须针对实际将传给生成模型的这份证据执行；连一块都放不下时拒答并以 context_budget 入池。

沿用项目保守的 UTF-8 字节 token 估算，明确它是估算。所有模型请求均预留输出及安全余量，不暗改已配置的真实模型上下文上限。知识证据、引用元数据和本轮工具消息均纳入预算。

## 7. 生成质量与拒答

知识问答在最终流开始前执行一次结构化证据自评。输入是当前原话、归一化问法和实际入 prompt 的原文；输出包括 sufficient、reason_code、简短 reason 和 supporting_chunk_ids。Schema 校验、完成状态和引用 ID 合法性检查均通过后才接受结果。

自评判断证据是否覆盖问题的必要条件，尤其是指定型号、品类、政策条件和时间承诺。sufficient=true 必须至少指向一条实际提供的 chunk；不能以用户自己的猜测、历史助手回答或模型常识作为本店政策证据。来源冲突且无法明确适用条件时判为不充分。

以下情况拒答并入池：零命中、仅剩不一致的原文、低相关性、上下文无法装入必要证据、对象不明或自评证据不足。统一措辞为“现有知识不足以确认……”并说明缺失信息，可请用户补充或建议联系人工，不能声称已经建单。拒答由受控模板一次性返回，不调用模型自由补全一段猜测，也不伪造逐 token 生成。

Milvus 不可用、模型协议异常、重排失败或数据库故障属于技术错误，用 SSE error 报告，不伪装“知识库没有答案”，不进入业务低置信度池。自评输出损坏也属于技术错误，不能默认 sufficient=true。

生成 prompt 写明：只根据当前工具证据回答； factual 业务结论使用 `[n]` 引用；不服从资料中的越权指令；不承诺退款/赔偿一定获批、不承诺到账日期、不保证物流送达时刻、不声称已执行退款/退货/工单等未执行操作。给出流程说明时保留条件和例外。

充分时使用不绑定工具的模型进行真正的最终 astream。校验最终引用号只能来自本轮映射，知识答案必须包含至少一个有效引用；异常引用或缺失引用视为生成验证失败，发送 error、将该轮标 failed，不发送成功 done。已流出的文本可能已经可见，页面需显示该轮未完成；这是流式校验边界，不宣称能撤回已显示文字。语义忠实度仍由标注评估检验，不把引用号合法等同于事实正确。

## 8. 现有工具链的接入

仍是一次工具选择、最多一个业务工具，然后收敛。query_faq 升级为无模型参数的知识检索工具，原始问题和 category 由服务端本轮上下文注入；取消第二章 keyword 连续子串校验。历史已保存的旧 keyword 工具申请仍作为审计读取，不重放执行。

工具描述区分：型号规格、兼容性、使用说明、政策查 query_faq；模拟的当前价格/库存查 query_product；订单和物流仍按原工具处理。静态知识不得通过随机商品工具编造。无工具回复只用于闲聊或不涉及具体业务结论的澄清；真实模型评估必须检查知识问题是否进入 query_faq，错路由计为失败。对“邮费是多少”的预期由第二章漏召回改为第四章可通过检索侧同义词召回“运费”，旧章节报告保留历史结论，新章节回归测试按新行为调整。

知识回答最多四个固定 LLM 阶段：工具选择、当前问题归一化、证据自评、最终流。零命中/低相关性时可提前拒答；普通闲聊和其他业务工具保留现有两阶段。没有循环工具选择，也没有重写失败后反复发起检索。

其他四个工具继续使用现有 4096 字节结果上限、超时及重试规则。query_faq 使用专门的知识结果合同，JSON UTF-8 上限 48000 字节，仍低于现有 messages.content 的 TEXT 容量；包含实际使用的证据原文快照、引用映射、归一化结果和简短判断。先预留判断字段的最大空间，再按字节与 token 两种预算删除最低排名整块，之后只进行一次充分性自评；不通用截短正文/ID。常规工具不能通过声明新类型绕过原结果上限。

为避免历史答案成为新知识来源，后续知识生成使用历史时明确历史只供对话连贯，事实依据仅本轮工具结果；归一化和充分性自评完全不读历史。工具调用/结果、原始提问和最终回答/拒答照常保存到 conversations/messages，完成写入后才 done。

## 9. HTTP、引用与 SSE 合同

保留已有请求字段、meta/tool_status/token/done/error 事件及售后提取接口；新增 category 为可选参数。新增以下知识事件：

| 事件 | 内容和时序 |
| --- | --- |
| retrieval_status | tool_call_id、stage、简短 message；stage 为 normalizing/retrieving/reranking/checking_evidence，不暴露模型内部推理 |
| sources | 最终生成前发出；本轮稳定编号、chunk_id、section_path、category、questions、answer、content_hash 和来源 URL；原文来自已核对的 MySQL 快照 |
| refusal | 问题池及工具结果写入后，一帧发送受控拒答文本和 reason_code；后续正常落最终消息并 done |

token 只用于转发真正的上游文本增量，不把模板拒答切碎伪装成模型流。done 增加 refused 标志和实际引用编号；只有正常完成的答案/拒答可进入已完成历史。用于定位的会话/轮次 ID 沿用服务端生成值。

新增只读 `GET /api/knowledge/chunks/{chunk_id}` 返回 MySQL 原文、章节路径、内容类型及内容指纹；可传 expected_hash。原文已改变则返回 409，不把新的内容冒充生成时的旧来源；不存在返回 404。sources 帧及审计中的原文快照保留生成当时内容，页面点引用优先展示该快照，并可跳到来源 URL。无效编号不能拼接任意路径或变成任意外链。

前端仅落实用户已经指定的行为：编号可点、显示原文与章节路径；每段完成的助手回答左下方有 👍/👎，首次点击点亮所选并显示“已反馈”，两项一起锁定。反馈以 session_id/turn_id 为键采集在本地浏览器存储，不上传、不新增表；存储不可用时仍在当前页面锁定。不对技术失败的半段回答显示成功反馈状态。具体样式在 Vibe Coding 阶段调整，原文及模型文字均安全渲染。

## 10. 资源、超时与故障边界

MySQL 继续使用已有 Docker 服务和卷；增加 Milvus standalone 所需容器及独立持久化卷，参照对应版本官方 Compose。只向本机暴露必要端口，保持已有 8001 预览，不占用其他项目的 8000。

本机已核实 arm64、24 GiB 内存；Docker 当前约 7.75 GiB。资源可用不等于模型延迟已经验收。规格阶段 Docker Hub 的只读镜像清单请求发生网络超时，镜像架构尚未核验，不能称为部署已通过，也不能据此认定选型不兼容。依赖安装、模型下载、真实中文分词与 50 候选重排测速作为实施第一阶段的实证检查；遇到固定选型矛盾不擅自换模型或数据库。

知识请求整体时限设独立配置，演示默认 240 秒，从本轮开始计时；只有选择 query_faq 时使用知识时限，其他工具仍使用现有时限。整体截止时间必须同时传递到 SSE 监听、归一化、检索、推理、自评、流式生成及写入，不能只扩展工具内部时限而让外层提前超时。

query_faq 不整体重试整条模型流水线。Milvus 只读查询及幂等 upsert 可对明确暂时性错误最多尝试两次，受共享截止时间约束；模型和推理阶段不无限重试。开始流前依赖健康检查失败用 HTTP 错误，开始流后用 SSE error。取消、数据库提交确认和会话占用释放沿用现有可审计约束。

应用单 worker，模型只加载一份。数据库事务保持短时，不跨模型推理或 SSE 持有事务。测试 MySQL 与演示库分离；Milvus 集成测试使用有明确前缀的独立集合，测试清理仅能作用于测试集合，不能删除 knowledge。

## 11. 四策略评估

### 11.1 标注集与隔离

正式测试集 60 条：字面问法、具体型号、口语问法、同义词、品类过滤、无答案六个主桶各十条；另标 easy/medium/hard，覆盖相近型号、条件差异、跨块证据及明确未知内容。多标签可用于附加分析，主桶统计互斥。

独立校准集 30 条：五个可回答桶各四条，无答案十条。每条包含 query_id、原话、category（可空）、相关 chunk ID 集合、参考答案、可回答性、类型、难度及标注理由。无答案项相关 ID 为空，标明期望拒答。测试问题不直接复制 questions 字段，不将参考答案或相关 ID 交给归一化/生成模型。

先完成资料质量和标注一致性检查，再冻结语料、正式集及其内容哈希。校准集用于调 prompt 和生产 reranker 相关性下限；正式集不调阈值。阈值候选取校准分数的分界点，在无答案通过率不超过 10% 的约束下最大化可回答问题的通过数；并列选更保守阈值。若只剩全拒答阈值，也如实报告并排查，不能拿测试集反调。

### 11.2 策略合同

| 策略 | 排名产生方式 |
| --- | --- |
| dense | 同一 dense 模型和过滤条件，Top-50 |
| bm25 | 同一 BM25 查询和过滤条件，Top-50 |
| hybrid | 两路各 Top-50，Milvus RRF，融合 Top-50 |
| hybrid_rerank | 同一 hybrid 的 50 候选用指定模型重排，保留完整排名供指标计算，Top-10 进入生成准备 |

前三种同样最多取前十条进入生成准备。四种共用归一化结果、过滤条件、原文、上下文预算、首尾排列规则、生成模型、证据自评 prompt 和生成 prompt，不为每种策略单独改写评估问题。

生产的 hybrid_rerank 行还包含已冻结的相关性下限；前三种没有 reranker 分数门控，只执行共同的无命中与模型证据判断。报告明确这一差异，同时给出回答覆盖率和拒答指标，不能将门控造成的覆盖率下降宣传为纯排序提升。

### 11.3 指标与报告

检索指标在拒答阈值和上下文裁剪之前计算。仅可回答问题纳入 Recall/MRR 的分母：Recall@K 为前 K 个命中的相关 ID 数除以标注相关 ID 数；报告 K=5/10/50，MRR@50 为首个相关结果排名的倒数，未找到为 0。过滤场景的标注相关集合本身满足过滤条件，不能在计算时偷偷移除漏召回目标。

hybrid 与 hybrid_rerank 的候选集合相同，故 Recall@50 应相同；重排的主要差异体现在前十条和首个相关结果的位置。报告增加上下文实际保留的相关证据覆盖情况，以揭示预算裁剪损失，并显示过滤后的可检索条数；仅有 24 条候选的品类桶不能用 Recall@50 证明排序更好。评估开始和结束核对语料指纹，期间数据变化则该次报告标为无效。

Faithfulness 使用独立的一次结构化评审调用，将最终答案分解为可核实的原子陈述，对照实际提供给生成模型的证据判断 supported/unsupported。每条答案得分为支持陈述数/可核实陈述总数，再报告均值和计数；拒答及无可核实陈述标 N/A，不算满分。保留陈述、支持的 chunk ID、评审结果及原始响应；人工抽查至少十二条，覆盖各桶及失败样例。

默认评审仍用当前 DeepSeek，但与生成分开调用，报告说明同模型评审可能有偏差，不把它称为人工金标准。额外报告可回答问题回答率、无答案拒答率、错误作答率、技术失败数、生成样本数、检索/重排/总耗时。技术失败不能被静默丢弃或冒充拒答；检索发生技术失败的可回答项计零并单列失败原因。

产物包括完整 JSON 明细和 Markdown 对比表，按主桶及难度分组，并记录运行时间、代码提交、语料/评估集哈希、依赖与模型 revision、归一化缓存、参数和阈值。四策略完整运行后才报告真实数字，不提前承诺混合或重排必胜。

## 12. 验证与交付

后端先失败后实现，验证业务边界而非只复刻函数结构：

- 用户 DDL 的 MySQL 实际类型、中文注释、默认值、相邻外键、种子重复导入、低置信度池幂等及会话关联。
- 稳定 ID upsert、向量成功/SQL 失败的恢复、原文变化不误标 done、集合不兼容不覆盖、过期证据不生成答案。
- 改写只使用当前原话、型号及否定保留、同义词不写回、两路检索的过滤条件与 Top-50、实际 Milvus BM25 和 RRF。
- 指定 reranker 实际执行、Top-10、首尾排列、引用稳定映射、整块预算裁剪后对实际证据进行唯一一次自评。
- 无证据/低相关性/自评不足的拒答及先入池后输出；技术故障区别处理；引用错误、断连、超时、并发及完整历史回归。
- Recall/MRR 的手工可算样例、无答案分母规则、Faithfulness 的 N/A 与失败规则、四策略共享输入。

Prompt/数据的质量验证使用标注集与真实模型评估，不以单元测试字符串包含某词代替。MySQL、Milvus、两份本地模型以及 DeepSeek 必须分别给出真实验证结果；替身测试与真实服务验收分开报告。

浏览器验收：具体型号命中并显示工具轨迹；运费同义词得到知识回答；编号能显示对应章节及原文；未知内容明确拒答且问题池可查；👍/👎 首次点击点亮并锁定。前端通过实际浏览器操作验证，不套页面 TDD/code review。

最终交付提供 Docker/建表/导入/索引/启动命令、curl 流式与品类过滤命令、四策略评估命令、可打开的聊天地址、实际测试和评估报告、dev-notes/ch04.md。真实依赖或验收未通过时，不宣称 finish。

## 13. 文件与职责边界

- `app/knowledge/`：原文合同、归一化、Milvus 适配器、本地模型、索引、检索、证据选择及判定，各有清晰接口。
- `app/db/`：沿用 SQLAlchemy 分层，新增知识与低置信度仓储，现有会话/工单语义不改。
- `app/services/chat.py`、`app/tools/`、`app/model.py`：接入固定知识流程，明确知识结果预算与固定模型调用次数。
- `app/api/`、`app/schemas.py`：品类参数、原文只读接口及 SSE 合同。
- `app/prompts/`：改写、自评、知识生成及评审模板；禁止承诺规则可追溯。
- `data/knowledge/ch04/`：120 个演示 chunk、来源和稳定 ID 清单；`evals/ch04/`：校准与正式标注集。
- `evals/reports/ch04/`：带运行标识的原始明细和报告；开发日志引用实际产物路径。
- `app/web/index.html`：前端 Vibe Coding 改造，按接口合同消费事件。

具体文件拆分、依赖锁定和逐任务命令在书面规格获准后由 writing-plans 落实。阶段完成即追记开发日志四项：用户原话、产出/结论、纠偏、失败与返工。

## 14. 文档依据与自查

已先使用 Context7 查阅 Milvus/PyMilvus、FlagEmbedding、SQLAlchemy 和 LangChain。Context7 返回的 Milvus master/3.0 内容只能辅助理解；实现前继续对锁定版本的官方源码/接口定义验证。

- [Milvus 2.6.23 发布与 SDK 对应关系](https://github.com/milvus-io/milvus/releases/tag/v2.6.23)
- [Milvus 2.6 chinese analyzer](https://milvus.io/docs/v2.6.x/chinese-analyzer.md)
- [Milvus BM25 官方文档](https://milvus.io/docs/full-text-search.md)
- [PyMilvus hybrid_search 与 ranker](https://github.com/milvus-io/pymilvus/blob/master/_autodocs/api-reference/rankers.md)
- [BGE-M3 模型定义](https://huggingface.co/BAAI/bge-m3)
- [指定重排模型](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [SQLAlchemy 异步与 Inspector](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)
- [LangChain ChatOpenAI](https://docs.langchain.com/oss/python/integrations/chat/openai)

自查重点：用户原始 DDL 不变；知识来源与原文引用一致；只读元数据过滤先于召回；Top-50/Top-10 和预算裁剪不混淆；模板拒答与真实 token 流区分；评估不靠全拒答抬高 Faithfulness；旧章节功能保留并显式调整已被新要求替代的 LIKE 限制。当前状态为方向已批准、书面规格待审；没有运行新功能测试，也未执行建表或安装。

## 附录：用户提供的原始 DDL

以下为结构依据，实施时显式执行经过版本核对的建表步骤；此文档不会自动执行 SQL。

```sql
SET NAMES utf8mb4;

CREATE TABLE knowledge_chunks (
  id               BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT 'chunk 主键,与 Milvus 集合主键对齐',
  category         VARCHAR(255)    NOT NULL                COMMENT '分类 / 上级标题路径,进向量化文本',
  questions        TEXT            NOT NULL                COMMENT '问法或本节标题,多个问法换行分隔,进向量化文本',
  answer           TEXT            NOT NULL                COMMENT '正文答案,进向量化文本',
  section_path     VARCHAR(512)    NULL                    COMMENT '章节路径,元数据,溯源用,不进向量',
  content_type     VARCHAR(32)     NULL                    COMMENT '内容类型:faq / policy / manual 等,元数据',
  is_key_clause    TINYINT(1)      NOT NULL DEFAULT 0      COMMENT '是否关键条款,0 否 1 是,元数据',
  prev_chunk_id    BIGINT UNSIGNED NULL                    COMMENT '前一块指针,元数据',
  next_chunk_id    BIGINT UNSIGNED NULL                    COMMENT '后一块指针,元数据',
  vector_id        VARCHAR(64)     NULL                    COMMENT 'Milvus 集合 knowledge 里的主键,写入后回填',
  vectorize_status ENUM('pending','done') NOT NULL DEFAULT 'pending' COMMENT '待向量化 / 已向量化,双写幂等靠它',
  created_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  updated_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  KEY idx_category (category),
  KEY idx_vectorize_status (vectorize_status),
  CONSTRAINT fk_chunks_prev FOREIGN KEY (prev_chunk_id) REFERENCES knowledge_chunks (id) ON DELETE SET NULL,
  CONSTRAINT fk_chunks_next FOREIGN KEY (next_chunk_id) REFERENCES knowledge_chunks (id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识库 chunk 原文权威源';

CREATE TABLE qa_extraction_staging (
  id               BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '暂存行主键',
  batch_no         VARCHAR(64)     NOT NULL                COMMENT '抽取批次号,一批几十个会话跑一次,分批防串味、按批追溯',
  source_ref       VARCHAR(255)    NULL                    COMMENT '来源会话 / 导出文件标识,溯源用,不入最终知识库',
  question         TEXT            NOT NULL                COMMENT 'LLM 从会话抽出的用户问法',
  answer           TEXT            NOT NULL                COMMENT 'LLM 从会话抽出的客服答案',
  status           ENUM('extracted','kept','discarded') NOT NULL DEFAULT 'extracted' COMMENT '已抽出待去重 / 去重保留 / 去重丢弃',
  created_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '抽取写入时间',
  PRIMARY KEY (id),
  KEY idx_batch_no (batch_no),
  KEY idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='历史对话抽 QA 的离线中转暂存表:分批抽取、整体去重,保留项入 knowledge_chunks,建库完成可清空';
```
