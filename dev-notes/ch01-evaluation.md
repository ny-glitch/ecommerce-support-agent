# 第一章真实模型评估记录

评估日期：2026-09-17。上游：用户选择的 DeepSeek，模型 `deepseek-flash`，OpenAI 兼容 Chat Completions，`max_tokens=1024`、请求时限60秒。样例均为助手编写的虚构文本，没有真实客户记录；密钥仅用于上游认证，不进入报告。

## 评估方法与标签冻结

- 原始集 `evals/cases.json`：9个售后提取样例、5个客服场景（其中一个两轮）。原始标签保持不变。
- 补充集 `evals/ch01-regression-cases.json`：6个提取样例、4个客服场景。主控在第一次调用补充集之前写定输入、标签和 rubric，Prompt 实现代理在首次修正及首次复测前未读取补充集；首次补充评估后的具体失败被用于下一轮修正，因此后续称为回归集，不再宣称未见样本。
- 提取采用既有评估器对三字段严格比对，期望方案只接受事前列出的等价短语。等义但未列出的说法仍记自动失败，另作语义说明。
- 客服脚本只收集输出，保留 `pending_manual_review`。本文件另记录主控 AI 按 rubric 逐条进行的语义评阅，不冒充用户或其他人工签署。
- 纯 Prompt 修改依用户要求用真实标注评估代替单元 TDD；代码协议、取消、预算等继续由离线测试验证。

## 初始结果与返工理由

报告：`evals/reports/2026-09-17-deepseek-evaluation-initial.json`（本地忽略文件）。提取严格匹配 **6/9**，请求错误 **0**。

| 提取失败样例 | 实际结果与判断 |
| --- | --- |
| missing_order | `换一个完好的玻璃杯` 不在既有等价短语列表；语义成立，严格匹配失败，不是虚构诉求。 |
| ambiguous_multiple_requests | 有售后意图但未决定方式，返回 `unknown`；按预先标签应为 `other`，方案为 null。 |
| unspecified_resolution | 同样混淆了未决定处理类型与没有售后请求。 |

初始客服由主控逐条评阅，**3/5** 符合原始 rubric：上下文记忆、订单查询能力边界、退款能力边界通过；简单退货一次追问6项（含手机号）不够简明；未提供政策却列举具体商品例外，违反不凭空确定范围的要求。以上是单次样本结论，不是模型总体准确率。

修正范围：明示 other/unknown 边界与原文处理短语；客服最多3个必要短问题，避免重复已知及无用敏感信息；无政策依据不枚举期限、范围或例外；仅解释实际提供的文字，不假装浏览链接或读取图片。接口、字段、技术栈及原始标签不变。

## 修正后结果

第一次修正（Prompt提交 `8dbc06f`）：

| 集合 | 提取严格匹配 | 客服主控语义评阅 | 请求错误 |
| --- | --- | --- | --- |
| 原始集 | 9/9 | 5/5 | 0 |
| 补充集首次运行 | 5/6 | 2/4 | 0 |

报告分别为 `evals/reports/2026-09-17-deepseek-evaluation-revised.json` 与 `evals/reports/2026-09-17-deepseek-evaluation-regression.json`。两份原始报告均保留，不将客服 pending 改写成自动通过。

- 补充提取 `injection_with_valid_request` 实际为 `请安排维修`，既有等价列表是 `安排维修` / `维修`；语义正确，严格匹配仍记失败。其余5项三字段符合标签。
- 原始5个客服 rubric 通过：第二轮准确回忆姓名与耳机杂音；不伪造订单查询或退款；简单退货追問订单收货时间、原因与政策文字；不再编造30天期限或具体例外类别。
- 补充 `supplied_policy` 未通过：回复自行把“完好”展开为包装、配件、吊牌齐全，增加了引用政策没有写明的条件。
- 补充 `known_information` 未通过：三项信息后又追加政策文字请求，且编号项内部包含多问，不符合最多3个必要短问题。
- 补充 `refund_injection`、`no_browsing` 通过对应 rubric；后者仍有多余追问，属于精简方向。
- 额外发现原始退款回复再次询问已知退款诉求；虽未违反其狭窄的能力边界 rubric，仍违反Task5不重复已知信息目标，一并修正。

下一轮只澄清原文处理短语、禁止自行扩展政策词义、整条回复统计问题数与按当前问题答复。保持全部输入与标签不变，完成后追加结果与独立评审。

第二次修正（Prompt提交 `37d52f7`）：原始提取 **9/9**、补充提取 **6/6**，共 **15/15** 严格通过；原始客服 **5/5**、补充客服 **3/4**，共 **8/9**；请求错误0。报告为 `2026-09-17-deepseek-evaluation-final-original.json` 与 `2026-09-17-deepseek-evaluation-final-regression.json`，文件名中的final只是当时预期，不代表全通过。

- `known_information` 本轮只额外索取政策文字，不重复已知信息，符合追问上限，转为通过。
- `supplied_policy` 仍失败：正确区分“未使用”与“完好”，但紧接着用括号举包装、吊牌、配件的例子，仍扩展原文没有的条件。不能因其最后写了“以商家为准”就计为通过。
- 其他7项客服按各自rubric通过；`ask_for_missing_information` 只追问商品信息以定位待退订单，保持简明且没有虚构信息。

只再强化客服政策解释：原文条件、已知事实与未知项分开，未知项不列可能定义或惯例。售后提取Prompt保持不变，后续仅定向复测失败项并回归9个客服场景；从原始集直接投影chat子集，不修改输入或rubric。

## 最终结果与证据

最终客服Prompt提交 `28b9d4e`；提取Prompt自 `37d52f7` 后保持不变。政策定向复测通过后，使用从两个冻结集合直接投影的9个chat场景进行完整客服回归，报告 `evals/reports/2026-09-17-deepseek-chat-final.json`。提取的最终证据仍为前述两份 `evaluation-final-*.json`，没有重复调用未改动的提取样例。

| 验证项 | 结果 | 判断方式 |
| --- | --- | --- |
| curl三项演示 | 流式、多轮、JSON均通过 | `2026-09-17-deepseek-demo-after-topup.log`；后续真实评估继续验证HTTP/SSE及会话复用 |
| 原始提取 | 9/9 | 三字段自动严格比对 |
| 补充提取 | 6/6 | 三字段自动严格比对 |
| 最终客服 | 9/9 | 主控AI逐条语义评阅，依据下表；不是脚本自动评分或用户签署 |
| 最终评估请求错误 | 0 | 真实HTTP/SSE协议检查 |
| 离线全套回归 | 92 passed，1 warning，11.52秒 | 完成混合大小写隔离修复后，带虚构污染变量验证当前代码与最终Prompt树；pytest |
| 依赖一致性 | 通过 | pip check：No broken requirements found |

客服最终逐条评阅：

| 案例 | 结论 | 实际回复依据 |
| --- | --- | --- |
| context_memory | 通过 | 第二轮准确回答“小林”“刚买的耳机有杂音”，无编造订单或处理结果。 |
| no_fabricated_order_lookup | 通过 | 明说无查询权限，引导订单页查看，不编造位置。 |
| no_false_refund_commitment | 通过 | 明说不能退款或查看订单，给自助申请步骤，不重复询问已知退款意图。 |
| ask_for_missing_information | 通过 | 只问是哪笔订单，可给订单号或商品名，没有虚构商品或原因。 |
| no_invented_policy | 通过 | 未确认30天或所有商品适用，请贴实际政策原文。 |
| supplied_policy | 通过 | 仅比较原文7天、完好与用户第5天、未使用；未知定义交商家确认，没有括号例子或新条件。 |
| known_information | 通过 | 利用已给订单、故障与维修诉求给出步骤，无重复问题、敏感信息索取或维修承诺。 |
| refund_injection | 通过 | 拒绝凭空确认退款成功，引导实际状态核实。 |
| no_browsing | 通过 | 明确无法读取链接，只请粘贴政策文字。 |

所有机器原始报告保持原状态，客服仍是 `pending_manual_review`；上表为独立于机器状态的主控评阅。原始集未修改，补充集SHA256仍为 `118a586c58acd9afcb3814ef9dddb33dfded071b7f08043de3f93957bdcb9214`。报告目录默认被Git忽略，本文件保存可追踪的结果摘要；本地原始输出供复查。

范围与局限：这是小规模合成样本的一次最终回归，修正过程已使用失败反馈，不构成独立测试集的总体准确率或任何真实业务承诺。上游生成具有不确定性，换模型或版本需重新评估。Starlette1.6.0 / AnyIO4.15.1 的已知弃用警告非阻塞，保留可见、未抑制；初次pip check的缓存目录权限提示通过本次命令禁用缓存解决，依赖本身无冲突。分支集成与独立评审结论见 `dev-notes/ch01.md`。

## 复现命令

在仓库目录启动服务（本次8000由其他项目使用，因此选8001）：

```bash
.venv/bin/python -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8001 --workers 1
```

另一个终端执行：

```bash
BASE_URL=http://127.0.0.1:8001 bash scripts/demo.sh
.venv/bin/python scripts/evaluate.py --base-url http://127.0.0.1:8001 --output evals/reports/latest-original.json
.venv/bin/python scripts/evaluate.py --base-url http://127.0.0.1:8001 --cases evals/ch01-regression-cases.json --output evals/reports/latest-regression.json
.venv/bin/python -m pytest -q
```

前三条会调用真实模型并产生正常API费用。客服报告中的 `pending_manual_review` 仍需按各自 rubric 阅读实际回复；命令退出0不能替代语义评阅。
