# E-commerce Customer Service Agent

一个持续演进的电商客服 Agent 项目。仓库始终维护单一可运行版本，通过 Git 提交和版本标签记录从最小聊天服务到 RAG、Tool Calling、Workflow/HITL、Memory、Trace 和 Evaluation 的演进过程。

## v0.18.0

当前版本提供：

- FastAPI 服务与 `POST /chat`；
- OpenAI-compatible 聊天模型调用；
- `session_id` 和可信 Runtime Context 接入；
- 受控的电商客服身份与业务事实边界；
- 规则优先、轻量分类模型兜底的结构化意图识别；
- 稳定的 `intent_result`（意图、来源、置信度、命中词和说明）；
- 集中管理客服身份、事实优先级和高风险回答边界；
- 从仓库内 Markdown 原文解析带来源、章节和 metadata 的稳定知识块；
- 使用固定长度与重叠窗口切分长章节，保留可追溯的 `chunk_id`；
- 调用 OpenAI-compatible Embedding 接口批量生成向量并在进程内缓存；
- 通过余弦相似度、分数阈值与 Top-K 执行真正的向量检索；
- 将候选召回阈值与可回答置信阈值分开，避免把弱命中当成可靠依据；
- 过滤非当前有效知识，只有明确查询历史时才召回历史规则；
- 只把真实命中的知识块放入回答上下文，并返回顶层 `citations`；
- 低置信时清空 `citations`，使用补充信息或转人工的确定性兜底；
- 使用固定问题集计算 `recall@k`、`precision@k` 和用例通过状态；
- 在 `session_state.rag` 暴露候选阈值、低置信阈值、最高分和兜底动作；
- 在 `session_state.rag_quality` 暴露固定问题集质量摘要；
- 保留用户原话，为向量检索单独生成 `rewritten_query`；
- 对“耳麦”“叠券”“促销”等口语表达进行可观测归一化；
- 按意图补齐活动或售后检索词，但不读取可信 Runtime 身份字段；
- 合并原始查询和改写查询的候选知识，避免单路检索遗漏；
- 在检索前识别活动、售后、物流、商品、订单和投诉场景，并限制候选知识领域；
- 增加精确关键词召回，补足“赠品”“包装盒”“压坏”等长尾边界词；
- 合并原始向量、改写向量和关键词三路证据，并保留每条候选的来源和命中词；
- 根据规范化知识块内容生成稳定的 SHA256 索引指纹与版本号；
- 构建 chunk 快照和关键词倒排表，并拒绝重复 `chunk_id` 的不完整索引；
- 按“知识索引版本 + Embedding 服务/模型”复用知识向量，重建索引时统一失效；
- 使用有容量上限的 LRU 缓存复用稳定知识的 Hybrid RAG 候选，不缓存最终回答；
- 订单、物流、库存和退款进度等实时问题禁止进入检索缓存，也不会生成知识引用；
- 通过受控电商业务客户端查询订单、物流、商品价格/库存和退款申请实时状态；
- 使用 LangChain `create_agent` 让模型在五个只读工具 schema 中生成结构化调用；
- 后端严格校验工具白名单、必填参数和业务编号格式，拒绝额外参数；
- 模型不能提交或覆盖用户 ID，`runtime_user_id` 只由后端注入业务接口委托请求；
- 对工具 Observation 进行字段白名单脱敏，回答只引用脱敏后的实时事实；
- 顶层返回可观测的 `tool_calls` Action/Observation，实时路由不执行 RAG、不生成 citations；
- 业务服务、工具依赖或模型调用不可用时安全降级，不猜测实时状态；
- 工具执行前生成可观测的 `ClarificationPlan`，后端重新计算必填字段；
- 模型只可润色澄清问题，不能注入订单号、退款号、用户 ID 或清除缺失字段；
- 从可信 Runtime Context 读取当前用户订单摘要，关联订单或唯一订单可安全补全；
- 缺少订单号时返回结构化候选项，让用户确认目标而不是让模型代选；
- 支持按月份筛选当前用户候选订单，并显式标记候选上下文是否截断；
- 商品查询得到多个匹配时执行工具后澄清，同时保留本轮 Action/Observation；
- 在工具执行层显式区分内部 `ToolResult` 与模型/响应可见的 `Observation`；
- 原始订单、物流、商品和退款 payload 只在后端内部短暂存在，不进入 LangChain 消息；
- 按工具白名单压缩为安全摘要和关键 `facts`，并通过 `omitted_fields` 说明省略字段；
- 完整物流轨迹、运单号、用户身份、订单备注、商品长描述和退款内部字段不会进入回答上下文；
- 顶层返回 `next_action=answer_user/ask_clarification/fallback_answer/transfer_to_human`；
- 将工具失败归一为超时、参数错误、未找到、无权限、业务错误、模型不可用、系统错误等稳定类别；
- 只读工具仅在超时时最多重试一次，并通过 `attempts` 与 `retry_count` 暴露实际尝试次数；
- 参数错误、无权限、未找到和业务错误不会自动重试，避免无意义请求和副作用风险；
- 工具或模型失败时使用确定性安全模板降级，不把异常详情、响应体或凭证交给模型；
- 在 RAG 和 Tool Calling 前拦截直接退款、取消订单和赔付等高风险写请求，不执行写操作；
- 顶层返回 `risk_level`、`needs_human_approval` 与 `degraded`，并用 `transfer_to_human` 表示人工边界；
- 将商品问题细分为纯实时工具、纯稳定知识 RAG、Tool + RAG 联合回答三条路由；
- 同时询问商品库存/价格与卖点/活动规则时，在一个响应中返回 `tool_calls` 和 `citations`；
- 当前标价、活动价、库存、活动名称和会员条件只来自商品业务接口的安全 Observation；
- 商品卖点与平台活动规则只来自 Hybrid RAG 可靠命中，不用知识库猜当前价格或库存；
- 联合检索平衡选择商品知识与活动规则，避免促销扩展词挤掉商品卖点证据；
- 联合回答模型只接收脱敏 Observation 与可靠知识块，缺少引用标记时回退确定性回答；
- 在 `session_state.tool_rag` 暴露回答来源、事实边界、工具名、citation chunk 和联合完成状态；
- 使用请求级 Hooks 统一治理 `pre_tool_call`、`post_tool_call`、`on_error` 与 `on_completion` 生命周期；
- 工具执行前记录白名单、参数、只读边界和可信身份是否存在，业务读取后统一脱敏 Observation；
- 对手机号、邮箱、凭证字段和外部指令污染做递归清理，Hook 摘要不暴露 `runtime_user_id`；
- 顶层返回有序 `hook_events` 与一次性 `hook_completion`，仅表示公开治理轨迹而非隐藏推理链；
- Hooks 不执行退款、取消或赔付审批，高风险写操作仍停留在人工边界；
- 增加本地 MCP-style Catalog，统一提供五个只读工具的定义、Resource 和 Prompt 绑定；
- 现有 Tool Use 契约从 Catalog 转换生成，避免 Agent 版本之间重复维护工具 schema；
- 顶层返回 `mcp_context`，公开实际选中工具、可用工具及关联 Resource/Prompt URI；
- 未调用工具与高风险拦截会生成不同绑定摘要，高风险 Resource 不会绕过 Workflow/HITL；
- Catalog 不保存业务假数据，订单、物流、退款和商品当前事实仍来自可信业务接口或 Runtime Context；
- 当前实现是课程阶段的本地 MCP-style 组织层，不宣称已连接完整远程 MCP Server；
- 模型最终措辞只在所有 Observation 成功且允许直接回答时采用，否则使用确定性安全结果；
- 默认使用透明的轻量 reranker 重排，可选接入 OpenAI-compatible `/rerank` 服务；
- 商业 reranker 异常时回退轻量重排，并只公开安全的错误类型；
- 优先解析模型平台 `usage`，缺失时使用本地 token 估算；
- 返回输入、输出、总 token 与人民币估算成本；
- 记录会话级成本观察事件；
- 不向外部模型披露 Runtime Context 中的用户身份具体值；
- 可公开展示的 `reasoning_summary` 执行摘要；
- `/health` 与 `/capabilities`；
- 模型缺失或调用失败时的安全话术回退。

当前版本形成三条可观察路由：活动规则、售后政策等稳定知识走“版本化索引 → 查询改写 → Hybrid RAG → Reranker → Grounded Answer/Citations”；订单、物流、商品当前价格/库存和退款进度等纯实时事实走“MCP-style Catalog → ClarificationPlan → pre-tool Hook → LangChain Tool Use → 内部 ToolResult → post-tool/error Hook → 安全 Observation”；同时询问商品当前事实与稳定知识时走“Catalog 商品工具 Observation + 商品/活动平衡检索 + 联合 Grounded Answer”。每条路由结束时都生成一次公开安全的 completion Hook，并返回本轮 MCP 绑定摘要。Runtime Context 不进入 Embedding、缓存键、Reranker 或联合回答 Prompt；原始 ToolResult 和隐藏推理链不进入 Hook 或公开响应，RAG 规则也不会被冒充为当前 SKU 的实时事实。

关键词检索仍是透明的轻量精确词实现，不是完整 BM25/搜索引擎；索引和缓存均为进程内实现，不是独立向量数据库或分布式缓存。当前工具只支持只读查询；已经支持缺参/多候选澄清、ToolResult 压缩、错误分类、超时有限重试、安全降级、商品 Tool + RAG 联合回答、Hooks 治理以及本地 MCP-style 能力目录。高风险写请求只会被拦截并返回人工确认信号，MCP 与 Hooks 都不等同于 HITL；尚未实现可恢复工作流、真实 HITL 审批、多轮澄清状态记忆或远程 MCP Server 连接。

## 项目结构

```text
backend/
  agents/       # Agent 编排
  api/          # HTTP 路由与请求响应契约
  config/       # 环境变量与能力清单
  cost/         # Token usage 解析与估算成本
  degradation/  # 错误分类、有限重试决策、高风险边界与安全降级模板
  knowledge/    # 活动、售后、商品、订单等 Markdown 知识原文
  embeddings/   # OpenAI-compatible Embedding 客户端与文本缓存
  hooks/        # 工具前后、异常与完成阶段的公开安全治理
  integrations/ # 电商业务后端客户端与安全错误映射
  mcp_catalog/  # MCP-style 工具、Resource、Prompt 统一目录与绑定摘要
  models/       # OpenAI-compatible 分类和回答模型客户端
  observability/ # ToolResult 到安全 Observation 的压缩层
  rag/          # 文档切片、版本化索引、检索缓存、混合召回、重排与质量检查
  tools/        # 只读工具契约、规划、可信执行与 LangChain Tool Calling
  rag_quality_cases.json # 固定 RAG 质量问题集
  main.py       # FastAPI 应用入口
```

## 本地运行

要求 Python 3.13+。

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

在 `.env` 中配置真实的 `AGENT_OPENAI_API_KEY`；`AGENT_EMBEDDING_MODEL` 可单独指定向量模型。实时业务查询还需要启动配套电商后端，并配置：

```dotenv
AGENT_ECOMMERCE_BASE_URL=http://127.0.0.1:8081
AGENT_ECOMMERCE_SERVICE_TOKEN=replace-me
```

服务令牌必须与电商后端一致，不要把真实令牌提交到 Git。默认使用轻量重排；如需商业 reranker，可将 `AGENT_RAG_RERANK_ENABLED` 设为 `1` 并配置对应地址、模型和可选独立 Key。然后启动：

```powershell
Set-Location backend
..\.venv\Scripts\python main.py
```

访问：

- 健康检查：`http://localhost:8000/health`
- 接口文档：`http://localhost:8000/docs`

无需模型 Key 的离线回归测试：

```powershell
python -m unittest discover -s tests -v
```

请求示例：

```json
{
  "session_id": "demo-session-001",
  "runtime_user_id": "U1001",
  "runtime_nickname": "张三",
  "runtime_member_level": "gold",
  "runtime_risk_level": "low",
  "user_message": "你好，请介绍一下你能做什么"
}
```

响应中的核心结构化字段示例：

```json
{
  "intent": "refund_request",
  "intent_result": {
    "intent": "refund_request",
    "source": "rules",
    "confidence": 0.95,
    "matched_keywords": ["退款"],
    "explanation": "用户在询问退款、退货或质量问题，规则高置信标记为售后退款类消息。"
  }
}
```

业务知识命中时，顶层 `citations` 会返回：

- `citation_id`、`source_title` 和 `source_path`；
- `section`、`chunk_id`、余弦相似度分数和原文片段。

`session_state.rag` 会返回知识索引版本与指纹、缓存策略及命中状态、查询改写、检索场景、三路候选、向量/关键词/重排分数、召回来源和实时业务缺口，并继续暴露低置信门槛及 citations。`session_state.rag_quality` 返回固定问题集数量、通过数量以及平均 `recall@k`、`precision@k`。`/health` 也会返回当前索引版本和 chunk 数量。

实时业务问题会在顶层返回 `tool_calls`。每条记录包含模型提出且经后端校验的 `action`，以及由内部 ToolResult 压缩得到的 `observation`；Observation 只包含 `summary`、安全 `facts`、`omitted_fields`、`next_action`、`error_category`、`attempts` 和必要的安全候选数据。缺参或多候选时，顶层 `clarification` 返回 `clarification_field`、问题和安全候选项；顶层 `next_action` 告诉调用方应回答、继续澄清、安全兜底还是转人工。`session_state.degradation` 返回是否降级、稳定错误类别、重试次数、是否使用兜底和安全原因码。`session_state.tool_calling` 同时暴露后端校验后的计划、`pre_tool`/`post_tool` 阶段以及 `raw_tool_result_exposed=false`。这条路由的 `session_state.rag.status` 为 `skipped_realtime_tool_route`，`citations` 为空；业务接口会使用 `X-Agent-Service-Token` 和后端注入的 `X-Agent-User-Id` 完成身份委托。

商品混合问题会同时返回 `tool_calls` 与 `citations`。`session_state.tool_rag` 明确记录 `current_price_inventory=tool`、`product_and_promotion_knowledge=rag`，并暴露最终采用的工具与 citation chunk；活动价和当前活动取自商品接口中的 `promotion`，活动 ID、起止时间和未白名单字段不会进入模型。纯库存/价格问题仍只走工具，纯卖点/规则问题仍只走 RAG。

“直接退款、取消订单或赔付”等写请求会在任何检索或工具调用之前被拦截：响应返回 `risk_level=high`、`needs_human_approval=true`、`degraded=true` 和 `next_action=transfer_to_human`。这只是明确的人工接管边界，不代表已经创建工单、执行退款或完成 HITL 审批。

运行中修改知识文件后，可重启服务或在受控维护流程中调用 `rebuild_knowledge_index()` 重建索引；项目不暴露无鉴权的 HTTP 重建接口。重建会原子替换索引快照，并清空依赖旧版本的向量和检索缓存。

顶层 `cost_summary` 继续返回：

- `prompt_tokens`、`answer_tokens` 和 `total_tokens`；
- `token_source=model_usage` 或 `local_estimate`；
- 模型平台返回的 reasoning/cache usage 明细；
- 分开的输入、输出及总成本估算。

成本只是趋势观察，不替代模型平台的真实账单。

## 演进原则

- 每个版本保持可运行、可验证；
- 通过 Git 历史演进，不复制多个 lesson 目录；
- 实时业务事实必须来自业务 API，不让模型猜测；
- 高风险操作必须经过确定性规则和人工边界；
- 密钥、隐私和内部推理不进入仓库或公开 Trace。
