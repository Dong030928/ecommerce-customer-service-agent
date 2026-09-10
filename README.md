# E-commerce Customer Service Agent

一个持续演进的电商客服 Agent 项目。仓库始终维护单一可运行版本，通过 Git 提交和版本标签记录从最小聊天服务到 RAG、Tool Calling、Workflow/HITL、Memory、Trace 和 Evaluation 的演进过程。

## v0.30.0

当前版本提供：

- FastAPI 服务与 `POST /chat`；
- OpenAI-compatible 聊天模型调用；
- `session_id` 和可信 Runtime Context 接入；
- 将 Runtime Context 拆分为低风险的 `trusted_for_model` 与后端校验专用的 `system_only` 视图；
- 会员等级、账号风险和工具权限只接受可信调用方字段，聊天里的 VIP 或其他用户身份自称不能覆盖登录态；
- `conflict_notes` 公开身份/会员自称冲突，`permission_decision` 公开订单工具是否完成可信身份校验；
- 公开 Runtime Context 不返回原始用户 ID，也不展开当前用户订单、手机号或地址；
- 新增 Context Builder，将用户消息、Runtime Context、Session Memory、Tool Observation、RAG 片段和 Workflow State 统一组织；
- 每个 `ContextItem` 都记录来源类型、信任级别、模型可见性、冲突组和采用或排除原因；
- 顶层 `context_report` 和 `session_state.context_builder` 返回一致的上下文选择报告；
- 上下文按 trusted、verified、session、external、untrusted 的优先级排序，不把多来源事实扁平混合；
- Runtime Context 和工具事实优先于用户自述与 Session Memory，Workflow State 不被历史审批说法覆盖；
- 系统身份与授权信号进入排除项，恢复令牌、幂等键和冻结字段不会复制进模型上下文候选；
- 新增上下文压缩层，在 Context Builder 之后生成可审计的 `compression_report`；
- 当前用户消息、Runtime Context、Tool Observation 和 Workflow State 作为保护项，不会被旧聊天挤出窗口；
- 最近 4 条历史消息通过 Sliding Window 保留，中间历史命中当前订单号时按相关性召回；
- 旧低相关上下文折叠为公开摘要，`model_context` 只包含本轮保留项和压缩摘要；
- Workflow checkpoint 独立保存在服务端，审批恢复不依赖聊天历史或压缩摘要；
- 新增 Prompt Injection 防护层，统一扫描用户、历史消息、工具 Observation 和 RAG 外部文本；
- 外部文本按 user、tool、rag 标记来源，并在 `safety_decision` 中公开污染类别和处理结果；
- “忽略系统规则”“直接批准退款”等越权指令在进入模型前被隔离；
- 系统提示词、密钥、工具细节和隐藏推理请求在规划与回答模型调用前直接阻断；
- 手机号和收货地址在进入 `sanitized_context`、RAG Prompt 或 Tool Message 前完成脱敏；
- 安全层不替代 Workflow/HITL，外部文本仍不能绕过 checkpoint、恢复令牌、角色和事实复核；
- 新增公开 Trace 可观测层，将关键执行节点规范化为 `trace_event_v1`；
- Trace 覆盖 Runtime Context、Context、路由、RAG、工具、Workflow/HITL、Hooks、成本和最终回答；
- 新增 `GET /sessions/{session_id}/trace`，可按会话读取有序 Trace 事件；
- `/chat` 和 `/chat/resume` 的 `session_state.trace` 公开事件数量、Schema 版本及查询地址；
- Trace payload 递归移除系统提示词、隐藏推理、恢复令牌、幂等键、原始工具结果和身份字段；
- 每条事件明确 `public_trace=true`、`hidden_cot_exposed=false`，Trace 是执行证据而不是思维链；
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
- 顶层返回 `risk_level`、`needs_human_approval` 与 `degraded`；成功创建待审批请求不标记为降级，证据不足或审批冒充仍安全降级；
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
- 增加入口 `TaskPlanner`，统一生成可校验的 `RoutePlan`，区分 General、RAG、Tool、Tool + RAG 和 Workflow 信号；
- 高置信业务规则直接生成路线，低置信问题才调用轻量规划模型补充结构化草案；
- 规划模型只接收用户问题和公开工具候选，不接收 Runtime 用户身份或业务 payload；
- 模型提交的工具、知识域、实体引用和上下文字段均经过允许列表收窄；
- `RoutePlan.required_tools` 会限制 LangChain Tool Use 实际可见的工具集合；
- 高风险写操作会被最终安全规则覆盖为 `requires_workflow=true`，并进入固定 LangGraph 工作流；
- 顶层返回 `route_plan` 与公开安全的 `planner_trace`，不暴露隐藏推理链；
- 高风险售后在写动作被阻断后，只执行订单状态和物流状态的确定性只读查询；
- 结合真实 Hybrid RAG 售后政策 citation，返回结构化 `after_sale_assessment` 申请资格评估；
- 缺少订单号时先澄清；证据不完整时明确阻断，不让模型猜测资格或业务状态；
- `blocked_write_actions` 明确禁止退款、批准退款、取消订单和创建补偿；
- 使用 LangGraph `StateGraph` 固定售后类型识别、订单校验、物流读取、政策检索、资格判断和提交前停止节点；
- 顶层 `workflow` 与 `session_state.workflow` 返回工作流类型、状态、当前节点及完整节点历史；
- 未发货退款仅在订单已支付、未出库且未发货时进入待人工审批；
- 签收后退货独立核验签收状态、七天窗口、商品可退属性、退货原因与政策依据后进入待人工审批；
- 商品可退属性可由已认证应用网关注入的精确订单上下文补齐，业务 API 已有事实保持更高优先级；
- 资格通过后创建结构化 `ApprovalRequest(status=pending)`，工作流转为 `paused` 并返回 `require_human_approval`；
- 普通聊天中的“主管同意”“审批通过”等说法会被阻断，不能伪装成审批结果；
- 待审批工作流保存进程内 checkpoint，并返回不可预测的 `resume_token`、稳定幂等键和公开冻结字段；
- 新增独立 `POST /chat/resume`，校验会话、工作流、恢复令牌、审批角色和审批决定；
- 审批通过前重新读取订单与物流事实，冻结字段发生变化时拒绝沿用旧审批结果；
- 重复审批恢复命中相同幂等键并返回原申请编号，不重复记录售后申请；
- 当前提交仍是进程内售后申请记录，不调用真实退款、退货或支付写接口；
- 新增短期 Session Memory，只保存受控工具确认的最近订单、最近商品、最近意图和少量低风险偏好；
- 用户追问“刚才那个订单”时，从当前会话记忆恢复订单引用，并再次通过可信用户身份查询业务事实；
- 记忆按 `session_id + runtime_user_id` 隔离，不把跨用户的同名会话视为同一记忆空间；
- 顶层返回 `memory_update` 与 `memory_snapshot`，公开每项接受或拒绝写入的原因和 session TTL；
- 手机号、地址、审批令牌、系统提示词请求、聊天原文和用户自称不会进入可复用记忆；
- Memory 只能辅助消歧，不能覆盖 HITL checkpoint、恢复令牌、冻结字段或幂等校验；
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

当前入口先扫描用户与历史文本，泄密请求在规划模型前阻断，普通污染指令经隔离后再进入意图和路由链路。可信 Runtime Context 与用户文本保持双通道，Session Memory 仅做受控消歧；稳定知识进入“版本化索引 → 查询改写 → Hybrid RAG → Reranker → 安全清洗 → Grounded Answer/Citations”，实时事实进入“MCP-style Catalog → ClarificationPlan → Hooks → LangChain Tool Use → ToolResult → 安全 Observation”。高风险写请求进入 LangGraph Action Boundary。每条路由结束时，Context Builder 按来源与信任级别汇总，上下文压缩层按保护项、相关性和最近窗口生成候选，安全层公开 `safety_decision`，Trace 层再把已完成的关键节点写成有序、脱敏的公共执行证据。原始用户身份、ToolResult、恢复令牌、幂等键、冻结字段和隐藏推理链不进入模型上下文或 Trace。

关键词检索仍是透明的轻量精确词实现，不是完整 BM25/搜索引擎；索引、缓存、checkpoint、Session Memory、Trace 和模拟售后申请记录均为进程内实现，不是独立数据库或分布式状态服务。Context Builder、压缩与 Prompt Injection 防护当前使用确定性规则，不调用模型生成历史摘要，也不是覆盖全部对抗表达的内容审核系统；Trace 只说明公开执行结果，不提供隐藏思维链，当前尚未接入 Evaluation、失败归因或持久化观测平台。TaskPlanner 只生成入口 RoutePlan，业务工具仍只支持只读查询；审批人字段仍依赖受信网关注入，尚未接入独立认证/RBAC、真实业务写入或远程 MCP Server。

## 项目结构

```text
backend/
  agents/       # Agent 编排
  api/          # HTTP 路由与请求响应契约
  config/       # 环境变量与能力清单
  context/      # Runtime Context、Context Builder、压缩与 Sliding Window
  safety/       # Prompt Injection 扫描、污染标记与隐私脱敏
  cost/         # Token usage 解析与估算成本
  degradation/  # 错误分类、有限重试决策、高风险边界与安全降级模板
  knowledge/    # 活动、售后、商品、订单等 Markdown 知识原文
  embeddings/   # OpenAI-compatible Embedding 客户端与文本缓存
  hooks/        # 工具前后、异常与完成阶段的公开安全治理
  integrations/ # 电商业务后端客户端与安全错误映射
  mcp_catalog/  # MCP-style 工具、Resource、Prompt 统一目录与绑定摘要
  memory/       # 有写入策略和排除策略的进程内 Session Memory
  models/       # OpenAI-compatible 分类和回答模型客户端
  observability/ # 公共 Trace 事件规范化与会话存储
  planner/      # TaskPlanner、RoutePlan 白名单约束与公开 PlannerTrace
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

“直接退款、取消订单或赔付”等写请求会在普通工具调用之前进入高风险工作流：符合退款或退货资格时返回 `ApprovalRequest(status=pending)` 并暂停；这不代表人工已经批准，更不代表已经执行退款或退货。

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
