# Changelog

## 0.33.0

- 将负反馈改为“向量推荐、人工确认、重跑评测、失败归因、候选审核、批准入库”的分阶段闭环。
- 新增真实 Embedding 驱动的 Top-K Case 相似度推荐，检索问题组合原问题、观察回答与反馈内容。
- 新增反馈待办/详情、Case 目录、Case 确认和 Case 审核接口，支持无匹配场景以及批准、拒绝、合并三类结果。
- 人工确认已有 Case 后才触发 Agent 重跑；待审核 Case 不参与回归，只有批准 Case 并入 `/eval/run`。
- 新增反馈、事故 Trace 快照、候选 Case 和已审批 Case 的本地 JSON 原子持久化，并支持环境变量覆盖路径。
- 增加向量召回、状态机、审批隔离、持久化重载、输入校验与敏感信息脱敏回归测试。
- FastAPI、健康检查和能力声明版本同步升级到 `0.33.0`。

## 0.32.0

- 合并 lesson39 的失败归因与反馈闭环能力，新增 `POST /feedback/submit`。
- 将反馈绑定到已有会话 Trace 与可选 Eval 结果，按 Prompt、RAG、Tool、Context、Workflow 和测试期望归因。
- 负反馈生成可执行的进程内回归用例，并自动并入后续 `/eval/run`；正向和中性反馈不回填失败用例。
- 回填已有 case 时保留原始可复现输入、运行上下文和断言，公开响应不回显输入及 Runtime Context。
- 增加反馈字段约束、敏感文本脱敏、未知会话/用例处理及线程安全存储。
- FastAPI、健康检查、Workflow、Trace 和测试断言版本同步升级到 `0.32.0`。

## 0.31.0

- 合并 lesson38 的固定用例回归评测能力，保持单一工程目录。
- 新增 `backend/cases.yml` 与 `POST /eval/run`，支持全量或按 `case_id` 运行。
- 使用当前 Agent 的真实配置，检查回答、工具、引用、Trace、Workflow/HITL 与禁止输出，返回 `eval_report_v1`。
- 将课程工具、引用和审批断言适配为当前工程协议，不注入课程订单快照。
- 补齐课程未发货退款 SOP 对应的 Markdown 知识块，使引用断言检查实际检索结果。
- 修正未发货退款查询被统一扩展为签收后退货的问题，保留支付与审批相关检索信号。
- 每个用例分配唯一会话；异常记为失败并继续，未知用例返回 404，错误配置返回受控错误。
- 新增评测器正反例及 API 回归测试，同步版本与能力声明为 `0.31.0`。

## 0.30.0

- 新增 `trace_event_v1` 公共 Trace 契约和线程安全的进程内 TraceStore；
- 新增 `GET /sessions/{session_id}/trace` 会话 Trace 查询接口；
- Trace 覆盖 Runtime Context、Context、RoutePlan、RAG、工具、Workflow/HITL、Hooks、成本和最终回答；
- 工具调用记录 started/finished 配对事件，不写入原始参数值或 ToolResult；
- Workflow 暂停与人工审批需求分别记录，`/chat/resume` 记录恢复和审批结果；
- Trace payload 递归移除系统提示词、隐藏推理、身份、恢复令牌、幂等键、冻结字段和原始工具结果；
- 手机号、地址和密钥形态在 Trace 写入前统一脱敏；
- 每条 Trace 明确 `public_trace=true` 与 `hidden_cot_exposed=false`；
- `/chat` 与 `/chat/resume` 的 `session_state.trace` 返回 Schema、事件数和查询入口；
- 保持 Prompt Injection、上下文压缩、Memory、Hybrid RAG、Tool Calling 和 Workflow/HITL 能力；
- FastAPI 应用与健康检查版本同步升级到 `0.30.0`。

## 0.29.0

- 新增 Prompt Injection 防护层，统一扫描用户、历史、工具和 RAG 外部文本；
- 新增 `ExternalText`、`SafetyScan` 与 `SafetyDecision` 公开契约；
- 用户文本在路由与模型调用前清洗，受保护信息请求直接由确定性安全边界阻断；
- 系统提示词、密钥、工具细节和隐藏推理不对外泄露；
- Tool Observation 在作为 LangChain Tool Message 返回模型前递归清洗字符串字段；
- RAG citations、回答 Prompt 与确定性回退中的知识片段统一清洗；
- 手机号和收货地址在进入安全上下文前脱敏；
- `safety_decision` 公开污染来源、类别、处理方式和拒绝主题，不公开原始敏感文本；
- Prompt Injection 防护不替代 Workflow/HITL，普通聊天不能绕过审批恢复通道；
- 保持 Context Builder、上下文压缩、Sliding Window、Memory、Tool Calling 和 Hybrid RAG 能力；
- FastAPI 应用与健康检查版本同步升级到 `0.29.0`。

## 0.28.0

- 新增确定性上下文压缩层，在 Context Builder 之后生成 `compression_report`；
- 新增调用方历史消息契约，并将其始终视为不可信上下文；
- 当前用户消息、Runtime Context、Tool Observation 和 Workflow State 标记为保护项；
- 最近 4 条历史消息通过 Sliding Window 保留，不被旧历史挤出窗口；
- 中间历史命中当前订单号时按 Context Relevance 召回，缓解 Lost in the Middle；
- RAG 片段按高相关性保留，Session Memory 仍只能辅助消歧；
- 旧低相关上下文压缩为公开摘要，不直接进入 `model_context` 候选；
- Workflow checkpoint 继续独立保存在服务端，不依赖压缩摘要恢复；
- 能力清单新增保护项、相关性、Sliding Window、Lost in the Middle 和历史摘要信号；
- FastAPI 应用与健康检查版本同步升级到 `0.28.0`。

## 0.27.0

- 新增多来源 Context Builder，统一组织用户消息、Runtime Context、Session Memory、Tool Observation、RAG citations 和 Workflow State；
- 新增 `ContextItem` 与 `ContextBuildReport` 契约，公开来源、信任级别、模型可见性、冲突组和选择理由；
- 上下文按 trusted、verified、session、external、untrusted 排序，避免不同可信度事实被扁平拼接；
- 页面 Runtime Context 与 Session Memory 冲突时采用页面线索，并继续通过业务工具重新校验；
- 用户自称会员身份不能覆盖 Runtime Context，历史退款批准说法不能覆盖 Workflow State；
- 系统身份和权限信号作为排除项保留审计说明，不暴露原始 Runtime 用户 ID；
- Workflow 上下文只暴露安全状态摘要，不复制恢复令牌、幂等键或冻结字段；
- 顶层新增 `context_report`，并同步写入 `session_state.context_builder`；
- 保持现有 Hybrid RAG、Tool Calling、Session Memory、LangGraph、HITL 和幂等恢复能力；
- FastAPI 应用与健康检查版本同步升级到 `0.27.0`。

## 0.26.0

- 新增结构化 `RuntimeContextView`，拆分模型可见低风险字段与后端系统校验信号；
- 可信 Runtime Context 提供登录状态、会员等级、账号风险、页面摘要和权限列表；
- 用户在聊天中自称 VIP 或其他用户身份时生成冲突提示，但不能覆盖可信登录态；
- 新增确定性会员咨询路由，会员回答只采用 Runtime Context 的系统等级；
- 订单工具执行后生成公开 `permission_decision`，明确是否完成可信身份绑定及业务放行；
- 公开上下文不返回原始用户 ID、完整订单列表、手机号或地址；
- Session Memory 继续只辅助消歧，不能证明身份、订单归属或替代 HITL 恢复校验；
- 保持现有 Hybrid RAG、Tool Calling、LangGraph、checkpoint 与幂等能力；
- FastAPI 应用与健康检查版本同步升级到 `0.26.0`。

## 0.25.0

- 新增进程内 Session Memory，记录最近订单、最近商品、最近意图和少量低风险偏好；
- 订单与商品只有在受控业务工具成功返回后才允许写入，未验证订单明确拒写；
- 支持在同一会话中将“刚才那个订单”解析为最近已验证订单，并重新执行实时业务查询；
- 记忆以会话和可信 Runtime 用户联合隔离，避免同名 session 在不同登录用户间串用；
- 顶层响应新增 `memory_update` 和 `memory_snapshot`，公开写入决策、Session TTL 与排除项；
- 手机号、地址、审批信息、系统提示词请求、未验证身份声明和聊天原文不写入记忆；
- Session Memory 不参与审批恢复凭证、checkpoint、冻结字段或幂等事实的覆盖；
- 保持短期、进程内边界，尚未实现长期画像、持久化、Context Builder 或上下文压缩；
- FastAPI 应用与健康检查版本同步升级到 `0.25.0`。

## 0.24.0

- 新增独立 `POST /chat/resume` 审批恢复接口，与普通聊天通道隔离；
- 待审批工作流保存进程内 checkpoint，并返回不可预测的恢复令牌、稳定幂等键和公开冻结字段；
- 恢复时联合校验 session、workflow、resume token、审批人标识及售后主管角色；
- 审批通过前通过原可信用户身份重新读取订单和物流，冻结事实发生漂移时阻断提交；
- 支持批准、拒绝和补充信息三类审批结果，普通聊天仍不能充当审批通道；
- 幂等记录售后申请，重复恢复返回同一申请编号且不重复记录；
- 售后申请仍为进程内模拟记录，明确 `external_business_write_executed=false`，不执行真实退款或退货；
- 新增 token 拦截、事实漂移、审批恢复和幂等重放专项回归；
- FastAPI 应用与健康检查版本同步升级到 `0.24.0`。

## 0.23.0

- 新增 HITL 待审批请求契约，退款或退货资格通过后创建 `ApprovalRequest(status=pending)`；
- 符合条件的售后工作流转为 `paused`，公开待处理动作 `require_human_approval` 与关联审批 ID；
- 审批请求声明售后主管角色和允许的决策选项，但当前不接收或执行审批决定；
- 新增普通聊天审批冒充防线，“主管同意”“审批通过”等自然语言不会被当成受信审批结果；
- 审批请求只保存公开安全的提交者标记，不将 Runtime 用户身份复制到审批对象；
- 保持退款、退货和其他业务写动作未执行，尚不提供 `/chat/resume`、checkpoint 或幂等恢复；
- 新增待审批状态、审批关联、聊天审批阻断和无业务读取专项回归；
- FastAPI 应用与健康检查版本同步升级到 `0.23.0`。

## 0.22.0

- 细化未发货退款工作流，仅对已支付、未出库且未发货订单返回可准备退款申请；
- 新增签收后退货工作流，独立核验签收状态、七天窗口、商品可退属性、退货原因与真实政策依据；
- 工作流对符合条件的退款与退货分别返回 `prepare_refund_application` 和 `prepare_return_application`；
- 扩展安全 Observation 白名单，提供履约状态、签收时间和商品可退属性，同时继续隔离原始业务载荷；
- 支持从可信 Runtime Context 的精确订单摘要补齐商品可退属性，且不覆盖业务 API 已返回事实；
- 增加七天无理由退货的高风险工作流路由，信息咨询和进度查询仍保持只读分流；
- 两条售后路径继续固定停在 `stop_before_submission`，不提交申请、不执行退款或退货、不完成人工审批；
- 新增退货成功资格、超出七天窗口、缺少原因以及退款待处理动作专项回归；
- FastAPI 应用与健康检查版本同步升级到 `0.22.0`。

## 0.21.0

- 新增 LangGraph `StateGraph` 售后工作流，将高风险路径从普通 Agent 编排中隔离；
- 固定执行售后类型识别、订单校验、物流读取、政策检索、资格判断和提交前停止节点；
- 新增公开 `WorkflowSummary`，返回工作流 ID、类型、状态、当前节点、待处理动作和节点历史；
- 复用 lesson26 的确定性只读工具、真实 Hybrid RAG 政策 citations 与结构化 `HighRiskAssessment`；
- 订单校验失败时通过条件边提前停止，缺订单号或动作类型不明时停在澄清边界；
- 工作流最终固定停在 `stop_before_submission`，不创建申请、不执行退款、不完成人工审批；
- 保持 `workflow_started=true` 与 `write_executed=false` 的可观察边界，尚不提供 checkpoint 或 `resume_token`；
- 新增 LangGraph 工作流顺序、条件分支、公开摘要与隐私隔离专项回归；
- FastAPI 应用与健康检查版本同步升级到 `0.21.0`。

## 0.20.0

- 新增高风险售后 Action Boundary，在 `RoutePlan.requires_workflow` 路径中区分退款、退货、取消和补偿动作；
- 新增结构化 `HighRiskAssessment`，公开申请资格、证据清单、政策依据、风险原因和被禁止的写动作；
- 有明确订单号时确定性执行订单状态与物流状态只读查询，不让模型选择或扩展高风险工具；
- 使用真实 Hybrid RAG 检索售后政策，只有可靠知识命中才生成 `policy_basis` citations；
- 缺少订单号时返回结构化澄清，业务或政策证据不完整时保持阻断，不猜测资格；
- 高风险 MCP 摘要可以展示实际使用的只读证据工具，同时继续绑定高风险 Resource 与转人工 Prompt；
- 所有退款、批准退款、取消订单和创建补偿写动作继续被禁止，`workflow_started=false` 且不宣称已完成 HITL；
- 保留 TaskPlanner、MCP-style Catalog、Hooks、Hybrid RAG、Tool + RAG、结构化澄清和安全降级能力；
- FastAPI 应用与健康检查版本同步升级到 `0.20.0`。

## 0.19.0

- 新增 `TaskPlanner`，在 RAG、Tool、Tool + RAG 和高风险受控路径之前生成统一 `RoutePlan`；
- 新增 `RoutePlan`、`ToolCandidate` 与公开安全 `PlannerTrace` 契约，顶层响应和会话状态同步返回；
- 高置信规则直接规划，低置信场景通过 OpenAI-compatible 轻量模型生成结构化路线草案；
- 规划模型只接收用户消息和公开工具候选，不接收 Runtime 身份值、业务事实或原始 ToolResult；
- 对模型输出的 intent、工具名、知识域、实体引用、上下文要求、风险和 fallback 策略执行允许列表约束；
- `RoutePlan.required_tools` 直接限制 LangChain Agent 本轮可见的工具集合，未知工具不会进入执行面；
- 高风险退款、取消和赔付请求由确定性安全规则最终覆盖为 Workflow 路由，不执行普通工具或写操作；
- 保留 MCP-style Catalog、Hooks、Hybrid RAG、商品 Tool + RAG、结构化澄清和安全降级能力；
- FastAPI 应用与健康检查版本同步升级到 `0.19.0`；
- 新增四类轻路径、低置信模型规划、候选约束、高风险覆盖和工具可见范围回归测试。

## 0.18.0

- 新增本地 `MCPCatalog`，统一组织五个只读工具定义、边界 Resource 和 Observation Prompt；
- `tools/contracts.py` 改为从 Catalog 转换生成 `ToolSpec`，现有规划、LangChain Tool Use 和 ToolRuntime 继续复用统一契约；
- 新增 `MCPToolDefinition`、`MCPResource`、`MCPPrompt` 与 `MCPBindingSummary` API 契约；
- 顶层响应和 `session_state.mcp` 返回工具来源、实际选中工具、可用工具及 Resource/Prompt URI；
- 工具前置、后置、异常和完成 Hook 增加 MCP Tool Use 语义，同时继续隐藏 Runtime 身份值、原始 ToolResult 与隐藏推理链；
- 高风险写请求只绑定高风险 Resource 和转人工 Prompt，不把退款、取消或赔付降级为普通工具调用；
- Catalog 不维护业务假数据，实时事实仍来自电商业务接口和可信 Runtime Context；
- 明确当前为本地 MCP-style 组织层，`remote_server_connected=false`，不冒充完整远程 MCP Server；
- 保留 Tool + RAG 联合回答、Hybrid RAG、结构化澄清、有限重试、Hooks 治理和安全降级能力；
- 新增 Catalog 单一工具来源、绑定解析、工具/非工具/高风险响应摘要回归测试。

## 0.17.0

- 新增请求级 `HookManager`，统一承载 `pre_tool_call`、`post_tool_call`、`on_error` 和 `on_completion` 生命周期；
- 将前置 Hook 接入真实 `ToolRuntime`，确保工具白名单、参数与只读边界信号发生在业务读取之前；
- 将后置 Hook 接入 Observation 出口，递归清理手机号、邮箱、凭证字段和外部指令污染；
- Hook 仅公开可信身份是否存在，不记录 `runtime_user_id` 具体值，也不暴露原始 ToolResult 或隐藏推理链；
- 工具错误、模型异常和检索异常统一生成安全的降级事件，保留稳定错误类别与尝试次数；
- 每条响应路由都生成一次 `hook_completion`，汇总工具、脱敏、污染、降级和高风险命中数量；
- Hooks 只负责横切治理，不执行退款、取消、赔付等写操作，也不冒充真实 HITL 审批；
- 保留商品 Tool + RAG 联合回答、Hybrid RAG、结构化澄清、有限重试和原始结果隔离能力；
- 新增 Hook 顺序、业务读取前置校验、结果脱敏、异常治理和公开摘要回归测试。

## 0.16.0

- 新增商品 Tool + RAG 联合路由，在同一响应中返回实时 `tool_calls` 与稳定知识 `citations`；
- 保留纯实时工具与纯稳定知识 RAG 路由，只有混合商品问题才同时调用两类来源；
- 商品 Observation 增加活动价、活动名称、活动摘要、会员等级和活动条件等业务接口白名单字段；
- 活动 ID、起止时间、商品长描述和未允许的 promotion 字段继续由压缩层剔除；
- 商品联合检索在可靠候选中平衡选择商品知识与活动规则，避免单类证据挤占 Top-K；
- 联合回答 Prompt 明确当前价格/库存只信工具、卖点/平台规则只信 RAG，规则不得冒充当前 SKU 优惠；
- 模型回答缺少 citation 标记或模型不可用时回退到包含安全 Observation 与真实引用的确定性回答；
- `session_state.tool_rag` 增加来源列表、事实边界、工具名、citation chunk 与联合完成状态；
- 保留错误分类、超时有限重试、高风险写操作拦截、结构化澄清与原始 ToolResult 隔离能力；
- 新增联合路由、三路分流、来源隔离、商品/活动平衡召回和活动字段压缩回归测试。

## 0.15.0

- 增加稳定 `ErrorCategory`，统一描述超时、参数错误、未找到、无权限、业务错误、模型不可用和系统错误；
- 为 `ToolResult`、Observation 与 Action/Observation 记录增加实际 `attempts`；
- 只读工具仅在业务超时时最多重试一次，其他错误立即进入确定性降级；
- 电商客户端单独识别连接超时，同时继续隐藏网络详情、响应体和服务凭证；
- 工具或模型失败时使用按错误类别映射的安全模板，不让模型猜测实时业务结果；
- 在 RAG 与 Tool Calling 前拦截直接退款、取消订单和赔付等高风险写请求；
- 顶层响应增加 `risk_level`、`needs_human_approval` 和 `degraded`，高风险写请求返回 `transfer_to_human`；
- 保留 Hybrid RAG、Reranker、结构化澄清、ToolResult/Observation 隔离与可信身份注入链路；
- 新增超时映射、有限重试、非超时不重试、写操作拦截、退款状态误判防护和模型不可用降级测试。

## 0.14.0

- 增加内部 `ToolResult` 契约，原始业务 payload 只保留在后端工具执行边界；
- 新增 Observation 压缩层，将订单、物流、商品、退款和候选订单结果按字段白名单转换；
- Observation 返回安全摘要、关键 `facts`、`omitted_fields` 和 `next_action`；
- 完整物流轨迹、运单号、用户身份、订单备注、商品长描述和退款内部字段不进入模型或公开响应；
- LangChain 工具只返回压缩后的 Observation，模型最终措辞仅在 Observation 可直接回答时采用；
- 顶层响应增加 `next_action`，区分回答用户、继续澄清和安全兜底；
- 保留结构化工具前/工具后澄清、真实业务查询以及稳定 Hybrid RAG/citations 链路；
- 新增 ToolResult 隔离、字段省略、模型可见上下文和 next_action 回归测试。

## 0.13.0

- 增加顶层 `clarification`、`ClarificationPlan`、结构化候选项和澄清阶段状态；
- 模型可生成澄清问题草案，但后端重新计算工具必填字段并忽略模型提交的业务参数；
- 从可信 Runtime Context 读取当前用户订单摘要，关联订单或唯一订单可安全补全；
- 缺少订单号时返回当前用户候选订单，不让模型替用户选择；
- 增加按月份筛选当前用户候选订单的只读工具，并标记上下文截断状态；
- 商品工具返回多个匹配时保留本轮 `tool_calls`，转为工具后结构化澄清；
- 保留 lesson18 的 LangChain Tool Calling、真实业务接口、Hybrid RAG 与 citations 链路；
- 新增模型参数注入防护、Runtime 身份隔离、工具前/工具后澄清和月份候选回归测试。

## 0.12.0

- 接入电商业务后端客户端，通过服务令牌和可信 Runtime 用户身份执行委托查询；
- 增加订单状态、订单物流、商品库存/价格和退款进度四个只读工具契约；
- 使用 LangChain `create_agent` 驱动模型生成结构化工具调用，后端严格校验工具名和参数；
- 禁止模型传入或覆盖用户身份，由后端在调用业务接口时注入 `runtime_user_id`；
- 返回经过脱敏的 `tool_calls` Action/Observation 记录，最终回答只使用受控 Observation；
- 将实时业务事实与稳定知识分流，工具路由不执行 RAG、不生成 citations；
- 增加业务接口错误映射、缺少业务编号兜底和依赖/模型不可用时的安全降级；
- 新增工具契约、身份委托、结果脱敏、真实退款端点和 Agent 实时路由离线回归测试。

## 0.11.0

- 根据规范化知识块生成 SHA256 指纹、稳定索引版本、chunk 快照和关键词倒排表；
- 拒绝重复 `chunk_id`，避免索引构建时静默覆盖知识块；
- 按知识索引版本与 Embedding 服务/模型身份缓存知识向量；
- 增加容量受限的进程内 LRU 检索缓存，只保存 Hybrid RAG 候选，不保存最终回答；
- 缓存键覆盖原始/改写查询、场景、允许领域、关键词项、知识版本和 Embedding 身份；
- 重建知识索引时同时清空向量与检索缓存，防止旧知识结果继续复用；
- 实时订单、物流、库存和退款进度不进入检索缓存、不生成 citations，也不调用回答模型；
- 在 `/health` 和 `session_state.rag` 暴露索引版本、指纹、缓存策略与命中状态；
- 新增索引版本、缓存命中、失效及实时业务边界测试，并保留既有 RAG 与隐私回归。

## 0.10.0

- 增加可观测的 pre-retrieval 计划，按问题与结构化意图识别知识场景；
- 使用知识 metadata 中的 `domain` 限制活动、售后、物流、商品、订单和投诉候选范围；
- 增加轻量精确关键词召回，覆盖赠品、配件、包装盒和压坏等长尾售后表述；
- 合并原始查询向量、改写查询向量和关键词三路候选，保留召回来源与命中词；
- 轻量 reranker 增加精确词与向量/关键词双路命中信号，商业 reranker 仍可选且可安全回退；
- 固定 RAG 质量集改为验证完整 Hybrid RAG 候选链路；
- 在 `session_state.rag` 暴露场景计划、三路命中 ID 以及向量/关键词/重排分数；
- 保留查询改写、citations、低置信兜底、成本观察和 Runtime Context 隐私边界。

## 0.9.0

- 保留用户原话，为检索单独生成可观测的 `rewritten_query`；
- 增加口语归一化及按活动、售后意图补充检索词的查询改写；
- 查询改写不读取可信 Runtime 身份字段，避免向外部模型披露用户上下文；
- 分别召回原始查询和改写查询的向量候选，并按最佳向量分合并去重；
- 增加透明轻量 reranker，对当前规则、活动领域和精确约束进行加权或降权；
- 支持显式启用 OpenAI-compatible 商业 reranker，失败时安全回退轻量重排；
- 在会话状态暴露改写原因、候选顺序、重排原因以及向量/重排/最终分数；
- 保留固定质量集、低置信兜底、citations、模型 usage 与成本观察。

## 0.8.0

- 将向量候选入场阈值和回答低置信阈值拆分，分别负责召回与可靠性判断；
- 增加固定 RAG 问题集，覆盖当前活动、售后规则和知识库外问题；
- 计算单条用例的 `recall@k`、`precision@k`、fallback 和通过状态；
- 在 `session_state.rag_quality` 返回轻量质量汇总；
- 低置信命中不再进入模型上下文，也不生成 citations；
- 增加补充信息或转人工的确定性兜底，并暴露最高分、置信等级和兜底动作；
- 保留结构化意图、模型 usage、成本观察和 Runtime Context 隐私边界。

## 0.7.0

- 将 Markdown 章节切分为带来源、章节、metadata 和稳定 ID 的知识块；
- 使用重叠窗口避免长章节切分时丢失相邻上下文；
- 增加 OpenAI-compatible Embedding 客户端、批量向量生成和文本缓存；
- 建立进程内向量索引，通过余弦相似度、阈值和 Top-K 完成真实向量检索；
- 继续过滤非当前有效知识，仅在明确查询历史时召回历史规则；
- 将真实检索命中转换为顶层 `citations`，返回来源、章节、知识块、分数和原文片段；
- 保留结构化意图识别、模型 usage、成本观察和 Runtime Context 隐私边界；
- 明确当前索引尚未持久化，实时订单、物流和售后数据仍需业务工具提供。

## 0.6.0

- 增加仓库内 Markdown 知识库及结构化文档解析；
- 建立“检索相关知识后再回答”的基础 RAG 链路；
- 使用关键词重合和结构化意图加权进行 Top-K 检索；
- 过滤非当前有效知识，只有明确查询历史时才允许召回历史片段；
- 暴露候选数量、命中 ID、分数和关键词等检索信号；
- 保留模型 usage、成本观察和 Runtime Context 隐私边界；
- 明确当前版本尚未实现 Embedding、向量检索和 citations。

## 0.5.0

- 增加 Prompt Registry，将提示词拆分为可启用、可排序、按意图选择的片段；
- 将选中片段及优先级暴露到会话状态；
- 标准化模型平台 token usage，并兼容常见字段名称；
- 平台未返回 usage 时使用本地估算兜底；
- 增加输入、输出、总 token 和人民币估算成本摘要；
- 记录会话级成本观察事件，同时避免向外部模型披露 Runtime Context 身份值。

## 0.4.0

- 增加集中式 Prompt 装配模块；
- 声明客服身份、事实优先级和高风险回答边界；
- 全量注入当前及历史规则文档，并暴露上下文长度统计；
- 检测新旧规则冲突线索，但不将其伪装成自动裁决；
- 保留结构化意图识别，并避免向外部模型披露 Runtime Context 身份值。

## 0.3.0

- 增加规则优先、轻量模型兜底的结构化意图识别；
- 返回稳定的 `intent` 与 `intent_result` API 字段；
- 根据结构化意图生成受业务边界约束的客服回复；
- 模型不可用时回退到确定性安全话术。

## 0.2.0

- 增加电商客服身份和业务回答边界；
- 将用户问题包装为受控的 system/user messages；
- 返回公开的 `reasoning_summary` 执行摘要；
- 显式声明活动、订单、物流、退款和业务工具尚未接入。

## 0.1.0

- 建立 FastAPI `/chat` 服务；
- 接入 OpenAI-compatible 聊天模型；
- 定义最小请求、响应和 Runtime Context 契约；
- 增加健康检查与能力声明接口。
