# E-commerce Customer Service Agent

一个持续演进的电商客服 Agent 项目。仓库始终维护单一可运行版本，通过 Git 提交和版本标签记录从最小聊天服务到 RAG、Tool Calling、Workflow/HITL、Memory、Trace 和 Evaluation 的演进过程。

## v0.10.0

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
- 默认使用透明的轻量 reranker 重排，可选接入 OpenAI-compatible `/rerank` 服务；
- 商业 reranker 异常时回退轻量重排，并只公开安全的错误类型；
- 优先解析模型平台 `usage`，缺失时使用本地 token 估算；
- 返回输入、输出、总 token 与人民币估算成本；
- 记录会话级成本观察事件；
- 不向外部模型披露 Runtime Context 中的用户身份具体值；
- 可公开展示的 `reasoning_summary` 执行摘要；
- `/health` 与 `/capabilities`；
- 模型缺失或调用失败时的安全话术回退。

当前版本已经建立“稳定切片 → 查询改写 → pre-retrieval 场景规划 → 原始/改写双路向量召回 + 关键词召回 → 候选合并 → Reranker → 低置信判断 → Grounded Answer/Citations 或安全兜底”的 Hybrid RAG 链路，并用固定问题集观察基础质量。用户原话保持不变；Runtime Context 只保存在本地状态，不会被补写进 Embedding、关键词检索、Reranker 或回答模型请求。关键词检索是透明的轻量精确词实现，不是完整 BM25/搜索引擎；固定问题集仍是轻量检查，不是完整 Evaluation 平台。当前还没有版本化索引更新，知识库也不能替代订单、物流、库存和售后业务接口。

## 项目结构

```text
backend/
  agents/       # Agent 编排
  api/          # HTTP 路由与请求响应契约
  config/       # 环境变量与能力清单
  cost/         # Token usage 解析与估算成本
  knowledge/    # 活动、售后、商品、订单等 Markdown 知识原文
  embeddings/   # OpenAI-compatible Embedding 客户端与文本缓存
  models/       # OpenAI-compatible 分类和回答模型客户端
  rag/          # 文档切片、检索规划、混合召回、重排与质量检查
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

在 `.env` 中配置真实的 `AGENT_OPENAI_API_KEY`；`AGENT_EMBEDDING_MODEL` 可单独指定向量模型。默认使用轻量重排；如需商业 reranker，可将 `AGENT_RAG_RERANK_ENABLED` 设为 `1` 并配置对应地址、模型和可选独立 Key。不要把密钥提交到 Git。然后启动：

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

`session_state.rag` 会返回查询改写、检索场景、允许领域、关键词项、三路候选、重排顺序、向量/关键词/重排/最终分数、召回来源、reranker 模式和安全错误类型，并继续暴露低置信门槛及 citations。`session_state.rag_quality` 返回固定问题集数量、通过数量以及平均 `recall@k`、`precision@k`。当前索引是进程内实现，不是独立向量数据库。

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
