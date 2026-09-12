# 架构图绘制记录

使用内置 imagegen 绘图工具生成，并核对中文文字与五条执行路径的连接。最终图片为 architecture.png；以下保留原始生成提示词和连线修正提示词，便于后续维护。

## 生成提示词

```text
Use case: infographic-diagram
Asset type: GitHub README architecture image for an existing Chinese Python customer-service backend.
Primary request: Draw a polished, accurate Chinese software architecture diagram, landscape 2400x1600 approximate, with crisp highly legible Simplified Chinese text. Flat vector-like editorial diagram on off-white background, navy typography, muted blue/teal/purple lanes, amber only for human approval. Thin orthogonal arrows, rounded rectangular cards, generous spacing, no 3D, no robot art, no watermark. Chinese narrative labels only, retaining exact technology names where given. Do not use Mermaid.
Title: 电商智能客服 · 系统架构
Subtitle: 知识问答 · 实时查询 · 人工审批 · 质量闭环
Show this architecture with 3 areas:
TOP horizontal entry pipeline: 用户请求 → FastAPI 接口 → 安全扫描与上下文构建 → 意图识别与任务规划.
Within context card small caption: 可信身份 · 会话记忆 · 上下文压缩
Within planner card caption: 模型优先 · 风险规则兜底
CENTRAL large outlined container labeled 客服 Agent 编排. From the planner, fan out to FIVE equal-width side-by-side execution cards:
1 常规对话 ; caption 基础交互
2 知识检索 ; captions 三路召回 · RRF 融合 / 重排 · 引用 · 置信度
3 只读工具 ; captions 订单 · 物流 / 商品 · 库存 · 退款进度
4 工具与知识联合 ; captions 实时商品事实 / 选购规则与依据
5 售后工作流 ; captions LangGraph / 事实核验 · 资格校验. Within this fifth lane show a vertical progression to amber 人工审批 and then 模拟售后申请, connected with arrows. Under 人工审批 small label 检查点 · 恢复令牌 · 幂等. Clearly emphasize this lane footer 不执行真实退款.
All five execution paths converge into a horizontal output bar labeled 安全响应与结构化证据 with smaller caption 回答 · 引用 · 工具记录 · 工作流状态.
BOTTOM two horizontally aligned wide strips:
Strip A labeled 依赖与数据: three separated modules 模型服务（对话 · 向量 · 可选重排） | Markdown 知识库（稳定规则） | 电商业务 API（实时事实 · 只读）. These are supporting resources, NOT downstream processing steps; use no solid downward arrow from output bar into this strip.
Strip B labeled 横向治理: 安全防护与降级 | 脱敏执行轨迹 | 请求成本与预算 | 评测与反馈审核.
Small footer: 当前为单进程后端 · 会话与检查点保存在内存 · 反馈使用本地 JSON
Layout priority: exact Chinese text, clean routing with no crossing arrows, visibly distinct five execution lanes, enough space within fifth lane. Do not invent databases, frontend implementation, Redis, Kubernetes, external business write APIs or remote MCP servers. This is a conceptual architecture overview, not a deployment chart.
```

## 连线修正提示词

```text
Edit this architecture diagram with exactly ONE structural correction. The horizontal navy routing bus coming down from 意图识别与任务规划 currently starts above lane 2 知识检索, so lane 1 常规对话 is disconnected. Extend that same horizontal bus LEFTWARD at its existing y coordinate across to x=189 approximately (center of lane 1), then connect it down to the existing downward arrow above lane 1. All FIVE lanes must visibly connect to the same planner routing bus. Keep every other element, Chinese text, icons, colors, card positions, font sizes, canvas size and layout identical. Do not introduce any new labels or new arrows elsewhere. The bus should pass below the container title 客服 Agent 编排 without overlapping text; if needed shift only that title upward slightly within its whitespace. Keep readable Chinese unchanged.
```
