# Changelog

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
