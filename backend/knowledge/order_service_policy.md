---
title: 电商平台订单服务规则
domain: order
effective_status: active
owner: order_service_team
tags: [订单, 取消, 地址, 发货, 服务]
---

# 电商平台订单服务规则

## 订单状态解释
<!-- chunk_id: order-status-definition; keywords: 订单,状态,待支付,待发货,已发货,已完成,关闭 -->

待支付表示尚未完成支付，待发货表示支付成功但仓库尚未出库，已发货后的配送进度以物流系统为准。关闭后的订单不能恢复。

## 发货前取消订单
<!-- chunk_id: order-cancel-before-shipping; keywords: 取消订单,待发货,退款,支付,订单,发货前 -->

待支付或待发货订单可在订单页尝试取消；进入拣货或出库流程后取消按钮可能不可用，客服不能强行取消。

## 订单拆单说明
<!-- chunk_id: order-split-package; keywords: 拆单,包裹,订单,仓库,物流,发货,商品 -->

订单可能因仓库、库存或体积拆成多个包裹，每个包裹有独立物流信息。部分包裹先到不代表漏发。

## 赠品和主商品关系
<!-- chunk_id: order-gift-main-item; keywords: 赠品,主商品,退货,订单,活动,库存,售后 -->

赠品资格以下单时结算页为准，退回主商品时通常需要一并退回赠品。赠品无库存时不能承诺单独补发或现金替代。

## 大客户批量订单
<!-- chunk_id: order-bulk-purchase; keywords: 批量,大客户,订单,发票,库存,采购,人工 -->

大批量或高金额采购可能涉及库存锁定、合同、对公支付和专属报价，需要转人工大客户服务，普通客服不能承诺折扣和账期。
