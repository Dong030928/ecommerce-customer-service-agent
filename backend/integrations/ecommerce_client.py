"""Read-only client for realtime facts from the e-commerce backend."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

import httpx

from config.settings import (
    DEFAULT_ECOMMERCE_BASE_URL,
    ECOMMERCE_TIMEOUT_SECONDS,
    load_project_env,
)


class EcommerceClientError(RuntimeError):
    """Safe integration error that never contains credentials or response bodies."""

    def __init__(
        self,
        code: str,
        safe_message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message
        self.status_code = status_code


class EcommerceClient:
    """Call only the business backend's read endpoints."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        service_token: str | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url
        self._service_token = service_token
        self._http_client = http_client

    def _settings(self) -> tuple[str, str | None]:
        load_project_env()
        base_url = (
            self._base_url
            or os.getenv("AGENT_ECOMMERCE_BASE_URL")
            or os.getenv("ECOMMERCE_BASE_URL")
            or DEFAULT_ECOMMERCE_BASE_URL
        ).rstrip("/")
        token = self._service_token
        if token is None:
            token = os.getenv("AGENT_ECOMMERCE_SERVICE_TOKEN")
        normalized_token = str(token or "").strip()
        if normalized_token in {"", "replace-me", "your-service-token"}:
            normalized_token = None
        return base_url, normalized_token

    def _delegated_headers(self, runtime_user_id: str) -> dict[str, str]:
        _base_url, service_token = self._settings()
        user_id = runtime_user_id.strip()
        if not user_id:
            raise EcommerceClientError(
                "runtime_identity_missing",
                "缺少可信 runtime_user_id，不能查询用户业务事实。",
            )
        if service_token is None:
            raise EcommerceClientError(
                "service_auth_missing",
                "电商后端服务凭证未配置，暂时不能查询实时业务事实。",
            )
        return {
            "X-Agent-Service-Token": service_token,
            "X-Agent-User-Id": user_id,
        }

    def _get(
        self,
        path: str,
        *,
        runtime_user_id: str | None = None,
        params: dict[str, str] | None = None,
    ) -> Any:
        base_url, _service_token = self._settings()
        headers = (
            self._delegated_headers(runtime_user_id)
            if runtime_user_id is not None
            else {}
        )
        try:
            if self._http_client is not None:
                response = self._http_client.get(
                    f"{base_url}{path}",
                    headers=headers,
                    params=params,
                )
            else:
                response = httpx.get(
                    f"{base_url}{path}",
                    headers=headers,
                    params=params,
                    timeout=ECOMMERCE_TIMEOUT_SECONDS,
                )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            if status_code in {401, 403}:
                code = "business_access_denied"
                message = "业务系统拒绝了本次查询，请确认订单归属和服务身份。"
            elif status_code == 404:
                code = "business_fact_not_found"
                message = "业务系统没有查到对应记录。"
            elif status_code >= 500:
                code = "business_service_unavailable"
                message = "业务系统暂时不可用，请稍后重试。"
            else:
                code = "business_request_failed"
                message = "业务事实查询失败，请核对参数后重试。"
            raise EcommerceClientError(
                code,
                message,
                status_code=status_code,
            ) from exc
        except httpx.RequestError as exc:
            raise EcommerceClientError(
                "business_service_unavailable",
                "暂时无法连接电商业务系统，请稍后重试。",
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise EcommerceClientError(
                "business_response_invalid",
                "业务系统返回了无法解析的响应。",
            ) from exc
        if not isinstance(payload, dict) or payload.get("success") is not True:
            raise EcommerceClientError(
                "business_response_invalid",
                "业务系统没有返回可信的成功结果。",
            )
        return payload.get("data")

    def get_order(self, order_id: str, runtime_user_id: str) -> dict[str, Any]:
        data = self._get(
            f"/api/orders/{quote(order_id, safe='')}",
            runtime_user_id=runtime_user_id,
        )
        if not isinstance(data, dict):
            raise EcommerceClientError(
                "business_response_invalid",
                "订单接口没有返回有效订单。",
            )
        return data

    def get_logistics(self, order_id: str, runtime_user_id: str) -> dict[str, Any]:
        data = self._get(
            f"/api/orders/{quote(order_id, safe='')}/logistics",
            runtime_user_id=runtime_user_id,
        )
        if not isinstance(data, dict):
            raise EcommerceClientError(
                "business_response_invalid",
                "物流接口没有返回有效物流记录。",
            )
        return data

    def list_products(self, query: str) -> list[dict[str, Any]]:
        data = self._get("/api/products", params={"keyword": query})
        if not isinstance(data, list):
            raise EcommerceClientError(
                "business_response_invalid",
                "商品接口没有返回有效商品列表。",
            )
        return [item for item in data if isinstance(item, dict)]

    def get_refund_status(
        self,
        refund_request_id: str,
        runtime_user_id: str,
    ) -> dict[str, Any]:
        data = self._get(
            f"/api/refund/requests/{quote(refund_request_id, safe='')}",
            runtime_user_id=runtime_user_id,
        )
        if not isinstance(data, dict):
            raise EcommerceClientError(
                "business_response_invalid",
                "退款接口没有返回有效退款申请。",
            )
        return data
