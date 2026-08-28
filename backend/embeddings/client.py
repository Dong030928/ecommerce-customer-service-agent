"""OpenAI-compatible embedding client with in-process text caching."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import httpx

from config.settings import (
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_MODEL,
    api_key_is_missing,
    load_project_env,
)


class EmbeddingClient:
    """Convert text into vectors without owning retrieval decisions."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.http_client = http_client
        self._cache: dict[str, list[float]] = {}

    @staticmethod
    def _cache_key(base_url: str, model: str, text: str) -> str:
        raw_key = json.dumps(
            [base_url, model, text],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def _settings(self) -> tuple[str, str, str]:
        load_project_env()
        api_key = (
            self.api_key
            if self.api_key is not None
            else os.getenv("AGENT_OPENAI_API_KEY")
        )
        if api_key_is_missing(api_key):
            raise RuntimeError("缺少有效的 AGENT_OPENAI_API_KEY，无法执行向量检索。")
        base_url = (
            self.base_url
            or os.getenv("AGENT_OPENAI_BASE_URL", DEFAULT_EMBEDDING_BASE_URL)
        ).rstrip("/")
        model = self.model or os.getenv(
            "AGENT_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        )
        return api_key, base_url, model

    def _post(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> httpx.Response:
        if self.http_client is not None:
            return self.http_client.post(url, headers=headers, json=payload)
        return httpx.post(url, headers=headers, json=payload, timeout=60)

    def embed(self, text: str) -> list[float]:
        """Embed one text and cache it by provider, model, and content."""

        api_key, base_url, model = self._settings()
        cache_key = self._cache_key(base_url, model, text)
        if cache_key in self._cache:
            return self._cache[cache_key]
        response = self._post(
            f"{base_url}/embeddings",
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            {"model": model, "input": text},
        )
        response.raise_for_status()
        vector = [float(value) for value in response.json()["data"][0]["embedding"]]
        if not vector:
            raise ValueError("Embedding 服务返回了空向量。")
        self._cache[cache_key] = vector
        return vector

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Batch missing texts and preserve the caller's original order."""

        if not texts:
            return []
        api_key, base_url, model = self._settings()
        missing_texts: list[str] = []
        missing_keys: list[str] = []
        seen_missing_keys: set[str] = set()
        for text in texts:
            cache_key = self._cache_key(base_url, model, text)
            if cache_key in self._cache or cache_key in seen_missing_keys:
                continue
            seen_missing_keys.add(cache_key)
            missing_texts.append(text)
            missing_keys.append(cache_key)

        if missing_texts:
            response = self._post(
                f"{base_url}/embeddings",
                {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                {"model": model, "input": missing_texts},
            )
            response.raise_for_status()
            data = sorted(
                response.json()["data"], key=lambda item: int(item.get("index", 0))
            )
            if len(data) != len(missing_texts):
                raise ValueError("Embedding 服务返回的向量数量与输入数量不一致。")
            for cache_key, item in zip(missing_keys, data):
                vector = [float(value) for value in item["embedding"]]
                if not vector:
                    raise ValueError("Embedding 服务返回了空向量。")
                self._cache[cache_key] = vector

        return [self._cache[self._cache_key(base_url, model, text)] for text in texts]


DEFAULT_EMBEDDING_CLIENT = EmbeddingClient()


def read_embedding_model_name() -> str:
    """Expose the configured embedding model for diagnostics."""

    load_project_env()
    return os.getenv("AGENT_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def embed_text(
    text: str,
    embedding_client: EmbeddingClient | None = None,
) -> list[float]:
    return (embedding_client or DEFAULT_EMBEDDING_CLIENT).embed(text)


def embed_texts(
    texts: list[str],
    embedding_client: EmbeddingClient | None = None,
) -> list[list[float]]:
    return (embedding_client or DEFAULT_EMBEDDING_CLIENT).embed_many(texts)
