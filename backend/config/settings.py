"""Runtime configuration for capabilities and model connectivity."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
CAPABILITIES_PATH = BACKEND_DIR / "agent_capabilities.json"
KNOWLEDGE_DIR = BACKEND_DIR / "knowledge"
QUALITY_CASES_PATH = BACKEND_DIR / "rag_quality_cases.json"
DEFAULT_ENV_PATH = BACKEND_DIR.parent / ".env"
PLACEHOLDER_API_KEYS = {"", "你的模型平台 Key", "your-api-key", "YOUR_API_KEY"}
DEFAULT_INPUT_CNY_PER_1K = 0.001
DEFAULT_OUTPUT_CNY_PER_1K = 0.002
DEFAULT_EMBEDDING_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"
DEFAULT_ECOMMERCE_BASE_URL = "http://127.0.0.1:8081"
ECOMMERCE_TIMEOUT_SECONDS = 5.0
TOOL_CALLING_RECURSION_LIMIT = 4
CHUNK_SIZE = 420
CHUNK_OVERLAP = 80
CANDIDATE_K = 4
KEYWORD_CANDIDATE_K = 4
HYBRID_CANDIDATE_K = 6
RAG_RETRIEVAL_CACHE_MAX_ENTRIES = 256
FINAL_TOP_K = 2
RETRIEVAL_SCORE_THRESHOLD = 0.20
KEYWORD_SCORE_THRESHOLD = 0.15
# 重排后的最高分仍需通过可靠性门槛；更换模型或知识库后必须重新校准。
LOW_CONFIDENCE_THRESHOLD = 0.50
RERANK_MODEL_DEFAULT = "Qwen/Qwen3-Reranker-8B"
RERANK_INSTRUCTION_DEFAULT = (
    "Rank passages that directly support the e-commerce customer-service answer higher. "
    "Prefer current effective rules over expired notes and passages containing the exact "
    "promotion, refund, shipping, or product constraints needed by the user."
)
RERANK_TIMEOUT_SECONDS = 10.0
NORMALIZATION_RULES: list[tuple[str, str, str]] = [
    ("那个", "", "去掉缺少上下文的指代词“那个”"),
    ("这个", "", "去掉缺少上下文的指代词“这个”"),
    ("那款", "", "去掉缺少上下文的指代词“那款”"),
    ("这款", "", "去掉缺少上下文的指代词“这款”"),
    ("耳麦", "耳机", "把口语“耳麦”对齐为知识库常用词“耳机”"),
    ("叠券", "叠加 优惠券", "把口语“叠券”展开为“叠加 优惠券”"),
    ("会员券", "优惠券", "把“会员券”归一为知识库里的“优惠券”"),
    ("能叠吗", "能否 叠加 优惠券", "把省略问法展开为优惠叠加问题"),
    ("促销", "活动", "把“促销”归一为活动规则用词"),
    ("少一根", "少了 配件", "把配件缺失口语展开为售后检索词"),
    ("线少了", "配件 少了", "把充电线缺失口语展开为售后检索词"),
    ("盒子", "包装盒", "把“盒子”归一为售后规则中的“包装盒”"),
    ("压了", "压坏", "把包装受压口语归一为“压坏”"),
]


def load_agent_capabilities() -> dict[str, Any]:
    """Read the public capability manifest used by clients and debug tools."""

    with CAPABILITIES_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def load_project_env() -> Path | None:
    """Load repository-local model configuration without overriding process variables."""

    env_path = Path(os.getenv("AGENT_ENV_FILE", str(DEFAULT_ENV_PATH))).expanduser()
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)
    return env_path


def api_key_is_missing(api_key: str | None) -> bool:
    """Return whether the model key is missing or still a placeholder."""

    return api_key is None or api_key.strip() in PLACEHOLDER_API_KEYS


def env_flag_enabled(name: str, default: bool = False) -> bool:
    """Read a conventional boolean environment flag."""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}
