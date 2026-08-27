"""Runtime configuration for capabilities and model connectivity."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
CAPABILITIES_PATH = BACKEND_DIR / "agent_capabilities.json"
PROMPT_REGISTRY_PATH = BACKEND_DIR / "prompt_registry.json"
DEFAULT_ENV_PATH = BACKEND_DIR.parent / ".env"
PLACEHOLDER_API_KEYS = {"", "你的模型平台 Key", "your-api-key", "YOUR_API_KEY"}
DEFAULT_INPUT_CNY_PER_1K = 0.001
DEFAULT_OUTPUT_CNY_PER_1K = 0.002


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
