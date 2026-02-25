"""Centralised configuration – loaded from config.yaml + env vars.

Environment variables override config.yaml values.  Key mapping:
    config.yaml key   →  env var
    embedding_model   →  EMBEDDING_MODEL
    default_k         →  DEFAULT_K
    index_path        →  INDEX_PATH
    courses_path      →  COURSES_PATH
    log_path          →  LOG_PATH
    search_mode       →  SEARCH_MODE
    port              →  PORT
    ...
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env")

_CONFIG_PATH = _ROOT / "config.yaml"


def _load_yaml() -> Dict[str, Any]:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _get(key: str, default: Any = None, cast: type = str) -> Any:
    """Return env-var override or config.yaml value, cast to *cast*."""
    env_val = os.environ.get(key.upper())
    if env_val is not None:
        return cast(env_val)
    return _yaml.get(key, default)


_yaml = _load_yaml()

# ── Exported settings ────────────────────────────────────────────────

EMBEDDING_MODEL: str = _get("embedding_model", "BAAI/bge-base-en-v1.5")
DEFAULT_K: int = _get("default_k", 5, int)
INDEX_PATH: str = str(_ROOT / _get("index_path", "data/faiss.index"))
METADATA_PATH: str = str(_ROOT / _get("metadata_path", "data/metadata.pkl"))
COURSES_PATH: str = str(_ROOT / _get("courses_path", "data/courses.json"))
LOG_PATH: str = str(_ROOT / _get("log_path", "logs/queries.log"))
SEARCH_MODE: str = _get("search_mode", "hybrid")
FUSION_STRATEGY: str = _get("fusion_strategy", "weighted")
FUSION_ALPHA: float = _get("fusion_alpha", 0.7, float)
FUSION_BETA: float = _get("fusion_beta", 0.3, float)
HOST: str = _get("host", "127.0.0.1")
PORT: int = _get("port", 8000, int)
