from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

API_KEY_NAMES = ("DEEPSEEK_API_KEY", "OPENAI_API_KEY")
BASE_URL_NAMES = ("DEEPSEEK_BASE_URL", "OPENAI_BASE_URL")
DEFAULT_BASE_URL = "https://api.deepseek.com"


def load_env_file(env_path: str | None) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not env_path:
        return env

    path = Path(env_path)
    if not path.exists():
        return env

    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip("\"").strip("'")

    return env


def resolve_api_key(env_path: str | None) -> str | None:
    for name in API_KEY_NAMES:
        value = os.getenv(name)
        if value:
            return value

    env = load_env_file(env_path)
    for name in API_KEY_NAMES:
        value = env.get(name)
        if value:
            return value

    return None


def resolve_base_url(env_path: str | None) -> str:
    for name in BASE_URL_NAMES:
        value = os.getenv(name)
        if value:
            return value

    env = load_env_file(env_path)
    for name in BASE_URL_NAMES:
        value = env.get(name)
        if value:
            return value

    return DEFAULT_BASE_URL
