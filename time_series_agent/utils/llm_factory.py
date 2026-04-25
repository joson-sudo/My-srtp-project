"""
LLM factory utilities.
Provide a single entry point to build chat models for different providers.
"""

import os
from typing import Any, Dict, Optional


def _resolve_provider(config: Optional[Dict[str, Any]]) -> str:
    if not config:
        return "anthropic"
    return str(config.get("llm_provider", "anthropic")).strip().lower()


def _resolve_model_name(model: Optional[str], config: Optional[Dict[str, Any]], provider: str) -> str:
    if model:
        return model
    if config and config.get("llm_model"):
        return str(config["llm_model"])

    defaults = {
        "openai": "gpt-4o",
        "anthropic": "claude-3-5-sonnet-20241022",
    }
    return defaults.get(provider, "claude-3-5-sonnet-20241022")


def _resolve_temperature(config: Optional[Dict[str, Any]], default_temperature: float) -> float:
    if not config:
        return default_temperature
    return float(config.get("llm_temperature", default_temperature))


def _resolve_max_tokens(config: Optional[Dict[str, Any]], default_max_tokens: Optional[int]) -> Optional[int]:
    if config and config.get("llm_max_tokens") is not None:
        return int(config["llm_max_tokens"])
    return default_max_tokens


def _resolve_api_base(config: Optional[Dict[str, Any]], provider: str) -> Optional[str]:
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_API_BASE") or (config or {}).get("anthropic_api_base")
    if provider == "openai":
        return os.getenv("OPENAI_API_BASE") or (config or {}).get("openai_api_base")
    return None


def build_llm(
    model: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    default_temperature: float = 0.1,
    default_max_tokens: Optional[int] = None,
):
    """
    Build a chat model instance based on project configuration.
    """
    provider = _resolve_provider(config)
    model_name = _resolve_model_name(model, config, provider)
    temperature = _resolve_temperature(config, default_temperature)
    max_tokens = _resolve_max_tokens(config, default_max_tokens)
    api_base = _resolve_api_base(config, provider)

    if provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY is required when llm_provider=anthropic")
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError(
                "langchain-anthropic is required for llm_provider=anthropic"
            ) from exc

        kwargs: Dict[str, Any] = {
            "model": model_name,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return ChatAnthropic(**kwargs)

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required when llm_provider=openai")
        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": model_name,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        if api_base:
            kwargs["base_url"] = api_base

        return ChatOpenAI(**kwargs)

    raise ValueError(f"Unsupported llm_provider: {provider}")
