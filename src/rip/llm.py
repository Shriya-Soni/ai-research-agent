"""Shared LLM utilities — supports Azure OpenAI and standard OpenAI."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from rip.config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _load_model_registry() -> tuple[dict[str, dict], str]:
    """Load LLM_MODELS from project-root config.py."""
    root = Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import config as model_config  # noqa: PLC0415

    return model_config.LLM_MODELS, model_config.DEFAULT_LLM_MODEL


def _get_active_model_config() -> dict[str, Any] | None:
    settings = get_settings()
    models, default = _load_model_registry()
    model_id = settings.llm_model or default
    return models.get(model_id)


def llm_available() -> bool:
    settings = get_settings()
    if settings.openai_api_key:
        return True
    if settings.azure_openai_api_key and _get_active_model_config():
        return True
    return False


def get_llm(temperature: float = 0.2, max_tokens: int | None = None):
    """Return a LangChain chat model for the configured provider."""
    settings = get_settings()
    model_cfg = _get_active_model_config()
    kwargs: dict[str, Any] = {"temperature": temperature}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    if model_cfg and model_cfg.get("provider") == "azure":
        from langchain_openai import AzureChatOpenAI

        api_key = settings.azure_openai_api_key
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY is not set")

        return AzureChatOpenAI(
            azure_endpoint=model_cfg["endpoint"].rstrip("/"),
            api_key=api_key,
            api_version=model_cfg["api_version"],
            azure_deployment=model_cfg["deployment"],
            **kwargs,
        )

    from langchain_openai import ChatOpenAI

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        **kwargs,
    )


async def invoke_llm(system: str, user: str, max_tokens: int | None = None) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_llm(temperature=0.2, max_tokens=max_tokens)
    response = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
    return str(response.content)


async def invoke_structured(system: str, user: str, schema: type[T]) -> T | None:
    """Parse LLM output as structured JSON matching a Pydantic schema."""
    if not llm_available():
        return None

    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    prompt = (
        f"{user}\n\nRespond with valid JSON matching this schema:\n{schema_json}\n"
        "Return ONLY the JSON object, no markdown fences."
    )
    try:
        raw = await invoke_llm(system, prompt)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(raw)
        return schema.model_validate(data)
    except Exception as e:
        logger.warning("Structured LLM parse failed: %s", e)
        return None
