"""Application configuration loaded from environment and YAML."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Keep model downloads inside the project for portability
_MODEL_CACHE = Path(__file__).resolve().parent.parent.parent / ".cache" / "models"
_MODEL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_MODEL_CACHE))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(_MODEL_CACHE))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Standard OpenAI (optional)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # Azure OpenAI — used when LLM_MODEL=gpt-5.4-nano (see project config.py)
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = "https://kwik-help.cognitiveservices.azure.com/"
    azure_openai_api_version: str = "2025-04-01-preview"
    azure_openai_deployment: str = "gpt-5.4-nano"
    llm_model: str = "gpt-5.4-nano"

    tavily_api_key: str = ""

    coordinator_host: str = "0.0.0.0"
    coordinator_port: int = 8000

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "research_corpus"
    use_llm_planner: bool = True
    use_llm_reflection: bool = True

    web_agent_url: str = "http://localhost:8001"
    doc_agent_url: str = "http://localhost:8002"
    structured_agent_url: str = "http://localhost:8003"
    synthesis_agent_url: str = "http://localhost:8004"

    rrf_k: int = 60
    colbert_top_k: int = 20
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    max_research_iterations: int = 3
    sufficiency_threshold: float = 0.7

    config_path: Path = Field(default=Path("config/settings.yaml"))

    def load_yaml(self) -> dict[str, Any]:
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f) or {}
        return {}


@lru_cache
def get_settings() -> Settings:
    return Settings()
