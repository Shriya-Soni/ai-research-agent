"""Base agent logic shared by all specialist agents."""

from __future__ import annotations

import json
import logging
from typing import Any

import rip.agents._bootstrap  # noqa: F401 — set HF_HOME before model loads
from rip.config import get_settings
from rip.models.documents import RetrievedDocument, SourceType
from rip.retrieval.pipeline import RetrievalPipeline

logger = logging.getLogger(__name__)

_pipeline: RetrievalPipeline | None = None


def get_pipeline() -> RetrievalPipeline:
    """Build and warm up the retrieval pipeline once per agent process."""
    global _pipeline
    if _pipeline is None:
        logger.info("Initializing retrieval pipeline (first request may download models)...")
        _pipeline = build_pipeline()
        # Eager-load models so later requests don't block the event loop as long
        _pipeline._dense.model  # noqa: SLF001
        _pipeline._colbert.model  # noqa: SLF001
        logger.info("Retrieval pipeline ready")
    return _pipeline


def parse_mcp_json(raw: Any) -> list[dict[str, Any]]:
    """Safely parse JSON returned by an MCP tool."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError as e:
            logger.error("MCP returned invalid JSON: %s", e)
            return []
    return []


def build_pipeline() -> RetrievalPipeline:
    settings = get_settings()
    return RetrievalPipeline(
        rrf_k=settings.rrf_k,
        colbert_top_k=settings.colbert_top_k,
        embedding_model=settings.embedding_model,
        cross_encoder_model=settings.cross_encoder_model,
    )


def raw_to_documents(
    raw_items: list[dict[str, Any]],
    source_type: SourceType,
    id_prefix: str = "doc",
) -> list[RetrievedDocument]:
    """Convert raw MCP/API results into RetrievedDocument instances."""
    documents = []
    for i, item in enumerate(raw_items):
        documents.append(
            RetrievedDocument(
                id=item.get("id", f"{id_prefix}-{i}"),
                title=item.get("title", item.get("metric", f"Result {i}")),
                content=item.get("content", item.get("snippet", json.dumps(item))),
                source_type=source_type,
                source_uri=item.get("url", item.get("source_uri", "")),
                metadata={k: v for k, v in item.items() if k not in ("content", "snippet", "title")},
            )
        )
    return documents


async def run_retrieval(
    query: str,
    raw_documents: list[dict[str, Any]],
    source_type: SourceType,
) -> dict[str, Any]:
    """Shared retrieval flow: index corpus → RRF+ColBERT pipeline → return ranked docs."""
    pipeline = get_pipeline()
    docs = raw_to_documents(raw_documents, source_type)
    pipeline.index(docs)
    result = pipeline.retrieve(query)

    return {
        "query": query,
        "documents": [d.model_dump(mode="json") for d in result.documents],
        "pipeline_stages": result.pipeline_stages,
        "source_type": source_type.value,
    }
