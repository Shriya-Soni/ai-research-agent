"""Reflection — evaluates evidence sufficiency with LLM + heuristic fallback."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from rip.config import get_settings
from rip.llm import invoke_structured, llm_available
from rip.models.documents import RetrievedDocument, SourceType

logger = logging.getLogger(__name__)


class ReflectionResult(BaseModel):
    sufficiency_score: float = Field(ge=0.0, le=1.0)
    needs_more_research: bool
    gaps: list[str] = Field(default_factory=list)
    sub_queries: dict[str, str] = Field(default_factory=dict)
    reflection_notes: str = ""
    missing_perspectives: list[str] = Field(default_factory=list)


REFLECTION_SYSTEM = """You are a research quality evaluator. Assess whether gathered evidence
is sufficient to answer the original query. Consider source diversity, relevance scores,
recency, and depth. If insufficient, provide targeted sub-queries per agent (web, document, structured).
Score 0.0-1.0 where 0.7+ is typically sufficient for synthesis."""


async def evaluate_sufficiency(
    query: str,
    documents: list[RetrievedDocument],
    iteration: int,
    max_iterations: int,
) -> dict[str, Any]:
    """Evaluate evidence — LLM when available, heuristic fallback otherwise."""
    settings = get_settings()

    if settings.use_llm_reflection and llm_available() and documents:
        llm_result = await _evaluate_with_llm(query, documents, iteration, max_iterations)
        if llm_result:
            return llm_result

    return _evaluate_heuristic(query, documents, iteration, max_iterations)


async def _evaluate_with_llm(
    query: str,
    documents: list[RetrievedDocument],
    iteration: int,
    max_iterations: int,
) -> dict[str, Any] | None:
    evidence_summary = []
    for i, doc in enumerate(documents[:15], 1):
        evidence_summary.append(
            f"[{i}] ({doc.source_type.value}) {doc.title} "
            f"colbert={doc.colbert_score or 0:.2f} — {doc.content[:200]}"
        )

    result = await invoke_structured(
        REFLECTION_SYSTEM,
        (
            f"Query: {query}\nIteration: {iteration + 1}/{max_iterations}\n"
            f"Documents: {len(documents)}\n\nEvidence:\n" + "\n".join(evidence_summary)
        ),
        ReflectionResult,
    )
    if not result:
        return None

    needs_more = result.needs_more_research and iteration < max_iterations
    return {
        "sufficiency_score": result.sufficiency_score,
        "gaps": result.gaps,
        "reflection_notes": result.reflection_notes,
        "needs_more_research": needs_more,
        "sub_queries": result.sub_queries if needs_more else {},
        "missing_perspectives": result.missing_perspectives,
    }


def _evaluate_heuristic(
    query: str,
    documents: list[RetrievedDocument],
    iteration: int,
    max_iterations: int,
) -> dict[str, Any]:
    settings = get_settings()
    threshold = settings.sufficiency_threshold

    if not documents:
        return {
            "sufficiency_score": 0.0,
            "gaps": ["No evidence retrieved from any agent"],
            "reflection_notes": "All retrieval agents returned empty results.",
            "needs_more_research": iteration < max_iterations,
            "sub_queries": _generate_gap_queries(query),
            "missing_perspectives": ["all"],
        }

    source_types = {d.source_type for d in documents}
    avg_colbert = sum(d.colbert_score or 0 for d in documents) / len(documents)
    max_colbert = max(d.colbert_score or 0 for d in documents)

    gaps: list[str] = []
    sub_queries: dict[str, str] = {}

    if SourceType.WEB not in source_types:
        gaps.append("No web sources — may lack recent information")
        sub_queries["web"] = f"latest news and recent developments: {query}"

    if SourceType.LOCAL_DOC not in source_types:
        gaps.append("No local documents matched — internal knowledge gap")
        sub_queries["document"] = f"internal documentation and papers about: {query}"

    if SourceType.STRUCTURED not in source_types:
        gaps.append("No structured data — quantitative evidence missing")
        sub_queries["structured"] = f"statistics and metrics related to: {query}"

    if max_colbert < 0.3:
        gaps.append(f"Low ColBERT scores (max={max_colbert:.2f}) — weak semantic match")
        sub_queries.setdefault("web", f"more specific search: {query}")

    if len(documents) < 3:
        gaps.append(f"Only {len(documents)} documents — thin evidence base")

    diversity_score = len(source_types) / 3.0
    quality_score = min(1.0, avg_colbert)
    quantity_score = min(1.0, len(documents) / 5.0)
    sufficiency_score = diversity_score * 0.3 + quality_score * 0.4 + quantity_score * 0.3

    needs_more = sufficiency_score < threshold and iteration < max_iterations

    if not needs_more and gaps:
        reflection_notes = (
            f"Evidence sufficient (score={sufficiency_score:.2f}) despite gaps: "
            f"{'; '.join(gaps)}. Proceeding to synthesis."
        )
    elif needs_more:
        reflection_notes = (
            f"Evidence insufficient (score={sufficiency_score:.2f}). "
            f"Gaps: {'; '.join(gaps)}. Routing back for iteration {iteration + 1}."
        )
    else:
        reflection_notes = f"Evidence sufficient (score={sufficiency_score:.2f}). Ready for synthesis."

    return {
        "sufficiency_score": sufficiency_score,
        "gaps": gaps,
        "reflection_notes": reflection_notes,
        "needs_more_research": needs_more,
        "sub_queries": sub_queries if needs_more else {},
        "missing_perspectives": [],
    }


def _generate_gap_queries(query: str) -> dict[str, str]:
    return {
        "web": f"web search: {query}",
        "document": f"document search: {query}",
        "structured": f"structured data: {query}",
    }
