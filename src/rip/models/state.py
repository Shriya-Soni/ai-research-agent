"""LangGraph coordinator state schema."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from rip.models.documents import ResearchReport, RetrievedDocument


def merge_documents(
    existing: list[RetrievedDocument],
    new: list[RetrievedDocument],
) -> list[RetrievedDocument]:
    """Reducer: accumulate documents across parallel agent fan-out."""
    seen = {d.fingerprint for d in existing}
    merged = list(existing)
    for doc in new:
        if doc.fingerprint not in seen:
            seen.add(doc.fingerprint)
            merged.append(doc)
    return merged


def merge_events(existing: list[dict], new: list[dict]) -> list[dict]:
    return existing + new


class ResearchState(TypedDict, total=False):
    """State flowing through the LangGraph research coordinator."""

    # Input
    original_query: str
    current_query: str
    iteration: int
    max_iterations: int

    # Routing
    active_agents: list[str]
    sub_queries: dict[str, str]
    plan_reasoning: str
    focus_areas: list[str]

    # Evidence accumulation
    documents: Annotated[list[RetrievedDocument], merge_documents]
    agent_results: dict[str, Any]

    # Reflection
    sufficiency_score: float
    gaps: list[str]
    reflection_notes: str
    needs_more_research: bool

    # Observability
    trace: Annotated[list[dict], merge_events]

    # Output
    report: ResearchReport | None
    error: str | None
