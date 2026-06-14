"""LLM-powered query planner — decomposes research queries per agent."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from rip.llm import invoke_structured, llm_available

logger = logging.getLogger(__name__)


class ResearchPlan(BaseModel):
    reasoning: str = ""
    active_agents: list[str] = Field(default_factory=lambda: ["web", "document", "structured"])
    sub_queries: dict[str, str] = Field(default_factory=dict)
    focus_areas: list[str] = Field(default_factory=list)


PLANNER_SYSTEM = """You are a research query planner for a multi-agent system.
Given a research question, decompose it into specialized sub-queries for:
- web: public web search (recent news, papers, blogs, official docs)
- document: local knowledge base (internal docs, research papers, technical guides)
- structured: databases and APIs (statistics, metrics, tabular data)

Select only agents that are relevant. Return tailored sub-queries optimized for each source type.
For follow-up iterations, focus sub-queries on the identified gaps."""


async def plan_research(
    query: str,
    gaps: list[str] | None = None,
    iteration: int = 0,
    prior_sub_queries: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Plan agent routing and sub-queries. Falls back to heuristic plan without LLM."""
    if llm_available():
        gap_context = ""
        if gaps:
            gap_context = f"\nIdentified gaps from prior iteration: {'; '.join(gaps)}"
        if prior_sub_queries:
            gap_context += f"\nPrior sub-queries: {prior_sub_queries}"

        plan = await invoke_structured(
            PLANNER_SYSTEM,
            f"Research query: {query}\nIteration: {iteration + 1}{gap_context}",
            ResearchPlan,
        )
        if plan and plan.sub_queries:
            agents = [a for a in plan.active_agents if a in ("web", "document", "structured")]
            if not agents:
                agents = ["web", "document", "structured"]
            logger.info("LLM planner: %s → agents=%s", plan.reasoning[:80], agents)
            return {
                "active_agents": agents,
                "sub_queries": {a: plan.sub_queries.get(a, query) for a in agents},
                "plan_reasoning": plan.reasoning,
                "focus_areas": plan.focus_areas,
            }

    return _heuristic_plan(query, gaps)


def _heuristic_plan(query: str, gaps: list[str] | None = None) -> dict[str, Any]:
    sub_queries = {
        "web": f"recent developments and authoritative sources: {query}",
        "document": f"technical documentation and research papers: {query}",
        "structured": f"statistics, metrics, and quantitative data: {query}",
    }
    if gaps:
        for gap in gaps:
            lower = gap.lower()
            if "web" in lower or "recent" in lower:
                sub_queries["web"] = f"latest information addressing: {query} — {gap}"
            if "local" in lower or "document" in lower:
                sub_queries["document"] = f"internal knowledge about: {query} — {gap}"
            if "structured" in lower or "quantitative" in lower:
                sub_queries["structured"] = f"data and metrics for: {query} — {gap}"

    return {
        "active_agents": ["web", "document", "structured"],
        "sub_queries": sub_queries,
        "plan_reasoning": "Heuristic plan — set OPENAI_API_KEY for LLM decomposition",
        "focus_areas": [],
    }
