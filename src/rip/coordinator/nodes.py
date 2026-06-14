"""LangGraph node implementations for the research coordinator."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from rip.a2a.client import A2AClient
from rip.config import get_settings
from rip.coordinator.planner import plan_research
from rip.coordinator.reflection import evaluate_sufficiency
from rip.models.documents import RetrievedDocument
from rip.models.state import ResearchState

logger = logging.getLogger(__name__)


def _trace_event(node: str, event: str, data: dict[str, Any] | None = None) -> list[dict]:
    return [{
        "timestamp": datetime.now(UTC).isoformat(),
        "node": node,
        "event": event,
        "data": data or {},
    }]


def _get_agent_clients() -> dict[str, A2AClient]:
    settings = get_settings()
    return {
        "web": A2AClient(settings.web_agent_url),
        "document": A2AClient(settings.doc_agent_url),
        "structured": A2AClient(settings.structured_agent_url),
    }


async def router_node(state: ResearchState) -> dict[str, Any]:
    """LLM query planner — decompose query into agent-specific sub-queries."""
    settings = get_settings()
    query = state.get("current_query", state.get("original_query", ""))
    iteration = state.get("iteration", 0)

    if settings.use_llm_planner or iteration > 0:
        plan = await plan_research(
            query,
            gaps=state.get("gaps"),
            iteration=iteration,
            prior_sub_queries=state.get("sub_queries"),
        )
    else:
        plan = await plan_research(query, iteration=iteration)

    logger.info("Router: agents=%s", plan["active_agents"])
    return {
        "active_agents": plan["active_agents"],
        "sub_queries": plan["sub_queries"],
        "plan_reasoning": plan.get("plan_reasoning", ""),
        "focus_areas": plan.get("focus_areas", []),
        "trace": _trace_event("router", "planned", {
            "agents": plan["active_agents"],
            "sub_queries": plan["sub_queries"],
            "reasoning": plan.get("plan_reasoning", "")[:200],
        }),
    }


async def _call_agent(agent_name: str, query: str) -> list[RetrievedDocument]:
    clients = _get_agent_clients()
    client = clients[agent_name]
    try:
        result = await client.execute(query)
        raw_docs = result.get("documents", [])
        return [RetrievedDocument.model_validate(d) for d in raw_docs]
    except Exception as e:
        logger.error("Agent %s failed: %s", agent_name, e)
        return []


async def retrieval_fanout_node(state: ResearchState) -> dict[str, Any]:
    """Parallel fan-out: call specialist agents concurrently via A2A."""
    sub_queries = state.get("sub_queries", {})
    active_agents = state.get("active_agents", ["web", "document", "structured"])

    tasks = {
        agent: _call_agent(agent, sub_queries.get(agent, state.get("current_query", "")))
        for agent in active_agents
    }

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    all_documents: list[RetrievedDocument] = []
    agent_results: dict[str, Any] = {}
    trace_events: list[dict] = []

    for agent, result in zip(tasks.keys(), results, strict=True):
        if isinstance(result, Exception):
            logger.error("Agent %s raised: %s", agent, result)
            agent_results[agent] = {"error": str(result), "documents": []}
            trace_events.extend(_trace_event("retrieval_fanout", "agent_error", {
                "agent": agent, "error": str(result),
            }))
        else:
            agent_results[agent] = {
                "documents": [d.model_dump(mode="json") for d in result],
                "count": len(result),
                "top_score": max((d.colbert_score or 0 for d in result), default=0),
            }
            all_documents.extend(result)
            trace_events.extend(_trace_event("retrieval_fanout", "agent_complete", {
                "agent": agent,
                "count": len(result),
                "top_colbert": agent_results[agent]["top_score"],
            }))

    logger.info("Fan-out: %d documents from %d agents", len(all_documents), len(active_agents))
    return {
        "documents": all_documents,
        "agent_results": agent_results,
        "trace": trace_events,
    }


async def reflection_node(state: ResearchState) -> dict[str, Any]:
    """Evaluate evidence sufficiency — reflection loop."""
    query = state.get("original_query", "")
    documents = state.get("documents", [])
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    evaluation = await evaluate_sufficiency(query, documents, iteration, max_iterations)

    logger.info(
        "Reflection: score=%.2f, needs_more=%s",
        evaluation["sufficiency_score"],
        evaluation["needs_more_research"],
    )

    update: dict[str, Any] = {
        "sufficiency_score": evaluation["sufficiency_score"],
        "gaps": evaluation["gaps"],
        "reflection_notes": evaluation["reflection_notes"],
        "needs_more_research": evaluation["needs_more_research"],
        "trace": _trace_event("reflection", "evaluated", {
            "score": evaluation["sufficiency_score"],
            "needs_more": evaluation["needs_more_research"],
            "gaps": evaluation["gaps"],
            "iteration": iteration,
        }),
    }

    if evaluation["needs_more_research"]:
        update["iteration"] = iteration + 1
        update["sub_queries"] = evaluation["sub_queries"]
        update["current_query"] = query

    return update


async def synthesis_node(state: ResearchState) -> dict[str, Any]:
    """Delegate to synthesis agent for citation-backed report."""
    settings = get_settings()
    client = A2AClient(settings.synthesis_agent_url)

    query = state.get("original_query", "")
    documents = state.get("documents", [])

    try:
        result = await client.execute(
            query,
            documents=[d.model_dump(mode="json") for d in documents],
            gaps=state.get("gaps", []),
            trace=state.get("trace", []),
            agent_results=state.get("agent_results", {}),
            sufficiency_score=state.get("sufficiency_score", 0),
        )
        from rip.models.documents import ResearchReport

        report = ResearchReport.model_validate(result)
        return {
            "report": report,
            "trace": _trace_event("synthesis", "complete", {
                "confidence": report.confidence,
                "citations": len(report.citations),
            }),
        }
    except Exception as e:
        logger.error("Synthesis failed: %s", e)
        return {"error": str(e), "trace": _trace_event("synthesis", "error", {"error": str(e)})}


def route_after_reflection(state: ResearchState) -> str:
    if state.get("needs_more_research", False):
        return "router"
    return "synthesis"
