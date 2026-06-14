"""LangGraph state machine — coordinator with streaming support."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from langgraph.graph import END, StateGraph

from rip.config import get_settings
from rip.coordinator.nodes import (
    reflection_node,
    retrieval_fanout_node,
    route_after_reflection,
    router_node,
    synthesis_node,
)
from rip.models.documents import ResearchReport
from rip.models.state import ResearchState

logger = logging.getLogger(__name__)

NODE_LABELS = {
    "router": "Query Planner",
    "retrieval_fanout": "Parallel Retrieval",
    "reflection": "Evidence Reflection",
    "synthesis": "Report Synthesis",
}


def _initial_state(query: str, max_iterations: int) -> ResearchState:
    return {
        "original_query": query,
        "current_query": query,
        "iteration": 0,
        "max_iterations": max_iterations,
        "documents": [],
        "agent_results": {},
        "gaps": [],
        "needs_more_research": False,
        "trace": [],
        "plan_reasoning": "",
        "focus_areas": [],
        "report": None,
        "error": None,
    }


def build_research_graph() -> StateGraph:
    graph = StateGraph(ResearchState)

    graph.add_node("router", router_node)
    graph.add_node("retrieval_fanout", retrieval_fanout_node)
    graph.add_node("reflection", reflection_node)
    graph.add_node("synthesis", synthesis_node)

    graph.set_entry_point("router")
    graph.add_edge("router", "retrieval_fanout")
    graph.add_edge("retrieval_fanout", "reflection")
    graph.add_conditional_edges(
        "reflection",
        route_after_reflection,
        {"router": "router", "synthesis": "synthesis"},
    )
    graph.add_edge("synthesis", END)

    return graph


def compile_graph():
    return build_research_graph().compile()


async def run_research(query: str, max_iterations: int | None = None) -> ResearchReport:
    settings = get_settings()
    iterations = max_iterations or settings.max_research_iterations
    app = compile_graph()

    logger.info("Starting research: %s (max_iterations=%d)", query, iterations)
    final_state: dict[str, Any] = await app.ainvoke(_initial_state(query, iterations))

    if final_state.get("error"):
        raise RuntimeError(f"Research failed: {final_state['error']}")

    report = final_state.get("report")
    if report is None:
        raise RuntimeError("Research completed without generating a report")

    if isinstance(report, ResearchReport):
        report.metadata["trace"] = final_state.get("trace", [])
        return report

    validated = ResearchReport.model_validate(report)
    validated.metadata["trace"] = final_state.get("trace", [])
    return validated


async def stream_research(
    query: str,
    max_iterations: int | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Stream node-level progress events for the web UI."""
    settings = get_settings()
    iterations = max_iterations or settings.max_research_iterations
    app = compile_graph()

    yield {
        "event": "start",
        "data": {"query": query, "max_iterations": iterations},
    }

    final_report = None
    final_error = None
    all_trace: list[dict] = []

    try:
        async for update in app.astream(
            _initial_state(query, iterations),
            stream_mode="updates",
        ):
            for node_name, node_output in update.items():
                if node_output.get("trace"):
                    all_trace.extend(node_output["trace"])

                event_data: dict[str, Any] = {
                    "node": node_name,
                    "label": NODE_LABELS.get(node_name, node_name),
                }

                if node_name == "router":
                    event_data.update({
                        "agents": node_output.get("active_agents", []),
                        "sub_queries": node_output.get("sub_queries", {}),
                        "reasoning": node_output.get("plan_reasoning", ""),
                    })
                elif node_name == "retrieval_fanout":
                    event_data.update({
                        "total_documents": len(node_output.get("documents", [])),
                        "agent_results": {
                            k: {"count": v.get("count", 0), "top_score": v.get("top_score", 0)}
                            for k, v in node_output.get("agent_results", {}).items()
                            if "error" not in v
                        },
                    })
                elif node_name == "reflection":
                    event_data.update({
                        "sufficiency_score": node_output.get("sufficiency_score", 0),
                        "needs_more": node_output.get("needs_more_research", False),
                        "gaps": node_output.get("gaps", []),
                        "notes": node_output.get("reflection_notes", ""),
                        "iteration": node_output.get("iteration", 0),
                    })
                elif node_name == "synthesis":
                    if node_output.get("error"):
                        final_error = node_output["error"]
                    report = node_output.get("report")
                    if report:
                        r = report if isinstance(report, dict) else report.model_dump()
                        event_data["confidence"] = r.get("confidence", 0)
                        event_data["citation_count"] = len(r.get("citations", []))
                        final_report = r

                yield {"event": "node", "data": event_data}

        if final_error:
            yield {"event": "error", "data": {"error": final_error}}
            return

        if final_report:
            final_report.setdefault("metadata", {})
            final_report["metadata"]["trace"] = all_trace
            yield {"event": "complete", "data": final_report}
        else:
            yield {"event": "error", "data": {"error": "Research completed without a report"}}

    except Exception as e:
        logger.exception("Stream research failed")
        yield {"event": "error", "data": {"error": str(e)}}
