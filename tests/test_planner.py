"""Tests for query planner."""

import pytest

from rip.coordinator.planner import plan_research


@pytest.mark.asyncio
async def test_heuristic_plan_returns_all_agents():
    plan = await plan_research("How does ColBERT improve retrieval?")
    assert "web" in plan["active_agents"]
    assert "document" in plan["active_agents"]
    assert len(plan["sub_queries"]) >= 2


@pytest.mark.asyncio
async def test_plan_with_gaps_refines_queries():
    plan = await plan_research(
        "multi-agent systems",
        gaps=["No web sources — may lack recent information"],
        iteration=1,
    )
    assert "web" in plan["sub_queries"]
    assert "recent" in plan["sub_queries"]["web"].lower() or "latest" in plan["sub_queries"]["web"].lower()
