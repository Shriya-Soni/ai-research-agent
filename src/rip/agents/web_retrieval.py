"""Web retrieval specialist agent — uses deep_search for full page content."""

from __future__ import annotations

import logging

import uvicorn

from rip.a2a.server import create_a2a_app
import asyncio

from rip.agents.base import get_pipeline, parse_mcp_json, run_retrieval
from rip.mcp.client import call_tool
from rip.models.documents import SourceType
from rip.models.messages import AgentCard, AgentSkill

logger = logging.getLogger(__name__)

AGENT_CARD = AgentCard(
    name="web-retrieval-agent",
    description="Retrieves and ranks web information via Tavily/DuckDuckGo + RRF+ColBERT",
    url="http://localhost:8001",
    skills=[
        AgentSkill(
            id="web-search",
            name="Web Search",
            description="Deep web search with page extraction and hybrid re-ranking",
            tags=["web", "retrieval", "search", "tavily"],
        )
    ],
)


async def handle_query(query: str, metadata: dict) -> dict:
    use_deep = metadata.get("deep_search", True)
    tool = "deep_search" if use_deep else "search_web_tool"
    raw_json = await call_tool(
        tool_name=tool,
        arguments={"query": query, "max_results": metadata.get("max_results", 8)},
        args=["-m", "rip.mcp.web_search_server"],
    )
    raw_items = parse_mcp_json(raw_json)

    for item in raw_items:
        if "content" not in item and "snippet" in item:
            item["content"] = item["snippet"]

    return await run_retrieval(query, raw_items, SourceType.WEB)


async def _warmup() -> None:
    await asyncio.to_thread(get_pipeline)


def run_server(host: str = "0.0.0.0", port: int = 8001) -> None:
    app = create_a2a_app(AGENT_CARD, handle_query, on_startup=_warmup)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
