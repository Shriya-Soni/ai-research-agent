"""Local document search specialist agent — A2A-deployable."""

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
    name="document-search-agent",
    description="Searches local document corpora via vector DB MCP + hybrid retrieval",
    url="http://localhost:8002",
    skills=[
        AgentSkill(
            id="doc-search",
            name="Document Search",
            description="Hybrid dense+sparse search over local document collections",
            tags=["documents", "rag", "local"],
        )
    ],
)


async def handle_query(query: str, metadata: dict) -> dict:
    collection = metadata.get("collection", "knowledge-base")
    raw_json = await call_tool(
        tool_name="search_documents",
        arguments={"query": query, "collection": collection, "top_k": 20},
        args=["-m", "rip.mcp.vector_db_server"],
    )
    raw_items = parse_mcp_json(raw_json)
    return await run_retrieval(query, raw_items, SourceType.LOCAL_DOC)


async def _warmup() -> None:
    await asyncio.to_thread(get_pipeline)


def run_server(host: str = "0.0.0.0", port: int = 8002) -> None:
    app = create_a2a_app(AGENT_CARD, handle_query, on_startup=_warmup)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
