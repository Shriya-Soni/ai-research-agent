"""Structured data specialist agent — SQL and API queries via MCP."""

from __future__ import annotations

import json
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
    name="structured-data-agent",
    description="Queries structured data sources (SQL, REST APIs) via MCP tools",
    url="http://localhost:8003",
    skills=[
        AgentSkill(
            id="sql-query",
            name="SQL Query",
            description="Execute read-only SQL against structured databases",
            tags=["sql", "structured", "database"],
        ),
        AgentSkill(
            id="api-call",
            name="API Call",
            description="Fetch data from REST API endpoints",
            tags=["api", "rest", "structured"],
        ),
    ],
)


async def handle_query(query: str, metadata: dict) -> dict:
    """Route to SQL or API based on metadata hint."""
    mode = metadata.get("mode", "sql")
    if mode == "api":
        raw_json = await call_tool(
            tool_name="call_api",
            arguments={"endpoint": metadata.get("endpoint", "/metrics"), "params": "{}"},
            args=["-m", "rip.mcp.sql_api_server"],
        )
    else:
        raw_json = await call_tool(
            tool_name="execute_sql",
            arguments={"query": query, "database": metadata.get("database", "research_db")},
            args=["-m", "rip.mcp.sql_api_server"],
        )

    parsed_list = parse_mcp_json(raw_json)
    parsed = parsed_list[0] if parsed_list else {}
    raw_items = parsed.get("rows", [parsed] if parsed else [])
    return await run_retrieval(query, raw_items, SourceType.STRUCTURED)


async def _warmup() -> None:
    await asyncio.to_thread(get_pipeline)


def run_server(host: str = "0.0.0.0", port: int = 8003) -> None:
    app = create_a2a_app(AGENT_CARD, handle_query, on_startup=_warmup)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
