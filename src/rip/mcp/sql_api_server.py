"""MCP server: structured data tools (SQL + REST APIs)."""

from __future__ import annotations

import json
import logging

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)
mcp = FastMCP("sql-api")


@mcp.tool()
async def execute_sql(query: str, database: str = "research_db") -> str:
    """
    Execute a read-only SQL query against a structured database.

    In production, connect to PostgreSQL, BigQuery, or Snowflake.
    """
    mock_rows = [
        {"metric": "papers_published_2025", "value": 1247, "domain": "AI/ML"},
        {"metric": "citations_colbert", "value": 8934, "domain": "Information Retrieval"},
        {"metric": "a2a_adoptions", "value": 156, "domain": "Agent Systems"},
    ]
    return json.dumps(
        {"database": database, "query": query, "rows": mock_rows, "row_count": len(mock_rows)},
        indent=2,
    )


@mcp.tool()
async def call_api(
    endpoint: str,
    method: str = "GET",
    params: str = "{}",
) -> str:
    """Call a REST API endpoint and return structured JSON response."""
    parsed_params = json.loads(params) if params else {}
    return json.dumps(
        {
            "endpoint": endpoint,
            "method": method,
            "params": parsed_params,
            "response": {
                "status": "ok",
                "data": f"Mock API response for {endpoint}",
            },
        },
        indent=2,
    )


@mcp.tool()
async def list_data_sources() -> str:
    """List available structured data sources."""
    return json.dumps(
        {
            "sources": [
                {"name": "research_db", "type": "postgresql", "tables": ["papers", "citations", "authors"]},
                {"name": "metrics_api", "type": "rest", "base_url": "https://api.example.com/v1"},
            ]
        },
        indent=2,
    )


if __name__ == "__main__":
    mcp.run()
