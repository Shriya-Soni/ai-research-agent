"""MCP server: web search tools with Tavily/DuckDuckGo providers."""

from __future__ import annotations

import json
import logging

from mcp.server.fastmcp import FastMCP

from rip.mcp.providers.web_search import fetch_page_content, search_web

logger = logging.getLogger(__name__)
mcp = FastMCP("web-search")


@mcp.tool()
async def search_web_tool(query: str, max_results: int = 10) -> str:
    """Search the public web. Uses Tavily if configured, else DuckDuckGo."""
    results = await search_web(query, max_results=max_results)
    return json.dumps(results, indent=2)


@mcp.tool()
async def fetch_page(url: str) -> str:
    """Fetch and extract readable text content from a web page."""
    result = await fetch_page_content(url)
    return json.dumps(result, indent=2)


@mcp.tool()
async def deep_search(query: str, max_results: int = 5) -> str:
    """Search web and fetch full content from top results."""
    results = await search_web(query, max_results=max_results)
    enriched = []
    for r in results[:3]:
        if r.get("url"):
            page = await fetch_page_content(r["url"])
            r["content"] = page.get("content", r.get("snippet", ""))
        enriched.append(r)
    return json.dumps(enriched, indent=2)


if __name__ == "__main__":
    mcp.run()
