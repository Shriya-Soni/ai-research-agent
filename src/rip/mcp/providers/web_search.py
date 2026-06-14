"""Web search providers: Tavily → DuckDuckGo → mock fallback."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from rip.config import get_settings

logger = logging.getLogger(__name__)


async def search_tavily(query: str, max_results: int = 10) -> list[dict[str, Any]] | None:
    settings = get_settings()
    if not settings.tavily_api_key:
        return None

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": settings.tavily_api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": False,
                "search_depth": "advanced",
            },
        )
        resp.raise_for_status()
        data = resp.json()

    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", r.get("snippet", "")),
            "published": r.get("published_date", ""),
            "score": r.get("score", 0.0),
            "provider": "tavily",
        }
        for r in data.get("results", [])
    ]


async def search_duckduckgo(query: str, max_results: int = 10) -> list[dict[str, Any]] | None:
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("href", r.get("link", "")),
                "snippet": r.get("body", r.get("snippet", "")),
                "published": "",
                "provider": "duckduckgo",
            }
            for r in results
        ]
    except Exception as e:
        logger.warning("DuckDuckGo search failed: %s", e)
        return None


def search_mock(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    return [
        {
            "title": f"Research finding on: {query}",
            "url": f"https://example.com/research/{i}",
            "snippet": (
                f"Analysis of {query} covering developments, methodologies, "
                f"and findings from authoritative sources (result {i})."
            ),
            "published": "2025-06-01",
            "provider": "mock",
        }
        for i in range(1, min(max_results, 5) + 1)
    ]


async def search_web(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    """Cascade: Tavily → DuckDuckGo → mock."""
    results = await search_tavily(query, max_results)
    if results:
        logger.info("Web search via Tavily: %d results", len(results))
        return results

    results = await search_duckduckgo(query, max_results)
    if results:
        logger.info("Web search via DuckDuckGo: %d results", len(results))
        return results

    logger.info("Web search falling back to mock results")
    return search_mock(query, max_results)


async def fetch_page_content(url: str) -> dict[str, Any]:
    """Fetch and extract readable text from a URL."""
    try:
        async with httpx.AsyncClient(
            timeout=20.0,
            follow_redirects=True,
            headers={"User-Agent": "ResearchIntelligencePlatform/0.1"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text

        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            title = soup.title.string if soup.title else url
            paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
            content = "\n\n".join(paragraphs[:50])[:8000]
        except ImportError:
            title = url
            content = html[:8000]

        return {"url": url, "title": title or url, "content": content or html[:4000]}
    except Exception as e:
        logger.warning("Page fetch failed for %s: %s", url, e)
        return {"url": url, "title": url, "content": f"Failed to fetch: {e}", "error": str(e)}
