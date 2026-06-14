"""Tests for web search providers."""

import pytest

from rip.mcp.providers.web_search import search_mock, search_web


def test_mock_search_returns_results():
    results = search_mock("test query", max_results=3)
    assert len(results) == 3
    assert results[0]["title"]


@pytest.mark.asyncio
async def test_search_web_fallback():
    results = await search_web("Python programming", max_results=3)
    assert len(results) >= 1
    assert "title" in results[0]
