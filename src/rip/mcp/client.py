"""MCP client helper for agents to call tool servers."""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


def mcp_python() -> str:
    """Use the same Python interpreter as the running agent (venv-safe)."""
    return sys.executable


@asynccontextmanager
async def mcp_session(command: str | None = None, args: list[str] | None = None):
    """Connect to an MCP server via stdio transport."""
    server_params = StdioServerParameters(command=command or mcp_python(), args=args or [])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def call_tool(
    tool_name: str,
    arguments: dict[str, Any],
    args: list[str] | None = None,
    command: str | None = None,
) -> Any:
    """Call a single MCP tool and return the result."""
    async with mcp_session(command, args) as session:
        result = await session.call_tool(tool_name, arguments)
        if result.content:
            return result.content[0].text if hasattr(result.content[0], "text") else result.content
        return None
