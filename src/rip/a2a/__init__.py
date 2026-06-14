"""A2A protocol — agent discovery and JSON-RPC communication."""

from rip.a2a.client import A2AClient
from rip.a2a.server import create_a2a_app

__all__ = ["A2AClient", "create_a2a_app"]
