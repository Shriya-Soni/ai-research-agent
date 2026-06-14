"""MCP server: Qdrant vector database tools."""

from __future__ import annotations

import json
import logging

from mcp.server.fastmcp import FastMCP

from rip.retrieval.vector_store import get_vector_store

logger = logging.getLogger(__name__)
mcp = FastMCP("vector-db")


@mcp.tool()
async def list_collections() -> str:
    """List available document collections."""
    store = get_vector_store()
    return json.dumps({"collections": store.list_collections()})


@mcp.tool()
async def search_documents(
    query: str,
    collection: str = "knowledge-base",
    top_k: int = 20,
) -> str:
    """Semantic search over document collections via Qdrant (or in-memory fallback)."""
    store = get_vector_store()
    results = store.search(query, collection=collection, top_k=top_k)
    return json.dumps(results, indent=2)


@mcp.tool()
async def get_document(document_id: str) -> str:
    """Retrieve a document by ID from the corpus."""
    store = get_vector_store()
    results = store.search(document_id, top_k=50)
    for doc in results:
        if doc.get("id") == document_id:
            return json.dumps(doc, indent=2)
    return json.dumps({"error": f"Document {document_id} not found"})


@mcp.tool()
async def ingest_document(
    title: str,
    content: str,
    collection: str = "knowledge-base",
    source_uri: str = "",
    tags: str = "[]",
) -> str:
    """Ingest a new document into the vector store."""
    import uuid

    store = get_vector_store()
    doc = {
        "id": f"doc-{uuid.uuid4().hex[:8]}",
        "title": title,
        "content": content,
        "collection": collection,
        "source_uri": source_uri or f"local://{collection}/{title}",
        "tags": json.loads(tags) if tags else [],
    }
    count = store.upsert_documents([doc])
    return json.dumps({"ingested": count, "document": doc}, indent=2)


if __name__ == "__main__":
    mcp.run()
