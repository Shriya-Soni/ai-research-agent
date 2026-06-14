"""FastAPI coordinator API with web UI, streaming, and document ingestion."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from rip.a2a.client import A2AClient
from rip.config import get_settings
from rip.coordinator.graph import run_research, stream_research
from rip.models.documents import ResearchReport
from rip.retrieval.vector_store import get_vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Research Intelligence Platform",
    description="Multi-agent deep research with RRF+ColBERT, LangGraph, A2A, and MCP",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=3)
    max_iterations: int = Field(default=3, ge=1, le=5)


class DocumentIngestRequest(BaseModel):
    title: str
    content: str
    collection: str = "knowledge-base"
    source_uri: str = ""
    tags: list[str] = Field(default_factory=list)


class ResearchResponse(BaseModel):
    query: str
    report: ResearchReport
    metadata: dict[str, Any] = Field(default_factory=dict)


@app.get("/")
async def index():
    ui = STATIC_DIR / "index.html"
    if ui.exists():
        return FileResponse(ui, media_type="text/html")
    return {"message": "Research Intelligence Platform API", "docs": "/docs"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "coordinator", "version": "0.2.0"}


AGENT_LABELS = {
    "web": "Web Retrieval",
    "document": "Document Search",
    "structured": "Structured Data",
    "synthesis": "Synthesis",
}


async def _check_agent(name: str, url: str) -> dict[str, Any]:
    """Fast health check first, then agent card for skills."""
    import httpx

    base = {"url": url, "label": AGENT_LABELS.get(name, name)}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            health = await client.get(f"{url.rstrip('/')}/health")
            health.raise_for_status()
            try:
                card_resp = await client.get(f"{url.rstrip('/')}/.well-known/agent.json")
                card_resp.raise_for_status()
                card = card_resp.json()
                return {
                    **base,
                    "status": "online",
                    "name": card.get("name", name),
                    "skills": len(card.get("skills", [])),
                }
            except Exception:
                return {**base, "status": "online", "name": name, "skills": 0}
    except httpx.TimeoutException:
        return {**base, "status": "busy", "name": name, "skills": 0, "error": "Timed out — agent may be loading models"}
    except Exception as e:
        return {**base, "status": "offline", "name": name, "skills": 0, "error": str(e)}


@app.get("/api/agents/status")
async def agents_status() -> dict[str, Any]:
    """Check health of all A2A specialist agents."""
    import asyncio

    settings = get_settings()
    agents = {
        "web": settings.web_agent_url,
        "document": settings.doc_agent_url,
        "structured": settings.structured_agent_url,
        "synthesis": settings.synthesis_agent_url,
    }
    results = await asyncio.gather(
        *[_check_agent(name, url) for name, url in agents.items()],
        return_exceptions=True,
    )
    status = {}
    for (name, _url), result in zip(agents.items(), results, strict=True):
        if isinstance(result, Exception):
            status[name] = {"status": "offline", "name": name, "skills": 0, "error": str(result)}
        else:
            status[name] = result
    return {"agents": status}


@app.get("/architecture")
async def architecture() -> dict[str, Any]:
    settings = get_settings()
    return {
        "version": "0.2.0",
        "features": [
            "LLM query planner",
            "LLM reflection loop",
            "Tavily/DuckDuckGo web search",
            "Qdrant vector store",
            "SSE streaming",
            "Structured synthesis reports",
            "Web UI",
        ],
        "layers": [
            {"name": "Interface", "component": "FastAPI + Web UI"},
            {"name": "Coordinator", "component": "LangGraph", "features": ["planner", "fan-out", "reflection"]},
            {"name": "Agents (A2A)", "agents": [
                {"name": "web-retrieval", "url": settings.web_agent_url},
                {"name": "document-search", "url": settings.doc_agent_url},
                {"name": "structured-data", "url": settings.structured_agent_url},
                {"name": "synthesis", "url": settings.synthesis_agent_url},
            ]},
            {"name": "Tools (MCP)", "servers": ["web-search", "vector-db", "sql-api"]},
            {"name": "Data", "component": "Qdrant", "url": settings.qdrant_url},
        ],
        "retrieval_pipeline": ["dense", "sparse (BM25)", "RRF fusion", "ColBERT re-rank"],
    }


@app.post("/api/documents")
async def ingest_document(req: DocumentIngestRequest) -> dict[str, Any]:
    """Ingest a document into the Qdrant vector store."""
    store = get_vector_store()
    import uuid

    doc = {
        "id": f"doc-{uuid.uuid4().hex[:8]}",
        "title": req.title,
        "content": req.content,
        "collection": req.collection,
        "source_uri": req.source_uri or f"local://{req.collection}/{req.title}",
        "tags": req.tags,
    }
    count = store.upsert_documents([doc])
    return {"ingested": count, "document": doc}


@app.post("/api/documents/seed")
async def seed_corpus() -> dict[str, Any]:
    """Seed the vector store with default research corpus."""
    store = get_vector_store()
    count = store.seed()
    return {"seeded": count, "collections": store.list_collections()}


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest) -> ResearchResponse:
    try:
        report = await run_research(request.query, max_iterations=request.max_iterations)
        return ResearchResponse(
            query=request.query,
            report=report,
            metadata={"max_iterations": request.max_iterations},
        )
    except Exception as e:
        logger.exception("Research failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/research/stream")
async def research_stream(query: str, max_iterations: int = 3) -> EventSourceResponse:
    """SSE stream with per-node progress events."""

    async def event_generator():
        async for event in stream_research(query, max_iterations=max_iterations):
            yield {
                "event": event["event"],
                "data": json.dumps(event["data"]),
            }

    return EventSourceResponse(event_generator())


def run_coordinator() -> None:
    settings = get_settings()
    uvicorn.run(
        "rip.api.main:app",
        host=settings.coordinator_host,
        port=settings.coordinator_port,
        reload=False,
    )


if __name__ == "__main__":
    run_coordinator()
