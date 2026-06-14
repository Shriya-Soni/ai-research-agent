# Research Intelligence Platform

Multi-agent deep research system combining **LangGraph**, **A2A**, **MCP**, and **RRF + ColBERT** hybrid retrieval.

**v0.2** adds an LLM query planner, LLM reflection, Tavily/DuckDuckGo web search, Qdrant vector store, live SSE streaming, structured synthesis reports, and a web UI.

Given a research query, a coordinator agent spawns specialist sub-agents — one for web retrieval, one for local document search, one for structured data (SQL/APIs). Each agent re-ranks results with RRF fusion and ColBERT cross-encoding. A synthesis agent merges, deduplicates, and generates a citation-backed report.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface (FastAPI)                  │
│                    POST /research · SSE stream               │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│              Coordinator — LangGraph State Machine           │
│   router → fan-out → reflection ──┐                         │
│       ↑                           │ insufficient?          │
│       └───────────────────────────┘                         │
│                           │ sufficient                      │
│                           ▼                                 │
│                      synthesis → report                     │
└──────┬──────────────┬──────────────┬───────────────────────┘
       │ A2A          │ A2A          │ A2A
┌──────▼──────┐ ┌─────▼──────┐ ┌─────▼──────────┐
│ Web Agent   │ │ Doc Agent  │ │ Structured    │
│             │ │            │ │ Data Agent    │
└──────┬──────┘ └─────┬──────┘ └─────┬──────────┘
       │ MCP          │ MCP          │ MCP
┌──────▼──────┐ ┌─────▼──────┐ ┌─────▼──────────┐
│ web-search  │ │ vector-db  │ │ sql-api       │
│ MCP server  │ │ MCP server │ │ MCP server    │
└─────────────┘ └────────────┘ └───────────────┘
```

### Two-Protocol Stack

| Layer | Protocol | Purpose |
|-------|----------|---------|
| Coordinator ↔ Agents | **A2A** (JSON-RPC over HTTP) | Independently deployable specialist agents |
| Agents ↔ Tools | **MCP** (stdio/SSE) | External capabilities without hard-coding |

### Retrieval Pipeline (per agent)

1. **Dense** — embedding similarity via sentence-transformers (ANN-ready for FAISS/Qdrant)
2. **Sparse** — BM25 keyword matching (proper nouns, version numbers)
3. **RRF** — `score(d) = Σ 1/(k + rank)`, k=60 — no score normalization needed
4. **ColBERT** — cross-encoder re-rank on top-20 for token-level precision

### Reflection Loop

After merging results from all agents, the reflection node evaluates evidence sufficiency:
- Source diversity (web + docs + structured)
- ColBERT score quality
- Document quantity

If gaps are found (e.g., no recent web results, low ColBERT scores), the graph routes back to the router with revised sub-queries. This is what makes the system genuinely agentic rather than a static pipeline.

## Quick Start

### Prerequisites

- Python 3.11+
- Optional: `OPENAI_API_KEY` for LLM-powered synthesis

### Install

```bash
cd research-intelligence-platform
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

### Run (local — 5 terminals)

```bash
# Terminal 1: Coordinator
python -m rip.api.main

# Terminal 2-5: Specialist agents
python -m rip.agents.web_retrieval
python -m rip.agents.document_search
python -m rip.agents.structured_data
python -m rip.agents.synthesis
```

### Run (Docker — includes Qdrant)

```bash
docker compose up --build
```

### Web UI

Open [http://localhost:8000](http://localhost:8000) for the research console with live agent status, pipeline visualization, and SSE streaming.

### Research Query

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "How do A2A and MCP work together in multi-agent systems?"}'
```

### Stream Research (SSE)

```bash
curl -N "http://localhost:8000/research/stream?query=ColBERT+vs+BM25&max_iterations=2"
```

### Ingest Documents

```bash
curl -X POST http://localhost:8000/api/documents \
  -H "Content-Type: application/json" \
  -d '{"title": "My Paper", "content": "...", "collection": "research-papers"}'

curl -X POST http://localhost:8000/api/documents/seed
```

### Architecture Endpoint

```bash
curl http://localhost:8000/architecture
```

## Project Structure

```
src/rip/
├── api/              # User-facing FastAPI coordinator
├── coordinator/      # LangGraph state machine + reflection loop
├── agents/           # A2A-deployable specialist agents
├── a2a/              # A2A protocol (Agent Cards, JSON-RPC)
├── mcp/              # MCP tool servers (web, vector DB, SQL/API)
├── retrieval/        # Dense + sparse → RRF → ColBERT pipeline
└── models/           # Shared state, documents, messages
```

## Configuration

See `config/settings.yaml` and `.env.example`:

| Variable | Default | Description |
|----------|---------|-------------|
| `RRF_K` | 60 | RRF rank constant |
| `COLBERT_TOP_K` | 20 | Documents sent to cross-encoder |
| `MAX_RESEARCH_ITERATIONS` | 3 | Reflection loop limit |
| `SUFFICIENCY_THRESHOLD` | 0.7 | Evidence quality bar |

## Testing

```bash
pytest tests/ -v
```

## v0.2 Features

| Feature | Description |
|---------|-------------|
| **LLM Query Planner** | Decomposes queries into agent-specific sub-queries |
| **LLM Reflection** | Smarter gap analysis with targeted follow-up queries |
| **Tavily + DuckDuckGo** | Real web search with page extraction fallback chain |
| **Qdrant Vector Store** | Persistent document search with in-memory fallback |
| **SSE Streaming** | Per-node progress events for the web UI |
| **Structured Reports** | Executive summary, sections, methodology, limitations |
| **Document Ingestion** | `POST /api/documents` to add to the knowledge base |
| **Agent Status** | `GET /api/agents/status` for A2A health checks |

## Configuration

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | LLM planner, reflection, synthesis |
| `TAVILY_API_KEY` | Premium web search (optional) |
| `QDRANT_URL` | Vector store endpoint |
| `USE_LLM_PLANNER` | Enable LLM query decomposition |
| `USE_LLM_REFLECTION` | Enable LLM evidence evaluation |

## Extending

- **Full ColBERT**: `pip install -e ".[colbert]"` and swap `ColBERTReranker`
- **New agent**: Create an A2A agent with `create_a2a_app()`, register in `coordinator/nodes.py`
- **New MCP tool**: Add `@mcp.tool()` in a server under `src/rip/mcp/`

## Skills

The `skills/research/SKILL.md` file provides Cursor agent guidance for using this platform in research workflows.
