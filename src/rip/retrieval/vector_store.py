"""Qdrant vector store with in-memory fallback."""

from __future__ import annotations

import logging
import uuid
from typing import Any

import numpy as np

from rip.config import get_settings

logger = logging.getLogger(__name__)

SEED_CORPUS: list[dict[str, Any]] = [
    {
        "id": "doc-001",
        "title": "Hybrid Retrieval with RRF and ColBERT",
        "content": (
            "Reciprocal Rank Fusion combines dense and sparse retrieval without "
            "score normalization. The formula score(d) = sum(1/(k+rank)) rewards "
            "documents appearing high in either dense or sparse lists. ColBERT "
            "provides token-level cross-encoding for precision re-ranking."
        ),
        "source_uri": "local://papers/hybrid-retrieval.pdf",
        "collection": "research-papers",
        "tags": ["retrieval", "rrf", "colbert"],
    },
    {
        "id": "doc-002",
        "title": "LangGraph Multi-Agent Orchestration",
        "content": (
            "LangGraph state machines support conditional routing, parallel fan-out, "
            "and reflection loops. The coordinator spawns specialist agents via A2A "
            "and evaluates evidence sufficiency before synthesis."
        ),
        "source_uri": "local://docs/langgraph-agents.md",
        "collection": "internal-docs",
        "tags": ["langgraph", "agents", "orchestration"],
    },
    {
        "id": "doc-003",
        "title": "A2A and MCP Protocol Stack",
        "content": (
            "A2A enables agent-to-agent communication via JSON-RPC and Agent Cards "
            "at /.well-known/agent.json. MCP exposes tools to agents without "
            "hard-coding implementations. Together they form a two-protocol stack."
        ),
        "source_uri": "local://docs/protocol-stack.md",
        "collection": "knowledge-base",
        "tags": ["a2a", "mcp", "protocols"],
    },
    {
        "id": "doc-004",
        "title": "ColBERT Late Interaction Models",
        "content": (
            "ColBERT encodes queries and documents into token-level embeddings. "
            "Scoring uses MaxSim: each query token matches its best document token. "
            "This captures semantic nuance that single-vector pooling misses."
        ),
        "source_uri": "local://papers/colbert.pdf",
        "collection": "research-papers",
        "tags": ["colbert", "retrieval", "nlp"],
    },
    {
        "id": "doc-005",
        "title": "Multi-Agent Research Systems Design Patterns",
        "content": (
            "Effective research agents use specialist decomposition: web retrieval "
            "for recency, document search for depth, structured data for metrics. "
            "Reflection loops prevent premature synthesis on thin evidence."
        ),
        "source_uri": "local://docs/research-patterns.md",
        "collection": "knowledge-base",
        "tags": ["research", "agents", "patterns"],
    },
]


class VectorStore:
    """Qdrant-backed vector store with numpy fallback."""

    def __init__(self):
        self.settings = get_settings()
        self._client = None
        self._embedder = None
        self._memory_docs: list[dict[str, Any]] = []
        self._memory_embeddings: np.ndarray | None = None
        self._use_qdrant = False

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.settings.embedding_model)
        return self._embedder

    def _connect_qdrant(self) -> bool:
        if self._client is not None:
            return self._use_qdrant
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            self._client = QdrantClient(url=self.settings.qdrant_url)
            collections = [c.name for c in self._client.get_collections().collections]
            if self.settings.qdrant_collection not in collections:
                dim = self._get_embedder().get_sentence_embedding_dimension()
                self._client.create_collection(
                    collection_name=self.settings.qdrant_collection,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
            self._use_qdrant = True
            logger.info("Connected to Qdrant at %s", self.settings.qdrant_url)
            return True
        except Exception as e:
            logger.warning("Qdrant unavailable (%s), using in-memory store", e)
            self._use_qdrant = False
            return False

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self._get_embedder().encode(texts, normalize_embeddings=True)

    def seed(self) -> int:
        """Seed default corpus. Returns number of documents indexed."""
        return self.upsert_documents(SEED_CORPUS)

    def upsert_documents(self, documents: list[dict[str, Any]]) -> int:
        if self._connect_qdrant():
            from qdrant_client.models import PointStruct

            texts = [f"{d['title']}. {d['content']}" for d in documents]
            embeddings = self._embed(texts)
            points = [
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, doc["id"])),
                    vector=embeddings[i].tolist(),
                    payload=doc,
                )
                for i, doc in enumerate(documents)
            ]
            self._client.upsert(collection_name=self.settings.qdrant_collection, points=points)
            return len(documents)

        self._memory_docs.extend(documents)
        texts = [f"{d['title']}. {d['content']}" for d in self._memory_docs]
        self._memory_embeddings = self._embed(texts)
        return len(documents)

    def search(
        self,
        query: str,
        collection: str | None = None,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        if self._connect_qdrant():
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            query_vec = self._embed([query])[0].tolist()
            query_filter = None
            if collection:
                query_filter = Filter(
                    must=[FieldCondition(key="collection", match=MatchValue(value=collection))]
                )
            hits = self._client.search(
                collection_name=self.settings.qdrant_collection,
                query_vector=query_vec,
                query_filter=query_filter,
                limit=top_k,
            )
            return [{**hit.payload, "score": hit.score} for hit in hits]

        if not self._memory_docs:
            self.seed()

        query_vec = self._embed([query])[0]
        scores = self._memory_embeddings @ query_vec
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_idx:
            doc = self._memory_docs[idx]
            if collection and doc.get("collection") != collection:
                continue
            results.append({**doc, "score": float(scores[idx])})
        return results[:top_k]

    def list_collections(self) -> list[str]:
        if self._connect_qdrant():
            return list({d.get("collection", "default") for d in SEED_CORPUS})
        return list({d.get("collection", "default") for d in self._memory_docs or SEED_CORPUS})


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
