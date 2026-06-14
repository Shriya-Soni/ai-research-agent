"""Sparse retrieval via BM25 keyword matching."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from rip.models.documents import RetrievedDocument


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class SparseRetriever:
    """BM25 sparse retrieval — captures exact keyword and proper-noun signal."""

    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._documents: list[RetrievedDocument] = []

    def index(self, documents: list[RetrievedDocument]) -> None:
        self._documents = documents
        tokenized = [tokenize(f"{d.title} {d.content}") for d in documents]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 50) -> list[tuple[RetrievedDocument, float]]:
        if not self._documents or self._bm25 is None:
            return []

        query_tokens = tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for idx, score in ranked:
            if score <= 0:
                continue
            doc = self._documents[idx].model_copy()
            doc.sparse_score = float(score)
            results.append((doc, float(score)))
        return results
