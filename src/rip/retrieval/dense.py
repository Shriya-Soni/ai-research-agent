"""Dense retrieval via embedding similarity (ANN-ready)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from rip.models.documents import RetrievedDocument

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Semantic retrieval using sentence embeddings and cosine similarity."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._corpus_embeddings: np.ndarray | None = None
        self._documents: list[RetrievedDocument] = []

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def index(self, documents: list[RetrievedDocument]) -> None:
        """Build embedding index for a document corpus."""
        self._documents = documents
        texts = [f"{d.title}. {d.content}" for d in documents]
        self._corpus_embeddings = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )

    def search(self, query: str, top_k: int = 50) -> list[tuple[RetrievedDocument, float]]:
        if not self._documents or self._corpus_embeddings is None:
            return []

        query_emb = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self._corpus_embeddings @ query_emb

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            doc = self._documents[idx].model_copy()
            doc.dense_score = float(scores[idx])
            results.append((doc, float(scores[idx])))
        return results
