"""ColBERT-style token-level cross-encoding re-ranker.

Uses a cross-encoder as a practical ColBERT proxy. The interface is
swappable for full ColBERT (colbert-ai) when the optional dependency is installed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rip.models.documents import RetrievedDocument

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class ColBERTReranker:
    """
    Precision re-ranking layer.

    Full ColBERT scores query tokens against document tokens independently.
    This implementation uses a cross-encoder on the top-K RRF candidates —
    expensive but far more accurate than bi-encoder pooling alone.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder (ColBERT proxy): %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int = 20,
    ) -> list[RetrievedDocument]:
        if not documents:
            return []

        candidates = documents[:top_k]
        pairs = [(query, f"{d.title}. {d.content[:2000]}") for d in candidates]
        scores = self.model.predict(pairs)

        scored = sorted(
            zip(candidates, scores, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for doc, score in scored:
            reranked = doc.model_copy()
            reranked.colbert_score = float(score)
            results.append(reranked)
        return results
