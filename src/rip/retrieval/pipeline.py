"""End-to-end retrieval: dense + sparse → RRF → ColBERT."""

from __future__ import annotations

from dataclasses import dataclass, field

from rip.models.documents import RankedResult, RetrievedDocument
from rip.retrieval.colbert import ColBERTReranker
from rip.retrieval.dense import DenseRetriever
from rip.retrieval.rrf import reciprocal_rank_fusion
from rip.retrieval.sparse import SparseRetriever


@dataclass
class RetrievalPipeline:
    """Hybrid retrieval pipeline used by all specialist agents."""

    rrf_k: int = 60
    colbert_top_k: int = 20
    dense_top_k: int = 50
    sparse_top_k: int = 50
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    _dense: DenseRetriever = field(init=False)
    _sparse: SparseRetriever = field(init=False)
    _colbert: ColBERTReranker = field(init=False)

    def __post_init__(self) -> None:
        self._dense = DenseRetriever(self.embedding_model)
        self._sparse = SparseRetriever()
        self._colbert = ColBERTReranker(self.cross_encoder_model)

    def index(self, documents: list[RetrievedDocument]) -> None:
        self._dense.index(documents)
        self._sparse.index(documents)

    def retrieve(self, query: str) -> RankedResult:
        dense_results = self._dense.search(query, top_k=self.dense_top_k)
        sparse_results = self._sparse.search(query, top_k=self.sparse_top_k)

        fused = reciprocal_rank_fusion([dense_results, sparse_results], k=self.rrf_k)
        reranked = self._colbert.rerank(query, fused, top_k=self.colbert_top_k)

        return RankedResult(
            query=query,
            documents=reranked,
            pipeline_stages={
                "dense_count": len(dense_results),
                "sparse_count": len(sparse_results),
                "rrf_count": len(fused),
                "colbert_count": len(reranked),
                "top_colbert_score": reranked[0].colbert_score if reranked else 0.0,
            },
        )
