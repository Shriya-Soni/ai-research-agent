"""Tests for RRF fusion."""

from rip.models.documents import RetrievedDocument, SourceType
from rip.retrieval.rrf import reciprocal_rank_fusion


def _doc(doc_id: str, rank_hint: str = "") -> RetrievedDocument:
    return RetrievedDocument(
        id=doc_id,
        title=f"Doc {doc_id}",
        content=f"Content {rank_hint}",
        source_type=SourceType.WEB,
    )


def test_rrf_favors_documents_in_both_lists():
    dense = [(_doc("a"), 0.9), (_doc("b"), 0.8), (_doc("c"), 0.7)]
    sparse = [(_doc("b"), 5.0), (_doc("d"), 4.0), (_doc("a"), 3.0)]

    fused = reciprocal_rank_fusion([dense, sparse], k=60)

    ids = [d.id for d in fused]
    assert "b" in ids[:2], "Doc appearing high in both lists should rank near top"
    assert len(fused) == 4


def test_rrf_score_decreases_with_rank():
    dense = [(_doc("top"), 0.9)]
    sparse = [(_doc("top"), 5.0)]

    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    assert fused[0].rrf_score == 2 / (60 + 1)
