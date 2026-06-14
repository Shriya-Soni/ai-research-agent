"""Reciprocal Rank Fusion — merges ranked lists without score normalization."""

from __future__ import annotations

from rip.models.documents import RetrievedDocument


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[RetrievedDocument, float]]],
    k: int = 60,
) -> list[RetrievedDocument]:
    """
    Fuse multiple ranked lists using RRF.

    score(d) = Σ 1 / (k + rank_i(d))

    Documents appearing high in any list are rewarded; rank position only,
    no cross-space score normalization needed.
    """
    doc_scores: dict[str, float] = {}
    doc_map: dict[str, RetrievedDocument] = {}

    for ranked_list in ranked_lists:
        for rank, (doc, _score) in enumerate(ranked_list, start=1):
            key = doc.id
            doc_scores[key] = doc_scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in doc_map:
                doc_map[key] = doc

    sorted_ids = sorted(doc_scores.keys(), key=lambda d: doc_scores[d], reverse=True)

    results = []
    for doc_id in sorted_ids:
        doc = doc_map[doc_id].model_copy()
        doc.rrf_score = doc_scores[doc_id]
        results.append(doc)
    return results
