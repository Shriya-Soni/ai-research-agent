"""Tests for the reflection loop."""

import pytest

from rip.coordinator.reflection import evaluate_sufficiency
from rip.models.documents import RetrievedDocument, SourceType


def _doc(source: SourceType, colbert: float, doc_id: str = "test-1") -> RetrievedDocument:
    return RetrievedDocument(
        id=doc_id,
        title="Test",
        content="Test content about hybrid retrieval",
        source_type=source,
        source_uri="http://example.com",
        colbert_score=colbert,
    )


@pytest.mark.asyncio
async def test_empty_documents_triggers_more_research():
    result = await evaluate_sufficiency("test query", [], iteration=0, max_iterations=3)
    assert result["needs_more_research"] is True
    assert result["sufficiency_score"] == 0.0


@pytest.mark.asyncio
async def test_diverse_high_quality_evidence_is_sufficient():
    docs = [
        _doc(SourceType.WEB, 0.8, "w1"),
        _doc(SourceType.LOCAL_DOC, 0.7, "d1"),
        _doc(SourceType.STRUCTURED, 0.6, "s1"),
    ]

    result = await evaluate_sufficiency("test query", docs, iteration=0, max_iterations=3)
    assert result["sufficiency_score"] > 0.5
    assert result["needs_more_research"] is False
