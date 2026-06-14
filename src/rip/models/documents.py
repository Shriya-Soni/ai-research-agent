"""Document and evidence models used across retrieval and synthesis."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    WEB = "web"
    LOCAL_DOC = "local_doc"
    STRUCTURED = "structured"
    SYNTHESIS = "synthesis"


class RetrievedDocument(BaseModel):
    """A single piece of evidence returned by a retrieval agent."""

    id: str
    title: str
    content: str
    source_type: SourceType
    source_uri: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float | None = None
    colbert_score: float | None = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def fingerprint(self) -> str:
        """Content fingerprint for deduplication."""
        normalized = " ".join(self.content.lower().split())[:500]
        return f"{self.source_uri}:{normalized}"


class RankedResult(BaseModel):
    """Output of the full retrieval pipeline for one query."""

    query: str
    documents: list[RetrievedDocument]
    pipeline_stages: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    index: int
    document_id: str
    title: str
    source_uri: str
    excerpt: str


class ReportSection(BaseModel):
    title: str
    content: str


class ResearchReport(BaseModel):
    """Final citation-backed research output."""

    query: str
    summary: str
    full_text: str = ""
    findings: list[str]
    citations: list[Citation]
    confidence: float
    gaps: list[str] = Field(default_factory=list)
    sections: list[ReportSection] = Field(default_factory=list)
    methodology: str = ""
    limitations: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
