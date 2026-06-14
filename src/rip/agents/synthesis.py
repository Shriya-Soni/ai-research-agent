"""Synthesis agent — generates full citation-backed research papers."""

from __future__ import annotations

import logging
import re
from typing import Any

import uvicorn

from rip.a2a.server import create_a2a_app
from rip.llm import invoke_llm, llm_available
from rip.models.documents import Citation, ReportSection, ResearchReport, RetrievedDocument
from rip.models.messages import AgentCard, AgentSkill

logger = logging.getLogger(__name__)

MAX_SOURCES = 20
MIN_WORDS = 1200

AGENT_CARD = AgentCard(
    name="synthesis-agent",
    description="Merges multi-agent evidence into full research papers with citations",
    url="http://localhost:8004",
    skills=[
        AgentSkill(
            id="synthesize",
            name="Research Synthesis",
            description="Generate comprehensive research papers with inline citations and references",
            tags=["synthesis", "report", "citations"],
        )
    ],
)

PAPER_SYSTEM = """You are an expert research author writing comprehensive, publication-quality research papers.

RULES (strict):
1. Write ONLY from the provided evidence sources — do not invent facts.
2. Cite EVERY factual claim with inline bracket notation [1], [2], etc. matching source numbers.
3. Minimum length: {min_words} words across all sections.
4. Use markdown with these exact section headers:
   ## Abstract
   ## 1. Introduction
   ## 2. Background and Related Work
   ## 3. Analysis and Findings
   ## 4. Discussion
   ## 5. Conclusion
5. Each section must be substantive (multiple paragraphs), not bullet points.
6. In Analysis, synthesize across sources — compare, contrast, identify themes.
7. Note contradictions or gaps in the evidence in Discussion.
8. Do NOT include a References section — it will be appended automatically."""


def deduplicate(documents: list[RetrievedDocument]) -> list[RetrievedDocument]:
    seen: set[str] = set()
    unique: list[RetrievedDocument] = []
    for doc in sorted(documents, key=lambda d: d.colbert_score or 0, reverse=True):
        if doc.fingerprint not in seen:
            seen.add(doc.fingerprint)
            unique.append(doc)
    return unique[:MAX_SOURCES]


def build_citations(documents: list[RetrievedDocument]) -> list[Citation]:
    return [
        Citation(
            index=i,
            document_id=doc.id,
            title=doc.title,
            source_uri=doc.source_uri,
            excerpt=doc.content[:400],
        )
        for i, doc in enumerate(documents, start=1)
    ]


def format_evidence(documents: list[RetrievedDocument]) -> str:
    blocks = []
    for i, doc in enumerate(documents, start=1):
        blocks.append(
            f"--- SOURCE [{i}] ---\n"
            f"Type: {doc.source_type.value}\n"
            f"Title: {doc.title}\n"
            f"URI: {doc.source_uri}\n"
            f"Relevance (ColBERT): {doc.colbert_score or 0:.3f}\n"
            f"Content:\n{doc.content[:2000]}"
        )
    return "\n\n".join(blocks)


def append_references(paper: str, citations: list[Citation]) -> str:
    if not citations:
        return paper + "\n\n## References\n\n_No sources were retrieved for this research run._"
    refs = ["\n## References\n"]
    for c in citations:
        uri = f" — {c.source_uri}" if c.source_uri else ""
        refs.append(f"[{c.index}] {c.title}{uri}")
    return paper.rstrip() + "\n\n" + "\n".join(refs)


def parse_sections(full_text: str) -> list[ReportSection]:
    sections: list[ReportSection] = []
    parts = re.split(r"(?=^## )", full_text, flags=re.MULTILINE)
    for part in parts:
        part = part.strip()
        if not part.startswith("##"):
            continue
        lines = part.split("\n", 1)
        title = lines[0].lstrip("#").strip()
        content = lines[1].strip() if len(lines) > 1 else ""
        if title.lower() != "references":
            sections.append(ReportSection(title=title, content=content))
    return sections


def extract_findings(full_text: str) -> list[str]:
    """Pull bullet-like sentences with citations from Analysis section."""
    findings = []
    for line in full_text.split("\n"):
        line = line.strip()
        if re.search(r"\[\d+\]", line) and len(line) > 40:
            findings.append(line.lstrip("-•* "))
    return findings[:12] or [line.strip() for line in full_text.split("\n") if re.search(r"\[\d+\]", line)][:8]


def _fallback_paper(query: str, documents: list[RetrievedDocument], gaps: list[str]) -> str:
    citations = build_citations(documents)
    parts = [f"# Research Paper: {query}\n", "## Abstract\n"]
    if documents:
        parts.append(
            f"This report synthesizes {len(documents)} sources on '{query}' "
            f"retrieved via multi-agent hybrid search (RRF + ColBERT).\n"
        )
    else:
        parts.append(
            "Insufficient evidence was retrieved to produce a grounded research paper. "
            "Ensure all specialist agents are running and restart them after code updates.\n"
        )
    parts.append("\n## 1. Introduction\n")
    parts.append(f"This paper examines: {query}\n")
    parts.append("\n## 3. Analysis and Findings\n")
    for c, doc in zip(citations, documents, strict=True):
        parts.append(f"- [{c.index}] **{doc.title}**: {doc.content[:300]}...\n")
    if gaps:
        parts.append("\n## 4. Discussion\n")
        parts.append("Evidence gaps: " + "; ".join(gaps) + "\n")
    return append_references("\n".join(parts), citations)


async def synthesize_paper(
    query: str,
    documents: list[RetrievedDocument],
    gaps: list[str],
    agent_results: dict[str, Any],
) -> dict[str, Any]:
    citations = build_citations(documents)

    if not llm_available():
        full_text = _fallback_paper(query, documents, gaps)
        return {
            "summary": full_text.split("\n\n")[1][:500] if documents else "No LLM or sources available.",
            "full_text": full_text,
            "findings": extract_findings(full_text),
            "sections": parse_sections(full_text),
            "methodology": "Multi-agent RRF+ColBERT retrieval with A2A coordination.",
            "limitations": "; ".join(gaps) if gaps else "LLM unavailable — template report only.",
            "confidence": min(0.7, len(documents) * 0.1) if documents else 0.1,
        }

    if not documents:
        full_text = _fallback_paper(query, documents, gaps)
        return {
            "summary": "No evidence retrieved — unable to produce a grounded research paper.",
            "full_text": full_text,
            "findings": [],
            "sections": parse_sections(full_text),
            "methodology": "RRF + ColBERT hybrid retrieval across web, document, and structured agents.",
            "limitations": "No sources retrieved from any agent. " + "; ".join(gaps),
            "confidence": 0.1,
        }

    agent_summary = "\n".join(
        f"- {agent}: {info.get('count', 0)} documents (top ColBERT: {info.get('top_score', 0):.3f})"
        for agent, info in agent_results.items()
        if "error" not in info
    )

    user_prompt = (
        f"Research question: {query}\n\n"
        f"Retrieval summary:\n{agent_summary}\n\n"
        f"Evidence gaps to address in Discussion: {'; '.join(gaps) or 'none identified'}\n\n"
        f"SOURCES ({len(documents)} total — cite using [1] through [{len(documents)}]):\n\n"
        f"{format_evidence(documents)}"
    )

    system = PAPER_SYSTEM.format(min_words=MIN_WORDS)

    try:
        paper_body = await invoke_llm(system, user_prompt, max_tokens=4096)
        paper_body = paper_body.strip()
        if paper_body.startswith("```"):
            paper_body = paper_body.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        full_text = append_references(paper_body, citations)
        abstract = ""
        if "## Abstract" in paper_body:
            abstract = paper_body.split("## Abstract", 1)[-1].split("##", 1)[0].strip()

        word_count = len(full_text.split())
        confidence = min(0.95, 0.4 + len(documents) * 0.05 + (0.1 if word_count > MIN_WORDS else 0))

        return {
            "summary": abstract or paper_body[:600],
            "full_text": full_text,
            "findings": extract_findings(full_text),
            "sections": parse_sections(full_text),
            "methodology": (
                "Evidence gathered by specialist agents (web, document, structured) using "
                "dense+sparse hybrid retrieval, RRF fusion (k=60), and ColBERT re-ranking. "
                f"Synthesis used {len(documents)} sources ({word_count} words)."
            ),
            "limitations": "; ".join(gaps) if gaps else "See Discussion section for caveats.",
            "confidence": confidence,
        }
    except Exception as e:
        logger.error("LLM synthesis failed: %s", e)
        full_text = _fallback_paper(query, documents, gaps + [str(e)])
        return {
            "summary": full_text[:500],
            "full_text": full_text,
            "findings": extract_findings(full_text),
            "sections": parse_sections(full_text),
            "methodology": "Multi-agent retrieval pipeline.",
            "limitations": str(e),
            "confidence": 0.3,
        }


async def handle_query(query: str, metadata: dict) -> dict:
    raw_docs = metadata.get("documents", [])
    documents = deduplicate([RetrievedDocument.model_validate(d) for d in raw_docs])

    logger.info("Synthesizing paper for '%s' with %d sources", query, len(documents))

    synthesis = await synthesize_paper(
        query,
        documents,
        gaps=metadata.get("gaps", []),
        agent_results=metadata.get("agent_results", {}),
    )

    report = ResearchReport(
        query=query,
        summary=synthesis["summary"],
        full_text=synthesis["full_text"],
        findings=synthesis["findings"],
        citations=build_citations(documents),
        confidence=synthesis["confidence"],
        gaps=metadata.get("gaps", []),
        sections=synthesis.get("sections", []),
        methodology=synthesis.get("methodology", ""),
        limitations=synthesis.get("limitations", ""),
        metadata={
            "source_count": len(documents),
            "word_count": len(synthesis["full_text"].split()),
            "agent_results": metadata.get("agent_results", {}),
            "sufficiency_score": metadata.get("sufficiency_score", 0),
        },
    )
    return report.model_dump(mode="json")


def run_server(host: str = "0.0.0.0", port: int = 8004) -> None:
    app = create_a2a_app(AGENT_CARD, handle_query)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
