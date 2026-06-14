"""Hybrid retrieval pipeline: dense + sparse → RRF → ColBERT re-rank."""

from rip.retrieval.pipeline import RetrievalPipeline

__all__ = ["RetrievalPipeline"]
