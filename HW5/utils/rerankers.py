"""Reranking helpers for Experiment 4 variations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

from langchain_core.documents import Document


@dataclass
class RerankResult:
    document: Document
    score: float


def _parse_score(raw_text: str) -> float:
    clean = raw_text.strip()
    if not clean:
        return 0.0
    first_token = clean.split()[0]
    try:
        return float(first_token)
    except ValueError:
        digits = "".join(ch for ch in first_token if ch.isdigit() or ch == ".")
        return float(digits) if digits else 0.0


def llm_rerank(
    documents: List[Document],
    question: str,
    scorer: Callable[[str], str],
    *,
    top_k: int = 3,
) -> List[Document]:
    """Apply an LLM scoring function to reorder documents."""
    ranked: List[RerankResult] = []
    for doc in documents:
        prompt = (
            "Rate how relevant this chunk is to the user question on a scale of 0-10.\n"
            f"Question: {question}\nChunk: {doc.page_content}\nRating:"
        )
        score_text = scorer(prompt)
        score_str = str(getattr(score_text, "content", score_text))
        score = _parse_score(score_str)
        ranked.append(RerankResult(document=doc, score=score))
    ranked.sort(key=lambda item: item.score, reverse=True)
    return [item.document for item in ranked[:top_k]]
