"""Helpers for contextual chunk enrichment and reranking used in Experiment 4."""

from __future__ import annotations

from typing import Callable, List

from langchain_core.documents import Document


def summarize_chunk(chunk_text: str, *, llm_call: Callable[[str], str]) -> str:
    prompt = (
        "You are preparing retrieval hints for a BOI compliance document. "
        "Write a 1-2 sentence summary capturing the key concepts in:\n\n"
        f"Chunk:{chunk_text}\n\nSummary:"
    )
    result = llm_call(prompt)
    return str(getattr(result, "content", result)).strip()


def build_contextual_chunks(
    chunks: List[Document],
    *,
    summary_fn: Callable[[str], str],
    metadata_key: str = "context_summary",
    include_in_content: bool = True,
) -> List[Document]:
    """Return chunks enriched with contextual summaries for embedding."""

    enriched: List[Document] = []
    for chunk in chunks:
        summary = summary_fn(chunk.page_content)
        new_meta = {**chunk.metadata, metadata_key: summary}
        new_content = (
            f"Summary: {summary}\n\n{chunk.page_content}"
            if include_in_content
            else chunk.page_content
        )
        enriched.append(Document(page_content=new_content, metadata=new_meta))
    return enriched
