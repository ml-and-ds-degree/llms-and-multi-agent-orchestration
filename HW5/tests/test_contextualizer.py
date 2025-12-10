"""Tests for HW5/utils/contextualizer.py"""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.contextualizer import (
    summarize_chunk,
    build_contextual_chunks,
)


class TestSummarizeChunk:
    """Tests for summarize_chunk function."""

    def test_summarize_chunk_basic(self):
        """Test basic chunk summarization."""

        def mock_llm_call(prompt):
            return "This is a summary of the chunk."

        result = summarize_chunk(
            "This is a long chunk of text about BOI compliance.",
            llm_call=mock_llm_call,
        )

        assert result == "This is a summary of the chunk."

    def test_summarize_chunk_with_content_attr(self):
        """Test summarization when LLM returns object with content attribute."""

        class MockResponse:
            def __init__(self, content):
                self.content = content

        def mock_llm_call(prompt):
            return MockResponse("Summary content")

        result = summarize_chunk(
            "Chunk text",
            llm_call=mock_llm_call,
        )

        assert result == "Summary content"

    def test_summarize_chunk_strips_whitespace(self):
        """Test that result is stripped of whitespace."""

        def mock_llm_call(prompt):
            return "  Summary with spaces  \n"

        result = summarize_chunk(
            "Test chunk",
            llm_call=mock_llm_call,
        )

        assert result == "Summary with spaces"

    def test_summarize_chunk_prompt_format(self):
        """Test that prompt is formatted correctly."""
        prompts_received = []

        def mock_llm_call(prompt):
            prompts_received.append(prompt)
            return "Summary"

        chunk_text = "Test chunk content"
        summarize_chunk(chunk_text, llm_call=mock_llm_call)

        assert len(prompts_received) == 1
        prompt = prompts_received[0]
        assert "BOI compliance document" in prompt
        assert "1-2 sentence summary" in prompt
        assert chunk_text in prompt


class TestBuildContextualChunks:
    """Tests for build_contextual_chunks function."""

    def test_build_contextual_chunks_with_content(self):
        """Test building contextual chunks with summary in content."""
        chunks = [
            Document(page_content="Chunk 1 text", metadata={"page": 1}),
            Document(page_content="Chunk 2 text", metadata={"page": 2}),
        ]

        def mock_summary_fn(text):
            return f"Summary of: {text}"

        result = build_contextual_chunks(
            chunks,
            summary_fn=mock_summary_fn,
            include_in_content=True,
        )

        assert len(result) == 2

        # Check first chunk
        assert result[0].page_content.startswith("Summary: Summary of: Chunk 1 text")
        assert "Chunk 1 text" in result[0].page_content
        assert result[0].metadata["context_summary"] == "Summary of: Chunk 1 text"
        assert result[0].metadata["page"] == 1

        # Check second chunk
        assert result[1].page_content.startswith("Summary: Summary of: Chunk 2 text")
        assert "Chunk 2 text" in result[1].page_content
        assert result[1].metadata["context_summary"] == "Summary of: Chunk 2 text"
        assert result[1].metadata["page"] == 2

    def test_build_contextual_chunks_without_content(self):
        """Test building contextual chunks with summary only in metadata."""
        chunks = [
            Document(page_content="Original text", metadata={"source": "doc1"}),
        ]

        def mock_summary_fn(text):
            return "Generated summary"

        result = build_contextual_chunks(
            chunks,
            summary_fn=mock_summary_fn,
            include_in_content=False,
        )

        assert len(result) == 1
        # Content should remain unchanged
        assert result[0].page_content == "Original text"
        # Metadata should have summary
        assert result[0].metadata["context_summary"] == "Generated summary"
        assert result[0].metadata["source"] == "doc1"

    def test_build_contextual_chunks_custom_metadata_key(self):
        """Test using custom metadata key for summaries."""
        chunks = [
            Document(page_content="Test", metadata={}),
        ]

        result = build_contextual_chunks(
            chunks,
            summary_fn=lambda x: "summary",
            metadata_key="custom_summary_key",
            include_in_content=False,
        )

        assert "custom_summary_key" in result[0].metadata
        assert result[0].metadata["custom_summary_key"] == "summary"

    def test_build_contextual_chunks_empty_list(self):
        """Test with empty chunks list."""
        result = build_contextual_chunks(
            [],
            summary_fn=lambda x: "summary",
        )

        assert result == []

    def test_build_contextual_chunks_preserves_all_metadata(self):
        """Test that all original metadata is preserved."""
        chunks = [
            Document(
                page_content="Text",
                metadata={"page": 5, "source": "file.pdf", "section": "intro"},
            ),
        ]

        result = build_contextual_chunks(
            chunks,
            summary_fn=lambda x: "summary",
            include_in_content=False,
        )

        assert result[0].metadata["page"] == 5
        assert result[0].metadata["source"] == "file.pdf"
        assert result[0].metadata["section"] == "intro"
        assert result[0].metadata["context_summary"] == "summary"
