"""Tests for HW5/utils/rag_pipeline.py"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.documents import Document

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.rag_pipeline import (
    SequentialOllamaEmbeddings,
    _percentile,
    summarize_chunks,
    build_embeddings,
    attach_contextual_metadata,
)


class TestPercentile:
    """Tests for _percentile helper function."""

    def test_percentile_empty_list(self):
        """Test percentile with empty list."""
        result = _percentile([], 0.5)
        assert result == 0.0

    def test_percentile_single_value(self):
        """Test percentile with single value."""
        result = _percentile([10], 0.5)
        assert result == 10.0

    def test_percentile_median(self):
        """Test percentile for median (50th percentile)."""
        values = [1, 2, 3, 4, 5]
        result = _percentile(values, 0.5)
        assert result == 3.0

    def test_percentile_25th(self):
        """Test 25th percentile."""
        values = [10, 20, 30, 40, 50]
        result = _percentile(values, 0.25)
        assert result == 20.0

    def test_percentile_75th(self):
        """Test 75th percentile."""
        values = [10, 20, 30, 40, 50]
        result = _percentile(values, 0.75)
        assert result == 40.0

    def test_percentile_interpolation(self):
        """Test percentile with interpolation."""
        values = [1, 2, 3, 4]
        result = _percentile(values, 0.5)
        # Should interpolate between 2 and 3
        assert result == pytest.approx(2.5, rel=0.01)


class TestSummarizeChunks:
    """Tests for summarize_chunks function."""

    def test_summarize_chunks_empty(self):
        """Test with empty chunk list."""
        result = summarize_chunks([])
        assert result == {"count": 0}

    def test_summarize_chunks_single(self):
        """Test with single chunk."""
        chunks = [Document(page_content="Hello World")]
        result = summarize_chunks(chunks)

        assert result["count"] == 1
        assert result["min"] == 11
        assert result["max"] == 11
        assert result["mean"] == 11.0
        assert result["median"] == 11.0

    def test_summarize_chunks_multiple(self):
        """Test with multiple chunks of varying lengths."""
        chunks = [
            Document(page_content="a" * 100),
            Document(page_content="b" * 200),
            Document(page_content="c" * 150),
            Document(page_content="d" * 50),
        ]
        result = summarize_chunks(chunks)

        assert result["count"] == 4
        assert result["min"] == 50
        assert result["max"] == 200
        assert result["mean"] == 125.0
        assert result["median"] == 125.0
        assert "p10" in result
        assert "p90" in result
        assert len(result["sample_first5"]) == 4


class TestSequentialOllamaEmbeddings:
    """Tests for SequentialOllamaEmbeddings wrapper."""

    @patch("utils.rag_pipeline.OllamaEmbeddings")
    def test_init(self, mock_ollama):
        """Test initialization."""
        embedder = SequentialOllamaEmbeddings(
            model="test-model",
            delay=0.01,
        )
        assert embedder.model == "test-model"
        assert embedder.delay == 0.01

    @patch("utils.rag_pipeline.OllamaEmbeddings")
    def test_embed_query(self, mock_ollama):
        """Test single query embedding."""
        mock_instance = Mock()
        mock_instance.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_ollama.return_value = mock_instance

        embedder = SequentialOllamaEmbeddings(model="test-model")
        result = embedder.embed_query("test text")

        assert result == [0.1, 0.2, 0.3]
        mock_instance.embed_query.assert_called_once_with("test text")

    @patch("utils.rag_pipeline.OllamaEmbeddings")
    @patch("time.sleep")
    def test_embed_documents_success(self, mock_sleep, mock_ollama):
        """Test embedding multiple documents successfully."""
        mock_instance = Mock()
        mock_instance.embed_query.side_effect = [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ]
        mock_ollama.return_value = mock_instance

        embedder = SequentialOllamaEmbeddings(model="test-model", delay=0.01)
        texts = ["text1", "text2", "text3"]
        results = embedder.embed_documents(texts)

        assert len(results) == 3
        assert results[0] == [0.1, 0.2]
        assert results[1] == [0.3, 0.4]
        assert results[2] == [0.5, 0.6]
        # Should call sleep between embeddings (n-1 times)
        assert mock_sleep.call_count == 2

    @patch("utils.rag_pipeline.OllamaEmbeddings")
    def test_embed_documents_with_failure(self, mock_ollama):
        """Test handling of embedding failures."""
        mock_instance = Mock()
        mock_instance.embed_query.side_effect = [
            [0.1, 0.2],
            Exception("Embedding failed"),
            [0.5, 0.6],
        ]
        mock_ollama.return_value = mock_instance

        embedder = SequentialOllamaEmbeddings(model="test-model", delay=0.0)
        texts = ["text1", "text2", "text3"]
        results = embedder.embed_documents(texts)

        assert len(results) == 3
        assert results[0] == [0.1, 0.2]
        # Failed embedding should return zero vector
        assert results[1] == [0.0] * 768
        assert results[2] == [0.5, 0.6]
        assert len(embedder._failed_indices) == 1
        assert embedder._failed_indices[0] == 1


class TestBuildEmbeddings:
    """Tests for build_embeddings function."""

    @patch("utils.rag_pipeline.SequentialOllamaEmbeddings")
    def test_build_embeddings_basic(self, mock_seq_embeddings):
        """Test building embeddings with basic parameters."""
        result = build_embeddings("test-model")

        mock_seq_embeddings.assert_called_once_with(
            model="test-model",
            delay=0.05,
            base_url=None,
            headers=None,
        )

    @patch("utils.rag_pipeline.SequentialOllamaEmbeddings")
    def test_build_embeddings_with_url(self, mock_seq_embeddings):
        """Test building embeddings with custom base URL."""
        result = build_embeddings(
            "test-model",
            base_url="http://custom:11434",
        )

        mock_seq_embeddings.assert_called_once_with(
            model="test-model",
            delay=0.05,
            base_url="http://custom:11434",
            headers=None,
        )


class TestAttachContextualMetadata:
    """Tests for attach_contextual_metadata function."""

    def test_attach_contextual_metadata_with_content(self):
        """Test attaching contextual summaries to chunks (including in content)."""
        chunks = [
            Document(page_content="Chunk 1 content", metadata={"page": 1}),
            Document(page_content="Chunk 2 content", metadata={"page": 2}),
        ]

        def mock_summary_fn(text):
            return f"Summary of: {text[:10]}"

        result = attach_contextual_metadata(
            chunks,
            summary_fn=mock_summary_fn,
            metadata_key="context_summary",
            include_in_content=True,
        )

        assert len(result) == 2

        # Check first chunk
        assert "Summary: Summary of: Chunk 1 co" in result[0].page_content
        assert "Chunk 1 content" in result[0].page_content
        assert result[0].metadata["context_summary"] == "Summary of: Chunk 1 co"
        assert result[0].metadata["page"] == 1

        # Check second chunk
        assert "Summary: Summary of: Chunk 2 co" in result[1].page_content
        assert "Chunk 2 content" in result[1].page_content
        assert result[1].metadata["context_summary"] == "Summary of: Chunk 2 co"

    def test_attach_contextual_metadata_without_content(self):
        """Test attaching contextual summaries only to metadata."""
        chunks = [
            Document(page_content="Chunk content", metadata={}),
        ]

        def mock_summary_fn(text):
            return "Test summary"

        result = attach_contextual_metadata(
            chunks,
            summary_fn=mock_summary_fn,
            include_in_content=False,
        )

        assert len(result) == 1
        # Content should not be modified
        assert result[0].page_content == "Chunk content"
        # But metadata should have summary
        assert result[0].metadata["context_summary"] == "Test summary"

    def test_attach_contextual_metadata_custom_key(self):
        """Test using custom metadata key."""
        chunks = [Document(page_content="Test", metadata={})]

        result = attach_contextual_metadata(
            chunks,
            summary_fn=lambda x: "summary",
            metadata_key="custom_key",
            include_in_content=False,
        )

        assert "custom_key" in result[0].metadata
        assert result[0].metadata["custom_key"] == "summary"
