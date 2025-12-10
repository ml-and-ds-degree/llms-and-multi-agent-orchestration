"""Tests for HW5/utils/rerankers.py"""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.rerankers import (
    RerankResult,
    _parse_score,
    llm_rerank,
)


class TestParseScore:
    """Tests for _parse_score helper function."""

    def test_parse_score_clean_number(self):
        """Test parsing clean numeric scores."""
        assert _parse_score("7.5") == 7.5
        assert _parse_score("10") == 10.0
        assert _parse_score("0") == 0.0

    def test_parse_score_with_whitespace(self):
        """Test parsing scores with whitespace."""
        assert _parse_score("  8.5  ") == 8.5
        assert _parse_score("\n7.0\n") == 7.0

    def test_parse_score_with_text(self):
        """Test parsing when score is followed by text."""
        assert _parse_score("8 out of 10") == 8.0
        assert _parse_score("9.5 - highly relevant") == 9.5

    def test_parse_score_messy_format(self):
        """Test parsing scores from messy LLM output."""
        # The function looks at first token and extracts digits
        assert _parse_score("7.5 out of 10") == 7.5
        assert _parse_score("8 points") == 8.0

    def test_parse_score_empty_string(self):
        """Test parsing empty string."""
        assert _parse_score("") == 0.0
        assert _parse_score("   ") == 0.0

    def test_parse_score_no_digits(self):
        """Test parsing when no valid number is found."""
        assert _parse_score("Not a number") == 0.0


class TestRerankResult:
    """Tests for RerankResult dataclass."""

    def test_rerank_result_creation(self):
        """Test creating a RerankResult instance."""
        doc = Document(page_content="Test content")
        result = RerankResult(document=doc, score=8.5)

        assert result.document == doc
        assert result.score == 8.5


class TestLlmRerank:
    """Tests for llm_rerank function."""

    def test_llm_rerank_basic(self):
        """Test basic reranking with clear scores."""
        docs = [
            Document(page_content="Low relevance content"),
            Document(page_content="High relevance content"),
            Document(page_content="Medium relevance content"),
        ]

        def mock_scorer(prompt):
            if "High relevance" in prompt:
                return "9.0"
            elif "Medium relevance" in prompt:
                return "6.0"
            else:
                return "3.0"

        result = llm_rerank(
            documents=docs,
            question="Find high relevance content",
            scorer=mock_scorer,
            top_k=3,
        )

        # Should be ordered by score: High (9.0), Medium (6.0), Low (3.0)
        assert len(result) == 3
        assert result[0].page_content == "High relevance content"
        assert result[1].page_content == "Medium relevance content"
        assert result[2].page_content == "Low relevance content"

    def test_llm_rerank_top_k(self):
        """Test that top_k limits results correctly."""
        docs = [
            Document(page_content="Content 1"),
            Document(page_content="Content 2"),
            Document(page_content="Content 3"),
            Document(page_content="Content 4"),
        ]

        scores = {
            "Content 1": 9.0,
            "Content 2": 7.0,
            "Content 3": 5.0,
            "Content 4": 3.0,
        }

        def mock_scorer(prompt):
            for content, score in scores.items():
                if content in prompt:
                    return str(score)
            return "0.0"

        result = llm_rerank(
            documents=docs,
            question="Test query",
            scorer=mock_scorer,
            top_k=2,
        )

        # Should only return top 2
        assert len(result) == 2
        assert result[0].page_content == "Content 1"
        assert result[1].page_content == "Content 2"

    def test_llm_rerank_with_content_attr(self):
        """Test reranking when scorer returns object with content attribute."""
        docs = [
            Document(page_content="Doc 1"),
            Document(page_content="Doc 2"),
        ]

        class MockResponse:
            def __init__(self, content):
                self.content = content

        def mock_scorer(prompt):
            if "Doc 1" in prompt:
                return MockResponse("8.0")
            else:
                return MockResponse("6.0")

        result = llm_rerank(
            documents=docs,
            question="Test",
            scorer=mock_scorer,
            top_k=2,
        )

        assert len(result) == 2
        assert result[0].page_content == "Doc 1"

    def test_llm_rerank_messy_scores(self):
        """Test reranking with messy score outputs."""
        docs = [
            Document(page_content="Doc A"),
            Document(page_content="Doc B"),
            Document(page_content="Doc C"),
        ]

        def mock_scorer(prompt):
            if "Doc A" in prompt:
                return "Rating: 7.5 out of 10"
            elif "Doc B" in prompt:
                return "9 - highly relevant"
            else:
                return "Score is 4.0"

        result = llm_rerank(
            documents=docs,
            question="Query",
            scorer=mock_scorer,
            top_k=3,
        )

        # Should parse and sort correctly: B (9), A (7.5), C (4.0)
        assert result[0].page_content == "Doc B"
        assert result[1].page_content == "Doc A"
        assert result[2].page_content == "Doc C"

    def test_llm_rerank_empty_list(self):
        """Test reranking with empty document list."""
        result = llm_rerank(
            documents=[],
            question="Test",
            scorer=lambda x: "5.0",
            top_k=3,
        )

        assert result == []

    def test_llm_rerank_prompt_format(self):
        """Test that prompts are formatted correctly."""
        docs = [Document(page_content="Test content")]
        prompts_received = []

        def mock_scorer(prompt):
            prompts_received.append(prompt)
            return "5.0"

        llm_rerank(
            documents=docs,
            question="What is this about?",
            scorer=mock_scorer,
            top_k=1,
        )

        assert len(prompts_received) == 1
        prompt = prompts_received[0]
        assert "Rate how relevant" in prompt
        assert "What is this about?" in prompt
        assert "Test content" in prompt
        assert "0-10" in prompt
