"""Tests for HW5/utils/metrics.py"""

import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import (
    ExperimentMetrics,
    QueryMetrics,
    Timer,
    print_metrics_summary,
)


class TestQueryMetrics:
    """Tests for QueryMetrics dataclass."""

    def test_query_metrics_creation(self):
        """Test creating a QueryMetrics instance."""
        qm = QueryMetrics(
            query="What is BOI?",
            response="Bank of India",
            response_time=1.5,
            chunks_retrieved=3,
        )
        assert qm.query == "What is BOI?"
        assert qm.response == "Bank of India"
        assert qm.response_time == 1.5
        assert qm.chunks_retrieved == 3
        assert qm.is_accurate is None

    def test_query_metrics_with_accuracy(self):
        """Test QueryMetrics with accuracy assessment."""
        qm = QueryMetrics(
            query="Test query",
            response="Test response",
            response_time=2.0,
            chunks_retrieved=5,
            is_accurate=True,
        )
        assert qm.is_accurate is True


class TestExperimentMetrics:
    """Tests for ExperimentMetrics dataclass."""

    def test_experiment_metrics_creation(self):
        """Test creating an ExperimentMetrics instance."""
        em = ExperimentMetrics(
            experiment_name="test_experiment",
            model_name="llama2",
            embedding_model="nomic-embed-text",
            chunk_size=500,
            chunk_overlap=50,
        )
        assert em.experiment_name == "test_experiment"
        assert em.model_name == "llama2"
        assert em.chunk_size == 500
        assert em.chunk_overlap == 50
        assert len(em.queries) == 0

    def test_avg_response_time_empty(self):
        """Test avg_response_time with no queries."""
        em = ExperimentMetrics(
            experiment_name="test",
            model_name="llama2",
            embedding_model="nomic",
            chunk_size=500,
            chunk_overlap=50,
        )
        assert em.avg_response_time == 0.0

    def test_avg_response_time_with_queries(self):
        """Test avg_response_time calculation."""
        em = ExperimentMetrics(
            experiment_name="test",
            model_name="llama2",
            embedding_model="nomic",
            chunk_size=500,
            chunk_overlap=50,
        )
        em.queries = [
            QueryMetrics("q1", "r1", 1.0, 3),
            QueryMetrics("q2", "r2", 2.0, 3),
            QueryMetrics("q3", "r3", 3.0, 3),
        ]
        assert em.avg_response_time == 2.0

    def test_accuracy_rate_empty(self):
        """Test accuracy_rate with no assessed queries."""
        em = ExperimentMetrics(
            experiment_name="test",
            model_name="llama2",
            embedding_model="nomic",
            chunk_size=500,
            chunk_overlap=50,
        )
        assert em.accuracy_rate == 0.0

    def test_accuracy_rate_none_assessed(self):
        """Test accuracy_rate when no queries have accuracy assessments."""
        em = ExperimentMetrics(
            experiment_name="test",
            model_name="llama2",
            embedding_model="nomic",
            chunk_size=500,
            chunk_overlap=50,
        )
        em.queries = [
            QueryMetrics("q1", "r1", 1.0, 3, None),
            QueryMetrics("q2", "r2", 2.0, 3, None),
        ]
        assert em.accuracy_rate == 0.0

    def test_accuracy_rate_with_assessments(self):
        """Test accuracy_rate calculation with mixed assessments."""
        em = ExperimentMetrics(
            experiment_name="test",
            model_name="llama2",
            embedding_model="nomic",
            chunk_size=500,
            chunk_overlap=50,
        )
        em.queries = [
            QueryMetrics("q1", "r1", 1.0, 3, True),
            QueryMetrics("q2", "r2", 2.0, 3, False),
            QueryMetrics("q3", "r3", 3.0, 3, True),
            QueryMetrics("q4", "r4", 4.0, 3, True),
        ]
        assert em.accuracy_rate == 0.75  # 3 out of 4

    def test_to_dict(self):
        """Test to_dict conversion."""
        em = ExperimentMetrics(
            experiment_name="test",
            model_name="llama2",
            embedding_model="nomic",
            chunk_size=500,
            chunk_overlap=50,
        )
        em.queries = [QueryMetrics("q1", "r1", 1.0, 3)]

        result = em.to_dict()
        assert isinstance(result, dict)
        assert result["experiment_name"] == "test"
        assert result["model_name"] == "llama2"
        assert "avg_response_time" in result
        assert "accuracy_rate" in result

    def test_save(self):
        """Test saving metrics to JSON file."""
        em = ExperimentMetrics(
            experiment_name="test",
            model_name="llama2",
            embedding_model="nomic",
            chunk_size=500,
            chunk_overlap=50,
            indexing_time=5.0,
            num_chunks=100,
        )
        em.queries = [QueryMetrics("q1", "r1", 1.0, 3, True)]

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_metrics.json"
            em.save(str(output_path))

            # Verify file exists
            assert output_path.exists()

            # Verify content
            with open(output_path) as f:
                data = json.load(f)
            assert data["experiment_name"] == "test"
            assert data["indexing_time"] == 5.0
            assert len(data["queries"]) == 1


class TestTimer:
    """Tests for Timer context manager."""

    def test_timer_measures_time(self):
        """Test that Timer measures elapsed time."""
        with Timer() as timer:
            time.sleep(0.1)  # Sleep for 100ms

        elapsed = timer.get_elapsed()
        assert elapsed >= 0.09  # Allow some margin
        assert elapsed < 0.2  # Should be less than 200ms

    def test_timer_get_elapsed_before_exit(self):
        """Test get_elapsed returns 0.0 before context exit."""
        timer = Timer()
        assert timer.get_elapsed() == 0.0

    def test_timer_get_elapsed_after_exit(self):
        """Test get_elapsed returns correct value after exit."""
        timer = Timer()
        with timer:
            time.sleep(0.05)

        elapsed1 = timer.get_elapsed()
        time.sleep(0.05)
        elapsed2 = timer.get_elapsed()

        # Elapsed should not change after context exit
        assert elapsed1 == elapsed2
        assert elapsed1 >= 0.04


class TestPrintMetricsSummary:
    """Tests for print_metrics_summary function."""

    def test_print_metrics_summary_basic(self, capsys):
        """Test printing metrics summary."""
        em = ExperimentMetrics(
            experiment_name="Test Experiment",
            model_name="llama2",
            embedding_model="nomic-embed-text",
            chunk_size=500,
            chunk_overlap=50,
            indexing_time=10.5,
            num_chunks=150,
        )
        em.queries = [
            QueryMetrics("What is BOI?", "Bank of India", 1.5, 3, True),
        ]

        print_metrics_summary(em)

        captured = capsys.readouterr()
        assert "EXPERIMENT SUMMARY: Test Experiment" in captured.out
        assert "llama2" in captured.out
        assert "Indexing Time: 10.50s" in captured.out
        assert "Number of Chunks: 150" in captured.out
        assert "What is BOI?" in captured.out

    def test_print_metrics_summary_no_accuracy(self, capsys):
        """Test printing when no accuracy assessments exist."""
        em = ExperimentMetrics(
            experiment_name="Test",
            model_name="llama2",
            embedding_model="nomic",
            chunk_size=500,
            chunk_overlap=50,
        )
        em.queries = [QueryMetrics("q1", "r1", 1.0, 3)]

        print_metrics_summary(em)

        captured = capsys.readouterr()
        # Should not show accuracy rate
        assert "Accuracy Rate:" not in captured.out
