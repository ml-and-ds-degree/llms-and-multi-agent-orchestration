"""Tests for HW5/utils/visualize.py"""

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.visualize import (
    load_results,
    plot_latency_comparison,
    plot_indexing_time,
    plot_chunk_counts,
    plot_accuracy_score,
)


class TestLoadResults:
    """Tests for load_results function."""

    def test_load_results_valid_files(self):
        """Test loading valid experiment result files."""
        with TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)

            # Create test JSON files
            exp1 = {
                "experiment_name": "Test 1",
                "avg_response_time": 2.5,
                "accuracy_rate": 0.8,
                "queries": [],
            }
            exp2 = {
                "experiment_name": "Test 2",
                "avg_response_time": 3.0,
                "accuracy_rate": 0.75,
                "queries": [{"ai_score": 0.9}, {"ai_score": 0.85}],
            }

            with open(results_dir / "experiment_1_baseline.json", "w") as f:
                json.dump(exp1, f)
            with open(results_dir / "experiment_2_llm_small.json", "w") as f:
                json.dump(exp2, f)

            # Mock RESULTS_DIR
            with patch("utils.visualize.RESULTS_DIR", results_dir):
                results = load_results()

            assert len(results) == 2
            assert results[0]["experiment_name"] == "Test 1"
            assert results[1]["experiment_name"] == "Test 2"
            # Check calculated_accuracy
            assert results[1]["calculated_accuracy"] == 0.875  # (0.9 + 0.85) / 2

    def test_load_results_with_display_labels(self):
        """Test that display labels are added correctly."""
        with TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)

            exp = {
                "experiment_name": "Test",
                "avg_response_time": 2.0,
            }

            with open(results_dir / "experiment_1_baseline_plus.json", "w") as f:
                json.dump(exp, f)

            with patch("utils.visualize.RESULTS_DIR", results_dir):
                results = load_results()

            assert len(results) == 1
            assert results[0]["display_label"] == "Baseline"
            assert results[0]["filename"] == "experiment_1_baseline_plus"

    def test_load_results_skips_non_experiment_files(self):
        """Test that non-experiment files are skipped."""
        with TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)

            # Create a valid experiment file
            with open(results_dir / "experiment_1_baseline.json", "w") as f:
                json.dump({"experiment_name": "Valid"}, f)

            # Create files that should be skipped
            with open(results_dir / "other_file.json", "w") as f:
                json.dump({"data": "skip me"}, f)
            with open(results_dir / "readme.txt", "w") as f:
                f.write("Not JSON")

            with patch("utils.visualize.RESULTS_DIR", results_dir):
                results = load_results()

            assert len(results) == 1
            assert results[0]["experiment_name"] == "Valid"

    def test_load_results_handles_corrupt_files(self):
        """Test that corrupt/invalid JSON files are handled gracefully."""
        with TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)

            # Create valid file
            with open(results_dir / "experiment_1_baseline.json", "w") as f:
                json.dump({"experiment_name": "Valid"}, f)

            # Create corrupt file
            with open(results_dir / "experiment_2_broken.json", "w") as f:
                f.write("{ invalid json")

            with patch("utils.visualize.RESULTS_DIR", results_dir):
                results = load_results()

            # Should only load the valid file
            assert len(results) == 1

    def test_load_results_empty_directory(self):
        """Test with no experiment files."""
        with TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)

            with patch("utils.visualize.RESULTS_DIR", results_dir):
                results = load_results()

            assert results == []


class TestPlotFunctions:
    """Tests for plotting functions."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "display_label": "Experiment 1",
                    "avg_response_time": 2.5,
                    "indexing_time": 10.0,
                    "num_chunks": 100,
                    "calculated_accuracy": 0.85,
                },
                {
                    "display_label": "Experiment 2",
                    "avg_response_time": 3.0,
                    "indexing_time": 12.0,
                    "num_chunks": 120,
                    "calculated_accuracy": 0.90,
                },
            ]
        )

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.bar")
    @patch("matplotlib.pyplot.title")
    @patch("matplotlib.pyplot.ylabel")
    @patch("matplotlib.pyplot.xticks")
    @patch("matplotlib.pyplot.grid")
    @patch("matplotlib.pyplot.text")
    @patch("matplotlib.pyplot.tight_layout")
    def test_plot_latency_comparison(
        self,
        mock_tight,
        mock_text,
        mock_grid,
        mock_xticks,
        mock_ylabel,
        mock_title,
        mock_bar,
        mock_figure,
        mock_close,
        mock_savefig,
        sample_dataframe,
    ):
        """Test latency comparison plot generation."""
        with TemporaryDirectory() as tmpdir:
            with patch("utils.visualize.CHARTS_DIR", Path(tmpdir)):
                plot_latency_comparison(sample_dataframe)

        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
        mock_bar.assert_called_once()

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.bar")
    @patch("matplotlib.pyplot.title")
    @patch("matplotlib.pyplot.ylabel")
    @patch("matplotlib.pyplot.xticks")
    @patch("matplotlib.pyplot.grid")
    @patch("matplotlib.pyplot.text")
    @patch("matplotlib.pyplot.tight_layout")
    def test_plot_indexing_time(
        self,
        mock_tight,
        mock_text,
        mock_grid,
        mock_xticks,
        mock_ylabel,
        mock_title,
        mock_bar,
        mock_figure,
        mock_close,
        mock_savefig,
        sample_dataframe,
    ):
        """Test indexing time plot generation."""
        with TemporaryDirectory() as tmpdir:
            with patch("utils.visualize.CHARTS_DIR", Path(tmpdir)):
                plot_indexing_time(sample_dataframe)

        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.bar")
    @patch("matplotlib.pyplot.title")
    @patch("matplotlib.pyplot.ylabel")
    @patch("matplotlib.pyplot.xticks")
    @patch("matplotlib.pyplot.grid")
    @patch("matplotlib.pyplot.text")
    @patch("matplotlib.pyplot.tight_layout")
    def test_plot_chunk_counts(
        self,
        mock_tight,
        mock_text,
        mock_grid,
        mock_xticks,
        mock_ylabel,
        mock_title,
        mock_bar,
        mock_figure,
        mock_close,
        mock_savefig,
        sample_dataframe,
    ):
        """Test chunk counts plot generation."""
        with TemporaryDirectory() as tmpdir:
            with patch("utils.visualize.CHARTS_DIR", Path(tmpdir)):
                plot_chunk_counts(sample_dataframe)

        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.bar")
    @patch("matplotlib.pyplot.title")
    @patch("matplotlib.pyplot.ylabel")
    @patch("matplotlib.pyplot.xticks")
    @patch("matplotlib.pyplot.grid")
    @patch("matplotlib.pyplot.ylim")
    @patch("matplotlib.pyplot.text")
    @patch("matplotlib.pyplot.tight_layout")
    def test_plot_accuracy_score(
        self,
        mock_tight,
        mock_text,
        mock_ylim,
        mock_grid,
        mock_xticks,
        mock_ylabel,
        mock_title,
        mock_bar,
        mock_figure,
        mock_close,
        mock_savefig,
        sample_dataframe,
    ):
        """Test accuracy score plot generation."""
        with TemporaryDirectory() as tmpdir:
            with patch("utils.visualize.CHARTS_DIR", Path(tmpdir)):
                plot_accuracy_score(sample_dataframe)

        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
        # Accuracy plot should set ylim
        mock_ylim.assert_called_once_with(0, 1.1)
