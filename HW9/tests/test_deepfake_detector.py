import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure correctness of imports regardless of where pytest is run
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from pydantic import ValidationError

from HW9.deepfake_detector import (
    DetectionOutput,
    JudgeOutput,
    extract_frames_as_jpegs,
    gemini_detector,
    judge_agent,
    ollama_detector,
    openai_detector,
    run_detection_pipeline,
)


# ---------------------------------------------------------------------
# Data Model Tests
# ---------------------------------------------------------------------
def test_detection_output_valid():
    """Test valid creation of DetectionOutput."""
    output = DetectionOutput(
        agent_name="TestAgent",
        verdict="Deepfake",
        confidence=0.9,
        reasons=["Reason 1", "Reason 2"],
    )
    assert output.verdict == "Deepfake"
    assert output.confidence == 0.9
    assert output.agent_name == "TestAgent"


def test_detection_output_invalid_verdict():
    """Test validation failure for invalid verdict."""
    with pytest.raises(ValidationError):
        DetectionOutput(
            agent_name="TestAgent",
            verdict="Maybe",  # Invalid
            confidence=0.9,
            reasons=["Reason 1"],
        )


def test_detection_output_invalid_confidence():
    """Test validation failure for out-of-range confidence."""
    with pytest.raises(ValidationError):
        DetectionOutput(
            agent_name="TestAgent",
            verdict="Real",
            confidence=1.5,  # Invalid > 1.0
            reasons=["Reason 1"],
        )


def test_judge_output_valid():
    """Test valid creation of JudgeOutput."""
    det_out = DetectionOutput(
        agent_name="TestAgent", verdict="Deepfake", confidence=0.9, reasons=["Reason 1"]
    )
    output = JudgeOutput(
        final_verdict="Deepfake",
        summary="Summary",
        votes={"Deepfake": 1, "Real": 0},
        details=[det_out],
    )
    assert output.final_verdict == "Deepfake"
    assert len(output.details) == 1


# ---------------------------------------------------------------------
# Functionality Tests (Mocking)
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_detection_pipeline_text_input():
    """
    Test the pipeline with text input, mocking all agents.
    Verifies that the judge aggregates the results from 3 agents.
    """
    # Mock return values for detection agents
    mock_detection_result = MagicMock()
    mock_detection_result.data = DetectionOutput(
        agent_name="MockAgent",
        verdict="Deepfake",
        confidence=0.95,
        reasons=["Mock reason"],
    )

    # Mock return value for judge agent
    mock_judge_result = MagicMock()
    mock_judge_result.data = JudgeOutput(
        final_verdict="Deepfake",
        summary="Mock Summary",
        votes={"Deepfake": 3, "Real": 0},
        details=[mock_detection_result.data] * 3,
    )

    # Patch the run method of the agents
    with (
        patch.object(gemini_detector, "run", new_callable=AsyncMock) as mock_gemini,
        patch.object(openai_detector, "run", new_callable=AsyncMock) as mock_openai,
        patch.object(ollama_detector, "run", new_callable=AsyncMock) as mock_ollama,
        patch.object(judge_agent, "run", new_callable=AsyncMock) as mock_judge,
    ):
        mock_gemini.return_value = mock_detection_result
        mock_openai.return_value = mock_detection_result
        mock_ollama.return_value = mock_detection_result
        mock_judge.return_value = mock_judge_result

        input_text = "A video description showing unnatural blinking"
        result = await run_detection_pipeline(input_text)

        assert result.final_verdict == "Deepfake"
        assert result.summary == "Mock Summary"
        assert len(result.details) == 3

        # Verify all agents were called
        assert mock_gemini.called
        assert mock_openai.called
        assert mock_ollama.called
        assert mock_judge.called


def test_extract_frames_as_jpegs_file_not_found():
    """Test that extract_frames raises error for missing file."""
    # We expect RuntimeError because cv2.VideoCapture opening fails or returns empty
    # In the code: if not cap.isOpened(): raise RuntimeError
    # We rely on cv2 behavior here, but since we didn't mock it yet,
    # we assume the file doesn't exist on disk.

    # We should mock cv2 to ensure consistent behavior regardless of installed libs
    with patch.dict(sys.modules, {"cv2": MagicMock()}):
        import cv2

        cv2.VideoCapture.return_value.isOpened.return_value = False

        with pytest.raises(RuntimeError):
            extract_frames_as_jpegs("non_existent_video.mp4")


def test_extract_frames_as_jpegs_success():
    """Test frame extraction logic with mocked OpenCV."""
    mock_cv2 = MagicMock()

    # Setup mock VideoCapture chain
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 100  # Total frames

    # Mock reading a frame (ret, frame)
    # Return a dummy frame array (height, width, channels)
    dummy_frame = MagicMock()
    dummy_frame.shape = (100, 100, 3)
    mock_cap.read.return_value = (True, dummy_frame)

    mock_cv2.VideoCapture.return_value = mock_cap

    # Mock JPEG encoding
    # success, encoded_image
    mock_enc = MagicMock()
    mock_enc.tobytes.return_value = b"fake_jpeg_bytes"
    mock_cv2.imencode.return_value = (True, mock_enc)

    mock_cv2.CAP_PROP_FRAME_COUNT = 7
    mock_cv2.CAP_PROP_POS_FRAMES = 1
    mock_cv2.IMWRITE_JPEG_QUALITY = 1

    # Apply the mock to sys.modules so import cv2 inside function gets it
    with patch.dict(sys.modules, {"cv2": mock_cv2}):
        frames = extract_frames_as_jpegs("test_video.mp4", max_frames=2)

        assert len(frames) == 2
        assert frames[0].media_type == "image/jpeg"
        # Since we mocked tobytes()
        assert frames[0].data == b"fake_jpeg_bytes"

        # Verify release called
        mock_cap.release.assert_called_once()
