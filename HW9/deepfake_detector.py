"""
Deepfake detection system using Pydantic AI.

- Three detection agents using LATEST SOTA models (January 2026):
    1) Gemini 3 Pro (Preview) - latest SOTA reasoning and multimodal with native video
    2) GPT-5.2 (December 2025) - new flagship multimodal from OpenAI
    3) Qwen 3 VL (Late 2025) - latest open separate vision-language model via Ollama

- One Judge agent ("LLM as Judge") that takes the three detection outputs,
  performs a strict majority vote, and returns a final decision and reasoning.

Usage:
    - Install dependencies:
        uv pip install pydantic-ai opencv-python

    - Configure API keys via environment variables:
        export GOOGLE_API_KEY="your-google-api-key"
        export OPENAI_API_KEY="your-openai-api-key"

    - Set model strings via environment variables if needed:
        export GEMINI_MODEL="gemini-3-pro-preview"
        export OPENAI_MODEL="openai:gpt-5.2"
        export OLLAMA_MODEL="ollama:qwen3-vl"
        export JUDGE_MODEL="openai:gpt-5.2"

    - Run:
        uv run HW9/deepfake_detector.py

Notes:
    - The agents accept a video file path or a text description.
    - Gemini 3 Pro receives the full video (native video support).
    - GPT-5.2 and Qwen 3 VL receive extracted frames (image analysis).
"""

import argparse
import asyncio
import mimetypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent

# ---------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------
# Loguru is pre-configured to output to stderr with nice formatting.
# If you want to force stdout or customize:
# logger.remove()
# logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")

# ---------------------------------------------------------------------
# Configuration: model identifiers (override via environment variables)
# Latest models as of January 2026:
# - Gemini 3 Pro (Preview) - latest SOTA reasoning and multimodal from Google (Gemini 3 family)
# - GPT-5.2 (December 2025) - new flagship multimodal from OpenAI
# - Qwen 3 VL (late 2025) - most powerful open-source vision-language model
# ---------------------------------------------------------------------
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL", "gemini-3-pro-preview"
)  # Replaces Gemini 2.0 Flash
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "openai:gpt-5.2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "ollama:qwen3-vl")
JUDGE_MODEL = os.getenv(
    "JUDGE_MODEL", "openai:gpt-5.2"
)  # Best for reasoning/aggregation


# ---------------------------------------------------------------------
# Structured output models
# ---------------------------------------------------------------------
class DetectionOutput(BaseModel):
    """
    Each detection agent MUST return this structured output.
    pydantic-ai will ensure validation and request reflection if the model's output
    does not match the schema.
    """

    agent_name: str = Field(
        ..., description="Name/id of the agent that produced this output"
    )
    verdict: Literal["Deepfake", "Real", "Uncertain"] = Field(
        ..., description="Strict final verdict from this agent"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Agent's confidence in the verdict (0.0-1.0)"
    )
    reasons: List[str] = Field(
        ..., description="List of concise reasoning points (1-5 items)"
    )


class JudgeOutput(BaseModel):
    final_verdict: Literal["Deepfake", "Real"] = Field(
        ..., description="Final strict decision"
    )
    summary: str = Field(..., description="Summarized reasoning for the final verdict")
    votes: Dict[str, int] = Field(
        ..., description="Counts of votes, e.g. {'Deepfake':2,'Real':1}"
    )
    details: List[DetectionOutput] = Field(
        ..., description="Original detection agent outputs used to decide"
    )


# ---------------------------------------------------------------------
# Optional dependencies dataclass (if you later need to inject DB/services)
# ---------------------------------------------------------------------
@dataclass
class EmptyDeps:
    """
    Placeholder for RunContext dependencies. Keep it here to show how to type deps
    if you later add tools or dynamic instructions that require dependencies.
    """

    pass


# ---------------------------------------------------------------------
# System prompts for each detector (specialized for deepfake detection)
# ---------------------------------------------------------------------
GEMINI_SYSTEM_PROMPT = (
    "You are an expert video forensics model specialized in detecting deepfakes using Gemini 3 Pro. "
    "You have native video understanding capabilities with multimodal analysis. Analyze the provided video for:\n"
    "- Visual artifacts (blending, warping, edge artifacts, aliasing)\n"
    "- Frame-level inconsistencies and temporal jitter\n"
    "- Lighting mismatches and shadow inconsistencies\n"
    "- Unnatural facial expressions and movements\n"
    "- Audio-visual synchronization issues (lip-sync)\n"
    "- Temporal continuity problems\n\n"
    "Return structured JSON with: agent_name='Gemini-3-Pro', verdict (Deepfake|Real|Uncertain), "
    "confidence (0.0-1.0), and 1-5 specific reasons for your judgment."
)

OPENAI_SYSTEM_PROMPT = (
    "You are a forensic AI assistant with advanced multimodal capabilities for detecting deepfakes using GPT-5.2. "
    "You will be provided with frames extracted from a video. Analyze them for:\n"
    "- Spatial anomalies (warped backgrounds, inconsistent details like earrings/reflections)\n"
    "- Motion plausibility between frames\n"
    "- Lighting and shadow consistency\n"
    "- Facial muscle movement naturalness\n"
    "- Compression artifacts typical of GANs or diffusion models\n"
    "- Temporal continuity issues\n\n"
    "Return structured JSON with: agent_name='GPT-5.2', verdict (Deepfake|Real|Uncertain), "
    "confidence (0.0-1.0), and a concise list of 1-5 reasons."
)

OLLAMA_SYSTEM_PROMPT = (
    "You are Qwen 3 VL, the latest open-source multimodal model specialized in deepfake detection. "
    "You will analyze video frames for visual anomalies:\n"
    "- Unnatural movement and head/eye motion patterns\n"
    "- Inconsistent shadows and lighting across frames\n"
    "- Pixel-level artifacts and blending issues\n"
    "- Facial feature inconsistencies\n"
    "- Compression artifacts from synthetic generation\n\n"
    "Return structured JSON with: agent_name='Qwen-3-VL', verdict (Deepfake|Real|Uncertain), "
    "confidence (0.0-1.0), and 1-5 bullet points explaining your assessment."
)

JUDGE_SYSTEM_PROMPT = (
    "You are a judge agent that aggregates deepfake detection results using GPT-5.2. "
    "You will receive structured outputs from three detection agents. "
    "Your task:\n"
    "1. Perform strict majority vote between 'Deepfake' and 'Real' based on verdict fields\n"
    "2. If tied or contains 'Uncertain', apply tie-breaker: prefer side with higher total confidence\n"
    "3. If still tied, prefer 'Deepfake' (conservative security posture)\n\n"
    "Return structured JSON with: final_verdict (Deepfake|Real), summary (brief reasoning), "
    "votes (counts dict), and details (the original agent outputs).\n"
    "DO NOT invent agent outputs - use only the provided data."
)

# ---------------------------------------------------------------------
# Build agents (Lazy initialization to allow import without env vars)
# ---------------------------------------------------------------------
_agents_cache = {}


def get_agents():
    if _agents_cache:
        return (
            _agents_cache["gemini"],
            _agents_cache["openai"],
            _agents_cache["ollama"],
            _agents_cache["judge"],
        )

    gemini_detector = Agent(
        GEMINI_MODEL,
        deps_type=EmptyDeps,
        result_type=DetectionOutput,
        system_prompt=GEMINI_SYSTEM_PROMPT,
    )

    openai_detector = Agent(
        OPENAI_MODEL,
        deps_type=EmptyDeps,
        result_type=DetectionOutput,
        system_prompt=OPENAI_SYSTEM_PROMPT,
    )

    ollama_detector = Agent(
        OLLAMA_MODEL,
        deps_type=EmptyDeps,
        result_type=DetectionOutput,
        system_prompt=OLLAMA_SYSTEM_PROMPT,
    )

    judge_agent = Agent(
        JUDGE_MODEL,
        deps_type=EmptyDeps,
        result_type=JudgeOutput,
        system_prompt=JUDGE_SYSTEM_PROMPT,
    )

    _agents_cache["gemini"] = gemini_detector
    _agents_cache["openai"] = openai_detector
    _agents_cache["ollama"] = ollama_detector
    _agents_cache["judge"] = judge_agent

    return gemini_detector, openai_detector, ollama_detector, judge_agent


# ---------------------------------------------------------------------
# Frame extraction helper
# ---------------------------------------------------------------------
def extract_frames_as_jpegs(
    video_path: str,
    max_frames: int = 10,
    resize_width: Optional[int] = 512,
) -> List[BinaryContent]:
    """
    Extract up to max_frames evenly spaced frames from the video and return them
    as a list of BinaryContent with media_type 'image/jpeg'.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        # fall back to sequential read until EOF
        frames_to_take = list(range(max_frames))
    else:
        step = max(1, total_frames // max_frames)
        frames_to_take = list(range(0, total_frames, step))[:max_frames]

    contents: List[BinaryContent] = []
    for i, frame_idx in enumerate(frames_to_take):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        # optional resize for bandwidth/size control
        if resize_width is not None:
            h, w = frame.shape[:2]
            if w != resize_width:
                new_h = int(resize_width * (h / w))
                frame = cv2.resize(frame, (resize_width, new_h))

        # encode as JPEG
        success, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not success:
            continue
        jpeg_bytes = enc.tobytes()
        contents.append(BinaryContent(data=jpeg_bytes, media_type="image/jpeg"))

    cap.release()
    if not contents:
        raise RuntimeError("No frames extracted from video.")
    return contents


# ---------------------------------------------------------------------
# Helper: format a textual prompt
# ---------------------------------------------------------------------
def build_detector_prompt() -> str:
    """
    Consistent wrapper prompt passed to each detector.
    """
    return (
        "Analyze the provided video content (frames or video file) for deepfake artifacts.\n"
        "Look for visual inconsistencies, temporal anomalies, and audio-visual mismatches.\n"
        "Return the structured detection output."
    )


# ---------------------------------------------------------------------
# Run detectors in parallel (async) and then consult the judge agent
# ---------------------------------------------------------------------
async def run_detection_pipeline(video_input: str) -> JudgeOutput:
    """
    Run the three detector agents in parallel, collect their structured outputs,
    and invoke the judge agent to produce the final verdict.

    Args:
        video_input: Either a file path to a video or a text description

    Returns:
        JudgeOutput with final verdict and analysis
    """
    gemini_detector, openai_detector, ollama_detector, judge_agent = get_agents()

    prompt_text = build_detector_prompt()

    # Determine if input is a local video file path
    is_video_file = os.path.exists(video_input) and os.path.isfile(video_input)

    tasks = []

    # --- Gemini Task (native video support) ---
    if is_video_file:
        # Pass video bytes directly to Gemini
        mime_type, _ = mimetypes.guess_type(video_input)
        mime_type = mime_type or "video/mp4"
        video_bytes = Path(video_input).read_bytes()
        gemini_content = [
            prompt_text,
            BinaryContent(data=video_bytes, media_type=mime_type),
        ]
    else:
        # Fallback to text description
        gemini_content = prompt_text + f"\n\nVideo Description:\n{video_input}"

    tasks.append(gemini_detector.run(gemini_content, deps=None))

    # --- OpenAI & Ollama Tasks (Frame Extraction) ---
    if is_video_file:
        try:
            # Extract frames once for both agents
            logger.info(f"Extracting frames from {video_input}...")
            frames = extract_frames_as_jpegs(
                video_input, max_frames=10, resize_width=512
            )
            logger.info(f"Extracted {len(frames)} frames")
            frame_content = [prompt_text, *frames]
        except Exception as e:
            logger.opt(exception=True).warning(
                f"Frame extraction failed: {e}. Falling back to text prompt."
            )
            frame_content = (
                prompt_text
                + f"\n\n(Video file provided but frame extraction failed: {video_input})"
            )
    else:
        frame_content = prompt_text + f"\n\nVideo Description:\n{video_input}"

    tasks.append(openai_detector.run(frame_content, deps=None))
    tasks.append(ollama_detector.run(frame_content, deps=None))

    # Await all three
    logger.info(
        f"Running detection agents in parallel for input: {video_input[:50]}..."
    )
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        logger.error(f"Critical error during parallel execution: {e}")
        raise

    # Extract validated outputs (DetectionOutput instances)
    detections: List[DetectionOutput] = []
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            logger.error(f"Agent {i} failed: {res}")
            # Optionally append a placeholder failure result or skip
            continue
        if hasattr(res, "data"):
            detections.append(res.data)

    if not detections:
        logger.error("All detection agents failed.")
        raise RuntimeError("All detection agents failed to produce results.")

    # Prepare the judge prompt: provide the three structured detection outputs as JSON text
    details_text = "\n\n".join([d.model_dump_json(indent=2) for d in detections])
    judge_prompt = (
        "You are being given the exact outputs from three detection agents (as JSON). "
        "Decide with strict majority logic (tie-breaker: higher total confidence, then prefer Deepfake) "
        "and return the final structured JSON.\n\n"
        f"Agent outputs:\n{details_text}\n\n"
        "Produce JSON exactly matching the JudgeOutput model."
    )

    logger.info("Consulting judge agent...")
    judge_result = await judge_agent.run(judge_prompt, deps=None)
    return judge_result.data


# ---------------------------------------------------------------------
# Synchronous convenience wrapper
# ---------------------------------------------------------------------
def run_detection_pipeline_sync(video_input: str) -> JudgeOutput:
    """Synchronous wrapper for convenience/testing."""
    return asyncio.run(run_detection_pipeline(video_input))


# ---------------------------------------------------------------------
# CLI / Main Execution
# ---------------------------------------------------------------------
async def main():
    parser = argparse.ArgumentParser(
        description="Deepfake Detection System (Multi-Agent)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="video.mp4",
        help="Path to video file or text description of the video.",
    )
    args = parser.parse_args()

    # If input is default "video.mp4" but file doesn't exist, warn user or fall back to demo text
    video_input = args.input
    if video_input == "video.mp4" and not os.path.exists(video_input):
        logger.warning(
            f"Default input '{video_input}' not found. Switching to demo text description."
        )
        video_input = (
            "Short video: A 20-second interview clip of a public figure speaking. "
            "Observations: occasional lip-sync misalignment in 00:06-00:08, slight flicker on the jawline at 00:04, "
            "shadows look slightly inconsistent on the right side, audio seems clean. Frame-by-frame, "
            "there is an unnatural blink pattern (two fast blinks at 00:10)."
        )

    logger.info("=" * 80)
    logger.info("DEEPFAKE DETECTION SYSTEM STARTING")
    logger.info("=" * 80)
    logger.info(f"Gemini Model: {GEMINI_MODEL}")
    logger.info(f"OpenAI Model: {OPENAI_MODEL}")
    logger.info(f"Ollama Model: {OLLAMA_MODEL}")
    logger.info(f"Judge Model:  {JUDGE_MODEL}")
    logger.info("-" * 80)

    try:
        judge_output = await run_detection_pipeline(video_input)
        logger.info("=" * 80)
        logger.info("FINAL JUDGE OUTPUT")
        logger.info("=" * 80)
        print(judge_output.model_dump_json(indent=2))
    except Exception as e:
        logger.opt(exception=True).critical(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
        sys.exit(130)
