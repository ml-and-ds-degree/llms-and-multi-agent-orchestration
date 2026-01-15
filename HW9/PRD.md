# Product Requirements Document (PRD): Deepfake Detection System

## 1. Overview
The **Deepfake Detection System** is a multi-agent AI application designed to identify synthetic media (deepfakes) with high accuracy. By leveraging the latest State-of-the-Art (SOTA) multimodal Large Language Models (LLMs) available as of January 2026, the system analyzes video content for visual artifacts, temporal inconsistencies, and audio-visual mismatches. It employs a "Council of Experts" architecture where multiple specialized agents analyze the input independently, and a "Judge" agent aggregates their findings to deliver a final verdict.

## 2. Problem Statement
With the proliferation of advanced generative AI models (GANs, diffusion models), deepfakes have become increasingly indistinguishable from real footage. Single-model detection systems can be prone to specific biases or blind spots. A multi-model approach enables cross-validation of artifacts, increasing the reliability of detection.

## 3. Goals & Objectives
- **High Accuracy**: Minimize false positives and false negatives by aggregating insights from top-tier models from different providers (Google, OpenAI, Open Source).
- **Robustness**: Handle both video files (via frame extraction or native video input) and textual descriptions of videos.
- **Explainability**: Provide clear, structured reasoning for every verdict, breaking down specific artifacts found.
- **Extensibility**: Build on `pydantic-ai` to easily swap underlying models as newer SOTA version emerge.

## 4. Architecture

### 4.1. Core Components
The system consists of three **Detection Agents** and one **Judge Agent**.

1.  **Gemini Detector (Google)**
    *   **Model**: `gemini-3-pro-preview` (Dec 2025)
    *   **Capability**: Native video understanding. It processes the full video stream to detect temporal anomalies, audio-sync issues, and motion artifacts.
    *   **Role**: Temporal and Audio-Visual Expert.

2.  **OpenAI Detector (OpenAI)**
    *   **Model**: `openai:gpt-5.2` (Dec 2025)
    *   **Capability**: Advanced frame-by-frame analysis.
    *   **Role**: Spatial Artifact Expert (lighting, shadows, reflections).

3.  **Open Source Detector (Meta/Alibaba)**
    *   **Model**: `ollama:qwen3-vl` (Late 2025)
    *   **Capability**: High-resolution image analysis via local/Ollama inference.
    *   **Role**: Pixel-level Anomaly Expert (compression artifacts, edge blending).

4.  **Judge Agent**
    *   **Model**: `openai:gpt-5.2` (LLM-as-a-Judge)
    *   **Role**: Aggregator. Receives structured outputs from all agents and determines the final verdict.
    *   **Logic**: Strict majority vote with a confidence-based tie-breaker. Defaults to "Deepfake" in unresolved ties (security-first posture).

### 4.2. Workflow
1.  **Input Ingestion**: User provides a video file path (e.g., `.mp4`) or a text description.
2.  **Preprocessing**:
    *   If video file:
        *   Frames are extracted (e.g., 10 frames) using OpenCV for frame-based agents (GPT-5.2, Qwen 3 VL).
        *   Raw video bytes are prepared for the native video agent (Gemini).
3.  **Parallel Execution**: All three detection agents run asynchronously.
4.  **Structured Output**: Each agent returns a `DetectionOutput` object containing:
    *   `verdict` (Deepfake | Real | Uncertain)
    *   `confidence` (0.0 - 1.0)
    *   `reasons` (List of observed artifacts)
5.  **Adjudication**: The Judge Agent analyzes the three reports and synthesizes a `JudgeOutput` containing the final decision, summary, and vote counts.

## 5. Technical Requirements

### 5.1. Tech Stack
-   **Language**: Python 3.12+
-   **Framework**: `pydantic-ai` for agent orchestration and structured output validation.
-   **Video Processing**: `opencv-python` for frame extraction.
-   **Models**: Access via API (Google AI source, OpenAI API) and local inference (Ollama).

### 5.2. Data Models
**DetectionOutput**
```json
{
  "agent_name": "string",
  "verdict": "Deepfake" | "Real" | "Uncertain",
  "confidence": float,
  "reasons": ["string"]
}
```

**JudgeOutput**
```json
{
  "final_verdict": "Deepfake" | "Real",
  "summary": "string",
  "votes": {"Deepfake": int, "Real": int},
  "details": [DetectionOutput]
}
```

## 6. Configuration & Environment
The system must be configurable via Environment Variables:
-   `GEMINI_MODEL`: Default `gemini-3-pro-preview`
-   `OPENAI_MODEL`: Default `openai:gpt-5.2`
-   `OLLAMA_MODEL`: Default `ollama:qwen3-vl`
-   `JUDGE_MODEL`: Default `openai:gpt-5.2`
-   API Keys: `GOOGLE_API_KEY`, `OPENAI_API_KEY`

## 7. Success Metrics
-   **Latency**: Full pipeline execution (video extraction + parallel inference) under 30 seconds for short clips.
-   **Resilience**: Graceful fallback to text description if video processing fails.
-   **Consistency**: JSON schemas must be strictly enforced; agents failing to adhere should trigger validation retries (handled by `pydantic-ai`).
