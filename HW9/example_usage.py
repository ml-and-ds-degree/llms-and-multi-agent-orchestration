"""
Example usage of the Deepfake Detection System

This script demonstrates how to use the deepfake detector
with both video files and text descriptions.
"""

import asyncio
import os

from deepfake_detector import run_detection_pipeline, run_detection_pipeline_sync


def example_with_text_description():
    """Example: Analyze a video using text description"""
    print("=" * 80)
    print("EXAMPLE 1: Text Description Analysis")
    print("=" * 80)

    description = """
    A 30-second video clip of a political figure giving a speech.
    Notable observations:
    - Slight flickering around the jaw and neck area (0:05-0:08)
    - Lip movements don't perfectly sync with audio (0:12-0:15)
    - Unnatural blinking pattern: 3 rapid blinks at 0:20
    - Shadows on the face appear inconsistent with the background lighting
    - Audio quality is suspiciously clean with no background noise
    """

    result = run_detection_pipeline_sync(description)

    print("\n✓ Analysis Complete!")
    print(f"\nFinal Verdict: {result.final_verdict}")
    print(f"Vote Breakdown: {result.votes}")
    print(f"\nSummary: {result.summary}")
    print("\nIndividual Agent Results:")
    for detail in result.details:
        print(f"\n  {detail.agent_name}:")
        print(f"    - Verdict: {detail.verdict}")
        print(f"    - Confidence: {detail.confidence:.2%}")
        print("    - Reasons:")
        for reason in detail.reasons:
            print(f"      • {reason}")


def example_with_video_file():
    """Example: Analyze an actual video file"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Video File Analysis")
    print("=" * 80)

    # Replace with your actual video path
    video_path = "path/to/your/video.mp4"

    if not os.path.exists(video_path):
        print(f"\n⚠ Video file not found: {video_path}")
        print("Please update the video_path variable with an actual video file.")
        return

    print(f"\nAnalyzing video: {video_path}")
    print("This will:")
    print("  1. Send full video to Gemini 1.5 Pro")
    print("  2. Extract 10 frames for GPT-4o")
    print("  3. Extract 10 frames for Qwen2-VL")
    print("  4. Aggregate results with Judge agent")
    print("\nProcessing...")

    result = run_detection_pipeline_sync(video_path)

    print("\n✓ Analysis Complete!")
    print(f"\nFinal Verdict: {result.final_verdict}")
    print(f"Vote Breakdown: {result.votes}")
    print(f"\nSummary: {result.summary}")


async def example_async():
    """Example: Use async API for better performance"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Async Processing (Multiple Videos)")
    print("=" * 80)

    descriptions = [
        "Video 1: Clear deepfake with obvious facial artifacts",
        "Video 2: Genuine video with natural movements",
        "Video 3: Subtle deepfake with minor lip-sync issues",
    ]

    # Process multiple videos in parallel
    tasks = [run_detection_pipeline(desc) for desc in descriptions]
    results = await asyncio.gather(*tasks)

    print("\n✓ Batch Analysis Complete!")
    for i, (desc, result) in enumerate(zip(descriptions, results), 1):
        print(f"\n--- Video {i} ---")
        print(f"Description: {desc[:60]}...")
        print(f"Verdict: {result.final_verdict}")
        print(f"Votes: {result.votes}")


def check_environment():
    """Check if required environment variables are set"""
    print("=" * 80)
    print("ENVIRONMENT CHECK")
    print("=" * 80)

    required = {
        "GOOGLE_API_KEY": "Required for Gemini 1.5 Pro",
        "OPENAI_API_KEY": "Required for GPT-4o",
    }

    optional = {
        "GEMINI_MODEL": "Optional (default: gemini-1.5-pro)",
        "OPENAI_MODEL": "Optional (default: openai:gpt-4o)",
        "OLLAMA_MODEL": "Optional (default: ollama:qwen2-vl)",
        "JUDGE_MODEL": "Optional (default: openai:gpt-4o)",
    }

    print("\nRequired:")
    all_set = True
    for key, desc in required.items():
        value = os.getenv(key)
        status = "✓ SET" if value else "✗ MISSING"
        print(f"  {status} {key}: {desc}")
        if not value:
            all_set = False

    print("\nOptional:")
    for key, desc in optional.items():
        value = os.getenv(key)
        status = "✓ SET" if value else "○ Using default"
        display = f"{value[:20]}..." if value and len(value) > 20 else (value or "")
        print(f"  {status} {key}: {desc}")
        if value:
            print(f"         Value: {display}")

    return all_set


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("DEEPFAKE DETECTION SYSTEM - EXAMPLES")
    print("=" * 80)

    # Check environment first
    env_ok = check_environment()

    if not env_ok:
        print("\n⚠ Warning: Some required environment variables are missing!")
        print("Please set GOOGLE_API_KEY and OPENAI_API_KEY before running.")
        print("\nExample:")
        print('  export GOOGLE_API_KEY="your-google-api-key"')
        print('  export OPENAI_API_KEY="your-openai-api-key"')
        return

    # Run examples
    try:
        # Example 1: Text description
        example_with_text_description()

        # Example 2: Video file (commented out by default)
        # example_with_video_file()

        # Example 3: Async batch processing
        print("\n" + "=" * 80)
        print("Running async batch example...")
        asyncio.run(example_async())

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 80)
    print("Examples complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
