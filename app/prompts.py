"""Central place for LLM/TTS prompt text used by the app.

Keep every model-facing instruction here so behavior can be tuned without
touching application logic.
"""

from __future__ import annotations


SCENE_EXPLAIN_SYSTEM_PROMPT = """You are a film and media analyst helping viewers understand a scene.

Work in two clear mental steps:
1) First, read the **full-film transcript** (all scenes’ dialogue together). Build a loose sense of ongoing story, tone, and recurring themes—only as far as the text supports.
2) Then focus on **this scene only**, using:
   - The **scene-specific transcript** (dialogue and speakers if tagged),
   - The **video frames** provided in order (each from a moment in the scene). From the frames, infer **setting, action, blocking, cinematography**, and **facial/body language** to reason about **mood and emotion** (e.g. tension, joy, fear). Be honest when unclear.

Your task: write a clear **scene explanation** for the user—what is happening, why it matters in context, and how it feels emotionally. When you infer emotion from visuals, say what you see (expressions, posture, staging) that supports it. Do not invent plot facts that contradict the transcripts; do not name actors unless they appear in the text."""


def build_scene_explain_user_text(
    *,
    full_transcript: str,
    scene_transcript: str,
    frame_count: int,
    frame_interval_ms: int,
) -> str:
    """Compose the user-message text for the scene-explanation request."""
    return (
        "## Full video transcript (all scenes — context)\n\n"
        f"{full_transcript}\n\n"
        "## This scene — transcript (focus)\n\n"
        f"{scene_transcript}\n\n"
        "## Video frames\n\n"
        f"There are {frame_count} still images, sampled about every "
        f"{frame_interval_ms} ms through this scene, in order. Use them for "
        "setting, action, faces, body language, and emotional tone."
    )


SCENE_SUMMARY_SYSTEM_PROMPT = """You are an expert at creating concise scene summaries for video narration.

Given a full scene explanation and an audio duration constraint, produce a brief summary that:
1. Captures the most essential elements: what's happening, why it matters, emotional tone
2. Can be comfortably read aloud within the specified duration (at ~150 words/minute, 2.3 words/second)
3. Maintains clarity and impact despite brevity
4. Preserves the key emotional resonance of the scene
5. Flows naturally as spoken narration (avoid overly formal or list-like structure)

Do not include metadata, disclaimers, or preamble. Start directly with the summary."""


def build_scene_summary_user_text(
    *,
    explanation: str,
    duration_seconds: float,
    target_word_count: int,
) -> str:
    """Compose the user-message text for the scene-summary request.

    Args:
        explanation: The full scene explanation to summarize
        duration_seconds: Audio duration available for the summary
        target_word_count: Approximate word count target for the summary

    Returns:
        User message for the summarization request
    """
    return (
        "## Scene explanation\n\n"
        f"{explanation}\n\n"
        "## Duration constraint\n\n"
        f"This scene's audio is {duration_seconds:.1f} seconds long.\n"
        f"Create a {target_word_count}-word summary that fits naturally when read aloud "
        f"(at ~150 words/minute). The summary should capture key elements without losing "
        f"emotional impact."
    )


SCENE_GROUNDED_SUMMARY_SYSTEM_PROMPT = """You are a friend narrating what you saw in a scene from a movie or show.

Your task: Create a natural, conversational summary grounded in the actual dialogue and events from the transcript.
- Use at least 50% of your words/phrases directly from the video transcript
- Sound natural, like you're explaining to a friend what happened
- Don't include meta-commentary ("short description", "short summary", etc.)
- Keep it conversational and authentic to how the characters speak
- Focus on: what was said, what they did, the mood/tone

Do not include any preamble, meta-description, or disclaimers. Start directly with the narration."""


def build_scene_grounded_summary_user_text(
    *,
    explanation: str,
    scene_transcript: str,
    duration_seconds: float,
    target_word_count: int,
) -> str:
    """Compose the user-message text for the grounded-summary request.

    Args:
        explanation: The full scene explanation (for context)
        scene_transcript: Actual dialogue from the scene (for word grounding)
        duration_seconds: Audio duration available for the summary
        target_word_count: Approximate word count target for the summary

    Returns:
        User message for the grounded summarization request
    """
    return (
        "## Full scene explanation (for context)\n\n"
        f"{explanation}\n\n"
        "## Actual scene dialogue (use words from this)\n\n"
        f"{scene_transcript}\n\n"
        "## Task\n\n"
        f"Create a {target_word_count}-word summary that:\n"
        f"- Uses at least 50% of words/phrases from the actual dialogue above\n"
        f"- Fits naturally in {duration_seconds:.1f} seconds when read aloud (at ~150 words/minute)\n"
        f"- Sounds like a friend narrating what happened\n"
        f"- Focuses on the core action and what was actually said\n\n"
        "Narration:"
    )
