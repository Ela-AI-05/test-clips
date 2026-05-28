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

If provided, use the **video context** to identify characters and objects:
- Character tracking shows which faces appear throughout the entire video
- Only mention character names if confidence is 70% or higher; otherwise use "the man", "the woman", etc.
- For objects: mention if they appear in earlier scenes (e.g., "the briefcase he carried from Scene 1")

Your task: write a clear **scene explanation** for the user—what is happening, why it matters in context, and how it feels emotionally. When you infer emotion from visuals, say what you see (expressions, posture, staging) that supports it. Do not invent plot facts that contradict the transcripts; do not name actors unless they appear in the text or video context."""


def build_scene_explain_user_text(
    *,
    full_transcript: str,
    scene_transcript: str,
    frame_count: int,
    frame_interval_ms: int,
    video_context_text: str = "",
) -> str:
    """Compose the user-message text for the scene-explanation request.

    Args:
        full_transcript: Full video transcript for narrative context
        scene_transcript: Scene-specific dialogue
        frame_count: Number of video frames provided
        frame_interval_ms: Frame sampling interval in milliseconds
        video_context_text: Character and object tracking info (optional)
    """
    text = (
        "## Full video transcript (all scenes — context)\n\n"
        f"{full_transcript}\n\n"
        "## This scene — transcript (focus)\n\n"
        f"{scene_transcript}\n\n"
    )

    if video_context_text:
        text += (
            "## Video context (character and object tracking)\n\n"
            f"{video_context_text}\n\n"
        )

    text += (
        "## Video frames\n\n"
        f"There are {frame_count} still images, sampled about every "
        f"{frame_interval_ms} ms through this scene, in order. Use them for "
        "setting, action, faces, body language, and emotional tone."
    )

    return text


def format_video_context_for_prompt(
    video_context,
    character_confidence_threshold: int = 70,
) -> str:
    """Format video context data as text for the prompt.

    Args:
        video_context: VideoContext object with characters and objects
        character_confidence_threshold: Only mention names above this confidence %

    Returns:
        Formatted text describing characters and objects
    """
    if not video_context:
        return ""

    lines = []

    # Character information
    if video_context.characters:
        lines.append("**Characters:**")
        for char in video_context.characters:
            conf_pct = int(char.confidence * 100)
            appearance_count = len(char.appearances)

            if conf_pct >= character_confidence_threshold:
                lines.append(
                    f"- Character #{char.id} ('{char.name}', {conf_pct}% confidence): "
                    f"appears in {appearance_count} frame(s) throughout the video"
                )
            else:
                lines.append(
                    f"- Character #{char.id}: appears in {appearance_count} frame(s) "
                    f"({conf_pct}% confidence, below naming threshold)"
                )
        lines.append("")

    # Object information
    if video_context.objects:
        lines.append("**Objects/Props tracked:**")
        for obj in video_context.objects:
            appearance_count = len(obj.appearances)
            lines.append(
                f"- '{obj.class_label}' (Object #{obj.id}): "
                f"appears in {appearance_count} frame(s) across video"
            )
        lines.append("")

    return "\n".join(lines)


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


SCENE_TRANSCRIPT_ENHANCEMENT_SYSTEM_PROMPT = """You are an expert video analyst and narrator. Your task is to enhance a scene's transcript with visual and contextual details derived from video analysis.

You are provided with:
1. **Scene Transcript** — Audio transcription with speaker labels and timestamps
   - Format: [start-end] [SPEAKER] dialogue text
   - Speakers labeled as: "Speaker 1", "Speaker 2", etc. (from audio diarization)

2. **Grounded Summary** — A detailed description created from the scene's video frames and audio transcript. This tells you what is ACTUALLY happening in the scene visually.

3. **Video Context** — Character and object detection data:
   - Characters detected in frames with confidence scores
   - Objects detected (COCO classes: person, chair, briefcase, phone, etc.)

Your goal: Enhance the scene transcript by adding visual/contextual details that would help an audience understand the full context of what's being said. Make "Speaker 1" more human by referencing what they're doing, holding, or their characteristics.

**Guidelines:**

1. **Speaker Identification:**
   - Infer gender/role from grounded summary (e.g., "the tall man in a suit" → male, business setting)
   - If video context identifies specific characters, use that data
   - Replace generic "Speaker N" labels with inferred characteristics: Male, Female, Business person, etc.
   - Keep original speaker ordering intact

2. **Actions and Gestures:**
   - Add what speakers are doing: [stands, walks, points, gestures]
   - Add how they're saying it: [emphatically, quietly, hesitantly]
   - Only include actions visible in grounded summary or video analysis

3. **Objects and Items:**
   - Add what speakers are holding/using: [holding briefcase] [with coffee cup]
   - Reference objects from video context if they appear near the speaker
   - Include items relevant to the dialogue context

4. **Scene Context:**
   - Add location/setting if clear from grounded summary: [in modern office] [at conference table]
   - Include relevant environmental details: [morning sunlight] [crowded room]

5. **Constraints:**
   - Keep enhancements MINIMAL and FACTUAL — only add what's clearly visible or mentioned
   - Do NOT add vague, generic, or speculative details
   - Do NOT change the original dialogue text — only enhance context around it
   - Do NOT add meta-commentary or disclaimers
   - Preserve ALL timestamps exactly as provided
   - ONLY enhance the scene duration segments (do not modify surrounding context)

**Output Format:**

Return the enhanced scene transcript using this exact format:
[start-end] [SPEAKER with context] dialogue text [enhancement if relevant]

Examples:
- Original: [45.2s–48.5s] [Speaker 1] I think we need to pause here.
- Enhanced: [45.2s–48.5s] [Male/Manager] I think we need to pause here. [taps folder on table]

- Original: [50.1s–52.3s] [Speaker 2] Absolutely, good idea.
- Enhanced: [50.1s–52.3s] [Female/Analyst] Absolutely, good idea. [nods, glancing at laptop screen]

Do NOT output explanations, preamble, or meta-analysis — only the enhanced transcript."""


def build_transcript_enhancement_user_text(
    *,
    scene_transcript: str,
    grounded_summary: str,
    video_context_text: str = "",
) -> str:
    """Compose the user-message text for the transcript-enhancement request.

    Args:
        scene_transcript: Scene transcript with timestamps and speaker labels
        grounded_summary: Grounded summary text describing visual and contextual details
        video_context_text: Character and object detection data (optional)

    Returns:
        User message for the transcript enhancement request
    """
    text = (
        "## Scene Transcript (Audio Diarization)\n\n"
        f"{scene_transcript}\n\n"
        "## Grounded Summary (Visual Analysis + Frames)\n\n"
        f"{grounded_summary}\n\n"
    )

    if video_context_text:
        text += (
            "## Video Context (Object & Character Detection)\n\n"
            f"{video_context_text}\n\n"
        )

    text += "Using the visual context from the grounded summary and any available character/object detection data, enhance the scene transcript with speaker characteristics, actions, objects, and scene context."

    return text
