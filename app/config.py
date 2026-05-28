"""Configuration settings for the scene detection and analysis application.

This file centralizes all environment-based configuration and default values
for the application. Modify values here or set corresponding environment variables.
"""

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Load .env file so environment variables are available
_config_root = Path(__file__).resolve().parent.parent
load_dotenv(_config_root / ".env")

# ============================================================================
# WHISPER (Speech Recognition)
# ============================================================================

WHISPER_MODEL: str = os.environ.get("WHISPER_MODEL", "base")
"""Whisper model to use for speech recognition. Options: tiny, base, small, medium, large"""

WHISPER_DEVICE: str = os.environ.get("WHISPER_DEVICE", "cpu")
"""Device to run Whisper on. Options: cpu, cuda, mps"""

WHISPER_COMPUTE_TYPE_GPU: str = os.environ.get("WHISPER_COMPUTE", "float16")
"""Compute type for GPU (CUDA). Options: float16, float32"""

WHISPER_COMPUTE_TYPE_CPU: str = os.environ.get("WHISPER_COMPUTE", "int8")
"""Compute type for CPU. Options: int8, float32"""


# ============================================================================
# SCENE DETECTION
# ============================================================================

SCENE_DETECTION_THRESHOLD: float = 27.0
"""ContentDetector threshold for scene detection (lower = more sensitive)"""

DEFAULT_SHOTS_PER_SCENE: int = 3
"""Default number of shots to merge into each scene"""

MIN_SHOTS_PER_SCENE: int = 1
"""Minimum shots per scene allowed"""

MAX_SHOTS_PER_SCENE: int = 50
"""Maximum shots per scene allowed"""


# ============================================================================
# FRAME EXTRACTION (for scene explanation)
# ============================================================================

EXPLAIN_MAX_FRAMES: int = int(os.environ.get("EXPLAIN_MAX_FRAMES", "18"))
"""Maximum number of frames to extract from a scene for AI analysis"""

EXPLAIN_FRAME_INTERVAL_MS_DEFAULT: int = 500
"""Default frame interval in milliseconds (500ms = 2 frames/sec)"""

EXPLAIN_FRAME_INTERVAL_MS_MIN: int = 200
"""Minimum frame interval in milliseconds"""

EXPLAIN_FRAME_INTERVAL_MS_MAX: int = 4000
"""Maximum frame interval in milliseconds"""


# ============================================================================
# OPENAI API (General)
# ============================================================================

OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "").strip()
"""OpenAI API key (required for all AI features)"""

OPENAI_BASE_URL: str = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
"""OpenAI API base URL (can be overridden for custom endpoints)"""


# ============================================================================
# OPENAI - SCENE EXPLANATION
# ============================================================================

OPENAI_EXPLAIN_MODEL: str = os.environ.get("OPENAI_EXPLAIN_MODEL", "gpt-4o-mini")
"""Model used for scene explanation. Options: gpt-4o, gpt-4o-mini, gpt-4-turbo, etc."""

OPENAI_EXPLAIN_MAX_TOKENS: int = int(os.environ.get("OPENAI_EXPLAIN_MAX_TOKENS", "1200"))
"""Max tokens for scene explanation response"""

OPENAI_EXPLAIN_TIMEOUT: int = int(os.environ.get("OPENAI_EXPLAIN_TIMEOUT", "120"))
"""Timeout in seconds for explanation API call"""

OPENAI_IMAGE_DETAIL: str = os.environ.get("OPENAI_IMAGE_DETAIL", "low")
"""Image detail level for vision. Options: low, high"""

EXPLAIN_FULL_TRANSCRIPT_MAX_CHARS: int = int(
    os.environ.get("EXPLAIN_FULL_TRANSCRIPT_MAX_CHARS", "16000")
)
"""Max characters to include from full video transcript for context"""


# ============================================================================
# OPENAI - SCENE SUMMARY
# ============================================================================

OPENAI_SUMMARY_MAX_TOKENS: int = int(os.environ.get("OPENAI_SUMMARY_MAX_TOKENS", "300"))
"""Max tokens for quick summary response"""

OPENAI_SUMMARY_TIMEOUT: int = int(os.environ.get("OPENAI_SUMMARY_TIMEOUT", "60"))
"""Timeout in seconds for summary API call"""

# Summary uses same model as explanation
OPENAI_SUMMARY_MODEL = OPENAI_EXPLAIN_MODEL


# ============================================================================
# OPENAI - GROUNDED SUMMARY (uses actual dialogue from transcript)
# ============================================================================

OPENAI_GROUNDED_SUMMARY_MAX_TOKENS: int = int(
    os.environ.get("OPENAI_GROUNDED_SUMMARY_MAX_TOKENS", "300")
)
"""Max tokens for grounded summary response"""

OPENAI_GROUNDED_SUMMARY_TIMEOUT: int = int(
    os.environ.get("OPENAI_GROUNDED_SUMMARY_TIMEOUT", "60")
)
"""Timeout in seconds for grounded summary API call"""

# Grounded summary uses same model as explanation
OPENAI_GROUNDED_SUMMARY_MODEL = OPENAI_EXPLAIN_MODEL


# ============================================================================
# OPENAI - TEXT-TO-SPEECH
# ============================================================================

OPENAI_TTS_MODEL: str = os.environ.get("OPENAI_TTS_MODEL", "tts-1")
"""TTS model to use. Options: tts-1 (lower latency), tts-1-hd (higher quality)"""

OPENAI_TTS_VOICE: str = os.environ.get("OPENAI_TTS_VOICE", "alloy")
"""TTS voice to use. Options: alloy, echo, fable, onyx, nova, shimmer"""

OPENAI_TTS_MAX_INPUT_CHARS: int = int(os.environ.get("OPENAI_TTS_MAX_INPUT_CHARS", "4096"))
"""Max characters per TTS request (OpenAI limit)"""

OPENAI_TTS_TIMEOUT: int = int(os.environ.get("OPENAI_TTS_TIMEOUT", "120"))
"""Timeout in seconds for TTS API call"""


# ============================================================================
# SUMMARY AUDIO TIMING
# ============================================================================

SUMMARY_WORDS_PER_SECOND: float = 2.3
"""Average speaking pace for summaries (~150 words/minute = 2.5 words/sec, slightly conservative)"""

SUMMARY_MIN_WORDS: int = 30
"""Minimum word count for summaries (even for very short scenes)"""

SUMMARY_MAX_WORDS: int = 500
"""Maximum word count for summaries (for very long scenes)"""
