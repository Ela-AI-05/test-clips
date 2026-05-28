# Configuration Guide

This document describes all configuration options for the Scene Detection and Analysis application.

Configuration can be set in two ways:
1. **Environment variables** - Set before starting the application
2. **Default values** - Defined in `app/config.py`

Environment variables take precedence over defaults. Set them in your `.env` file or export them before running the app.

## Whisper (Speech Recognition)

### `WHISPER_MODEL`
**Default:** `base`
**Type:** String
**Options:** `tiny`, `base`, `small`, `medium`, `large`

Model size for speech recognition. Larger models are more accurate but slower.

```bash
export WHISPER_MODEL=small
```

### `WHISPER_DEVICE`
**Default:** `cpu`
**Type:** String
**Options:** `cpu`, `cuda`, `mps`

Device to run Whisper on. Use `cuda` for NVIDIA GPUs, `mps` for Apple Silicon.

```bash
export WHISPER_DEVICE=cuda
```

### `WHISPER_COMPUTE`
**Default:** `float16` (GPU) / `int8` (CPU)
**Type:** String
**Options:** `float16`, `float32` (GPU) / `int8`, `float32` (CPU)

Computation precision. Lower precision (int8, float16) is faster but less accurate.

```bash
export WHISPER_COMPUTE=float32
```

## Scene Detection

### `SCENE_DETECTION_THRESHOLD`
**Default:** `27.0`
**Type:** Float

ContentDetector threshold for scene detection. Lower values = more sensitive (more scenes detected).
Typical range: 20-35

```python
# In app/config.py
SCENE_DETECTION_THRESHOLD: float = 27.0
```

### `DEFAULT_SHOTS_PER_SCENE`
**Default:** `3`
**Type:** Integer
**Range:** 1-50

Default number of shots to merge into each scene. Users can override this in the UI.

```python
# In app/config.py
DEFAULT_SHOTS_PER_SCENE: int = 3
```

## Frame Extraction (Scene Explanation)

### `EXPLAIN_MAX_FRAMES`
**Default:** `18`
**Type:** Integer

Maximum number of frames to extract from a scene for AI analysis. More frames = more context but slower processing.

```bash
export EXPLAIN_MAX_FRAMES=20
```

### Frame Interval (in milliseconds)
**Default:** 500ms (2 frames/second)
**Min:** 200ms (5 frames/second)
**Max:** 4000ms (0.25 frames/second)

Frame sampling interval. Hardcoded in `app/config.py`:

```python
EXPLAIN_FRAME_INTERVAL_MS_DEFAULT: int = 500
EXPLAIN_FRAME_INTERVAL_MS_MIN: int = 200
EXPLAIN_FRAME_INTERVAL_MS_MAX: int = 4000
```

## OpenAI API

### `OPENAI_API_KEY`
**Default:** Empty
**Type:** String
**Required:** Yes

Your OpenAI API key. Required for all AI features (explanation, summary, TTS).

```bash
export OPENAI_API_KEY=sk-...
```

Or in `.env`:
```env
OPENAI_API_KEY=sk-...
```

### `OPENAI_BASE_URL`
**Default:** `https://api.openai.com/v1`
**Type:** String

OpenAI API endpoint. Can be overridden for custom/proxy endpoints.

```bash
export OPENAI_BASE_URL=https://custom-endpoint.com/v1
```

## Scene Explanation

### `OPENAI_EXPLAIN_MODEL`
**Default:** `gpt-4o-mini`
**Type:** String
**Options:** `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, etc.

Model used for scene analysis (explanation, summary, grounded summary).

```bash
export OPENAI_EXPLAIN_MODEL=gpt-4o
```

### `OPENAI_EXPLAIN_MAX_TOKENS`
**Default:** `1200`
**Type:** Integer

Maximum tokens in the API response for scene explanations.

```bash
export OPENAI_EXPLAIN_MAX_TOKENS=1500
```

### `OPENAI_EXPLAIN_TIMEOUT`
**Default:** `120` seconds
**Type:** Integer

Request timeout for explanation API calls.

```bash
export OPENAI_EXPLAIN_TIMEOUT=180
```

### `OPENAI_IMAGE_DETAIL`
**Default:** `low`
**Type:** String
**Options:** `low`, `high`

Vision model detail level. `high` provides more visual detail but costs more tokens.

```bash
export OPENAI_IMAGE_DETAIL=high
```

### `EXPLAIN_FULL_TRANSCRIPT_MAX_CHARS`
**Default:** `16000`
**Type:** Integer

Maximum characters from the full video transcript to include as context. Reduces token usage.

```bash
export EXPLAIN_FULL_TRANSCRIPT_MAX_CHARS=20000
```

## Scene Summary

### `OPENAI_SUMMARY_MAX_TOKENS`
**Default:** `300`
**Type:** Integer

Maximum tokens for quick summary generation.

```bash
export OPENAI_SUMMARY_MAX_TOKENS=400
```

### `OPENAI_SUMMARY_TIMEOUT`
**Default:** `60` seconds
**Type:** Integer

Request timeout for summary API calls.

```bash
export OPENAI_SUMMARY_TIMEOUT=90
```

## Grounded Summary (Transcript-Based)

### `OPENAI_GROUNDED_SUMMARY_MAX_TOKENS`
**Default:** `300`
**Type:** Integer

Maximum tokens for grounded summary generation (uses actual dialogue from transcript).

```bash
export OPENAI_GROUNDED_SUMMARY_MAX_TOKENS=400
```

### `OPENAI_GROUNDED_SUMMARY_TIMEOUT`
**Default:** `60` seconds
**Type:** Integer

Request timeout for grounded summary API calls.

```bash
export OPENAI_GROUNDED_SUMMARY_TIMEOUT=90
```

## Summary Audio Timing

### Summary Speaking Pace
**Default:** `2.3` words/second
**Type:** Float

Average speaking rate for narration (~150 words/minute).

```python
# In app/config.py
SUMMARY_WORDS_PER_SECOND: float = 2.3
SUMMARY_MIN_WORDS: int = 30
SUMMARY_MAX_WORDS: int = 500
```

**Example:** For a 22-second scene:
- Target words = 22 × 2.3 ≈ 51 words
- This ensures the narration fits naturally within the scene duration

## Text-to-Speech (TTS)

### `OPENAI_TTS_MODEL`
**Default:** `tts-1`
**Type:** String
**Options:** `tts-1`, `tts-1-hd`

TTS model. `tts-1` is lower latency, `tts-1-hd` is higher quality.

```bash
export OPENAI_TTS_MODEL=tts-1-hd
```

### `OPENAI_TTS_VOICE`
**Default:** `alloy`
**Type:** String
**Options:** `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`

Voice for narration. Each has a different tone/accent.

```bash
export OPENAI_TTS_VOICE=nova
```

### `OPENAI_TTS_MAX_INPUT_CHARS`
**Default:** `4096`
**Type:** Integer

Maximum input length per TTS request (OpenAI API limit). Long explanations are automatically trimmed.

```bash
export OPENAI_TTS_MAX_INPUT_CHARS=4000
```

### `OPENAI_TTS_TIMEOUT`
**Default:** `120` seconds
**Type:** Integer

Request timeout for TTS API calls.

```bash
export OPENAI_TTS_TIMEOUT=180
```

## Example .env File

```env
# Whisper settings
WHISPER_MODEL=base
WHISPER_DEVICE=cpu
WHISPER_COMPUTE=int8

# OpenAI API
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1

# Models and analysis
OPENAI_EXPLAIN_MODEL=gpt-4o-mini
OPENAI_IMAGE_DETAIL=low
EXPLAIN_MAX_FRAMES=18
EXPLAIN_FULL_TRANSCRIPT_MAX_CHARS=16000

# Limits and timeouts
OPENAI_EXPLAIN_MAX_TOKENS=1200
OPENAI_EXPLAIN_TIMEOUT=120
OPENAI_SUMMARY_MAX_TOKENS=300
OPENAI_SUMMARY_TIMEOUT=60
OPENAI_GROUNDED_SUMMARY_MAX_TOKENS=300
OPENAI_GROUNDED_SUMMARY_TIMEOUT=60

# TTS
OPENAI_TTS_MODEL=tts-1
OPENAI_TTS_VOICE=alloy
OPENAI_TTS_MAX_INPUT_CHARS=4096
OPENAI_TTS_TIMEOUT=120
```

## Environment Setup Examples

### Quick Start (CPU, small models)
```bash
export OPENAI_API_KEY=sk-your-key
export WHISPER_MODEL=base
export OPENAI_EXPLAIN_MODEL=gpt-4o-mini
python -m uvicorn app.main:app --reload
```

### GPU Optimized (NVIDIA CUDA)
```bash
export OPENAI_API_KEY=sk-your-key
export WHISPER_DEVICE=cuda
export WHISPER_COMPUTE=float16
export OPENAI_EXPLAIN_MODEL=gpt-4o
export EXPLAIN_MAX_FRAMES=25
python -m uvicorn app.main:app --reload
```

### Apple Silicon (Metal Performance Shaders)
```bash
export OPENAI_API_KEY=sk-your-key
export WHISPER_DEVICE=mps
export OPENAI_EXPLAIN_MODEL=gpt-4o
python -m uvicorn app.main:app --reload
```

## Modifying Configuration

1. **For deployment:** Set environment variables via your deployment platform
2. **For development:** Create `.env` file in project root
3. **For testing:** Override in Python:
   ```python
   import os
   os.environ['EXPLAIN_MAX_FRAMES'] = '25'
   ```

All configuration values are read when the app starts. Changes to `.env` require restart.
