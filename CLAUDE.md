# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

**Setup:**
```bash
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

**Run the app:**
```bash
source .venv/bin/activate
python -m app.main
```
Then open http://127.0.0.1:8000 in your browser.

**Requirements:**
- Python 3.10+ (3.12–3.13 recommended)
- FFmpeg and ffprobe on PATH (e.g., `brew install ffmpeg` on macOS)
- `.env` file with `OPENAI_API_KEY` for AI features (see CONFIG.md for all options)

## Project Architecture

**test-clips** is a video analysis web app that breaks videos into scenes, transcribes dialogue, and generates AI-powered explanations with narration.

### High-Level Data Flow

1. **Upload** → user uploads video file
2. **Full-Video Audio Extraction** → extract complete video audio once for transcription
3. **Full-Video Transcription** → Whisper or AssemblyAI transcribes entire audio with speaker diarization
   - **Whisper** (default): optional pyannote speaker diarization via `HF_TOKEN`
   - **AssemblyAI**: built-in speaker diarization, always available
4. **Shot Detection** → PySceneDetect finds shot boundaries (configurable threshold)
5. **Scene Grouping** → merge N shots per scene (user-configurable, default 3)
6. **Per-Scene Transcript Filtering** → extract segments from full transcript matching scene timestamp ranges
7. **Video Analysis** → YOLOv8 + face clustering detect characters/objects across entire video (cached)
8. **Scene Explanation** → on-demand: extract frames from scene → send to OpenAI vision → generate explanation + optional TTS narration

### Backend (FastAPI in `app/main.py`)

**Key Endpoints:**
- `POST /api/upload` — upload video, returns `{id, filename, path_suffix}`
- `POST /api/process/{video_id}?provider=whisper|assemblyai&shots_per_scene=N` — detect shots, group into scenes, transcribe full video with selected provider. Returns full scene/shot metadata
- `GET /api/transcript/{video_id}/full-video` — fetch complete video transcript with all speakers and timestamps
- `GET /api/transcript/{video_id}/{filename}` — fetch scene-specific transcript JSON (e.g., `scene_001_assemblyai.json`)
- `POST /api/analyze-video/{video_id}` — analyze video for character/object context (caches result)
- `POST /api/explain-scene/{video_id}/{scene_index}` — generate explanation + cached narrations (summary, grounded summary)
- `GET /api/clips/{video_id}/{kind}/{filename}` — serve shots, scenes, or scene audio (kind: `shots|scenes|scene_audio`)
- `GET /api/explanations/{video_id}/{filename}` — fetch explanation MP3 narrations
- `GET /api/list-outputs` — list all previously processed videos
- `POST /api/clear` — delete all outputs

**Session Structure (under `outputs/{video_id}/`):**
```
outputs/{video_id}/
  audio.wav              # Full video audio (extracted once for transcription)
  shots/                 # Individual shot clips
    shot_001.mp4, ...
  scenes/                # Merged scene clips (1–N shots per scene)
    scene_001.mp4, ...
  scene_audio/           # Extracted audio for each scene (derived from full audio)
    scene_001.m4a, ...
  transcripts/           # Full video and per-scene transcriptions
    full_video_whisper.json         # Complete video transcription (if Whisper used)
    full_video_assemblyai.json      # Complete video transcription (if AssemblyAI used)
    scene_001_whisper.json          # Scene 1 segments (filtered from full_video_whisper)
    scene_001_assemblyai.json       # Scene 1 segments (filtered from full_video_assemblyai)
    scene_002_whisper.json
    scene_002_assemblyai.json
    ...
  explanations/          # Cached explanations and narrations
    scene_001.txt
    scene_001.mp3        # Full explanation narration
    scene_001_summary.txt
    scene_001_summary.mp3
    scene_001_grounded.txt
    scene_001_grounded.mp3
  video_context.json     # Character/object detection results (optional)
```

Each transcript JSON contains:
- `text` — full concatenated text
- `language` — detected language code
- `language_probability` — confidence (Whisper) or null (AssemblyAI)
- `provider` — "whisper" or "assemblyai"
- `diarization` — `{enabled: bool, speaker_count: int, note: string}`
- `segments` — array of `{start, end, text, speaker, words}` for each dialogue segment

### Configuration (`app/config.py`)

All settings are centralized in `app/config.py` and read from environment variables or defaults:
- **Transcription Provider** — `TRANSCRIPTION_PROVIDER` (whisper or assemblyai), `ASSEMBLYAI_API_KEY` (required for AssemblyAI)
- **Whisper** — model size, device (cpu/cuda/mps), compute precision, pyannote diarization token (`HF_TOKEN`)
- **Scene Detection** — ContentDetector threshold (lower = more scenes)
- **Frame Extraction** — max frames, interval (200–4000ms)
- **OpenAI** — API key, base URL, models for explanation/summary/TTS, timeouts, token limits
- **Summarization** — word-per-second pacing (default 2.3 for natural speech timing)
- **Video Context** — face clustering threshold, YOLOv8 model/confidence, character confidence threshold for mentioning names

See `CONFIG.md` for detailed reference.

### Video Context (Character & Object Detection)

`app/video_context.py` + `app/face_recognition_clustering.py` + `app/object_detection.py`:
- Runs automatically after video processing (non-blocking—won't fail processing if it fails)
- Extracts sparse frames (default 1 per 2 seconds)
- Clusters faces into character identities using face embeddings
- Detects objects using YOLOv8
- Caches results in `video_context.json` per video
- Used in scene explanations to mention "the man from Scene 1" or "the briefcase" for continuity

Character mentions in explanations only appear if detected with ≥70% confidence (tunable via `CHARACTER_CONFIDENCE_THRESHOLD_PERCENT`).

### Transcription Providers (`app/transcription_provider.py`)

Unified provider abstraction supporting both Whisper and AssemblyAI:
- **WhisperTranscriber** — uses faster-whisper library with optional pyannote speaker diarization
  - Fast, free (local), requires HF_TOKEN for speaker diarization
  - Returns word-level timestamps and speaker labels
- **AssemblyAITranscriber** — uses AssemblyAI SDK with built-in speaker diarization
  - Higher accuracy, requires API key (paid per minute)
  - Always includes speaker labels and structured segments
- **get_transcriber(provider)** — factory function returns appropriate provider instance
- **TranscriptionResult** — unified dataclass for consistent output across providers

Both providers return identical JSON format for easy switching. Full video is transcribed once; per-scene segments filtered by timestamp ranges. Preserves speaker diarization across entire video duration.

### Prompts (`app/prompts.py`)

System prompts for OpenAI are defined here to keep all model instructions in one place:
- `SCENE_EXPLAIN_SYSTEM_PROMPT` — instructs the model to analyze scene visuals, dialogue, and video context
- `SCENE_SUMMARY_SYSTEM_PROMPT` — time-constrained summary for TTS
- `SCENE_GROUNDED_SUMMARY_SYSTEM_PROMPT` — summary using actual transcript dialogue

Helper functions build user-message text for each request type, injecting transcript, frames, and video context.

## Key Patterns & Design Decisions

### Why Re-encode Instead of Stream Copy in FFmpeg

`app/main.py:_split_clip()` uses `-c:v libx264` (re-encode) instead of `-c copy` (stream copy). This ensures shot boundaries align exactly with detected timestamps. Stream copy aligns to keyframes, causing black frames or junk before the next keyframe—visible to users and bad UX.

### Explanation Caching

Explanations (full text + summaries + narrations) are cached in `explanations/` after first request. Subsequent requests return cached results instantly. Narrations are generated on-demand if the API key is available; if generation fails, the UI still shows the text explanation.

### Video Context Opt-In

Video context analysis is optional and runs in the background after processing completes. It's cached, so re-analyzing the same video returns immediately. If analysis fails, scene explanations still work (without character/object continuity hints).

### Environment Variable Precedence

`app/config.py` loads `.env` first, then reads environment variables. Environment variables override `.env` values at runtime—useful for Docker/deployment where you set vars before starting the app.

## Testing & Development Tips

**Check FFmpeg:**
```bash
ffmpeg -version
ffprobe -version
```

**Run with debug logging:**
Set `WHISPER_MODEL=tiny` for faster iteration (trades accuracy for speed).

**Test a specific scene explanation:**
```python
# In a Python shell with .venv activated:
from app.main import explain_scene
result = explain_scene(video_id="some_id", scene_index=1)
print(result)
```

**Clear all outputs to reset state:**
```bash
curl -X POST http://127.0.0.1:8000/api/clear
```

**Check loaded video context:**
```bash
cat outputs/{video_id}/video_context.json
```

## Important Notes

- **Transcription Provider Selection** — choose between Whisper (free, local) or AssemblyAI (requires API key) via radio buttons before processing. Selection affects provider suffix in transcript filenames.
- **Full-Video Transcription** — the entire video is transcribed once to preserve speaker diarization context. Per-scene transcripts are filtered from the full transcription based on timestamp ranges.
- **Speaker Diarization** — Whisper requires `HF_TOKEN` for speaker labels (optional). AssemblyAI always includes speaker identification (built-in).
- **API Key Required for Explanations** — without `OPENAI_API_KEY` in `.env`, the `/api/explain-scene` endpoint returns HTTP 503. Other features (shot detection, transcription, scene merging) work without it.
- **AssemblyAI Transcription Error** — if you select AssemblyAI but `ASSEMBLYAI_API_KEY` is not set, the endpoint returns HTTP 503. Sign up at https://assemblyai.com and add your API key to `.env`.
- **Large Model Downloads** — first run downloads Whisper weights (~1.4 GB for "base"), pyannote (~450 MB), and YOLOv8 (~50–100 MB). Network connectivity required.
- **Virtual Environment** — torch and pyannote.audio are large; use a fresh venv if dependency conflicts arise.
- **.env Requires Restart** — changes to `.env` are only picked up when the app restarts.
- **No GPU by Default** — Whisper and YOLOv8 run on CPU unless `WHISPER_DEVICE=cuda` or `WHISPER_DEVICE=mps` are set. GPU significantly speeds up processing.
- **Shots Display Toggle** — individual shots are hidden by default for cleaner UI. Check "Show Shots" to view them before scenes.

## Frontend (`static/index.html`)

Static HTML/JS UI with no build step—vanilla HTML, CSS, and ES6 JavaScript.

### Controls & Features

**Before Processing:**
- **File Input** — choose video file
- **Shots per Scene** — configure scene grouping (default 3 shots per scene)
- **Show Shots** — checkbox to toggle individual shot display (unchecked by default)
- **Transcription Provider** — radio buttons to select Whisper (default) or AssemblyAI
- **Process** — initiates shot detection, transcription, and scene grouping

**After Processing:**
- **Shots Section** — displays individual shot clips (if "Show Shots" is checked)
- **Scenes Section** — displays merged scene clips with:
  - **Transcript** button — opens modal with full scene transcript, speaker labels, timestamps, and download options
  - **Frames/sec selector** — controls frame extraction density for scene explanation
  - **Explain scene** button — generates AI-powered scene analysis with narration (requires OPENAI_API_KEY)
- **View Full Transcript** button — opens modal with complete video transcription including all speakers and timestamps across entire video

### Transcript Display

Scene and full-video transcripts display:
- **Metadata** — provider (Whisper/AssemblyAI), detected language, confidence, speaker count
- **Segments** — for each dialogue turn:
  - Timestamp range `[start – end]s`
  - Speaker label (e.g., "Speaker 1", "Speaker A")
  - Full text and word-level timestamps (if available)
- **Download Options**
  - TXT format: `[start - end] Speaker: text`
  - WebVTT format: video subtitle format for media players

### Architecture

Communicates with backend via fetch requests to `/api/` endpoints:
- Form data for uploads
- JSON for transcript display and explanation modal
- Audio/video streaming for playback
- Event listeners for user interactions (button clicks, checkbox toggle, escape key)

Modal system for transcript, explanation, and full-video transcript viewing with smooth transitions and backdrop dismissal.
