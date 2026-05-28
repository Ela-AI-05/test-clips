# Features Overview

## Latest Features (May 2026)

### 1. Dual Transcription Providers

**Whisper (Default)**
- Free, runs locally
- Requires ~1.4GB model download on first run
- Optional speaker diarization with `HF_TOKEN` (via pyannote)
- Speaker labels shown as "Speaker 1", "Speaker 2", etc.
- Word-level timestamps available

**AssemblyAI (New)**
- Higher accuracy transcription
- Built-in speaker diarization (always enabled)
- Requires API key (paid per minute, free trial credits available)
- Speaker labels shown as "Speaker A", "Speaker B", etc.
- Structured segment output with speaker metadata

**Selection:**
- Choose provider via radio buttons before processing
- Both providers return identical JSON format for seamless switching
- Environment variable default: `TRANSCRIPTION_PROVIDER=whisper` (or `assemblyai`)

### 2. Full-Video Transcription with Speaker Diarization

**Previous Approach (Per-Scene):**
- Each scene audio extracted separately and transcribed
- Speaker context lost between scenes
- Individual speaker detection per scene only

**New Approach (Full-Video):**
- Entire video audio transcribed once
- Preserves speaker continuity across all scenes
- All speaker diarization labels consistent throughout video
- Per-scene transcripts filtered from full transcription by timestamp ranges
- Both full-video and per-scene transcripts stored separately

**Benefits:**
- Accurate speaker identification across entire narrative
- Can track character speeches across multiple scenes
- Compare transcriptions from both providers without reprocessing
- Maintains speaker labels consistently (same speaker has same label across entire video)

### 3. Full Video Transcript Modal

**View Complete Transcription:**
- Click "View Full Transcript" button (appears after processing)
- Modal displays all segments with:
  - Timestamp ranges `[start – end]s`
  - Speaker labels with visual badges
  - Full text for each dialogue turn
  - Metadata: provider, language, speaker count, confidence

**Download Options:**
- **TXT Format**: Plain text with timestamps and speaker labels
  ```
  [0.00s - 2.34s] Speaker 1: Hello, how are you today?
  [2.50s - 4.10s] Speaker 2: I'm doing great, thanks for asking!
  ```
- **WebVTT Format**: Video subtitle format compatible with media players
  ```
  WEBVTT
  
  00:00:00.000 --> 00:00:02.340
  Speaker 1: Hello, how are you today?
  ```

### 4. Scene-Level Transcripts with Speaker Diarization

Each scene includes:
- Relevant segments from full-video transcription
- Same provider and speaker consistency as full video
- Download options (TXT and WebVTT)
- Audio player with word-level timestamp seeking
- Speaker badges and timing information

### 5. Shots Display Toggle

**Default Behavior:**
- Shots section hidden by default (cleaner UI)
- Only scenes visible on initial load

**Show Shots:**
- Checkbox "Show shots" in controls section
- Check to display all individual shot clips before scenes
- Uncheck to hide shots and focus on merged scenes
- Toggle at any time (before or after processing)
- Respects state when loading previous outputs

**Use Cases:**
- Power users: enable to see shot boundaries and timing
- Standard users: keep disabled for simplified workflow
- Quick review: toggle on to examine shot-level transitions

### 6. Scene Explanation with AI Vision Analysis

**For Each Scene:**
- Click "Explain scene" button
- AI analyzes:
  - Visual content (frames from the scene)
  - Dialogue (transcribed text with speakers)
  - Character/object context (from video analysis)
- Generates:
  - Full explanation (detailed analysis)
  - Summary (time-constrained for TTS narration)
  - Grounded summary (using actual transcript dialogue)
  - Optional narration (TTS audio for all three)

**Frame Extraction Control:**
- Selector: 5/sec, 2/sec (default), 1/sec, 0.5/sec
- Higher density = more detailed but slower analysis
- Can regenerate with different frame rates

### 7. Multi-Provider Comparison

**Storage Strategy:**
Each video can have transcripts from both providers:
```
transcripts/
  full_video_whisper.json
  full_video_assemblyai.json
  scene_001_whisper.json
  scene_001_assemblyai.json
  scene_002_whisper.json
  scene_002_assemblyai.json
  ...
```

**Benefits:**
- A/B compare accuracy between providers
- Switch providers without reprocessing
- Gradual migration from Whisper to AssemblyAI (or vice versa)
- Keep both for quality validation

## Core Features

### Shot & Scene Detection

**Shot Detection:**
- Uses PySceneDetect ContentDetector
- Threshold configurable via `CONTENT_THRESHOLD` (default 27.0)
- Lower values = more shots detected

**Scene Grouping:**
- Merge N consecutive shots into scenes (user-configurable, default 3)
- Scenes represent semantic units (often correspond to script scenes)
- Each scene gets video clip, audio extraction, and analysis

### Video Analysis (Background)

**Character & Object Detection:**
- Runs non-blocking after video processing
- Face clustering identifies recurring characters
- YOLOv8 detects objects and their context
- Results cached in `video_context.json`
- Used to enhance scene explanations with continuity cues

**Example Enhancement:**
Instead of: "A man enters the room..."
Enhanced: "The man from Scene 1 enters the room again..."

### Cached Explanations

**First Generation:**
- Call `/api/explain-scene/{video_id}/{scene_index}`
- AI analyzes frames, generates full, summary, and grounded explanations
- Narrations generated if `OPENAI_API_KEY` available

**Subsequent Calls:**
- Return cached results instantly (no re-analysis)
- Can regenerate with different frame rates or settings
- Narration generation can fail gracefully (text still shows)

### Audio & Video Playback

**Scene Audio:**
- Extracted from video for each scene
- Playable in transcript modal with word-level seeking
- Format: M4A (extracted via FFmpeg)

**Scene Video:**
- Re-encoded MP4 for exact boundary alignment
- Requires transcoding (ensures accuracy over speed)
- Playable in scene card with controls

**Full Video:**
- Original source file preserved
- Session structure stored separately from uploads

## Technical Implementation

### Transcription Provider Abstraction

**TranscriptionResult Dataclass:**
```python
{
  "text": str,                          # Full concatenated text
  "language": str,                      # Detected language
  "language_probability": float,        # Confidence (Whisper) or null
  "provider": str,                      # "whisper" or "assemblyai"
  "diarization": {
    "enabled": bool,
    "speaker_count": int,
    "note": str                         # Optional status message
  },
  "segments": [
    {
      "start": float,                   # Seconds
      "end": float,
      "text": str,
      "speaker": str,                   # Speaker label
      "words": [                        # Word-level detail (if available)
        {"start": float, "end": float, "word": str}
      ]
    }
  ]
}
```

### API Endpoints (Latest)

**Transcription:**
- `POST /api/process/{video_id}?provider=whisper|assemblyai&shots_per_scene=N`
  - Transcription provider selectable per video
  - Both approaches return identical format
  
- `GET /api/transcript/{video_id}/full-video`
  - Complete video transcription with all speakers
  - Prefers AssemblyAI if both exist
  
- `GET /api/transcript/{video_id}/scene_NNN_PROVIDER.json`
  - Scene-specific segments filtered from full transcription

**Scene & Shot Management:**
- `GET /api/clips/{video_id}/shots/shot_NNN.mp4`
- `GET /api/clips/{video_id}/scenes/scene_NNN.mp4`
- `GET /api/clips/{video_id}/scene_audio/scene_NNN.m4a`

**Analysis & Explanation:**
- `POST /api/analyze-video/{video_id}` - Character/object detection (background)
- `POST /api/explain-scene/{video_id}/{scene_index}?frame_interval_ms=500&regenerate=true`

### Frontend Controls

**Main Processing:**
- File input (video file selector)
- "Shots per scene" input (number 1-50, default 3)
- "Show shots" checkbox (toggle shots display, default unchecked)
- Transcription provider selector (Whisper/AssemblyAI radios)
- Process button (initiates pipeline)
- Clear All button (delete all outputs)

**After Processing:**
- Shots section (if enabled) with video clips
- Scenes section with:
  - Transcript button (modal with full scene transcript)
  - Frame rate selector (5/s, 2/s, 1/s, 0.5/s)
  - Explain scene button (AI analysis with narration)
- Full Transcript button (complete video transcription modal)

## Configuration

### Environment Variables

```bash
# Transcription Provider
TRANSCRIPTION_PROVIDER=whisper          # or "assemblyai"
ASSEMBLYAI_API_KEY=sk_...              # Required for AssemblyAI

# Whisper (if using Whisper provider)
WHISPER_MODEL=base                      # tiny, small, medium, large
WHISPER_DEVICE=cpu                      # cpu, cuda, mps
WHISPER_COMPUTE_TYPE_CPU=int8           # int8, float32
WHISPER_COMPUTE_TYPE_GPU=float16        # float16, int8

# Speaker Diarization (Whisper only)
HF_TOKEN=hf_...                         # Hugging Face token for pyannote

# OpenAI (for explanations)
OPENAI_API_KEY=sk-...                   # Required for scene explanation
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_EXPLAIN_MODEL=gpt-4o-mini
OPENAI_TTS_MODEL=tts-1
OPENAI_TTS_VOICE=alloy

# Scene Detection
CONTENT_THRESHOLD=27.0                  # Lower = more shots

# Video Context (Character detection)
CHARACTER_CONFIDENCE_THRESHOLD_PERCENT=70
```

## Performance Notes

| Task | Provider | Time (1 min video) | Cost |
|------|----------|-------------------|------|
| Shot Detection | SceneDetect | ~5-10s | Free |
| Full Transcription | Whisper (base) | ~30-60s | Free (local) |
| Full Transcription | AssemblyAI | ~15s | ~$0.20 |
| Scene Explanation | OpenAI GPT-4o-mini | ~5-10s | ~$0.01 |
| Character Detection | YOLOv8 | ~20-30s | Free (local) |

**GPU Optimization:**
- Whisper: 2-3x faster with CUDA/MPS
- YOLOv8: 3-5x faster with CUDA/MPS
- OpenAI API: same speed (remote)

## Error Handling

**Missing AssemblyAI API Key:**
- Returns HTTP 503 with helpful error message
- Other providers still work
- Can select Whisper as fallback

**Missing OpenAI API Key:**
- Shot/scene processing works fine
- Explanation feature returns HTTP 503
- Transcription unaffected

**Transcription Failure:**
- Returns HTTP 502 with error details
- Processing stops gracefully
- No partial results saved

**Video Analysis Failure:**
- Non-blocking, won't fail main processing
- Scene explanations work without context hints
- Video processes normally otherwise

## Future Enhancements

Potential additions (not implemented):
- Rename speakers manually in transcripts
- Search/filter transcripts by speaker or keyword
- Export full transcript with scene markers
- Speaker profile tracking across multiple videos
- Custom speaker naming instead of "Speaker 1"
- Confidence scores per segment
- Alternative language support per segment
