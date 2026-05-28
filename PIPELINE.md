# test-clips Pipeline Documentation

## High-Level Overview

**test-clips** is a video analysis web app that breaks videos into scenes, transcribes dialogue, and generates AI-powered explanations with enhanced context and narration.

The pipeline processes a video through multiple stages:

```
                            INPUT: Video File
                                  │
                    ┌─────────────┴──────────────┐
                    │                            │
                    ▼                            ▼
        ┌──────────────────────┐    ┌──────────────────────┐
        │  Extract Full Audio  │    │  Shot Detection      │
        │    audio.wav         │    │ (PySceneDetect)      │
        └──────────┬───────────┘    │ Finds shot boundaries│
                   │                └──────────┬───────────┘
                   │                           │
                   ├───────────────────────────┤
                   │                           │
                   ▼                           ▼
        ┌──────────────────────┐    ┌──────────────────────┐
        │Full Video Transcribe │    │  Scene Grouping      │
        │ • Whisper (default)  │    │ (Merge N shots/scene)│
        │ • AssemblyAI (alt)   │    │ Configurable (3 default)
        │ • Speaker Diarization│    └──────────┬───────────┘
        └──────────┬───────────┘               │
                   │                           │
                   ▼                           ▼
        ┌─────────────────────────────────────────────────┐
        │       Per-Scene Transcript Filtering             │
        │  (Extract segments from full transcription by   │
        │   timestamp ranges matching scene boundaries)   │
        │                                                 │
        │ Output: scene_001_whisper.json or              │
        │         scene_001_assemblyai.json              │
        └────────────────────┬────────────────────────────┘
                             │
         ┌───────────────────┴───────────────────┐
         │                                       │
         ▼                                       ▼
┌─────────────────────────────────┐   ┌──────────────────────────┐
│ Video Context Analysis (ASYNC)  │   │  Scene Explanation       │
│ • YOLOv8 object detection       │   │ (Per-Scene, On-Demand)   │
│ • Face clustering by character  │   │                          │
│ → video_context.json (cached)   │   │ 1. Extract frames        │
└──────────────────┬──────────────┘   │ 2. Send to OpenAI Vision │
                   │                  │ 3. Generate explanation  │
                   │                  │ 4. Generate summaries    │
                   │                  │ 5. Generate narration    │
                   │                  │                          │
                   └──────────┬───────→ (inject character/object)│
                              │        context for continuity    │
                              │                                  │
                              ▼                                  ▼
                    ┌──────────────────────────────────────────────┐
                    │     Cached Scene Explanations                │
                    │  • scene_001.txt (full explanation)          │
                    │  • scene_001.mp3 (narration)                 │
                    │  • scene_001_summary.txt (summary)           │
                    │  • scene_001_summary.mp3 (summary narration) │
                    │  • scene_001_grounded.txt (grounded summary) │
                    │  • scene_001_grounded.mp3 (grounded narration)
                    └────────────┬─────────────────────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────────────────────────┐
                    │  Enhanced Transcript (On-Demand)             │
                    │                                              │
                    │ • Load scene transcript                      │
                    │ • Load grounded summary (visual context)     │
                    │ • Load video context (char/object data)      │
                    │ • Send to second AI model                    │
                    │ • Add speaker ID, actions, objects           │
                    │ • Generate MP3 narration                     │
                    │                                              │
                    │ Output:                                      │
                    │ • scene_001_enhanced_transcript.txt          │
                    │ • scene_001_enhanced_transcript.mp3          │
                    └────────────┬─────────────────────────────────┘
                                 │
                                 ▼
                            ┌──────────┐
                            │ Frontend │
                            │ (Display)│
                            └──────────┘
```

---

## Detailed Stage Breakdown

### Stage 1: Video Upload & Storage

**Endpoint:** `POST /api/upload`

- User selects video file via web UI
- File stored in: `uploads/{filename}`
- Returns: `{id, filename, path_suffix}`

**Output:**
```
uploads/
  video_file.mp4
```

---

### Stage 2: Audio Extraction

**Called by:** `POST /api/process/{video_id}`

- FFmpeg extracts complete audio from video
- Stored once for full-video transcription
- Duration probed for scene timing constraints

**Command:**
```bash
ffmpeg -i video.mp4 -q:a 9 -n audio.wav
```

**Output:**
```
outputs/{video_id}/
  audio.wav
```

---

### Stage 3: Shot Detection

**Called by:** `POST /api/process/{video_id}`

**Process:**
1. Uses PySceneDetect with ContentDetector
2. Detects shot boundaries based on histogram changes
3. Threshold configurable (default 27.0 = ~8-10 shots per minute)
4. Each shot re-encoded using FFmpeg with `-c:v libx264` (not stream copy) to ensure exact frame alignment

**Why re-encode, not stream-copy:**
- Stream copy (`-c copy`) aligns only to keyframes
- Creates black frames or junk before next keyframe → bad UX
- Re-encoding takes longer but guarantees pixel-perfect alignment

**Output:**
```
outputs/{video_id}/shots/
  shot_001.mp4
  shot_002.mp4
  shot_003.mp4
  ...
```

---

### Stage 4: Scene Grouping (Shot Merging)

**Called by:** `POST /api/process/{video_id}?shots_per_scene=3`

**Process:**
1. Merges N consecutive shots into one scene
2. Default: 3 shots per scene (user-configurable)
3. Each scene re-encoded by combining shots with FFmpeg concat protocol

**Example:**
- 9 shots detected
- With `shots_per_scene=3`:
  - Scene 1 = Shots 1–3
  - Scene 2 = Shots 4–6
  - Scene 3 = Shots 7–9

**Output:**
```
outputs/{video_id}/scenes/
  scene_001.mp4  (3 shots concatenated)
  scene_002.mp4  (3 shots concatenated)
  scene_003.mp4  (3 shots concatenated)
```

---

### Stage 5: Full-Video Transcription

**Called by:** `POST /api/process/{video_id}?provider=whisper|assemblyai`

**Provider Options:**

#### Whisper (Local, Free)
- Model: `base`, `small`, `medium`, `large` (configurable)
- Device: CPU, CUDA, or MPS (Apple Silicon)
- Precision: Float32, Float16 (configurable)
- Speaker diarization: Optional via pyannote.audio (requires `HF_TOKEN`)
- Speed: ~2–5 min for 1 hour video (base model, CPU)

#### AssemblyAI (Cloud, Paid)
- Requires `ASSEMBLYAI_API_KEY`
- Built-in speaker diarization
- Higher accuracy, ~per-minute pricing
- Word-level timing included

**Output:**
```
outputs/{video_id}/transcripts/
  full_video_whisper.json      # or assemblyai
  
  # Content:
  {
    "text": "full concatenated dialogue",
    "segments": [
      {
        "start": 0.0,
        "end": 5.123,
        "text": "dialogue text",
        "speaker": "Speaker 1",  # if diarization enabled
        "words": [...]           # if available
      },
      ...
    ],
    "language": "en",
    "language_probability": 0.95,
    "provider": "whisper",
    "diarization": {
      "enabled": true,
      "speaker_count": 2
    }
  }
```

---

### Stage 6: Per-Scene Transcript Filtering

**Called by:** `POST /api/process/{video_id}`

**Process:**
1. Read full-video transcript
2. For each scene:
   - Extract segments where `start >= scene_start AND end <= scene_end`
   - Copy diarization metadata
   - Build scene-specific JSON with `timeline_in_source_video`

**Output:**
```
outputs/{video_id}/transcripts/
  scene_001_whisper.json       # filtered segments for scene 1
  scene_002_whisper.json       # filtered segments for scene 2
  ...
  
  # Each scene file includes:
  {
    "scene_index": 0,
    "timeline_in_source_video": {
      "start": 0.0,
      "end": 23.456
    },
    "segments": [
      {
        "start": 0.5,
        "end": 3.2,
        "text": "...",
        "speaker": "Speaker 1"
      }
    ],
    "text": "concatenated dialogue for this scene",
    "provider": "whisper",
    "diarization": {...},
    "full_video_transcript_file": "full_video_whisper.json"
  }
```

**Display Format (in UI):**
```
[0.5s–3.2s] [Speaker 1] dialogue text
[3.5s–7.8s] [Speaker 2] response text
```

---

### Stage 7: Video Context Analysis (Async, Optional)

**Called by:** `POST /api/process/{video_id}` (non-blocking)

**Process:**
1. Extract sparse frames (default 1 per 2 seconds)
2. Run YOLOv8 object detection on each frame (COCO 80 classes)
3. Cluster faces using histogram descriptors + hierarchical clustering
4. Build character identity map + object tracking map

**Character Detection:**
- Face extraction: OpenCV Haar Cascade
- Clustering threshold: 0.6 (histogram distance)
- Confidence: Average of detections per character
- Display threshold: Only mention if ≥70% confidence

**Object Detection:**
- YOLOv8 Medium model (50–100 MB download)
- Confidence threshold: 0.5 (configurable)
- Classes: COCO 80 classes (person, chair, briefcase, phone, etc.)
- Tracking: IoU-based matching across frames (IoU threshold 0.3)

**Output:**
```
outputs/{video_id}/video_context.json

{
  "characters": [
    {
      "id": 0,
      "name": "unknown",
      "appearances": [5, 12, 23, 45, ...],  # frame indices
      "confidence": 0.87
    }
  ],
  "objects": [
    {
      "id": 0,
      "class_label": "person",
      "appearances": [5, 6, 7, 8, ...],    # continuous track
      "first_scene": null
    },
    {
      "id": 1,
      "class_label": "briefcase",
      "appearances": [12, 34, 56, ...],    # can be non-continuous
      "first_scene": 1
    }
  ]
}
```

**Non-blocking:** If analysis fails, scene explanations still work (without character/object continuity hints).

---

### Stage 8: Scene Explanation (On-Demand, Per-Scene)

**Endpoint:** `POST /api/explain-scene/{video_id}/{scene_index}?frame_interval_ms=500`

**Process:**

#### 8a. Frame Extraction
- Default: 18 frames at 500ms intervals (configurable 200–4000ms)
- FFmpeg extracts JPEGs from scene video
- Base64-encoded for OpenAI Vision API

#### 8b. OpenAI Vision Call
**Model:** `gpt-4o-mini` (configurable)

**Input to AI:**
1. **Full video transcript** (all scenes, for narrative context)
2. **Scene-specific transcript** (filtered segments with timestamps)
3. **Video context** (character + object tracking, if ≥70% confidence)
4. **Frame images** (18 JPEGs, in order)

**System Prompt:** `SCENE_EXPLAIN_SYSTEM_PROMPT`
- Instructs model to:
  - Read full transcript for story context
  - Analyze frames for setting, action, blocking, emotion
  - Use video context to track characters across scenes
  - Generate clear, grounded explanation

**Output:** Explanation text (1000–2000 words)

#### 8c. Generate Summary
**Model:** Same as explanation

**Input:**
- Full explanation (for content)
- Scene duration (seconds)
- Target word count: `max(30, min(duration_s × 2.3, 500))`

**System Prompt:** `SCENE_SUMMARY_SYSTEM_PROMPT`
- Creates time-constrained summary for audio narration
- ~150 words/minute speaking pace
- Preserves emotional tone and key elements

**Output:** Summary text (30–500 words)

#### 8d. Generate Grounded Summary
**Model:** Same as explanation

**Input:**
- Full explanation (context)
- Scene transcript (actual dialogue)
- Duration (seconds)
- Target word count (same calculation)

**System Prompt:** `SCENE_GROUNDED_SUMMARY_SYSTEM_PROMPT`
- Requires ≥50% of words from actual transcript
- Sounds like a friend narrating
- Uses actual character dialogue

**Output:** Grounded summary text (30–500 words, ≥50% from transcript)

#### 8e. Generate TTS Audio (Optional)
**API:** OpenAI TTS

**For each text:**
- Explanation → `scene_001.mp3`
- Summary → `scene_001_summary.mp3`
- Grounded → `scene_001_grounded.mp3`

**Output:**
```
outputs/{video_id}/explanations/
  scene_001.txt
  scene_001.mp3
  scene_001_summary.txt
  scene_001_summary.mp3
  scene_001_grounded.txt
  scene_001_grounded.mp3
```

**Caching:** Once generated, subsequent requests return cached files instantly.

---

### Stage 9: Enhanced Transcript (On-Demand, Per-Scene) ⭐ NEW

**Endpoint:** `POST /api/enhance-transcript/{video_id}/{scene_index}?regenerate=false`

**Process:**

#### 9a. Load Prerequisites
- Scene transcript (from Stage 6)
- Grounded summary (from Stage 8d)
- Video context (from Stage 7, optional)

**Prerequisites check:**
- If grounded summary doesn't exist → 400 error
- User must run explanation first

#### 9b. Second AI Model Call
**Model:** `gpt-4o-mini` (same or separate, configurable)

**Input:**
1. **Scene Transcript (Audio Diarization)**
   - Timestamps and generic speaker labels ("Speaker 1", "Speaker 2")
   
2. **Grounded Summary (Visual Analysis + Frames)**
   - What actually happens in the scene visually
   - Actions, expressions, objects visible
   
3. **Video Context (Character & Object Detection)**
   - Character confidence scores
   - Object class labels (COCO)

**System Prompt:** `SCENE_TRANSCRIPT_ENHANCEMENT_SYSTEM_PROMPT`
- Instructs model to enhance transcript with:
  - **Speaker identification**: Infer gender/role from grounded summary, use video context if available
  - **Actions & gestures**: What speakers are doing [stands, walks, points, gestures]
  - **Objects & items**: What speakers are holding [holding briefcase] [with coffee cup]
  - **Scene context**: Location/setting [in modern office] [at conference table]

**Constraints:**
- Keep enhancements MINIMAL and FACTUAL
- Only add what's clearly visible or mentioned
- Do NOT change original dialogue text
- Do NOT add vague or speculative details
- Preserve ALL timestamps exactly
- Temperature 0.3 (very low creativity)

**Example Output:**
```
Original:  [45.2s–48.5s] [Speaker 1] I think we need to pause here.
Enhanced:  [45.2s–48.5s] [Male/Manager] I think we need to pause here. [taps folder on table]

Original:  [50.1s–52.3s] [Speaker 2] Absolutely, good idea.
Enhanced:  [50.1s–52.3s] [Female/Analyst] Absolutely, good idea. [nods, glancing at laptop screen]
```

#### 9c. Cache Result
```
outputs/{video_id}/explanations/
  scene_001_enhanced_transcript.txt
```

#### 9d. Generate TTS Audio
**API:** OpenAI TTS

```
outputs/{video_id}/explanations/
  scene_001_enhanced_transcript.mp3
```

**Output Response:**
```json
{
  "enhanced_transcript": "enhanced text with contextual details",
  "enhanced_transcript_audio_url": "/api/explanations/{video_id}/scene_001_enhanced_transcript.mp3",
  "cached": false
}
```

---

## Complete Data Flow Diagram

```
UPLOAD
  │
  ├─→ Extract Audio ──────────────┐
  │                               │
  ├─→ Shot Detection              │
  │      │                        │
  │      ├─→ Scene Grouping       │
  │      │      │                 │
  │      │      ▼                 ▼
  │      │   Transcribe (Full Video)
  │      │      │
  │      │      └─→ Per-Scene Filtering
  │      │           │
  │      │           ├─→ Video Context Analysis (async)
  │      │           │
  │      │           └─→ Scene Explanation (on-demand)
  │      │                │
  │      │                ├─→ Summary generation
  │      │                │
  │      │                ├─→ Grounded Summary
  │      │                │
  │      │                └─→ Enhanced Transcript (on-demand) ⭐
  │      │                     │
  │      │                     └─→ TTS Audio for all
  │      │
  │      └─→ Store clips
  │
  └─→ Frontend Display
```

---

## Data Structures

### TranscriptionResult (from both Whisper & AssemblyAI)
```python
{
    "text": str,                    # Full concatenated text
    "segments": [                   # Array of speech segments
        {
            "start": float,         # Timestamp in seconds
            "end": float,           # Timestamp in seconds
            "text": str,            # Segment dialogue
            "speaker": str,         # "Speaker 1", "Speaker 2", etc.
            "words": [              # Word-level timing (if available)
                {
                    "start": float,
                    "end": float,
                    "word": str
                }
            ]
        }
    ],
    "language": str,                # Language code (e.g., "en")
    "language_probability": float,  # Confidence (Whisper only)
    "provider": str,                # "whisper" or "assemblyai"
    "diarization": {
        "enabled": bool,
        "speaker_count": int,
        "note": str
    }
}
```

### VideoContext
```python
{
    "characters": [
        {
            "id": int,
            "name": str,
            "confidence": float,        # 0.0-1.0
            "appearances": [int, ...]   # Frame indices
        }
    ],
    "objects": [
        {
            "id": int,
            "class_label": str,         # COCO class name
            "appearances": [int, ...],  # Frame indices
            "first_scene": int | None
        }
    ]
}
```

---

## Configuration Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `SCENE_DETECTION_THRESHOLD` | 27.0 | ContentDetector threshold (lower = more scenes) |
| `DEFAULT_SHOTS_PER_SCENE` | 3 | Shots to merge per scene |
| `EXPLAIN_MAX_FRAMES` | 18 | Frames to extract per scene |
| `EXPLAIN_FRAME_INTERVAL_MS_DEFAULT` | 500 | Frame sampling interval (ms) |
| `SPARSE_FRAME_INTERVAL_MS` | 2000 | Frame sampling for video context (ms) |
| `FACE_CLUSTERING_THRESHOLD` | 0.6 | Histogram distance threshold |
| `CHARACTER_CONFIDENCE_THRESHOLD_PERCENT` | 70 | Min confidence to mention character names |
| `YOLOV8_CONF_THRESHOLD` | 0.5 | YOLOv8 confidence threshold |
| `SUMMARY_WORDS_PER_SECOND` | 2.3 | ~150 words/minute speaking pace |
| `SUMMARY_MIN_WORDS` | 30 | Minimum summary length |
| `SUMMARY_MAX_WORDS` | 500 | Maximum summary length |
| `TRANSCRIPTION_PROVIDER` | "whisper" | "whisper" or "assemblyai" |
| `OPENAI_EXPLANATION_MODEL` | "gpt-4o-mini" | Model for explanations & enhancement |
| `OPENAI_TTS_MODEL` | "tts-1" | TTS model (tts-1 or tts-1-hd) |

---

## Error Handling

| Error | Cause | Recovery |
|-------|-------|----------|
| `OPENAI_API_KEY` not set | Explanation/enhancement requires key | User sets `.env` or env var, restarts app |
| `ASSEMBLYAI_API_KEY` missing | Selected AssemblyAI but no key | Set key in `.env`, retry process |
| `HF_TOKEN` missing | Requested Whisper with pyannote diarization | Set token (optional), can omit speaker labels |
| Video processing fails | Invalid video format or corrupted file | Upload valid MP4/WebM with audio stream |
| Video context analysis fails | YOLOv8 or face clustering error | Non-blocking; scene explanations still work |
| Scene explanation fails | API timeout or quota exceeded | Retry with smaller frame count or longer interval |
| Enhanced transcript fails | Grounded summary not yet generated | Generate explanation first (creates grounded summary) |

---

## Performance Notes

| Operation | Typical Time | Constraints |
|-----------|--------------|-------------|
| Audio extraction | ~1–3s | Per video |
| Shot detection | ~2–5s | Per video (CPU only) |
| Full transcription (Whisper base) | ~2–5 min | Per video (CPU); faster on GPU |
| Full transcription (AssemblyAI) | ~1–2 min | Per video (API latency) |
| Per-scene filtering | ~100ms | Per scene |
| Video context analysis | ~30–60s | Per video (sparse frames + YOLOv8) |
| Frame extraction | ~1–3s | Per scene |
| Scene explanation | ~10–30s | Per scene (OpenAI API latency) |
| Summary generation | ~5–10s | Per scene (OpenAI API latency) |
| Grounded summary | ~5–10s | Per scene (OpenAI API latency) |
| **Enhanced transcript** | **~5–10s** | **Per scene (OpenAI API latency)** |
| TTS audio (all three) | ~5–15s | Per scene (per-text API calls) |

**Total time:** 30–45 minutes for typical 1-hour video (Whisper + all explanations, CPU)

---

## Storage Layout

```
outputs/
  {video_id}/
    audio.wav                              # Full extracted audio
    video_context.json                     # Character & object tracking
    
    shots/
      shot_001.mp4
      shot_002.mp4
      ...
    
    scenes/
      scene_001.mp4
      scene_002.mp4
      ...
    
    scene_audio/
      scene_001.m4a                        # Extracted from full audio
      scene_002.m4a
      ...
    
    transcripts/
      full_video_whisper.json              # Full transcription
      full_video_assemblyai.json
      scene_001_whisper.json               # Per-scene filtered
      scene_001_assemblyai.json
      scene_002_whisper.json
      ...
    
    explanations/
      scene_001.txt
      scene_001.mp3
      scene_001_summary.txt
      scene_001_summary.mp3
      scene_001_grounded.txt
      scene_001_grounded.mp3
      scene_001_enhanced_transcript.txt    # ⭐ NEW
      scene_001_enhanced_transcript.mp3    # ⭐ NEW
      ...
```

---

## Key Design Decisions

1. **Full-Video Transcription First** — Transcribe entire video once to preserve speaker diarization context across scenes, rather than transcribing per-scene (which loses speaker continuity).

2. **Re-encode vs. Stream Copy** — Re-encode video clips with `-c:v libx264` instead of stream copy (`-c copy`) to ensure exact frame-boundary alignment. Stream copy aligns only to keyframes, causing visible junk/black frames.

3. **Video Context Async, Non-blocking** — Character/object detection runs after processing completes. If it fails, scene explanations still work without continuity hints.

4. **Explanation Caching** — Cache all three text outputs (explanation, summary, grounded) and optional TTS audio. Subsequent requests return cached results instantly.

5. **Enhanced Transcript On-Demand** — Don't run automatically (keeps initial explanation fast). User clicks "Enhance Transcript" button. Uses second AI model with low temperature (0.3) for factual additions only.

6. **Sparse Frame Sampling** — Video context uses 1 frame per 2 seconds (2000ms); explanation uses configurable interval (default 500ms, user-selectable 200–4000ms) for different detail levels.

---

## Future Enhancements

- Multi-language support (transcription + explanation in different languages)
- Speaker identity linking (ask user to name detected characters)
- Custom character/object filtering (show only relevant entities)
- Batch processing (upload multiple videos)
- Real-time streaming transcription
- Custom AI model endpoint support
- Enhanced transcript diff view (highlight what changed from original)
- Confidence scores for each enhancement type
