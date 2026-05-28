# Enhanced Transcript Feature

## Overview

The Enhanced Transcript feature uses a second AI model to enrich scene transcripts with visual and contextual details derived from video analysis. This helps narration audiences better understand the full context of what's being said in a scene.

---

## How It Works

### Data Flow

```
User clicks "Enhance Transcript" button
    │
    ▼
POST /api/enhance-transcript/{video_id}/{scene_index}
    │
    ├─→ Load scene transcript (audio diarization with timestamps)
    │
    ├─→ Load grounded summary (visual context from frames)
    │
    ├─→ Load video context (character & object detection)
    │
    ├─→ Call OpenAI with second AI model
    │   • System: SCENE_TRANSCRIPT_ENHANCEMENT_SYSTEM_PROMPT
    │   • Input: All three data sources
    │   • Temperature: 0.3 (low creativity, factual only)
    │
    ├─→ Cache result: scene_001_enhanced_transcript.txt
    │
    ├─→ Generate TTS: scene_001_enhanced_transcript.mp3
    │
    └─→ Return JSON with enhanced text + audio URL
        │
        ├─→ Display in enhanced transcript section
        │
        └─→ Allow comparison with original
```

---

## What Gets Enhanced

The enhanced transcript adds:

### 1. **Speaker Identification**
- **Original:** `[Speaker 1] Excuse me, do you mind...`
- **Enhanced:** `[Female/Relaxed] Excuse me, do you mind...`
- Uses video context (character detection) + inferences from grounded summary

### 2. **Actions & Gestures**
- **Original:** `Free country. What's that?`
- **Enhanced:** `Free country. What's that? [rushes in with cool black box, playful grin]`
- Derived from grounded summary video analysis

### 3. **Objects & Items**
- What speakers are holding, carrying, or using
- Linked to detected objects from YOLOv8
- Provides continuity with earlier scenes

### 4. **Scene Context**
- Location/setting details: `[in modern office]`, `[at conference table]`
- Environmental details: `[morning sunlight]`, `[crowded room]`
- Mood/atmosphere clues

---

## Backend Implementation

### New Files/Functions

#### `app/prompts.py`
- `SCENE_TRANSCRIPT_ENHANCEMENT_SYSTEM_PROMPT` — Detailed instructions for enhancement
- `build_transcript_enhancement_user_text()` — Formats input data for the model

#### `app/main.py`
- `_format_transcript_segments(transcript_data: dict) -> str` — Formats transcript dict to readable text
- `_openai_enhance_scene_transcript()` — Calls OpenAI API for enhancement
- `POST /api/enhance-transcript/{video_id}/{scene_index}` — Main endpoint

### Configuration

All settings available in `app/config.py`:

```python
ENHANCEMENT_FEATURE_ENABLED = True
ENHANCEMENT_MODEL = OPENAI_EXPLAIN_MODEL  # Uses same model as explanations
ENHANCEMENT_TEMPERATURE = 0.3  # Low creativity for factual additions
ENHANCEMENT_TIMEOUT = 30  # API timeout
ENHANCEMENT_MAX_TOKENS = 2000  # Max response length
```

### Endpoint Details

**URL:** `POST /api/enhance-transcript/{video_id}/{scene_index}`

**Query Parameters:**
- `regenerate` (optional, default: false) — Force regeneration even if cached

**Response:**
```json
{
  "enhanced_transcript": "string with enhanced text",
  "enhanced_transcript_audio_url": "/api/explanations/...",
  "cached": boolean
}
```

**Error Codes:**
- `400` — Grounded summary not generated (run explanation first)
- `503` — API key not configured
- `500` — Enhancement failed (see error message)

---

## Frontend Implementation

### UI Components

#### Enhanced Transcript Section (in Scene Explanation Modal)

```
┌─ Enhanced Transcript ────────────────────────────────┐
│ [Compare] [Download] [Regenerate]                   │
├─────────────────────────────────────────────────────┤
│ [Single View] (default)                             │
│ [6.8s–9.7s] [Female/Relaxed] Excuse me...           │
│ [sitting calmly on bench]                           │
└─────────────────────────────────────────────────────┘
```

#### Comparison View (toggled via "Compare" button)

```
┌────────────────────────┬────────────────────────┐
│  Original Transcript   │  Enhanced Transcript   │
├────────────────────────┼────────────────────────┤
│ [6.8s–9.7s]            │ [6.8s–9.7s]            │
│ [Speaker 1]            │ [Female/Relaxed]       │
│ Excuse me...           │ Excuse me...           │
│                        │ [sitting calmly]       │
│                        │                        │
│ [11.7s–20.9s]          │ [11.7s–20.9s]          │
│ [Speaker 2]            │ [Male/Eager]           │
│ Free country...        │ Free country...        │
│                        │ [rushes in with box]   │
└────────────────────────┴────────────────────────┘
```

### JavaScript Functions

#### Main Functions
- `_fetchEnhancedTranscript(videoId, scene, regenerate)` — Fetches enhanced transcript from API
- `_populateComparisonView(scene, enhancedData)` — Populates side-by-side comparison views
- Toggle comparison view: `toggleComparisonBtn.addEventListener("click", ...)`

#### Download
- User can download enhanced transcript as `.txt` file
- Filename: `enhanced-transcript-scene-{index}.txt`

#### Audio
- Automatic TTS narration generation
- Audio player shows enhanced narration alongside text

---

## User Workflow

### 1. **Explanation Phase** (existing)
   1. Click "Explain Scene" button
   2. AI analyzes frames + dialogue → generates explanation
   3. Summary and grounded summary auto-generate
   4. Modal shows all three with audio narrations

### 2. **Enhancement Phase** (new, on-demand)
   1. Click "Enhance Transcript" button
   2. Second AI model enriches transcript with visual context
   3. Loading spinner shows: "Enhancing transcript with visual context…"
   4. Enhanced text appears in new section
   5. Audio narration generates automatically

### 3. **Comparison** (optional)
   1. Click "Compare" button
   2. Switch to side-by-side view
   3. Original on left, enhanced on right
   4. Easy to see what was added
   5. Click "Single View" to return

### 4. **Download**
   1. Click "Download" to save enhanced transcript as `.txt`
   2. File includes all enhancements

---

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Load data | < 1s | Scene transcript + grounded summary |
| Enhancement API call | 5–15s | OpenAI API latency |
| TTS generation | 3–10s | Per-text audio narration |
| Comparison view | < 1s | Loads original transcript from API |
| **Total** | **10–30s** | Per-scene, on-demand |

All results cached after first generation.

---

## Error Handling

### Common Errors & Solutions

#### 1. "Grounded summary not generated"
- **Cause:** User clicked enhance before generating explanation
- **Fix:** Generate scene explanation first (which auto-generates grounded summary)

#### 2. "OPENAI_API_KEY not configured"
- **Cause:** No API key in `.env` or environment
- **Fix:** Add `OPENAI_API_KEY=sk-...` to `.env`, restart app

#### 3. "Enhancement failed: [error message]"
- **Cause:** API timeout, rate limit, or service issue
- **Fix:** Retry after a few seconds; check API status

#### 4. "Could not load original: [error]" (in comparison view)
- **Cause:** Scene transcript file missing or corrupted
- **Fix:** Regenerate explanation (will recreate transcript files)

---

## Configuration Options

### Via Environment Variables

```bash
# Use different model for enhancements (optional)
export ENHANCEMENT_MODEL=gpt-4-turbo

# Adjust temperature (0.0 = deterministic, 1.0 = creative)
export ENHANCEMENT_TEMPERATURE=0.2

# Set different timeout
export ENHANCEMENT_TIMEOUT=45
```

### Via `app/config.py`

```python
# Master toggle
ENHANCEMENT_FEATURE_ENABLED = True

# Model selection
ENHANCEMENT_MODEL = OPENAI_EXPLAIN_MODEL  # Reuses explanation model

# API parameters
ENHANCEMENT_TEMPERATURE = 0.3  # Low for factual enhancements
ENHANCEMENT_TIMEOUT = 30
ENHANCEMENT_MAX_TOKENS = 2000
```

---

## Caching Strategy

**Enhanced transcripts are cached after first generation:**

```
outputs/{video_id}/explanations/
  scene_001_enhanced_transcript.txt   # Enhanced text (cached)
  scene_001_enhanced_transcript.mp3   # TTS audio (cached)
```

**Refresh strategy:**
- Click "Regenerate" button to force re-generation
- Caching prevents duplicate API calls for same scene
- Caching improves performance for modal toggling

---

## Data Privacy

### What Gets Sent to OpenAI

1. **Scene transcript** — User's own video content
2. **Grounded summary** — AI-generated description (no raw video)
3. **Video context** — Detection metadata only (no images)

**What does NOT get sent:**
- Raw video frames
- Full video context
- Entire transcript (only scene segments)

### Compliance

- Respects `OPENAI_API_KEY` requirement (user controls whether to use API)
- All data processed through OpenAI's official API
- Results stored locally in `outputs/` directory
- No external services beyond OpenAI

---

## Future Enhancements

1. **Detail level selector** — Allow users to choose minimal/balanced/detailed enhancement
2. **Custom prompts** — Let users provide custom enhancement instructions
3. **Character naming** — Allow users to assign names to detected characters
4. **Multi-language** — Enhance transcripts in user's preferred language
5. **Diff highlighting** — Show exactly what changed in comparison view
6. **Batch enhancement** — Enhance all scenes of a video at once
7. **Alternative models** — Support Claude API or other LLM providers

---

## Testing

### Manual Testing Steps

1. **Upload and process video**
   - Upload a video file
   - Run scene detection + transcription
   - Generate explanation for a scene

2. **Test enhancement**
   - Click "Enhance Transcript" button
   - Wait for completion
   - Verify enhanced text appears with contextual details

3. **Test comparison**
   - Click "Compare" button
   - Verify original and enhanced side-by-side
   - Click "Single View" to toggle back

4. **Test audio**
   - Verify narration audio plays
   - Check audio URL in browser dev tools

5. **Test caching**
   - Click "Enhance Transcript" again
   - Verify instant return (cached=true in response)

6. **Test download**
   - Click "Download" button
   - Verify file downloads as `enhanced-transcript-scene-001.txt`

### API Testing

```bash
# Test endpoint directly
curl -X POST "http://localhost:8000/api/enhance-transcript/{video_id}/1"

# With regeneration
curl -X POST "http://localhost:8000/api/enhance-transcript/{video_id}/1?regenerate=true"

# Check response structure
curl -s -X POST "http://localhost:8000/api/enhance-transcript/{video_id}/1" | jq '.enhanced_transcript'
```

---

## Troubleshooting

### Enhancement button disabled
- **Cause:** Explanation still generating
- **Fix:** Wait for explanation to complete

### "Enhance Transcript" button missing
- **Cause:** Feature disabled in config or browser cache
- **Fix:** Check `ENHANCEMENT_FEATURE_ENABLED`, clear cache

### Comparison view won't show
- **Cause:** Original transcript API failed
- **Fix:** Check browser console for network errors
- **Solution:** Make sure video was fully processed

### TTS audio won't play
- **Cause:** Missing `OPENAI_TTS_API_KEY` or generation failed
- **Fix:** Check if API key is configured; see error in browser console

### Large memory usage
- **Cause:** Comparison view loading many scenes at once
- **Fix:** Toggle comparison off when not needed

---

## Bug Fixes Applied

### Fixed Issues

1. **500 Error on Enhancement Click** ✓
   - **Root Cause:** `_scene_transcript_block()` called with dict instead of Path
   - **Fix:** Created `_format_transcript_segments()` for dict inputs
   - **Commit:** Fixed function signature mismatch

2. **Wrong Config Variable Name** ✓
   - **Root Cause:** Used `OPENAI_EXPLANATION_MODEL` instead of `OPENAI_EXPLAIN_MODEL`
   - **Fix:** Replaced all occurrences with correct variable name
   - **Commit:** Fixed config reference

---

## Summary

The Enhanced Transcript feature successfully augments scene transcripts with visual and contextual details from video analysis. Users can:

- ✓ Click "Enhance Transcript" for on-demand enrichment
- ✓ View enhanced text with speaker IDs, actions, objects
- ✓ Compare original vs. enhanced side-by-side
- ✓ Download enhanced transcript as text
- ✓ Listen to TTS narration of enhanced text
- ✓ Regenerate to refresh with different data

All features are cached, performant, and fully integrated with the existing explanation workflow.
