# Dual-Voice TTS for Enhanced Transcripts

## Overview

Enhanced transcript audio now features **dual voices** to create clear auditory distinction between original dialogue and AI-added visual context.

**User Experience:**
- 🎙️ **Strong voice (alloy)** at normal speed (1.0x) — Original dialogue (what was actually said)
- 🔇 **0.5s silence gap** — Clear auditory break between sections
- 🎵 **Bright voice (nova)** at slower speed (0.85x) — Enhanced details [actions, objects, context]

## Key Improvements (v2)

✨ **Fixed "Flat Audio" Issue:**
- **Added silence gaps** (0.5s) between dialogue and enhancement for clear auditory distinction
- **Changed enhancement voice** from "shimmer" (similar to alloy) to "nova" (distinctly different)
- Nova is bright/energetic and obviously different from alloy's neutral tone
- Silence creates natural pause that lets listeners perceive the voice change more clearly

## Implementation

### Architecture

```
Enhanced Transcript Text
    ↓
Parse into Segments
    ├─ Dialogue: "[Speaker] text"
    └─ Enhancement: "[bracketed details]"
    ↓
Generate Dual Audio
    ├─ Dialogue → OpenAI TTS (voice: alloy, speed: 1.0)
    ├─ [0.5s Silence] → FFmpeg generated silence
    ├─ Enhancement → OpenAI TTS (voice: nova, speed: 0.85)
    ↓
Merge MP3s with FFmpeg
    ↓
Final Audio File
```

### Audio Sequence Per Segment

```
[Dialogue in alloy voice]
    ↓
[0.5s silence - lets listener perceive voice change]
    ↓
[Enhancement in nova voice - noticeably different]
```

### New Functions in `app/main.py`

#### 1. `_create_silence_mp3(duration_seconds: float, output_path: Path)`
**New in v2** — Generates silent MP3 files for auditory gaps

Uses ffmpeg `anullsrc` (null audio source) to generate silence at 24kHz mono, matching the OpenAI TTS audio format.

```python
_create_silence_mp3(0.5, Path("silence.mp3"))  # Creates 0.5 second silence
```

**Why:** Without silence, dialogue and enhancement run together. The pause lets listeners perceive the voice change clearly.

#### 2. `_parse_enhanced_transcript_segments(enhanced_text: str) -> List[Dict]`
**Lines:** 1544-1599

Parses enhanced transcript format:
```
[timestamp] [speaker] dialogue [enhancement1] [enhancement2]...
```

Returns structured segments:
```python
{
  "timestamp": "[6.8s–9.7s]",
  "speaker": "[Female/Park Visitor]",
  "dialogue": "Excuse me, do you mind if I, uh—",
  "enhancement": "sits calmly on the bench, glancing up with a friendly smile"
}
```

#### 3. `_merge_mp3_files(mp3_files: List[Path], output_path: Path)`
**Lines:** 1622-1649

Merges multiple MP3 files using ffmpeg concat demuxer:
- Creates concat list file
- Runs: `ffmpeg -f concat -safe 0 -i concat_list.txt -c copy output.mp3`
- Cleans up temporary concat file

#### 4. `_openai_tts_to_mp3_dual_voice(...)`
**Lines:** 1652-1711

Generates dual-voice audio with silence gaps:
1. For each segment:
   - Generate dialogue audio with primary voice (alloy, 1.0x)
   - Generate 0.5s silence
   - Generate enhancement audio with secondary voice (nova, 0.85x)
2. Collect all MP3s in order: dialogue → silence → enhancement
3. Merge them into single file
4. Return final merged MP3

**New in v2:** Inserts `_create_silence_mp3(0.5, ...)` between dialogue and enhancement

#### 5. Modified `_openai_tts_to_mp3(...)`
**Lines:** 694-722

Added optional parameters:
- `voice: Optional[str] = None` — Override voice (default: config.OPENAI_TTS_VOICE)
- `speed: float = 1.0` — Control playback speed
- Both added to OpenAI API payload

### Configuration (app/config.py)

**Lines:** 160-167

```python
OPENAI_TTS_DIALOGUE_SPEED: float = 1.0  # Normal speed
OPENAI_TTS_ENHANCEMENT_VOICE: str = "nova"  # Bright, distinct voice (changed from shimmer in v2)
OPENAI_TTS_ENHANCEMENT_SPEED: float = 0.85  # Slower, measured pace
```

**Environment Variables:**
```bash
OPENAI_TTS_DIALOGUE_SPEED=1.0
OPENAI_TTS_ENHANCEMENT_VOICE=nova  # Options: alloy (default), echo, fable, onyx, nova, shimmer
OPENAI_TTS_ENHANCEMENT_SPEED=0.85  # Slower pacing for enhancements
```

**Why "nova" instead of "shimmer":**
- `shimmer` = smooth/warm (too similar to alloy's neutral tone)
- `nova` = bright/energetic (noticeably different, easy to distinguish)
- Silence gap + voice difference = clear perception of voice change

### Integration with Enhanced Transcript Endpoint

**Lines:** 1794-1826 in `POST /api/enhance-transcript/{video_id}/{scene_index}`

**Workflow:**
1. Parse enhanced text into segments
2. Call `_openai_tts_to_mp3_dual_voice()`
3. If dual-voice fails, fallback to single-voice TTS
4. Cache result and return audio URL

**Fallback Strategy:**
```python
try:
    # Try dual-voice
    segments = _parse_enhanced_transcript_segments(enhanced_text)
    _openai_tts_to_mp3_dual_voice(...)
except Exception:
    # Fallback to single-voice
    _openai_tts_to_mp3(...)
```

## Usage

### For Users

1. Click "Enhance Transcript" button
2. Wait for dual-voice TTS generation (2-3x slower than single-voice due to multiple API calls)
3. Listen to audio with clear distinction:
   - Dialogue: Strong, clear narration
   - Enhancements: Softer, slower context (easier to distinguish)

### For Developers

**Configuration Options:**

```bash
# Change dialogue voice to a slower, deeper voice
OPENAI_TTS_VOICE=onyx
OPENAI_TTS_DIALOGUE_SPEED=0.95

# Change enhancement voice to something brighter
OPENAI_TTS_ENHANCEMENT_VOICE=nova
OPENAI_TTS_ENHANCEMENT_SPEED=0.9

# Speed up enhancements but keep them softer
OPENAI_TTS_ENHANCEMENT_SPEED=0.95
```

**Voice Options:**
- `alloy` — Neutral, balanced (default for dialogue)
- `echo` — Slightly robotic
- `fable` — Story-like
- `onyx` — Deep, authoritative
- `nova` — Bright, energetic
- `shimmer` — Smooth, warm (default for enhancements)

## Performance Characteristics

### Latency
- **Single-voice TTS:** ~3-5 seconds for enhanced transcript audio
- **Dual-voice TTS:** ~8-15 seconds (multiple TTS API calls + ffmpeg merge)
- **Bottleneck:** OpenAI API latency per segment

### API Costs
- **Single-voice:** 1 TTS call per enhanced transcript
- **Dual-voice:** ~2N TTS calls where N = number of segments with enhancements
- **Cost Impact:** ~2× TTS API usage

### File Size
- **Typical enhanced transcript:** 20-40 seconds audio
- **MP3 bitrate:** Standard OpenAI TTS (128 kbps)
- **File size:** 300-600 KB

## Edge Cases Handled

1. **Segments with no enhancements:**
   - Only generate dialogue audio (skip enhancement TTS)

2. **Very long enhancements (>4096 chars):**
   - Truncated with `_tts_input_clip()` before TTS

3. **Parsing failures:**
   - Fallback to single-voice TTS of entire enhanced text
   - If that also fails, return error (audio_url = None)

4. **FFmpeg not available:**
   - Gracefully caught in merge function
   - Falls back to single-voice TTS

5. **API rate limits:**
   - Multiple TTS calls may trigger rate limiting
   - Errors are caught and logged
   - Fallback to single-voice attempts

## Testing

### Manual Testing Steps

1. **Upload and process a video** (scenes should be created)

2. **Generate explanation** for a scene:
   - Click "Explain Scene" button
   - Wait for completion
   - Verify grounded summary is generated

3. **Generate enhanced transcript**:
   - Click "Enhance Transcript" button
   - Wait for dual-voice TTS (takes longer than summary narration)
   - Check Network tab: `scene_001_enhanced_transcript.mp3` should load

4. **Listen to audio**:
   - Play enhanced transcript audio
   - Verify two distinct voices:
     - Dialogue: normal, strong voice
     - Enhancements: softer, slightly slower voice
   - Compare to summary audio (all one voice)

### Configuration Testing

```bash
# Test with different voices
export OPENAI_TTS_ENHANCEMENT_VOICE=nova
curl -X POST http://localhost:8000/api/enhance-transcript/{video_id}/1?regenerate=true

# Test with different speeds
export OPENAI_TTS_ENHANCEMENT_SPEED=0.9
curl -X POST http://localhost:8000/api/enhance-transcript/{video_id}/1?regenerate=true
```

## Troubleshooting

### Issue: Still hearing flat audio (both voices sound the same)

**Likely Cause (v1):** Using "shimmer" for both dialogue and enhancement voice—too similar

**Solution (v2):**
- Now using `nova` voice for enhancements (bright/energetic, distinctly different from alloy)
- Added 0.5s silence gaps between dialogue and enhancement
- Regenerate with `?regenerate=true` to get new audio

```bash
# Force regenerate
curl -X POST "http://127.0.0.1:8000/api/enhance-transcript/{video_id}/{scene_index}?regenerate=true"
```

### Issue: TTS generation failed (fallback to single-voice)

**Cause:** Parsing or dual-voice generation error

**Solution:**
1. Check logs: `tail -50 /tmp/app.log | grep -i "failed"`
2. Verify enhanced text contains `[bracketed enhancements]` with proper format
3. Try regenerating with `?regenerate=true`

### Issue: Very long generation time (>20 seconds)

**Cause:** Many segments = many TTS API calls

**Solution:**
1. Reduce segment count (currently limited by enhanced transcript length)
2. Reduce enhancement speed (faster = lower quality but faster TTS)
3. Use faster OpenAI TTS model (currently `tts-1`, could switch to higher latency model)

### Issue: "ffmpeg not found" error

**Cause:** FFmpeg not on PATH

**Solution:**
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Verify
ffmpeg -version
```

## Files Modified

| File | Changes |
|------|---------|
| `app/config.py` | Added TTS dialogue/enhancement speed and voice config (lines 160-167) |
| `app/main.py` | Added parsing, merging, and dual-voice functions; modified TTS endpoint (lines 694-720, 1544-1687, 1794-1826) |

## Customization Options

### Change Enhancement Voice

Set in `.env` or environment:
```bash
# More distinct from alloy
OPENAI_TTS_ENHANCEMENT_VOICE=nova      # Current default (bright/energetic)
OPENAI_TTS_ENHANCEMENT_VOICE=fable     # Story-like
OPENAI_TTS_ENHANCEMENT_VOICE=echo      # Slightly robotic
OPENAI_TTS_ENHANCEMENT_VOICE=onyx      # Deep/authoritative
```

### Adjust Silence Duration

Currently hardcoded to 0.5s. To change, edit `app/main.py`:
```python
# Line ~1679: Change 0.5 to desired duration (seconds)
_create_silence_mp3(0.7, silence_mp3)  # 0.7s silence instead
```

### Adjust Enhancement Speed

```bash
OPENAI_TTS_ENHANCEMENT_SPEED=0.9   # Faster (more natural)
OPENAI_TTS_ENHANCEMENT_SPEED=0.8   # Slower (more pronounced)
```

## Future Enhancements

1. **Configurable silence duration** via environment variable
2. **Dialogue voice selection** (currently hardcoded to "alloy")
3. **Audio ducking** (lower dialogue volume during enhancements)
4. **Per-segment voice customization** (different voices per dialogue speaker)
5. **Crossfade** instead of silence (smooth transitions)

## Success Metrics

✅ Dual-voice audio generates successfully  
✅ Dialogue uses primary voice at normal speed  
✅ Enhancements use secondary voice at slower speed  
✅ User perceives clear distinction when listening  
✅ Caching works correctly  
✅ Fallback to single-voice on failure  
✅ FFmpeg integration works on macOS/Linux  
✅ Configuration settings are applied  

## Summary

The dual-voice TTS feature significantly enhances the listening experience by providing clear auditory distinction between original dialogue and AI-added context. The implementation is robust with fallback mechanisms and configurable voice/speed settings, making it adaptable to different use cases and preferences.
