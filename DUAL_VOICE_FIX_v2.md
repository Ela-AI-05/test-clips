# Dual-Voice TTS Fix - Version 2

## Problem
Enhanced transcript audio was generating "flat" audio where both voices sounded identical, making it impossible to distinguish between dialogue and enhancements.

## Root Cause
- **Voice similarity:** Using "shimmer" for both dialogue and enhancement voices - too similar to "alloy" for clear distinction
- **No auditory break:** Both voices ran back-to-back without pause, so voice change wasn't perceptible

## Solution (v2)

### 1. Added Silence Gaps (0.5s)
New function `_create_silence_mp3()` generates silence using FFmpeg:
```
[Dialogue in alloy] → [0.5s silence] → [Enhancement in nova]
                       ↑ Creates clear auditory break
```

The silence lets listeners perceive the voice change more clearly.

### 2. Changed Enhancement Voice to "nova"
Changed default from `shimmer` to `nova`:
- **alloy** = neutral, balanced (dialogue voice)
- **nova** = bright, energetic (enhancement voice)
- Much more distinctive when separated by silence

### 3. Implementation Changes

**File: `app/main.py`**
- Added `_create_silence_mp3(duration_seconds, output_path)` function (~20 lines)
- Modified `_openai_tts_to_mp3_dual_voice()` to:
  - Generate dialogue TTS
  - Generate 0.5s silence
  - Generate enhancement TTS
  - Merge in order: dialogue → silence → enhancement

**File: `app/config.py`**
- Changed `OPENAI_TTS_ENHANCEMENT_VOICE` default from "shimmer" → "nova"

## Testing

**Server:** Restarted with updated code ✓  
**Unit tests:** All voice/silence/merge operations pass ✓  
**Integration:** Enhanced transcript regenerated with new dual-voice logic ✓  

## Audio Result

Per segment, listener hears:
1. Clear dialogue in neutral voice (alloy, 1.0x speed)
2. Brief silence (signals voice change coming)
3. Bright enhancement in energetic voice (nova, 0.85x speed)

This is audibly distinct and easy to perceive.

## How to Test

1. Refresh browser
2. Click "Enhance Transcript" button
3. Listen to the generated audio
4. You should now hear:
   - Dialogue in strong, neutral voice
   - A slight pause
   - Enhancements in a bright, different voice

## Customization

Change enhancement voice in `.env`:
```bash
OPENAI_TTS_ENHANCEMENT_VOICE=onyx   # Deep voice
OPENAI_TTS_ENHANCEMENT_VOICE=fable  # Story-like
```

Or adjust speed:
```bash
OPENAI_TTS_ENHANCEMENT_SPEED=0.9   # Faster
OPENAI_TTS_ENHANCEMENT_SPEED=0.8   # Even slower
```

## Summary

✅ Silence gaps provide clear auditory breaks  
✅ Nova voice is distinctly different from alloy  
✅ Combined = clear perception of two distinct voices  
✅ User can hear enhancements as separate content layer  
