# Scene Explanation Context - What the AI Model Receives

## Overview

When a user clicks "Explain scene", the application sends a comprehensive request to the OpenAI API with the following context layers:

---

## 1. **System Prompt** (defines the AI's role and instructions)

```
You are a film and media analyst helping viewers understand a scene.

Work in two clear mental steps:
1) First, read the **full-film transcript** (all scenes' dialogue together). 
   Build a loose sense of ongoing story, tone, and recurring themes—only as 
   far as the text supports.
   
2) Then focus on **this scene only**, using:
   - The **scene-specific transcript** (dialogue and speakers if tagged),
   - The **video frames** provided in order (each from a moment in the scene). 
     From the frames, infer **setting, action, blocking, cinematography**, 
     and **facial/body language** to reason about **mood and emotion** 
     (e.g. tension, joy, fear). Be honest when unclear.

Your task: write a clear **scene explanation** for the user—what is happening, 
why it matters in context, and how it feels emotionally. When you infer emotion 
from visuals, say what you see (expressions, posture, staging) that supports 
it. Do not invent plot facts that contradict the transcripts; do not name 
actors unless they appear in the text.
```

**Purpose:** Instructs the AI to act as a film analyst, use both text and visual context, and provide emotionally grounded analysis.

---

## 2. **Full Video Transcript** (entire video context)

### What's included:
- **All scenes' dialogues** concatenated together
- **Format:** `=== Scene N ===` followed by dialogue
- **Speaker tags:** `[Speaker Name] dialogue text` if available
- **Limit:** Truncated to ~16,000 characters (configurable via `EXPLAIN_FULL_TRANSCRIPT_MAX_CHARS`)

### Example:
```
=== Scene 1 ===
[Character A] Excuse me.
[Character B] What?

=== Scene 2 ===
[Character A] Do you want to go somewhere?
[Character B] Not really.
```

**Purpose:** Gives the AI understanding of the overall story, character relationships, and narrative context across the entire video.

---

## 3. **Scene-Specific Transcript** (focused dialogue for current scene)

### What's included:
- **Only dialogue from this scene**
- **Format:** `[start_time–end_time] [Speaker] dialogue text`
- **Timing:** Precise timestamps for each dialogue segment

### Example:
```
[0.0s–1.2s] [Character A] Excuse me.
[1.2s–2.5s] [Character B] What?
[2.5s–4.0s] [Character A] Do you mind if I uh, for your country?
[4.0s–5.0s] [Character B] What's that?
[5.0s–6.0s] [Character A] Oh nothing.
```

**Purpose:** Provides precise dialogue content and timing for the specific scene being analyzed.

---

## 4. **Video Frames** (visual context from the scene)

### What's included:
- **Still images** extracted from the video at regular intervals
- **Frame count:** Up to 18 frames (configurable via `EXPLAIN_MAX_FRAMES`)
- **Sampling rate:** User-selectable (default 2 frames/second = 500ms intervals)
  - 5/s (200ms) - Most detailed
  - 2/s (500ms) - Balanced (default)
  - 1/s (1000ms) - Faster
  - 0.5/s (2000ms) - Minimal

### Image properties:
- **Format:** JPEG, Base64 encoded
- **Detail level:** Configurable (low/high via `OPENAI_IMAGE_DETAIL`)
  - `low` - Faster processing, less token cost
  - `high` - More visual detail, more token cost

### Example metadata:
```
There are 10 still images, sampled about every 500 ms through this scene, 
in order. Use them for setting, action, faces, body language, and emotional tone.
```

**Purpose:** Provides visual evidence of setting, character actions, blocking, facial expressions, and emotional cues that aren't captured in dialogue alone.

---

## 5. **Configuration Metadata** (sent in request)

### Information provided:
- **Frame interval:** e.g., "sampled about every 500ms"
- **Total frame count:** e.g., "There are 10 still images"
- **Processing model:** e.g., "gpt-4o-mini"
- **Image detail level:** "low" or "high"

---

## Data Flow Example

When explaining Scene 1 with 2/s sampling (500ms intervals):

```
REQUEST TO OPENAI API:
{
  "model": "gpt-4o-mini",
  "max_tokens": 1200,
  "messages": [
    {
      "role": "system",
      "content": "[SYSTEM PROMPT above]"
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "## Full video transcript (all scenes — context)\n\n[FULL TRANSCRIPT]\n\n## This scene — transcript (focus)\n\n[SCENE TRANSCRIPT]\n\n## Video frames\n\nThere are 10 still images, sampled about every 500 ms through this scene..."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[FRAME 1 BASE64]",
            "detail": "low"
          }
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[FRAME 2 BASE64]",
            "detail": "low"
          }
        },
        ... [more frames] ...
      ]
    }
  ]
}
```

---

## Configuration Controls

These settings affect what context is sent to the AI:

### In `app/config.py`:

| Setting | Default | Controls | Effect |
|---------|---------|----------|--------|
| `EXPLAIN_MAX_FRAMES` | 18 | Frame extraction | How many frames to send to AI |
| `EXPLAIN_FULL_TRANSCRIPT_MAX_CHARS` | 16,000 | Transcript truncation | How much full-video context to include |
| `OPENAI_IMAGE_DETAIL` | "low" | Vision detail level | Visual analysis depth (token cost) |
| `EXPLAIN_FRAME_INTERVAL_MS_DEFAULT` | 500ms | Frame sampling | Default frames per second |
| `OPENAI_EXPLAIN_MODEL` | "gpt-4o-mini" | Model selection | AI model used for analysis |

### User Controls:

| Control | Options | Default | Effect |
|---------|---------|---------|--------|
| Frame sampling selector | 5/s, 2/s, 1/s, 0.5/s | 2/s | Frames sent per scene second |
| Regenerate with new settings | Any frame rate | 2/s | Re-analyze with different visual context |

---

## Summary: What the AI "Sees"

The AI model receives **three layers of context**:

1. **Narrative Context** 
   - Full video transcript (story, relationships, themes)
   - Scene-specific dialogue with timestamps

2. **Visual Context** 
   - Up to 18 stillframes from the scene
   - Base64-encoded JPEG images
   - Ordered chronologically through the scene

3. **Task Context** 
   - System prompt defining its role as film analyst
   - Configuration (frame count, sampling rate, detail level)
   - Instructions to use both text and visual analysis

---

## Constraints & Trade-offs

| Aspect | Benefit | Cost |
|--------|---------|------|
| **More frames** (5/s) | Better visual detail | Slower, higher token usage |
| **Fewer frames** (0.5/s) | Faster, lower cost | Less visual context |
| **High image detail** | Better visual analysis | Higher token usage |
| **Low image detail** | Lower token usage | Less visual precision |
| **Full transcript** (16K chars) | Better narrative context | Higher token usage |
| **Truncated transcript** | Lower token usage | Might miss story context |

---

## What the AI Does NOT Receive

- ❌ Audio/music information
- ❌ Scene duration or pacing information
- ❌ Color information (frames are JPEG)
- ❌ Video quality metadata
- ❌ Production notes or metadata
- ❌ Actor names (unless in transcript)
- ❌ Scene number or description from user
- ❌ Prior AI analysis (fresh analysis each time)

---

## Example Output

Based on all this context, the AI generates a scene explanation including:

- **Setting & Cinematography:** "The bright park setting with lush greenery..."
- **Action & Blocking:** "The man stands while the woman remains seated..."
- **Dialogue Analysis:** "The awkward exchange about 'for your country'..."
- **Emotional Tone:** "Light-hearted yet uncertain atmosphere..."
- **Visual Evidence:** "Her neutral expression, his forward posture suggests..."
- **Narrative Significance:** "This scene establishes character dynamics..."
