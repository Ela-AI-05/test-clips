# Video Context System - Usage Guide

## Overview

The new two-level context system automatically analyzes videos for character identification and object tracking, providing richer scene explanations with character names and object continuity.

## How to Use

### 1. Upload and Process a Video

```bash
# This is unchanged - use the existing upload/process flow
POST /api/upload                    # Upload video file
POST /api/process/{video_id}        # Process video (shots, scenes, transcripts, context analysis)
```

**Behind the scenes:**
- Video is split into shots and scenes
- Audio is extracted and transcribed
- **NEW:** Video context is automatically analyzed (character/object detection)

### 2. Explicit Video Analysis (Optional)

To explicitly trigger or re-run video analysis:

```bash
POST /api/analyze-video/{video_id}
```

**Response:**
```json
{
  "status": "success",
  "message": "Video analysis completed",
  "characters_count": 2,
  "objects_count": 5,
  "characters": [
    {
      "id": 0,
      "confidence": 0.92,
      "appearance_count": 15
    },
    {
      "id": 1,
      "confidence": 0.78,
      "appearance_count": 10
    }
  ],
  "objects": [
    {
      "id": 0,
      "class": "briefcase",
      "appearance_count": 8
    },
    ...
  ]
}
```

### 3. Get Scene Explanation (Enhanced with Context)

```bash
POST /api/explain-scene/{video_id}/{scene_index}?frame_interval_ms=500
```

**Response includes:**
- Scene explanation (now informed by character/object context)
- Summary and grounded summary
- Audio URLs
- Video context data (optional)

**Example explanation with context:**
> "A man (identified through video analysis as Character #0 with 92% confidence) enters carrying the briefcase we first saw in Scene 1. He approaches a woman (Character #1, 78% confidence)..."

## Configuration

### Default Settings

Located in `app/config.py`:

| Setting | Default | Purpose |
|---------|---------|---------|
| `SPARSE_FRAME_INTERVAL_MS` | 2000 | Sample every 2 seconds for video-wide context |
| `FACE_CLUSTERING_THRESHOLD` | 0.6 | Balance between same-person grouping and false matches |
| `CHARACTER_CONFIDENCE_THRESHOLD_PERCENT` | 70 | Only mention names if confidence ≥ 70% |
| `YOLOV8_MODEL` | yolov8m | Medium model (good accuracy/speed trade-off) |
| `YOLOV8_CONF_THRESHOLD` | 0.5 | Standard confidence threshold for object detection |

### Environment Variables

Override defaults via `.env` file:

```env
# Video context analysis settings
SPARSE_FRAME_INTERVAL_MS=2000
FACE_CLUSTERING_THRESHOLD=0.6
CHARACTER_CONFIDENCE_THRESHOLD_PERCENT=70
YOLOV8_MODEL=yolov8m
YOLOV8_CONF_THRESHOLD=0.5
```

## Understanding the Output

### Video Context JSON

Located at: `outputs/{video_id}/video_context.json`

```json
{
  "characters": [
    {
      "id": 0,
      "appearances": [0, 5, 10, 15, 20, ...],
      "name": "unknown",
      "confidence": 0.87
    }
  ],
  "objects": [
    {
      "id": 0,
      "class_label": "briefcase",
      "appearances": [0, 1, 2, 8, 15, ...],
      "first_scene": null
    }
  ]
}
```

**Interpretation:**
- **Character #0** appears in frames [0, 5, 10, ...] with average 87% confidence
- **Briefcase** is detected in frames [0, 1, 2, ...] throughout the video

### Scene Explanation with Context

The AI now sees this information:

```
Character #0: Appears in frames X, Y, Z (87% average confidence)
Character #1: Appears in frames A, B, C (65% average confidence - below naming threshold)

'briefcase' (Object #0): appears in X frames across video
'phone' (Object #1): appears in Y frames across video
```

The AI uses this context to:
- Mention "the man" (Character #0) by ID if 87% ≥ 70% threshold
- Use pronouns for Character #1 (65% < 70% threshold)
- Reference object persistence: "the briefcase from earlier scenes"

## Performance Impact

- **First-time analysis:** ~30-60 seconds (background processing during `/api/process`)
- **Scene explanations:** <100ms added (video context is text-based)
- **Caching:** Video context computed once per video, reused for all scenes
- **Storage:** ~50-200KB per video for context JSON

## Common Questions

### Q: What if face detection fails?
**A:** Scene explanations still work. Without face detection, the system falls back to pronouns ("the man", "the woman") instead of character IDs.

### Q: Why aren't character names mentioned?
**A:** Confidence is below the 70% threshold. This is intentional to avoid false identifications.

### Q: Can I re-run video analysis?
**A:** Yes, POST to `/api/analyze-video/{video_id}`. It will overwrite the cached context.

### Q: What objects does the system detect?
**A:** YOLO detects ~80 object classes including: person, car, dog, book, cup, chair, table, TV, phone, laptop, briefcase, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush, etc.

### Q: How does the confidence score work?
**A:** Face confidence is the average confidence across all frames where the character appears. Higher = more consistent detection across frames.

### Q: Can I adjust the naming confidence threshold?
**A:** Yes, set `CHARACTER_CONFIDENCE_THRESHOLD_PERCENT` in `.env`:

```env
# More aggressive naming (50% threshold)
CHARACTER_CONFIDENCE_THRESHOLD_PERCENT=50

# More conservative naming (80% threshold)
CHARACTER_CONFIDENCE_THRESHOLD_PERCENT=80
```

## Troubleshooting

### Video context analysis is slow

**Normal behavior:** First analysis takes 30-60 seconds per video. Subsequent scene explanations are fast because context is cached.

**To speed up:**
- Use faster GPU: Set YOLOV8 to faster model `yolov8s` (small) in config
- Increase sparse frame interval: `SPARSE_FRAME_INTERVAL_MS=4000` (less detailed but faster)

### No faces detected

**Possible causes:**
- Video quality too low
- Faces too small in frame
- Strong backlighting or side angles

**To improve:**
- Adjust face detection in `app/face_recognition_clustering.py` (scale_factor, minNeighbors)

### Objects not detected

**Possible causes:**
- Object not in YOLO training data
- Object too small or obscured

**To improve:**
- Lower confidence threshold: `YOLOV8_CONF_THRESHOLD=0.4` (detects more but less accurate)
- Switch to more accurate model: `YOLOV8_MODEL=yolov8l` (larger, slower)

## Advanced Usage

### Re-analyzing with different settings

1. Update config values in `.env`
2. Call `/api/analyze-video/{video_id}` to re-analyze
3. Scene explanations will use new context

### Disable context analysis

To disable automatic video analysis in `/api/process`:
- Edit `app/main.py` line ~1045
- Comment out the `analyze_video_for_context` call

### Custom face detection settings

Edit `app/face_recognition_clustering.py`:
```python
# Increase sensitivity (detect more faces, possibly false positives)
faces = cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,        # Lower = more sensitive
    minNeighbors=3,         # Lower = more sensitive
    minSize=(20, 20),       # Smaller = detects tiny faces
)

# Adjust clustering threshold for stricter/looser matching
clusters = fclusterdata(
    descriptors,
    t=threshold * 1.5,      # Higher = more permissive grouping
    ...
)
```

## Next Steps

- [ ] Upload a test video with 2-3 people
- [ ] Check scene explanations for character names
- [ ] Verify object references (e.g., "briefcase from earlier")
- [ ] Adjust confidence threshold if too strict/loose
- [ ] Fine-tune model selection for your hardware
