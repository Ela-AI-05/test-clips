"""Video-level context analysis for character and object tracking."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

import app.config as config
from app.face_recognition_clustering import extract_and_cluster_faces
from app.object_detection import ObjectDetector, track_objects_across_frames


@dataclass
class Character:
    """Identified character in the video."""

    id: int
    appearances: List[int]  # frame indices where this character appears
    name: str = "unknown"
    confidence: float = 0.0  # average confidence across appearances


@dataclass
class DetectedObject:
    """Object detected in the video."""

    id: int
    class_label: str  # e.g., "briefcase", "phone"
    appearances: List[int]  # frame indices
    first_scene: Optional[int] = None


@dataclass
class VideoContext:
    """Full video analysis context."""

    characters: List[Character]
    objects: List[DetectedObject]

    def to_dict(self):
        """Convert to dict for JSON serialization."""
        return {
            "characters": [asdict(c) for c in self.characters],
            "objects": [asdict(o) for o in self.objects],
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Load from dict (e.g., from JSON)."""
        characters = [Character(**c) for c in data.get("characters", [])]
        objects = [DetectedObject(**o) for o in data.get("objects", [])]
        return cls(characters=characters, objects=objects)


def extract_sparse_frames(
    video_path: Path,
    frame_interval_ms: int = config.SPARSE_FRAME_INTERVAL_MS,
) -> tuple:
    """
    Extract frames at regular intervals from entire video.

    Args:
        video_path: Path to video file
        frame_interval_ms: interval in milliseconds (2000 = 1 frame per 2 seconds)

    Returns:
        (frames, frame_indices)
        - frames: List[np.ndarray] - RGB frames
        - frame_indices: List[int] - frame numbers in video
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = int((frame_interval_ms / 1000.0) * fps)
    frame_interval = max(1, frame_interval)

    frames = []
    frame_indices = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
            frame_indices.append(frame_count)

        frame_count += 1

    cap.release()
    return frames, frame_indices


def analyze_video_for_context(
    video_path: Path,
    frame_interval_ms: int = config.SPARSE_FRAME_INTERVAL_MS,
) -> Optional[VideoContext]:
    """
    Analyze video for character and object continuity.

    Args:
        video_path: Path to video file
        frame_interval_ms: sparse sampling interval

    Returns:
        VideoContext with character and object tracking, or None if analysis fails
    """
    frames, frame_indices = extract_sparse_frames(video_path, frame_interval_ms)

    if not frames:
        return None

    # Extract and cluster faces
    character_id_counter = 0
    characters: List[Character] = []

    try:
        clusters, face_data = extract_and_cluster_faces(
            frames,
            threshold=config.FACE_CLUSTERING_THRESHOLD,
        )

        for cluster_face_indices in clusters:
            if not cluster_face_indices:
                continue

            # Get all frame indices where this character appears
            appearance_frame_indices = []
            confidence_scores = []

            for face_idx in cluster_face_indices:
                if face_idx < len(face_data):
                    frame_idx, _ = face_data[face_idx]
                    appearance_frame_indices.append(frame_idx)
                    # Note: face_recognition doesn't provide confidence directly
                    # We use 0.85 as default for detected faces
                    confidence_scores.append(0.85)

            if appearance_frame_indices:
                avg_confidence = (
                    sum(confidence_scores) / len(confidence_scores)
                    if confidence_scores
                    else 0.85
                )
                characters.append(
                    Character(
                        id=character_id_counter,
                        appearances=sorted(set(appearance_frame_indices)),
                        name="unknown",
                        confidence=avg_confidence,
                    )
                )
                character_id_counter += 1
    except Exception as e:
        # Graceful degradation if face detection fails
        print(f"Face detection failed: {e}")

    # Detect and track objects
    object_id_counter = 0
    objects: List[DetectedObject] = []

    try:
        detector = ObjectDetector()
        detections_by_frame = {}

        for frame_idx, frame in enumerate(frames):
            detections = detector.detect_objects(frame)
            if detections:
                detections_by_frame[frame_idx] = detections

        if detections_by_frame:
            # Track objects across frames
            tracks = track_objects_across_frames(
                detections_by_frame,
                iou_threshold=0.3,
            )

            for track in tracks:
                if not track:
                    continue

                # Get all frame indices for this object track
                appearance_indices = [frame_idx for frame_idx, _ in track]
                first_frame_idx = appearance_indices[0]

                # Get object class from first detection
                first_det_idx = track[0][1]
                class_label = detections_by_frame[first_frame_idx][first_det_idx].get(
                    "class", "object"
                )

                objects.append(
                    DetectedObject(
                        id=object_id_counter,
                        class_label=class_label,
                        appearances=sorted(set(appearance_indices)),
                        first_scene=None,  # Set during scene assignment if needed
                    )
                )
                object_id_counter += 1
    except Exception as e:
        # Graceful degradation if object detection fails
        print(f"Object detection failed: {e}")

    return VideoContext(characters=characters, objects=objects)


def save_video_context(
    context: VideoContext,
    video_id: str,
    outputs_dir: Path,
) -> Path:
    """
    Save video context to JSON file.

    Args:
        context: VideoContext object
        video_id: video identifier
        outputs_dir: path to outputs directory

    Returns:
        Path to saved JSON file
    """
    context_dir = outputs_dir / video_id / config.VIDEO_CONTEXT_CACHE_DIR
    context_dir.mkdir(parents=True, exist_ok=True)

    context_path = context_dir.parent / "video_context.json"
    context_path.write_text(
        json.dumps(context.to_dict(), indent=2),
        encoding="utf-8",
    )

    return context_path


def load_video_context(
    video_id: str,
    outputs_dir: Path,
) -> Optional[VideoContext]:
    """
    Load video context from JSON file.

    Args:
        video_id: video identifier
        outputs_dir: path to outputs directory

    Returns:
        VideoContext or None if file doesn't exist
    """
    context_path = outputs_dir / video_id / "video_context.json"

    if not context_path.is_file():
        return None

    try:
        data = json.loads(context_path.read_text(encoding="utf-8"))
        return VideoContext.from_dict(data)
    except (json.JSONDecodeError, OSError):
        return None
