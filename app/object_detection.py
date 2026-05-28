"""Object detection and tracking using YOLOv8."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

import app.config as config


class ObjectDetector:
    """YOLOv8-based object detection."""

    def __init__(self, model_name: str = config.YOLOV8_MODEL):
        """Initialize YOLOv8 model."""
        self.model = YOLO(model_name)
        self.conf_threshold = config.YOLOV8_CONF_THRESHOLD

    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in a frame.

        Args:
            frame: numpy array (HxWxC, RGB or BGR)

        Returns:
            List of detections: [{"class": "person", "conf": 0.95, "box": [x1, y1, x2, y2]}, ...]
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)

        if not results or not results[0].boxes:
            return []

        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0]

            class_name = self.model.names.get(cls_id, f"class_{cls_id}")

            detections.append({
                "class": class_name,
                "confidence": conf,
                "box": [float(x) for x in xyxy],  # [x1, y1, x2, y2]
            })

        return detections


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1, box2: [x1, y1, x2, y2] format

    Returns:
        IoU score (0-1)
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0

    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def track_objects_across_frames(
    detections_by_frame: Dict[int, List[Dict]],
    iou_threshold: float = 0.3,
) -> List[List[Tuple[int, int]]]:
    """
    Track objects across frames using IoU matching.

    Args:
        detections_by_frame: {frame_idx: [{"class": "...", "box": [...], "confidence": ...}]}
        iou_threshold: minimum IoU to consider same object (0.3)

    Returns:
        List of object tracks: [[(frame_idx, det_idx), ...], ...]
        Each track is a list of (frame_index, detection_index) tuples
    """
    if not detections_by_frame:
        return []

    frame_indices = sorted(detections_by_frame.keys())
    if not frame_indices:
        return []

    # Track state: list of active tracks
    # Each track: [(frame_idx, det_idx), ...]
    tracks: List[List[Tuple[int, int]]] = []

    # Initialize tracks from first frame
    first_frame = frame_indices[0]
    for det_idx in range(len(detections_by_frame[first_frame])):
        tracks.append([(first_frame, det_idx)])

    # Process subsequent frames
    for frame_idx in frame_indices[1:]:
        current_detections = detections_by_frame[frame_idx]
        matched = [False] * len(current_detections)

        # Try to match each detection to existing tracks
        for track in tracks:
            last_frame, last_det_idx = track[-1]
            last_det = detections_by_frame[last_frame][last_det_idx]

            best_iou = 0.0
            best_det_idx = -1

            for det_idx, det in enumerate(current_detections):
                if matched[det_idx]:
                    continue

                # Only consider same class
                if det.get("class") != last_det.get("class"):
                    continue

                iou = compute_iou(last_det["box"], det["box"])
                if iou > best_iou and iou > iou_threshold:
                    best_iou = iou
                    best_det_idx = det_idx

            if best_det_idx >= 0:
                track.append((frame_idx, best_det_idx))
                matched[best_det_idx] = True

        # Create new tracks for unmatched detections
        for det_idx, is_matched in enumerate(matched):
            if not is_matched:
                tracks.append([(frame_idx, det_idx)])

    return tracks
