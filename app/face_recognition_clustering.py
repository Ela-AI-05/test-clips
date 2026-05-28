"""Face detection and clustering using OpenCV."""

from typing import List, Optional

import cv2
import numpy as np
from scipy.cluster.hierarchy import fclusterdata


def extract_face_regions(frame: np.ndarray, scale_factor: float = 1.3) -> List[np.ndarray]:
    """
    Extract face regions from a frame using OpenCV Haar cascade.

    Args:
        frame: numpy array (BGR or RGB, HxWxC)
        scale_factor: cascade scale factor (1.3 is default)

    Returns:
        List of face region arrays (may be empty if no faces found)
    """
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.shape[2] == 3 else frame
    else:
        gray = frame

    # Load cascade classifier
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Detect faces
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=5,
        minSize=(30, 30),
    )

    if len(faces) == 0:
        return []

    # Extract face regions
    face_regions = []
    for x, y, w, h in faces:
        # Add some padding
        y1, y2 = max(0, y - 5), min(frame.shape[0], y + h + 5)
        x1, x2 = max(0, x - 5), min(frame.shape[1], x + w + 5)
        face_region = frame[y1:y2, x1:x2].copy()
        if face_region.size > 0:
            face_regions.append(face_region)

    return face_regions


def compute_face_descriptor(face_region: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute a simple descriptor for a face region (histogram-based).

    Args:
        face_region: Face image array

    Returns:
        Descriptor vector (simple histogram) or None if face too small
    """
    if face_region.shape[0] < 20 or face_region.shape[1] < 20:
        return None

    # Simple histogram descriptor
    if len(face_region.shape) == 3 and face_region.shape[2] == 3:
        # Multi-channel: compute histogram for each channel
        hist = []
        for i in range(3):
            h = cv2.calcHist([face_region], [i], None, [32], [0, 256])
            hist.extend(h.flatten())
        return np.array(hist)
    else:
        # Single channel
        h = cv2.calcHist([face_region], [0], None, [32], [0, 256])
        return h.flatten()


def compute_histogram_distance(desc1: np.ndarray, desc2: np.ndarray) -> float:
    """
    Compute distance between two histogram descriptors.

    Args:
        desc1, desc2: Histogram descriptors

    Returns:
        Distance metric (0-1, lower = more similar)
    """
    if desc1 is None or desc2 is None:
        return 1.0

    # Normalize descriptors
    d1 = cv2.normalize(desc1.flatten(), desc1.flatten(), norm=cv2.NORM_L2)
    d2 = cv2.normalize(desc2.flatten(), desc2.flatten(), norm=cv2.NORM_L2)

    # Compute chi-square distance
    return cv2.compareHist(d1.reshape(-1, 1), d2.reshape(-1, 1), cv2.HISTCMP_CHISQR)


def extract_faces_from_frames(frames: List[np.ndarray]) -> List[tuple]:
    """
    Extract face descriptors from frames.

    Args:
        frames: List of numpy arrays (RGB or BGR, HxWxC)

    Returns:
        List of (frame_idx, descriptor) tuples
        where descriptor is a histogram feature vector
    """
    faces = []

    for frame_idx, frame in enumerate(frames):
        face_regions = extract_face_regions(frame)

        for face_region in face_regions:
            descriptor = compute_face_descriptor(face_region)
            if descriptor is not None:
                faces.append((frame_idx, descriptor))

    return faces


def cluster_faces(
    faces: List[tuple],
    threshold: float = 0.6,
) -> List[List[int]]:
    """
    Cluster faces using hierarchical clustering on histogram descriptors.

    Args:
        faces: List of (frame_idx, descriptor) from extract_faces_from_frames()
        threshold: distance threshold for clustering (0.6 = moderate)

    Returns:
        List of clusters: [[face_idx, ...], ...]
        Each cluster represents one detected character
    """
    if not faces:
        return []

    descriptors = np.array([f[1] for f in faces])

    if len(descriptors) == 1:
        return [[0]]

    # Hierarchical clustering using chi-square distance
    # Threshold around 1.0-2.0 works well for histogram distances
    # Scale the input threshold appropriately
    cluster_threshold = threshold * 2.0  # Scale for histogram distance

    clusters = fclusterdata(
        descriptors,
        t=cluster_threshold,
        criterion='distance',
        method='complete',
        metric='euclidean',
    )

    # Group face indices by cluster
    cluster_groups: List[List[int]] = [[] for _ in range(clusters.max())]
    for face_idx, cluster_id in enumerate(clusters):
        cluster_groups[cluster_id - 1].append(face_idx)

    return [g for g in cluster_groups if g]


def extract_and_cluster_faces(
    frames: List[np.ndarray],
    threshold: float = 0.6,
) -> tuple:
    """
    Extract and cluster faces in frames.

    Args:
        frames: List of numpy arrays (RGB or BGR)
        threshold: clustering threshold (0.6)

    Returns:
        (clusters, face_appearances)
        - clusters: List[List[int]] - face indices grouped by character
        - face_appearances: List[(frame_idx, descriptor)] - original face data
    """
    faces = extract_faces_from_frames(frames)
    clusters = cluster_faces(faces, threshold=threshold)

    return clusters, faces
