"""
utils/face_crop.py
Facial region detection and cropping utilities using MediaPipe Face Mesh.

Provides:
  - FaceRegionCropper: Detects facial landmarks and crops per-region patches
    for eyes, nose, lips, skin (forehead/cheek), hairline, and ears.

The cropper is designed to be reusable across dataset loading and inference.
Falls back to fixed bounding boxes if MediaPipe is unavailable.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

REGIONS: List[str] = ["eyes", "nose", "lips", "skin", "hairline", "ears"]

# MediaPipe face mesh landmark indices for each region
# (Based on MediaPipe 468-landmark face mesh topology)
_LANDMARK_INDICES: Dict[str, List[int]] = {
    "eyes": [
        # Left eye
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
        # Right eye
        362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
    ],
    "nose": [1, 2, 3, 4, 5, 6, 19, 20, 94, 125, 141, 235, 236, 3],
    "lips": [
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
    ],
    "skin": [
        # Forehead + cheeks region
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
        400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162,
    ],
    "hairline": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397],
    "ears": [
        # Left ear region
        127, 234, 93, 132, 58, 172, 136, 150, 149,
        # Right ear
        356, 454, 323, 361, 288, 397, 365, 379, 378,
    ],
}


def _get_bbox(
    landmarks: np.ndarray,
    indices: List[int],
    H: int,
    W: int,
    margin: float = 0.2,
) -> Tuple[int, int, int, int]:
    """
    Compute bounding box (with margin) from selected landmark indices.

    Args:
        landmarks: (N, 2) array of (x, y) pixel coordinates.
        indices: List of landmark indices.
        H, W: Image height and width.
        margin: Fractional margin to add around the tight bbox.

    Returns:
        (x_min, y_min, x_max, y_max) pixel coordinates, clamped to image bounds.
    """
    pts = landmarks[indices]
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
    bw, bh = x_max - x_min, y_max - y_min
    pad_x, pad_y = bw * margin, bh * margin
    return (
        max(0, int(x_min - pad_x)),
        max(0, int(y_min - pad_y)),
        min(W, int(x_max + pad_x)),
        min(H, int(y_max + pad_y)),
    )


class FaceRegionCropper:
    """
    Detects facial landmarks with MediaPipe Face Mesh and crops per-region patches.

    Args:
        min_detection_confidence (float): MediaPipe face detection threshold.
        margin (float): Bounding box margin fraction around each region.
        fallback_on_failure (bool): Return zero-sized fallback crops if
                                    detection fails instead of None values.
        max_num_faces (int): Maximum faces to detect (we always use the first).
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        margin: float = 0.25,
        fallback_on_failure: bool = True,
        max_num_faces: int = 1,
    ) -> None:
        self.margin = margin
        self.fallback_on_failure = fallback_on_failure
        self._face_mesh = None
        self._init_kwargs = dict(
            static_image_mode=True,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
        )

    def _get_face_mesh(self):
        """Lazy-load MediaPipe FaceMesh."""
        if self._face_mesh is None:
            try:
                import mediapipe as mp
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(**self._init_kwargs)
            except ImportError:
                self._face_mesh = None
        return self._face_mesh

    def _detect_landmarks(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        Detect 468 facial landmarks.

        Args:
            image: PIL RGB image.

        Returns:
            (468, 2) array of pixel (x, y) coordinates, or None if no face found.
        """
        face_mesh = self._get_face_mesh()
        if face_mesh is None:
            return None

        img_np = np.array(image.convert("RGB"))
        H, W = img_np.shape[:2]
        results = face_mesh.process(img_np)

        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0].landmark
        pts = np.array([[l.x * W, l.y * H] for l in lm], dtype=np.float32)
        return pts

    def crop_regions(
        self,
        image: Image.Image,
    ) -> Dict[str, Optional[Image.Image]]:
        """
        Detect landmarks and crop each facial region.

        Args:
            image: PIL RGB face image.

        Returns:
            Dict[region_name → PIL crop | None].
            None values indicate failed detection for that region.
        """
        landmarks = self._detect_landmarks(image)
        H, W = image.height, image.width
        crops: Dict[str, Optional[Image.Image]] = {}

        if landmarks is None:
            if self.fallback_on_failure:
                # Fall back to fixed proportional crops
                for region in REGIONS:
                    crops[region] = self._fallback_crop(image, region)
            else:
                for region in REGIONS:
                    crops[region] = None
            return crops

        for region, indices in _LANDMARK_INDICES.items():
            # Clamp indices to valid range
            valid_idx = [i for i in indices if i < len(landmarks)]
            if not valid_idx:
                crops[region] = None
                continue
            x1, y1, x2, y2 = _get_bbox(landmarks, valid_idx, H, W, margin=self.margin)
            if x2 <= x1 or y2 <= y1:
                crops[region] = None
            else:
                crops[region] = image.crop((x1, y1, x2, y2))

        return crops

    @staticmethod
    def _fallback_crop(image: Image.Image, region: str) -> Image.Image:
        """
        Fixed proportional crops when landmark detection fails.

        Rough face-region approximate bounding boxes as fractions of (W, H).
        """
        W, H = image.size
        region_boxes: Dict[str, Tuple[float, float, float, float]] = {
            "eyes":     (0.1, 0.28, 0.90, 0.48),
            "nose":     (0.3, 0.40, 0.70, 0.65),
            "lips":     (0.25, 0.60, 0.75, 0.80),
            "skin":     (0.1, 0.05, 0.90, 0.75),
            "hairline": (0.1, 0.00, 0.90, 0.25),
            "ears":     (0.0, 0.25, 1.00, 0.65),
        }
        fx1, fy1, fx2, fy2 = region_boxes.get(region, (0.0, 0.0, 1.0, 1.0))
        box = (int(fx1 * W), int(fy1 * H), int(fx2 * W), int(fy2 * H))
        return image.crop(box)
