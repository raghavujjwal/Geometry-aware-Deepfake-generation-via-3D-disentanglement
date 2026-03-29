"""
inference/postprocess.py
Post-processing utilities for face swap outputs.

Provides:
  - blend_face_regions: Alpha composite generated face onto target background.
  - color_correction: Histogram-based color matching to target image palette.
  - sharpen_face: Edge-aware sharpening for the face region.
  - FacePostProcessor: Pipeline wrapper for all post-processing steps.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageFilter


# ─────────────────────────────────────────────────────────────────────────────
# Face mask extraction
# ─────────────────────────────────────────────────────────────────────────────


def _get_face_mask(
    image: Image.Image,
    feather_radius: int = 15,
) -> np.ndarray:
    """
    Generate a soft elliptical face mask for blending.

    Uses MediaPipe face mesh if available, otherwise falls back to
    an elliptical approximation centred in the image.

    Args:
        image: PIL RGB image.
        feather_radius: Gaussian blur radius for soft edges.

    Returns:
        Float32 mask array (H, W), values in [0, 1].
    """
    try:
        import mediapipe as mp

        mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1
        )
        img_rgb = np.array(image)
        results = mp_face_mesh.process(img_rgb)
        mp_face_mesh.close()

        if results.multi_face_landmarks:
            H, W = img_rgb.shape[:2]
            lm = results.multi_face_landmarks[0].landmark
            pts = np.array([[int(l.x * W), int(l.y * H)] for l in lm], dtype=np.int32)
            hull = cv2.convexHull(pts)
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull, 255)
            # Feather edges
            mask = cv2.GaussianBlur(mask, (feather_radius * 2 + 1, feather_radius * 2 + 1), feather_radius)
            return mask.astype(np.float32) / 255.0
    except Exception:
        pass

    # Fallback: elliptical mask
    H, W = image.height, image.width
    Y, X = np.ogrid[:H, :W]
    cx, cy = W // 2, int(H * 0.45)
    rx, ry = int(W * 0.38), int(H * 0.45)
    mask = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2
    mask = (1.0 - np.clip(mask, 0, 1)).astype(np.float32)
    # Feather
    mask_u8 = (mask * 255).astype(np.uint8)
    mask_u8 = cv2.GaussianBlur(mask_u8, (feather_radius * 2 + 1, feather_radius * 2 + 1), feather_radius)
    return mask_u8.astype(np.float32) / 255.0


# ─────────────────────────────────────────────────────────────────────────────
# Blending
# ─────────────────────────────────────────────────────────────────────────────


def blend_face_regions(
    generated: Image.Image,
    target: Image.Image,
    alpha: float = 0.95,
    feather_radius: int = 15,
) -> Image.Image:
    """
    Alpha-composite the generated face over the target background.

    Args:
        generated: Swapped face PIL image (same size as target).
        target: Original target face PIL image.
        alpha: Overall blend strength (1.0 = full swap in face region).
        feather_radius: Edge feathering radius for soft blending.

    Returns:
        Blended PIL image.
    """
    generated = generated.convert("RGB").resize(target.size)
    mask = _get_face_mask(target, feather_radius=feather_radius) * alpha

    gen_np = np.array(generated, dtype=np.float32)
    tgt_np = np.array(target, dtype=np.float32)
    mask_3ch = mask[:, :, np.newaxis]  # broadcast over channels

    blended = gen_np * mask_3ch + tgt_np * (1.0 - mask_3ch)
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


# ─────────────────────────────────────────────────────────────────────────────
# Color correction
# ─────────────────────────────────────────────────────────────────────────────


def color_correction(
    source: Image.Image,
    reference: Image.Image,
    method: str = "histogram",
) -> Image.Image:
    """
    Match the color distribution of ``source`` to ``reference``.

    Args:
        source: Image whose colors will be adjusted.
        reference: Image providing the target color distribution.
        method: 'histogram' (global) | 'lab' (Lab mean/std shift).

    Returns:
        Color-corrected PIL image.
    """
    if method == "lab":
        return _lab_color_transfer(source, reference)
    return _histogram_match(source, reference)


def _histogram_match(
    source: Image.Image,
    reference: Image.Image,
) -> Image.Image:
    """Per-channel histogram matching."""
    src_np = np.array(source.convert("RGB"), dtype=np.float32)
    ref_np = np.array(reference.convert("RGB"), dtype=np.float32)
    result = np.zeros_like(src_np)

    for c in range(3):
        src_hist, bins = np.histogram(src_np[:, :, c].flatten(), 256, [0, 256])
        ref_hist, _ = np.histogram(ref_np[:, :, c].flatten(), 256, [0, 256])

        src_cdf = src_hist.cumsum().astype(np.float64)
        ref_cdf = ref_hist.cumsum().astype(np.float64)

        src_cdf /= src_cdf[-1]
        ref_cdf /= ref_cdf[-1]

        lut = np.interp(src_cdf, ref_cdf, np.arange(256))
        result[:, :, c] = lut[src_np[:, :, c].astype(np.uint8)]

    return Image.fromarray(result.clip(0, 255).astype(np.uint8))


def _lab_color_transfer(
    source: Image.Image,
    reference: Image.Image,
) -> Image.Image:
    """Mean/std color transfer in Lab color space."""
    src_lab = cv2.cvtColor(np.array(source.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(np.array(reference.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)

    for c in range(3):
        src_mean, src_std = src_lab[:, :, c].mean(), src_lab[:, :, c].std() + 1e-6
        ref_mean, ref_std = ref_lab[:, :, c].mean(), ref_lab[:, :, c].std() + 1e-6
        src_lab[:, :, c] = (src_lab[:, :, c] - src_mean) * (ref_std / src_std) + ref_mean

    result_rgb = cv2.cvtColor(np.clip(src_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return Image.fromarray(result_rgb)


# ─────────────────────────────────────────────────────────────────────────────
# Sharpening
# ─────────────────────────────────────────────────────────────────────────────


def sharpen_face(
    image: Image.Image,
    strength: float = 1.0,
) -> Image.Image:
    """
    Apply unsharp-mask sharpening to the face region.

    Args:
        image: PIL image to sharpen.
        strength: Sharpening strength (0 = none, 1 = standard, >1 = aggressive).

    Returns:
        Sharpened PIL image.
    """
    img_np = np.array(image.convert("RGB"), dtype=np.float32)
    blurred = cv2.GaussianBlur(img_np, (0, 0), sigmaX=2.0)
    sharpened = img_np + strength * (img_np - blurred)
    return Image.fromarray(sharpened.clip(0, 255).astype(np.uint8))


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline wrapper
# ─────────────────────────────────────────────────────────────────────────────


class FacePostProcessor:
    """
    Composable post-processing pipeline for face swap outputs.

    Args:
        blend_alpha (float): Face region blend strength.
        color_correction_method (str): 'histogram' | 'lab' | None.
        sharpen_strength (float): Sharpening strength (0 = disabled).
        feather_radius (int): Mask feathering radius.
    """

    def __init__(
        self,
        blend_alpha: float = 0.95,
        color_correction_method: Optional[str] = "histogram",
        sharpen_strength: float = 0.5,
        feather_radius: int = 15,
    ) -> None:
        self.blend_alpha = blend_alpha
        self.color_correction_method = color_correction_method
        self.sharpen_strength = sharpen_strength
        self.feather_radius = feather_radius

    def __call__(
        self,
        generated: Image.Image,
        target: Image.Image,
    ) -> Image.Image:
        """
        Apply full post-processing pipeline.

        Args:
            generated: Raw generated face image.
            target: Original target face image.

        Returns:
            Post-processed face swap result.
        """
        result = blend_face_regions(
            generated, target,
            alpha=self.blend_alpha,
            feather_radius=self.feather_radius,
        )
        if self.color_correction_method:
            result = color_correction(result, target, method=self.color_correction_method)
        if self.sharpen_strength > 0:
            result = sharpen_face(result, strength=self.sharpen_strength)
        return result
