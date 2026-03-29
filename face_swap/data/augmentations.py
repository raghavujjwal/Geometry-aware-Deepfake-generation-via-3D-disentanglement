"""
data/augmentations.py
Face-specific augmentation pipeline for geometry-aware face swapping training.

Provides:
- FaceAugmentationPipeline: Composed pipeline applied to face images.
- PairedAugmentation: Applies consistent geometric transforms to
  (source, target) pairs while allowing independent photometric jitter.
- RegionCropAugmentation: Augmentation applied to extracted facial region crops.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Geometric Augmentations
# ─────────────────────────────────────────────────────────────────────────────


class RandomHorizontalFlipFace:
    """Horizontal flip that mirrors landmark / region maps consistently."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(
        self,
        image: Image.Image,
        landmarks: Optional[np.ndarray] = None,
    ) -> Tuple[Image.Image, Optional[np.ndarray]]:
        if random.random() < self.p:
            image = TF.hflip(image)
            if landmarks is not None:
                w = image.width
                landmarks = landmarks.copy()
                landmarks[:, 0] = w - landmarks[:, 0]
        return image, landmarks


class RandomAffine:
    """Affine transform: rotation, translation, scale with landmark propagation."""

    def __init__(
        self,
        degrees: float = 10.0,
        translate: Tuple[float, float] = (0.05, 0.05),
        scale: Tuple[float, float] = (0.95, 1.05),
        p: float = 0.5,
    ) -> None:
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.p = p

    def __call__(
        self,
        image: Image.Image,
        landmarks: Optional[np.ndarray] = None,
    ) -> Tuple[Image.Image, Optional[np.ndarray]]:
        if random.random() > self.p:
            return image, landmarks

        angle, translations, scale, shear = T.RandomAffine.get_params(
            degrees=(-self.degrees, self.degrees),
            translate=self.translate,
            scale_ranges=self.scale,
            shears=(0, 0),
            img_size=image.size,
        )
        image = TF.affine(image, angle, translations, scale, shear)

        if landmarks is not None:
            w, h = image.size
            cx, cy = w / 2.0, h / 2.0
            rad = np.deg2rad(-angle)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            lm = landmarks.astype(np.float32).copy()
            lm[:, 0] -= cx
            lm[:, 1] -= cy
            rotated_x = cos_a * lm[:, 0] - sin_a * lm[:, 1]
            rotated_y = sin_a * lm[:, 0] + cos_a * lm[:, 1]
            lm[:, 0] = rotated_x * scale + cx + translations[0]
            lm[:, 1] = rotated_y * scale + cy + translations[1]
            landmarks = lm

        return image, landmarks


class RandomCropResize:
    """Random crop then resize back, simulating camera zoom variations."""

    def __init__(
        self,
        crop_ratio: Tuple[float, float] = (0.85, 1.0),
        p: float = 0.3,
    ) -> None:
        self.crop_ratio = crop_ratio
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        ratio = random.uniform(*self.crop_ratio)
        w, h = image.size
        new_w, new_h = int(w * ratio), int(h * ratio)
        left = random.randint(0, w - new_w)
        top = random.randint(0, h - new_h)
        image = TF.crop(image, top, left, new_h, new_w)
        image = TF.resize(image, (h, w))
        return image


# ─────────────────────────────────────────────────────────────────────────────
# Photometric Augmentations
# ─────────────────────────────────────────────────────────────────────────────


class RandomColorJitter:
    """Torchvision ColorJitter with configurable probabilities."""

    def __init__(
        self,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.2,
        hue: float = 0.05,
        p: float = 0.8,
    ) -> None:
        self.p = p
        self._jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return self._jitter(image)
        return image


class RandomGrayscale:
    """Convert to grayscale with probability p (keeps 3 channels)."""

    def __init__(self, p: float = 0.05) -> None:
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return TF.to_grayscale(image, num_output_channels=3)
        return image


class RandomGaussianBlur:
    """Apply Gaussian blur to simulate motion blur / low-res captures."""

    def __init__(
        self,
        kernel_sizes: List[int] = [3, 5, 7],
        sigma: Tuple[float, float] = (0.1, 2.0),
        p: float = 0.2,
    ) -> None:
        self.kernel_sizes = kernel_sizes
        self.sigma = sigma
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.p:
            k = random.choice(self.kernel_sizes)
            sig = random.uniform(*self.sigma)
            return TF.gaussian_blur(image, kernel_size=k, sigma=sig)
        return image


class RandomJPEGCompression:
    """Simulate JPEG compression artifacts common in face datasets."""

    def __init__(self, quality_range: Tuple[int, int] = (30, 95), p: float = 0.2) -> None:
        self.quality_range = quality_range
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.p:
            import io

            quality = random.randint(*self.quality_range)
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            return Image.open(buf).convert("RGB")
        return image


class RandomNoise:
    """Add Gaussian noise to the image tensor."""

    def __init__(self, std_range: Tuple[float, float] = (0.0, 0.05), p: float = 0.1) -> None:
        self.std_range = std_range
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            std = random.uniform(*self.std_range)
            noise = torch.randn_like(tensor) * std
            return (tensor + noise).clamp(0.0, 1.0)
        return tensor


# ─────────────────────────────────────────────────────────────────────────────
# Composed Pipelines
# ─────────────────────────────────────────────────────────────────────────────


class FaceAugmentationPipeline:
    """
    Full augmentation pipeline for a single face image.

    Applies geometric and photometric transforms sequentially,
    followed by a normalization step to [-1, 1] for SDXL VAE input.

    Args:
        image_size (int): Target output resolution (square).
        flip_p (float): Probability for horizontal flip.
        affine_p (float): Probability for affine transform.
        jitter_p (float): Probability for color jitter.
        blur_p (float): Probability for Gaussian blur.
        jpeg_p (float): Probability for JPEG compression.
        noise_p (float): Probability for Gaussian noise.
        normalize (bool): Whether to normalize to [-1, 1]. Default True.
    """

    def __init__(
        self,
        image_size: int = 512,
        flip_p: float = 0.5,
        affine_p: float = 0.3,
        jitter_p: float = 0.8,
        blur_p: float = 0.2,
        jpeg_p: float = 0.2,
        noise_p: float = 0.1,
        normalize: bool = True,
    ) -> None:
        self.image_size = image_size
        self.normalize = normalize

        self.flip = RandomHorizontalFlipFace(p=flip_p)
        self.affine = RandomAffine(p=affine_p)
        self.crop_resize = RandomCropResize(p=0.3)
        self.jitter = RandomColorJitter(p=jitter_p)
        self.grayscale = RandomGrayscale(p=0.05)
        self.blur = RandomGaussianBlur(p=blur_p)
        self.jpeg = RandomJPEGCompression(p=jpeg_p)
        self.noise = RandomNoise(p=noise_p)

        self.to_tensor = T.ToTensor()
        self.norm = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __call__(
        self,
        image: Image.Image,
        landmarks: Optional[np.ndarray] = None,
    ) -> Dict[str, torch.Tensor | np.ndarray]:
        """
        Args:
            image: PIL RGB image.
            landmarks: Optional (N, 2) numpy array of facial landmark coordinates.

        Returns:
            Dict with keys 'image' (tensor) and optionally 'landmarks'.
        """
        # Resize to target
        image = TF.resize(image, (self.image_size, self.image_size))

        # Geometric
        image, landmarks = self.flip(image, landmarks)
        image, landmarks = self.affine(image, landmarks)
        image = self.crop_resize(image)

        # Photometric (PIL-level)
        image = self.jitter(image)
        image = self.grayscale(image)
        image = self.blur(image)
        image = self.jpeg(image)

        # To tensor + noise (tensor-level)
        tensor = self.to_tensor(image)
        tensor = self.noise(tensor)

        if self.normalize:
            tensor = self.norm(tensor)

        result: Dict = {"image": tensor}
        if landmarks is not None:
            result["landmarks"] = landmarks
        return result


class PairedAugmentation:
    """
    Paired augmentation for (source, target) face image tuples.

    Geometric transforms (flip, affine) are applied identically to both.
    Photometric transforms are applied independently.

    Args:
        image_size (int): Output resolution.
        geometric_p (float): Probability scalar for geometric transforms.
    """

    def __init__(self, image_size: int = 512, geometric_p: float = 0.5) -> None:
        self.image_size = image_size
        self.flip = RandomHorizontalFlipFace(p=geometric_p)
        self.affine = RandomAffine(p=geometric_p * 0.6)

        # Independent photometric per image
        self._photo_src = FaceAugmentationPipeline(
            image_size=image_size, flip_p=0.0, affine_p=0.0
        )
        self._photo_tgt = FaceAugmentationPipeline(
            image_size=image_size, flip_p=0.0, affine_p=0.0
        )

    def __call__(
        self,
        source: Image.Image,
        target: Image.Image,
        src_landmarks: Optional[np.ndarray] = None,
        tgt_landmarks: Optional[np.ndarray] = None,
    ) -> Tuple[Dict, Dict]:
        """
        Args:
            source: Source face PIL image.
            target: Target face PIL image.
            src_landmarks: Optional source landmarks (N, 2).
            tgt_landmarks: Optional target landmarks (N, 2).

        Returns:
            Tuple of (source_dict, target_dict) with 'image' tensors.
        """
        # Fix seed for shared geometric ops
        seed = random.randint(0, 2**31)

        def apply_geometric(img: Image.Image, lm: Optional[np.ndarray]) -> Tuple:
            random.seed(seed)
            img, lm = self.flip(img, lm)
            random.seed(seed)
            img, lm = self.affine(img, lm)
            return img, lm

        source, src_landmarks = apply_geometric(source, src_landmarks)
        target, tgt_landmarks = apply_geometric(target, tgt_landmarks)

        src_dict = self._photo_src(source, src_landmarks)
        tgt_dict = self._photo_tgt(target, tgt_landmarks)
        return src_dict, tgt_dict


class RegionCropAugmentation:
    """
    Augmentation for extracted facial region crops (eyes, nose, lips, etc.).

    Lighter augmentation than the full-face pipeline — no heavy geometric
    distortion (since region detectors are sensitive to shape).

    Args:
        image_size (int): Target crop resolution.
    """

    def __init__(self, image_size: int = 64) -> None:
        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(p=0.3),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.03),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __call__(self, crop: Image.Image) -> torch.Tensor:
        """
        Args:
            crop: PIL RGB image of the facial region.
        Returns:
            Normalized tensor of shape (3, image_size, image_size).
        """
        return self.transform(crop)
