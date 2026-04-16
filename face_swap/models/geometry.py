"""
models/geometry.py
Lightweight MediaPipe Face Mesh geometry conditioning.

This module keeps the original GeometryConditioning API used by the trainer,
but replaces live DECA/DPT inference with a cheap facial-mesh projection:
  - depth_map       (B, 3, H, W): 3-channel depth proxy for ControlNet
  - normal_map      (B, 3, H, W): normals derived from the depth proxy
  - depth_map_raw   (B, 1, H, W): single-channel mesh depth proxy
  - param_embedding (B, hidden_dim): compact normalized landmark embedding

The trainer still loads per-image cache files named "<image>.deca.pt". The
extension is intentionally unchanged for compatibility with the existing
dataset/training code.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MediaPipeMeshGeometry(nn.Module):
    """
    CPU-light geometry extractor using MediaPipe Face Mesh.

    MediaPipe is lazy-loaded so cached training does not require constructing
    the detector unless a cache miss occurs or the precompute script is run.
    """

    def __init__(
        self,
        hidden_dim: int = 320,
        image_size: int = 256,
        min_detection_confidence: float = 0.5,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.image_size = image_size
        self.min_detection_confidence = min_detection_confidence
        self._face_mesh = None

    def _get_face_mesh(self):
        if self._face_mesh is None:
            try:
                import mediapipe as mp
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=self.min_detection_confidence,
                )
            except (ImportError, AttributeError):
                self._face_mesh = False
        return None if self._face_mesh is False else self._face_mesh

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: (B, 3, H, W) normalized to [-1, 1].

        Returns:
            depth_raw:       (B, 1, image_size, image_size)
            param_embedding: (B, hidden_dim)
        """
        imgs_01 = ((images.detach().cpu().float() + 1.0) / 2.0).clamp(0, 1)
        imgs_np = (imgs_01 * 255).byte().permute(0, 2, 3, 1).numpy()

        depths: List[torch.Tensor] = []
        embeddings: List[torch.Tensor] = []
        for img_np in imgs_np:
            depth_np, embedding_np = self._process_one(img_np)
            depths.append(torch.from_numpy(depth_np))
            embeddings.append(torch.from_numpy(embedding_np))

        depth = torch.stack(depths, dim=0).float()
        embedding = torch.stack(embeddings, dim=0).float()
        return depth, embedding

    def _process_one(self, img_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        H, W = img_np.shape[:2]
        face_mesh = self._get_face_mesh()
        if face_mesh is None:
            return self._fallback_depth(img_np), np.zeros(self.hidden_dim, dtype=np.float32)

        results = face_mesh.process(img_np)
        if not results.multi_face_landmarks:
            return self._fallback_depth(img_np), np.zeros(self.hidden_dim, dtype=np.float32)

        landmarks = results.multi_face_landmarks[0].landmark
        xyz = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

        depth = self._landmarks_to_depth(xyz, H, W)
        embedding = self._landmarks_to_embedding(xyz)
        return depth, embedding

    def _landmarks_to_depth(self, xyz: np.ndarray, H: int, W: int) -> np.ndarray:
        try:
            import cv2
        except ImportError:
            return self._sparse_depth(xyz, H, W)

        xs = np.clip((xyz[:, 0] * W).round().astype(np.int32), 0, W - 1)
        ys = np.clip((xyz[:, 1] * H).round().astype(np.int32), 0, H - 1)
        z = xyz[:, 2]
        z = (z - z.min()) / (z.max() - z.min() + 1e-6)
        z = 1.0 - z

        depth = np.zeros((H, W), dtype=np.float32)
        counts = np.zeros((H, W), dtype=np.float32)
        for x, y, d in zip(xs, ys, z):
            depth[y, x] += d
            counts[y, x] += 1.0
        mask = counts > 0
        depth[mask] /= counts[mask]

        hull = cv2.convexHull(np.stack([xs, ys], axis=1))
        face_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(face_mask, hull, 255)

        unknown = ((face_mask > 0) & (~mask)).astype(np.uint8) * 255
        depth_u8 = np.clip(depth * 255.0, 0, 255).astype(np.uint8)
        depth_filled = cv2.inpaint(depth_u8, unknown, 3, cv2.INPAINT_TELEA).astype(np.float32) / 255.0
        depth_filled = cv2.GaussianBlur(depth_filled, (0, 0), sigmaX=5.0)
        depth_filled[face_mask == 0] = 0.0

        depth_resized = cv2.resize(
            depth_filled,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        return depth_resized[None, :, :].astype(np.float32)

    def _sparse_depth(self, xyz: np.ndarray, H: int, W: int) -> np.ndarray:
        xs = np.clip((xyz[:, 0] * W).round().astype(np.int32), 0, W - 1)
        ys = np.clip((xyz[:, 1] * H).round().astype(np.int32), 0, H - 1)
        z = xyz[:, 2]
        z = 1.0 - ((z - z.min()) / (z.max() - z.min() + 1e-6))
        depth = np.zeros((1, self.image_size, self.image_size), dtype=np.float32)
        sx = np.clip((xs * self.image_size / max(W, 1)).astype(np.int32), 0, self.image_size - 1)
        sy = np.clip((ys * self.image_size / max(H, 1)).astype(np.int32), 0, self.image_size - 1)
        depth[0, sy, sx] = z
        return depth

    def _fallback_depth(self, img_np: np.ndarray) -> np.ndarray:
        gray = img_np.astype(np.float32).mean(axis=2) / 255.0
        gray = 1.0 - gray
        depth = torch.from_numpy(gray).view(1, 1, gray.shape[0], gray.shape[1])
        depth = F.interpolate(
            depth,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return depth[0].numpy().astype(np.float32)

    def _landmarks_to_embedding(self, xyz: np.ndarray) -> np.ndarray:
        centered = xyz.copy()
        centered[:, :2] -= centered[:, :2].mean(axis=0, keepdims=True)
        centered[:, 2] -= centered[:, 2].mean()
        scale = np.linalg.norm(centered[:, :2], axis=1).max() + 1e-6
        centered /= scale

        flat = centered.reshape(-1)
        emb = np.zeros(self.hidden_dim, dtype=np.float32)
        n = min(self.hidden_dim, flat.shape[0])
        emb[:n] = flat[:n]
        return emb


def _depth_to_normal(depth: torch.Tensor) -> torch.Tensor:
    """
    Compute surface normals from a depth map via finite differences.

    Args:
        depth: (B, 1, H, W)

    Returns:
        normal: (B, 3, H, W) in [0, 1]
    """
    padded = F.pad(depth, [1, 1, 1, 1], mode="replicate")
    dz_dx = (padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2]) / 2.0
    dz_dy = (padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1]) / 2.0
    ones = torch.ones_like(dz_dx)
    normal = torch.cat([-dz_dx, -dz_dy, ones], dim=1)
    norm = normal.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normal = normal / norm
    return (normal.clamp(-1.0, 1.0) + 1.0) / 2.0


class GeometryConditioning(nn.Module):
    """
    MediaPipe-based geometry conditioning pipeline.

    The constructor keeps the DECA-compatible signature so existing trainer,
    loss, and inference code can instantiate it without changes.
    """

    def __init__(
        self,
        deca_model_path: str = "",
        deca_cfg_path: str = "",
        hidden_dim: int = 320,
        image_size: int = 256,
        device: str = "cuda",
        dpt_model_id: str = "",
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)
        self.mesh = MediaPipeMeshGeometry(hidden_dim=hidden_dim, image_size=image_size)

    @torch.no_grad()
    def forward(
        self,
        face_images: torch.Tensor,
        return_depth: bool = True,
        return_normal: bool = True,
        image_paths: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            face_images:  (B, 3, H, W) normalized to [-1, 1].
            return_depth: Include depth_map and depth_map_raw in output.
            return_normal: Include normal_map in output.
            image_paths:  Optional paths for .deca.pt cache lookup.
        """
        dev = face_images.device
        B = face_images.shape[0]

        if image_paths is not None:
            cached = self._load_cache_batch(image_paths, dev)
            if cached is not None:
                return cached

        depth_raw, param_embedding = self.mesh(face_images)
        depth_raw = depth_raw.to(dev)
        param_embedding = param_embedding.to(dev)

        result: Dict[str, torch.Tensor] = {
            "param_embedding": param_embedding,
            "codedict": None,
        }
        if return_depth:
            result["depth_map_raw"] = depth_raw
            result["depth_map"] = depth_raw.repeat(1, 3, 1, 1)
        if return_normal:
            result["normal_map"] = _depth_to_normal(depth_raw)

        if not return_depth:
            result["depth_map_raw"] = torch.zeros(B, 1, self.image_size, self.image_size, device=dev)
            result["depth_map"] = torch.zeros(B, 3, self.image_size, self.image_size, device=dev)
        if not return_normal:
            result["normal_map"] = torch.zeros(B, 3, self.image_size, self.image_size, device=dev)

        return result

    def _load_cache_batch(
        self,
        image_paths: List[str],
        device: torch.device,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Load pre-computed features from .deca.pt cache files if all present."""
        from pathlib import Path as _Path

        records = []
        for p in image_paths:
            cache_path = str(p) + ".deca.pt"
            if not _Path(cache_path).exists():
                return None
            records.append(torch.load(cache_path, map_location=device, weights_only=True))

        return {
            "param_embedding": torch.stack([r["param_embedding"] for r in records]).to(device),
            "depth_map": torch.stack([r["depth_map"] for r in records]).to(device),
            "depth_map_raw": torch.stack([r["depth_map_raw"] for r in records]).to(device),
            "normal_map": torch.stack([r["normal_map"] for r in records]).to(device),
            "codedict": None,
        }
