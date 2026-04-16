"""
models/geometry.py
DPT-based geometry encoder for ControlNet conditioning.

Replaces DECA with a lightweight HuggingFace DPT monocular depth estimator.
No DECA installation, no Cython compilation — works on Kaggle via pip.

Produces the same output format as the DECA version:
  - depth_map       (B, 3, H, W)  — 3-ch depth for ControlNet conditioning
  - normal_map      (B, 3, H, W)  — surface normals from depth gradients
  - depth_map_raw   (B, 1, H, W)  — raw single-channel depth
  - param_embedding (B, hidden_dim) — zeros (geometry loss disabled)

Install: pip install transformers
Model:   Intel/dpt-large (downloaded automatically from HuggingFace)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# DPT Depth Estimator
# ─────────────────────────────────────────────────────────────────────────────


class DPTDepthEstimator(nn.Module):
    """
    Monocular depth estimator using HuggingFace DPT (Intel/dpt-large).
    Lazy-loaded on first forward pass to avoid slowing down model init.

    Args:
        model_id: HuggingFace model ID.
        device: Target device.
    """

    DEFAULT_MODEL_ID = "Intel/dpt-large"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str = "cuda") -> None:
        super().__init__()
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None

    def _load(self) -> None:
        from transformers import DPTForDepthEstimation, DPTImageProcessor
        self._processor = DPTImageProcessor.from_pretrained(self.model_id)
        self._model = (
            DPTForDepthEstimation.from_pretrained(self.model_id)
            .to(self.device)
            .float()
            .eval()
        )
        for p in self._model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor, output_size: int = 256) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W) normalised to [-1, 1].
            output_size: Output depth map resolution.

        Returns:
            depth: (B, 1, output_size, output_size) normalised to [0, 1].
        """
        if self._model is None:
            self._load()

        # Convert [-1, 1] → PIL images for DPT processor
        imgs_01 = ((images.detach().cpu().float() + 1.0) / 2.0).clamp(0, 1)
        imgs_np = (imgs_01 * 255).byte().permute(0, 2, 3, 1).numpy()

        from PIL import Image as _PILImage
        pil_images = [_PILImage.fromarray(imgs_np[i]) for i in range(imgs_np.shape[0])]

        inputs = self._processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.amp.autocast("cuda", enabled=False):
            depth = self._model(**inputs).predicted_depth  # (B, H', W')

        B = depth.shape[0]
        depth = depth.unsqueeze(1).float()  # (B, 1, H', W')
        depth = F.interpolate(depth, size=(output_size, output_size), mode="bilinear", align_corners=False)

        # Normalise per-image to [0, 1]
        d_min = depth.view(B, -1).min(dim=1)[0].view(B, 1, 1, 1)
        d_max = depth.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
        depth = (depth - d_min) / (d_max - d_min + 1e-8)
        return depth


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# Full Geometry Conditioning Pipeline
# ─────────────────────────────────────────────────────────────────────────────


class GeometryConditioning(nn.Module):
    """
    DPT-based geometry conditioning pipeline (DECA-free).

    Drop-in replacement for the DECA-based version — same __init__ signature,
    same forward output keys. The deca_model_path / deca_cfg_path args are
    accepted but ignored (kept for config compatibility).

    Args:
        deca_model_path: Ignored. Kept for API compatibility.
        deca_cfg_path:   Ignored. Kept for API compatibility.
        hidden_dim:      Param embedding dimension (output is zeros).
        image_size:      Output map resolution.
        device:          Torch device.
        dpt_model_id:    HuggingFace model ID for DPT.
    """

    def __init__(
        self,
        deca_model_path: str = "",
        deca_cfg_path: str = "",
        hidden_dim: int = 320,
        image_size: int = 256,
        device: str = "cuda",
        dpt_model_id: str = DPTDepthEstimator.DEFAULT_MODEL_ID,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.hidden_dim = hidden_dim

        self.dpt = DPTDepthEstimator(model_id=dpt_model_id, device=device)

        # 1→3 channel depth projection for ControlNet input
        self.depth_project = nn.Conv2d(1, 3, kernel_size=1, bias=True).to(device)
        nn.init.ones_(self.depth_project.weight)
        nn.init.zeros_(self.depth_project.bias)

    def forward(
        self,
        face_images: torch.Tensor,
        return_depth: bool = True,
        return_normal: bool = True,
        image_paths: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            face_images:  (B, 3, H, W) normalised to [-1, 1].
            return_depth: Include depth_map and depth_map_raw in output.
            return_normal: Include normal_map in output.
            image_paths:  Optional paths for .deca.pt cache lookup.

        Returns:
            Dict with keys:
                'param_embedding': (B, hidden_dim) — zeros
                'depth_map':       (B, 3, H, W)
                'depth_map_raw':   (B, 1, H, W)
                'normal_map':      (B, 3, H, W)
                'codedict':        None
        """
        dev = self.depth_project.weight.device
        B = face_images.shape[0]

        # Try loading from pre-computed cache (.deca.pt files)
        if image_paths is not None:
            cached = self._load_cache_batch(image_paths, dev)
            if cached is not None:
                return cached

        result: Dict[str, torch.Tensor] = {
            "param_embedding": torch.zeros(B, self.hidden_dim, device=dev),
            "codedict": None,
        }

        if return_depth or return_normal:
            depth_raw = self.dpt(face_images, output_size=self.image_size).to(dev)
            if return_depth:
                result["depth_map_raw"] = depth_raw
                result["depth_map"] = self.depth_project(depth_raw)
            if return_normal:
                result["normal_map"] = _depth_to_normal(depth_raw)

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
            "param_embedding": torch.stack([r["param_embedding"] for r in records]),
            "depth_map":       torch.stack([r["depth_map"]       for r in records]),
            "depth_map_raw":   torch.stack([r["depth_map_raw"]   for r in records]),
            "normal_map":      torch.stack([r["normal_map"]      for r in records]),
            "codedict":        None,
        }
