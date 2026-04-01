"""
losses/geometry_loss.py
3D landmark consistency loss for geometry-aware face swap training.

Computes L2 distance between 3D facial landmarks predicted from the
generated face (via DECA re-encoding) and the target pose landmarks.

Optionally includes:
  - 2D projection consistency (re-projected vs target 2D landmarks)
  - Shape coefficient consistency (L2 distance in FLAME shape space)

Reference:
  DECA + HifiFace — 3D shape + semantic prior guidance pipeline.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryConsistencyLoss(nn.Module):
    """
    3D landmark consistency loss between generated and target faces.

    L_geo = ||L_3D(generated) - L_3D(target)||_2

    where L_3D(·) denotes DECA-decoded 68 3D landmark positions.

    Optionally:
      - ``shape_weight > 0``: penalises deviation of generated face shape
        coefficients from source shape (identity preservation in 3D).
      - ``proj_weight > 0``: 2D projection consistency on normalised coords.

    Args:
        deca_model_path (str): Path to DECA pretrained model.
        deca_cfg_path (str): Path to DECA config YAML.
        weight (float): Global geometry loss weight.
        shape_weight (float): Additional shape coefficient loss weight.
        proj_weight (float): 2D projection consistency weight.
        device (str): Torch device.
    """

    def __init__(
        self,
        deca_model_path: str = "pretrained/deca_model.tar",  # TODO: Set path
        deca_cfg_path: str = "configs/deca_cfg.yml",          # TODO: Set path
        weight: float = 0.5,
        shape_weight: float = 0.0,
        proj_weight: float = 0.0,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.weight = weight
        self.shape_weight = shape_weight
        self.proj_weight = proj_weight

        # Lazy-import to avoid hard dependency if DECA is not installed
        self._geometry: Optional[nn.Module] = None
        self._deca_kwargs = dict(
            deca_model_path=deca_model_path,
            deca_cfg_path=deca_cfg_path,
            device=device,
        )

    def _get_geometry(self) -> nn.Module:
        """Lazy-load the geometry conditioning module."""
        if self._geometry is None:
            from models.geometry import GeometryConditioning

            self._geometry = GeometryConditioning(**self._deca_kwargs)
            # Freeze DECA encoder (no gradients needed through it)
            for p in self._geometry.deca.parameters():
                p.requires_grad_(False)
        return self._geometry

    @staticmethod
    def _resize_for_deca(images: torch.Tensor) -> torch.Tensor:
        """Resize images to 224×224 as required by DECA."""
        return F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)

    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor,
        source_codedict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Compute geometry consistency loss.

        Args:
            generated: Swapped face images (B, 3, H, W), range [-1, 1].
            target: Target pose face images (B, 3, H, W), range [-1, 1].
            source_codedict: Optional pre-computed source DECA params dict
                             (avoids redundant encoding). Used for shape loss.

        Returns:
            Scalar geometry consistency loss.
        """
        geo = self._get_geometry()

        gen_resized = self._resize_for_deca(generated)
        tgt_resized = self._resize_for_deca(target)

        # Encode to get landmarks
        gen_lm = geo.get_landmarks(gen_resized)   # (B, 68, 3)
        tgt_lm = geo.get_landmarks(tgt_resized)   # (B, 68, 3)

        # Landmark L2 loss
        lm_loss = F.mse_loss(gen_lm, tgt_lm.detach())
        total_loss = self.weight * lm_loss

        # Optional shape coefficient consistency (source → generated)
        if self.shape_weight > 0 and source_codedict is not None:
            gen_codedict = geo.deca.encode(gen_resized)
            shape_loss = F.mse_loss(gen_codedict["shape"], source_codedict["shape"].detach())
            total_loss = total_loss + self.shape_weight * shape_loss

        # Optional 2D projection consistency
        if self.proj_weight > 0:
            gen_lm_2d = gen_lm[..., :2]  # drop z
            tgt_lm_2d = tgt_lm[..., :2]
            proj_loss = F.l1_loss(gen_lm_2d, tgt_lm_2d.detach())
            total_loss = total_loss + self.proj_weight * proj_loss

        return total_loss

    def compute_landmark_error(
        self,
        generated: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute mean Euclidean landmark error in normalised 3D space
        (evaluation metric, not for training).

        Args:
            generated: Swapped images (B, 3, H, W).
            target: Target pose images (B, 3, H, W).

        Returns:
            Mean per-landmark L2 distance scalar.
        """
        geo = self._get_geometry()
        gen_lm = geo.get_landmarks(self._resize_for_deca(generated))
        tgt_lm = geo.get_landmarks(self._resize_for_deca(target))
        # (B, 68, 3) → mean L2 per landmark per sample
        dist = torch.norm(gen_lm - tgt_lm, p=2, dim=-1)  # (B, 68)
        return dist.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Pixel Reconstruction Loss
# ─────────────────────────────────────────────────────────────────────────────


class PixelReconstructionLoss(nn.Module):
    """
    Pixel-level reconstruction loss (L1) between generated and target images.

    Args:
        weight (float): Loss weight scalar.
        loss_type (str): 'l1' | 'l2'. Default 'l1'.
    """

    def __init__(self, weight: float = 1.0, loss_type: str = "l1") -> None:
        super().__init__()
        self.weight = weight
        if loss_type == "l1":
            self.criterion = nn.L1Loss()
        elif loss_type == "l2":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type!r}")

    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            generated: Generated image (B, 3, H, W).
            target: Ground truth image (B, 3, H, W).
        Returns:
            Scalar reconstruction loss.
        """
        return self.weight * self.criterion(generated, target)
