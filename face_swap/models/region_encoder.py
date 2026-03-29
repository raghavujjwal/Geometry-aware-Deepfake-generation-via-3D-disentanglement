"""
models/region_encoder.py
ViT-based per-region facial feature encoders.

Architecture:
  - Separate ViT-B/16 encoder per facial region (eyes, nose, lips, skin, hairline, ears)
  - Each encoder takes a (3, 64, 64) region crop and outputs a (1, projection_dim) feature vector
  - A lightweight projection MLP maps ViT CLS token → projection_dim
  - All encoders share the same ViT backbone weights (with separate projections)
    unless ``shared_backbone=False``

Reference: IP-Adapter (https://arxiv.org/abs/2308.06721) — image prompt encoder design.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm  # Used for pre-built ViT models
except ImportError:
    timm = None  # type: ignore[assignment]


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

REGIONS: List[str] = ["eyes", "nose", "lips", "skin", "hairline", "ears"]


# ─────────────────────────────────────────────────────────────────────────────
# Projection MLP
# ─────────────────────────────────────────────────────────────────────────────


class ProjectionMLP(nn.Module):
    """
    Two-layer MLP that projects ViT CLS token → (num_tokens, projection_dim).

    Args:
        in_dim (int): ViT hidden dimension (e.g. 768 for ViT-B).
        projection_dim (int): Output feature dimension per token.
        num_tokens (int): Number of learned query tokens to produce.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        in_dim: int = 768,
        projection_dim: int = 512,
        num_tokens: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.projection_dim = projection_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim * 2, num_tokens * projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: CLS token, shape (B, in_dim).
        Returns:
            Region features, shape (B, num_tokens, projection_dim).
        """
        out = self.net(x)  # (B, num_tokens * projection_dim)
        return out.view(-1, self.num_tokens, self.projection_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Single-Region Encoder
# ─────────────────────────────────────────────────────────────────────────────


class SingleRegionEncoder(nn.Module):
    """
    Encoder for one facial region.

    Wraps a frozen / fine-tuned ViT backbone and a trainable projection MLP.

    Args:
        vit_backbone (nn.Module): Shared ViT backbone (timm model or similar).
        vit_hidden_dim (int): CLS token dimension from the ViT.
        projection_dim (int): Output feature dimension.
        num_tokens (int): Number of output tokens per region.
        freeze_backbone (bool): Whether to freeze the ViT weights.
        dropout (float): MLP dropout.
    """

    def __init__(
        self,
        vit_backbone: nn.Module,
        vit_hidden_dim: int = 768,
        projection_dim: int = 512,
        num_tokens: int = 4,
        freeze_backbone: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = vit_backbone
        self.projection = ProjectionMLP(
            in_dim=vit_hidden_dim,
            projection_dim=projection_dim,
            num_tokens=num_tokens,
            dropout=dropout,
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(self, crop: torch.Tensor) -> torch.Tensor:
        """
        Args:
            crop: Region crop tensor, shape (B, 3, H, W).
        Returns:
            Feature tokens, shape (B, num_tokens, projection_dim).
        """
        # Extract CLS token from ViT
        cls_token = self._extract_cls(crop)  # (B, vit_hidden_dim)
        return self.projection(cls_token)  # (B, num_tokens, projection_dim)

    def _extract_cls(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract CLS / pooled feature from ViT backbone.

        Handles both timm ViT (forward_features) and HuggingFace models.
        """
        if hasattr(self.backbone, "forward_features"):
            # timm ViT: returns (B, seq_len+1, dim) or (B, dim) for some models
            feats = self.backbone.forward_features(x)
            if feats.ndim == 3:
                return feats[:, 0]  # CLS token
            return feats
        elif hasattr(self.backbone, "get_image_features"):
            # HuggingFace CLIP-style
            return self.backbone.get_image_features(pixel_values=x)
        else:
            return self.backbone(x)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Region Encoder
# ─────────────────────────────────────────────────────────────────────────────


class FaceRegionEncoder(nn.Module):
    """
    Multi-region facial encoder: produces identity feature tokens for each
    facial region via dedicated projection heads over a shared ViT backbone.

    Architecture:
        - 1 shared ViT-B/16 backbone (optionally frozen)
        - 6 independent projection MLPs (one per region)
        - Total output: (B, num_regions, num_tokens, projection_dim)

    Args:
        vit_model_name (str): timm model name, e.g. 'vit_base_patch16_224'.
        projection_dim (int): Output feature dimension per token.
        num_tokens (int): Number of output tokens per region (IP-Adapter style).
        regions (List[str]): Region names to encode. Defaults to REGIONS.
        shared_backbone (bool): Use one backbone for all regions. Default True.
        freeze_backbone (bool): Freeze ViT weights. Default True.
        pretrained (bool): Load timm pretrained weights. Default True.
        dropout (float): Dropout in projection MLPs.
    """

    def __init__(
        self,
        vit_model_name: str = "vit_base_patch16_224",
        projection_dim: int = 512,
        num_tokens: int = 4,
        regions: Optional[List[str]] = None,
        shared_backbone: bool = True,
        freeze_backbone: bool = True,
        pretrained: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if regions is None:
            regions = REGIONS
        self.regions = regions
        self.num_tokens = num_tokens
        self.projection_dim = projection_dim

        if timm is None:
            raise ImportError("timm is required for FaceRegionEncoder. Install with: pip install timm")

        # Build ViT backbone(s)
        backbone = timm.create_model(
            vit_model_name,
            pretrained=pretrained,
            num_classes=0,  # remove classification head
        )
        vit_hidden_dim: int = backbone.num_features  # typically 768 for ViT-B

        # One backbone per region or shared
        if shared_backbone:
            backbones = {r: backbone for r in regions}
        else:
            backbones = {
                r: timm.create_model(vit_model_name, pretrained=pretrained, num_classes=0)
                for r in regions
            }

        self.encoders = nn.ModuleDict(
            {
                r: SingleRegionEncoder(
                    vit_backbone=backbones[r],
                    vit_hidden_dim=vit_hidden_dim,
                    projection_dim=projection_dim,
                    num_tokens=num_tokens,
                    freeze_backbone=freeze_backbone,
                    dropout=dropout,
                )
                for r in regions
            }
        )

    def forward(
        self, region_crops: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Encode all facial region crops.

        Args:
            region_crops: Dict mapping region name → crop tensor (B, 3, H, W).
                          Missing regions are skipped.

        Returns:
            Dict mapping region name → feature tensor (B, num_tokens, projection_dim).
        """
        outputs: Dict[str, torch.Tensor] = {}
        for region in self.regions:
            if region not in region_crops:
                continue
            crop = region_crops[region]
            outputs[region] = self.encoders[region](crop)
        return outputs

    def get_concatenated_features(
        self,
        region_crops: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass returning all region features concatenated along token dim.

        Args:
            region_crops: Dict of region crops (B, 3, H, W) per region.

        Returns:
            Concatenated features (B, num_regions * num_tokens, projection_dim).
        """
        region_feats = self.forward(region_crops)
        # Concatenate in canonical region order
        feats_list = [region_feats[r] for r in self.regions if r in region_feats]
        return torch.cat(feats_list, dim=1)  # (B, R*T, projection_dim)

    @property
    def output_tokens(self) -> int:
        """Total number of output tokens (num_regions × num_tokens)."""
        return len(self.regions) * self.num_tokens

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Return only the trainable (projection) parameters."""
        params = []
        for enc in self.encoders.values():
            params.extend(list(enc.projection.parameters()))
        return params
