"""
models/region_encoder.py

Per-region facial identity encoder for geometry-aware face swapping.

Architecture per region (e.g. mouth):
  7-channel crop (RGB + normal + depth, 64x64)
         |
  Modified ResNet-50 (first conv: 3ch -> 7ch, pretrained weights kept)
         |
  Global average pool -> 2048-d feature vector
         |
  Linear projection  -> 512-d token
         |
  Small Transformer Encoder (self-attention across multiple views of same region)
         |
  Aggregated identity token [B, 1, 512]

Each region has its OWN ResNet-50 + transformer — fully independent.
Trained with contrastive loss (NT-Xent) on VGGFace2 triplets:
  anchor   = region crop, Person A, pose 1
  positive = region crop, Person A, pose 2  -> tokens should be CLOSE
  negative = region crop, Person B, any     -> tokens should be FAR

At inference: single image -> single crop per region -> single token per region
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Constants ──────────────────────────────────────────────────────────────────

REGIONS     = ["mouth", "eyes", "ears", "nose"]
IN_CHANNELS = 7       # RGB(3) + normal(3) + depth(1)
CROP_SIZE   = 64
TOKEN_DIM   = 512     # output token dimension (matches SDXL cross-attn dim)
RESNET_DIM  = 2048    # ResNet-50 global avg pool output


# ── ResNet-50 with 7-channel input ─────────────────────────────────────────────

def build_resnet50_7ch(pretrained: bool = True) -> nn.Module:
    """
    Load pretrained ResNet-50, replace first conv to accept 7-channel input.

    Extra 4 channels (beyond RGB) zero-initialised so the network starts
    identically to the 3-channel pretrained model and gradually learns
    to use depth + normal information.
    """
    import torchvision.models as tv_models

    resnet = tv_models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)

    # Replace first conv 3ch -> 7ch
    old_conv = resnet.conv1
    new_conv = nn.Conv2d(IN_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight   # copy RGB weights
        new_conv.weight[:, 3:] = 0.0               # zero-init geometry channels
    resnet.conv1 = new_conv
    resnet.fc    = nn.Identity()                   # remove classification head

    return resnet


# ── Single Region Encoder ──────────────────────────────────────────────────────

class RegionEncoder(nn.Module):
    """
    Encoder for ONE facial region.

    Consists of:
      - Dedicated ResNet-50 (7ch input, partially frozen)
      - Linear projection: 2048 -> TOKEN_DIM
      - Small transformer encoder: self-attention across multiple views
        (at inference, num_views=1 so transformer is a no-op effectively)

    Args:
        region_name      : Name of the region (for logging)
        token_dim        : Output token dimension
        pretrained       : Load pretrained ImageNet ResNet-50 weights
        freeze_stages    : Freeze first N ResNet stages (1-3), train layer4+
        transformer_layers: Depth of per-region transformer
        transformer_heads : Attention heads
        dropout          : Dropout rate
    """

    def __init__(
        self,
        region_name: str,
        token_dim: int = TOKEN_DIM,
        pretrained: bool = True,
        freeze_stages: int = 3,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.region_name = region_name
        self.token_dim   = token_dim

        # ── Dedicated ResNet-50 backbone ───────────────────────────────────
        self.backbone = build_resnet50_7ch(pretrained=pretrained)

        # Freeze early stages
        stages = [self.backbone.layer1, self.backbone.layer2, self.backbone.layer3]
        for i in range(min(freeze_stages, 3)):
            for p in stages[i].parameters():
                p.requires_grad_(False)
        # layer4 + avgpool always trained

        # ── Projection: 2048 -> token_dim ──────────────────────────────────
        self.projection = nn.Sequential(
            nn.Linear(RESNET_DIM, token_dim),
            nn.LayerNorm(token_dim),
            nn.Dropout(dropout),
        )

        # ── Per-region transformer (self-attention across views) ───────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=transformer_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(token_dim),
        )

        # Learnable positional embeddings (supports up to 16 views)
        self.pos_embed = nn.Parameter(torch.zeros(1, 16, token_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

    def encode_crop(self, crop: torch.Tensor) -> torch.Tensor:
        """
        Encode a single crop to a token vector.

        Args:
            crop: (B, 7, 64, 64)
        Returns:
            token: (B, token_dim)
        """
        feat = self.backbone(crop)      # (B, 2048)
        return self.projection(feat)    # (B, token_dim)

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        """
        Encode one or multiple views of the same region.

        Args:
            crops: (B, num_views, 7, 64, 64)  during training (multiple poses)
                   (B, 1,         7, 64, 64)  during inference (single image)
                   OR (B, 7, 64, 64) — single view, will be unsqueezed

        Returns:
            token: (B, 1, token_dim) — aggregated identity token
        """
        if crops.ndim == 4:
            crops = crops.unsqueeze(1)   # (B, 7, H, W) -> (B, 1, 7, H, W)

        B, V, C, H, W = crops.shape

        # Encode each view independently
        crops_flat = crops.view(B * V, C, H, W)          # (B*V, 7, H, W)
        tokens_flat = self.encode_crop(crops_flat)        # (B*V, token_dim)
        tokens = tokens_flat.view(B, V, self.token_dim)  # (B, V, token_dim)

        # Add positional embeddings
        tokens = tokens + self.pos_embed[:, :V, :]

        # Self-attention across views -> aggregate to identity token
        refined = self.transformer(tokens)   # (B, V, token_dim)

        # Mean pool across views -> single identity token
        identity_token = refined.mean(dim=1, keepdim=True)   # (B, 1, token_dim)
        return identity_token


# ── Full Face Region Encoder ───────────────────────────────────────────────────

class FaceRegionEncoder(nn.Module):
    """
    Full per-region facial encoder with 4 independent RegionEncoders.

    Each region (mouth, eyes, ears, nose) has its own:
      - ResNet-50 backbone (7ch input)
      - Linear projection
      - Transformer encoder

    Trained independently with contrastive loss per region.
    At inference, all 4 run in parallel on a single source image.

    Args:
        regions          : List of region names
        token_dim        : Output token dimension per region
        pretrained       : Load pretrained ResNet-50 weights
        freeze_stages    : Freeze first N ResNet stages (0-3)
        transformer_layers: Layers in each region's transformer
        transformer_heads : Attention heads
        dropout          : Dropout rate
    """

    def __init__(
        self,
        regions: Optional[List[str]] = None,
        token_dim: int = TOKEN_DIM,
        pretrained: bool = True,
        freeze_stages: int = 3,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.regions   = regions or REGIONS
        self.token_dim = token_dim

        # One independent encoder per region
        self.encoders = nn.ModuleDict({
            r: RegionEncoder(
                region_name=r,
                token_dim=token_dim,
                pretrained=pretrained,
                freeze_stages=freeze_stages,
                transformer_layers=transformer_layers,
                transformer_heads=transformer_heads,
                dropout=dropout,
            )
            for r in self.regions
        })

    def forward(
        self,
        region_crops: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Encode all region crops independently.

        Args:
            region_crops: Dict[region -> (B, num_views, 7, 64, 64)]
                          or Dict[region -> (B, 7, 64, 64)] at inference

        Returns:
            Dict[region -> (B, 1, token_dim)]
        """
        return {
            r: self.encoders[r](region_crops[r])
            for r in self.regions
            if r in region_crops
        }

    def get_tokens_for_cross_attention(
        self,
        region_crops: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Returns per-region tokens ready for U-Net cross-attention injection.

        Returns:
            Dict[region -> (B, 1, token_dim)]
        """
        return self.forward(region_crops)

    def get_region_encoder(self, region: str) -> RegionEncoder:
        """Return a single region's encoder (for per-region training)."""
        return self.encoders[region]

    def trainable_parameters(self, region: Optional[str] = None) -> List[nn.Parameter]:
        """
        Return trainable parameters.

        Args:
            region: If given, return only that region's parameters.
                    If None, return all trainable parameters.
        """
        if region is not None:
            return [p for p in self.encoders[region].parameters() if p.requires_grad]
        return [p for p in self.parameters() if p.requires_grad]
