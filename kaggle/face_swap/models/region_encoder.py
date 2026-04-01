"""
models/region_encoder.py

Per-region facial feature encoder for geometry-aware face swapping.

Architecture per region:
  stacked crop (7ch: RGB + normal + depth, 64x64)
       |
  Modified ResNet-50 (first conv: 3ch -> 7ch, rest pretrained)
       |
  Global average pool -> feature vector (2048-d)
       |
  Linear projection -> token (512-d)

All 4 region tokens -> sequence [B, 4, 512]
       |
  Small Transformer Encoder (4 layers, 8 heads, trained from scratch)
       |
  Refined tokens [B, 4, 512] -> U-Net cross-attention

What is trained:
  - ResNet-50 first conv layer (7ch input, initialised from 3ch pretrained)
  - ResNet-50 last layer (layer4 + avgpool) - fine-tuned
  - Linear projection per region - trained from scratch
  - Transformer encoder - trained from scratch

What is frozen:
  - ResNet-50 layers 1-3 (pretrained ImageNet features)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Constants ──────────────────────────────────────────────────────────────────

REGIONS = ["mouth", "eyes", "ears", "nose"]

IN_CHANNELS  = 7      # RGB(3) + normal(3) + depth(1)
CROP_SIZE    = 64
TOKEN_DIM    = 512    # output token dimension (matches SDXL cross-attn dim)
RESNET_DIM   = 2048   # ResNet-50 output feature dimension


# ── ResNet-50 with 7-channel input ─────────────────────────────────────────────

def build_resnet50_7ch(pretrained: bool = True) -> nn.Module:
    """
    Load pretrained ResNet-50, modify first conv to accept 7-channel input.

    The extra 4 channels (beyond RGB) are initialised to zero so the network
    starts behaving identically to the 3-channel pretrained model.

    Returns:
        ResNet-50 backbone with:
          - conv1: 7ch in, 64ch out
          - layer1-4: pretrained
          - avgpool: global average pool
          - No fc head (we add our own projection)
    """
    import torchvision.models as models

    resnet = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)

    # Replace first conv: 3ch -> 7ch
    old_conv = resnet.conv1   # (64, 3, 7, 7)
    new_conv = nn.Conv2d(
        IN_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    with torch.no_grad():
        # Copy pretrained RGB weights
        new_conv.weight[:, :3] = old_conv.weight
        # Zero-init extra channels (normal + depth)
        new_conv.weight[:, 3:] = 0.0
    resnet.conv1 = new_conv

    # Remove classification head
    resnet.fc = nn.Identity()

    return resnet


# ── Single-Region Encoder ──────────────────────────────────────────────────────

class SingleRegionEncoder(nn.Module):
    """
    Encodes one region's 7-channel crop into a single TOKEN_DIM feature token.

    Shared ResNet-50 backbone, region-specific linear projection.

    Args:
        backbone   : Shared ResNet-50 instance
        token_dim  : Output dimension (default 512)
        dropout    : Dropout before projection
    """

    def __init__(
        self,
        backbone: nn.Module,
        token_dim: int = TOKEN_DIM,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(RESNET_DIM, token_dim),
            nn.LayerNorm(token_dim),
        )

    def forward(self, crop: torch.Tensor) -> torch.Tensor:
        """
        Args:
            crop: (B, 7, 64, 64)
        Returns:
            token: (B, token_dim)
        """
        feat = self.backbone(crop)   # (B, 2048)  [after avgpool + identity fc]
        return self.proj(feat)       # (B, token_dim)


# ── Multi-Region Encoder ───────────────────────────────────────────────────────

class FaceRegionEncoder(nn.Module):
    """
    Multi-region facial encoder.

    Architecture:
      - 1 shared ResNet-50 backbone (7ch input, partially frozen)
      - 1 linear projection per region (trained from scratch)
      - 1 small Transformer Encoder across all region tokens (trained from scratch)

    Input:  Dict[region -> (B, 7, 64, 64)]
    Output: Dict[region -> (B, token_dim)]
            + concatenated (B, num_regions, token_dim) for cross-attention

    Args:
        regions       : List of region names to encode
        token_dim     : Feature token dimension (should match SDXL cross-attn dim)
        pretrained    : Load pretrained ResNet-50 weights
        freeze_stages : How many ResNet stages to freeze (0=none, 3=freeze layer1-3)
        transformer_layers: Depth of the cross-region transformer
        transformer_heads : Attention heads in transformer
        dropout       : Dropout rate
    """

    def __init__(
        self,
        regions: Optional[List[str]] = None,
        token_dim: int = TOKEN_DIM,
        pretrained: bool = True,
        freeze_stages: int = 3,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.regions   = regions or REGIONS
        self.token_dim = token_dim

        # ── Shared ResNet-50 backbone ──────────────────────────────────────
        backbone = build_resnet50_7ch(pretrained=pretrained)

        # Freeze early stages
        if freeze_stages >= 1:
            for p in backbone.layer1.parameters():
                p.requires_grad_(False)
        if freeze_stages >= 2:
            for p in backbone.layer2.parameters():
                p.requires_grad_(False)
        if freeze_stages >= 3:
            for p in backbone.layer3.parameters():
                p.requires_grad_(False)
        # layer4 + avgpool always trained

        self.backbone = backbone

        # ── Per-region projection heads ────────────────────────────────────
        self.region_encoders = nn.ModuleDict({
            r: SingleRegionEncoder(backbone=self.backbone, token_dim=token_dim, dropout=dropout)
            for r in self.regions
        })

        # ── Cross-region Transformer Encoder ──────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=transformer_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,     # expects (B, seq, dim)
            norm_first=True,      # pre-norm (more stable training)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(token_dim),
        )

        # Learnable positional embeddings for each region
        self.pos_embed = nn.Parameter(
            torch.zeros(1, len(self.regions), token_dim)
        )
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(
        self,
        region_crops: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Encode all region crops and refine with cross-region attention.

        Args:
            region_crops: Dict[region -> (B, 7, 64, 64)]

        Returns:
            Dict[region -> (B, token_dim)] — refined tokens per region
        """
        tokens = []
        present_regions = []

        for r in self.regions:
            if r not in region_crops:
                continue
            tok = self.region_encoders[r](region_crops[r])   # (B, token_dim)
            tokens.append(tok)
            present_regions.append(r)

        if not tokens:
            return {}

        # Stack into sequence: (B, num_regions, token_dim)
        seq = torch.stack(tokens, dim=1)

        # Add positional embeddings (only for present regions)
        n = seq.shape[1]
        seq = seq + self.pos_embed[:, :n, :]

        # Cross-region transformer
        refined = self.transformer(seq)   # (B, num_regions, token_dim)

        return {
            r: refined[:, i, :]
            for i, r in enumerate(present_regions)
        }

    def get_tokens_for_cross_attention(
        self,
        region_crops: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass returning all tokens concatenated for U-Net cross-attention.

        Returns:
            (B, num_regions, token_dim) tensor
        """
        region_feats = self.forward(region_crops)
        tokens = [region_feats[r] for r in self.regions if r in region_feats]
        return torch.stack(tokens, dim=1)   # (B, R, token_dim)

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Return all trainable parameters (for optimiser setup)."""
        params = []
        # ResNet first conv (7ch) + layer4
        params += list(self.backbone.conv1.parameters())
        params += list(self.backbone.layer4.parameters())
        params += list(self.backbone.avgpool.parameters())
        # All projection heads
        for enc in self.region_encoders.values():
            params += list(enc.proj.parameters())
        # Transformer + positional embeddings
        params += list(self.transformer.parameters())
        params += [self.pos_embed]
        return params
