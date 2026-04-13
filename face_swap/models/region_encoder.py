"""
models/region_encoder.py
ResNet-50 + Transformer based per-region facial feature encoders.

Architecture:
  - Separate pretrained ResNet-50 backbone per facial region (eyes, nose, mouth, ears)
  - Each backbone is frozen; only the transformer head + projection are trainable
  - A multi-layer conv block fuses multi-modal input (RGB + depth + normal -> 3ch)
  - Trainable transformer encoder layers on top of ResNet-50 spatial features
  - A projection MLP maps transformer output tokens -> cross-attention dim

For 64x64 region crops, ResNet-50 produces (B, 2048, 2, 2) spatial features
which are flattened to 4 tokens of 2048-dim, processed by the transformer,
and projected to (B, 4, projection_dim) for IP-Adapter style injection.

Reference: IP-Adapter (https://arxiv.org/abs/2308.06721) -- image prompt encoder design.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision.models as tv_models
except ImportError:
    tv_models = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGIONS: List[str] = ["eyes", "nose", "mouth", "ears"]


# ---------------------------------------------------------------------------
# Input conv block (fuses RGB + depth + normal -> 3-ch for ResNet)
# ---------------------------------------------------------------------------


class InputConvBlock(nn.Module):
    """
    Multi-layer convolutional block that fuses multi-modal region crops
    into 3 channels suitable for the frozen ResNet-50 backbone.

    Channel layout expected:
        channels 0-2 : RGB
        channels 3-5 : surface normal (X, Y, Z) in [0, 1]
        channel  6   : depth (scalar) in [0, 1]

    Uses a residual design: at initialisation the conv path outputs zeros
    so the block acts as a pure RGB pass-through.  During training it
    learns to incorporate depth and normal cues.

    Args:
        in_channels (int):  Number of input channels (7 = RGB+normal+depth).
        out_channels (int): Number of output channels (3 to match ResNet).
        mid_channels (int): Hidden channel width in the conv block.
    """

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 3,
        mid_channels: int = 32,
    ) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=True),
        )
        # Zero-init final conv so the learned path starts at zero;
        # output = RGB + 0 = pure RGB at the beginning of training.
        nn.init.zeros_(self.conv_block[-1].weight)
        nn.init.zeros_(self.conv_block[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W) -- [RGB(3) || normal(3) || depth(1)].
        Returns:
            (B, 3, H, W) fused representation.
        """
        rgb = x[:, :3]                 # (B, 3, H, W)
        residual = self.conv_block(x)  # (B, 3, H, W)  -- starts at zero
        return rgb + residual


# ---------------------------------------------------------------------------
# Transformer head
# ---------------------------------------------------------------------------


class TransformerHead(nn.Module):
    """
    Trainable transformer encoder layers applied to ResNet-50 spatial tokens.

    Adds learned positional embeddings and applies ``num_layers`` standard
    transformer encoder layers with pre-norm (LayerNorm before attention).

    Args:
        embed_dim (int): Token embedding dimension (2048 for ResNet-50).
        num_heads (int): Number of attention heads.
        num_layers (int): Number of transformer encoder layers.
        num_spatial_tokens (int): Expected number of spatial tokens (4 for 2x2).
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        embed_dim: int = 2048,
        num_heads: int = 8,
        num_layers: int = 2,
        num_spatial_tokens: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_spatial_tokens, embed_dim) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Spatial tokens (B, num_tokens, embed_dim).
        Returns:
            Transformed tokens (B, num_tokens, embed_dim).
        """
        x = x + self.pos_embed
        x = self.encoder(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Projection MLP
# ---------------------------------------------------------------------------


class ProjectionMLP(nn.Module):
    """
    Projects transformer output tokens to the cross-attention dimension.
    Applied independently per token.

    Args:
        in_dim (int): Input dimension per token (ResNet-50 = 2048).
        projection_dim (int): Output feature dimension per token.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        in_dim: int = 2048,
        projection_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.projection_dim = projection_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token features (B, num_tokens, in_dim).
        Returns:
            Projected tokens (B, num_tokens, projection_dim).
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# Single-Region Encoder
# ---------------------------------------------------------------------------


class SingleRegionEncoder(nn.Module):
    """
    Encoder for one facial region.

    Architecture (for 64x64 region crops with in_channels=7):
        InputConvBlock : (B, 7, 64, 64) -> (B, 3, 64, 64)
        ResNet-50      : (B, 3, 64, 64) -> (B, 2048, 2, 2)   [frozen]
        Flatten spatial: (B, 2048, 2, 2) -> (B, 4, 2048)
        TransformerHead: (B, 4, 2048)    -> (B, 4, 2048)      [trainable]
        ProjectionMLP  : (B, 4, 2048)    -> (B, 4, proj_dim)  [trainable]

    Args:
        resnet_backbone (nn.Module): Pretrained ResNet-50 (layers only, no FC).
        resnet_dim (int): ResNet-50 output channel dimension (2048).
        projection_dim (int): Output feature dimension.
        num_tokens (int): Number of output tokens per region (4 = 2x2 spatial).
        in_channels (int): Input channels per crop. 7 = RGB(3)+normal(3)+depth(1).
        freeze_backbone (bool): Whether to freeze the ResNet-50 weights.
        num_transformer_layers (int): Transformer encoder layers in the head.
        num_heads (int): Attention heads in the transformer.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        resnet_backbone: nn.Module,
        resnet_dim: int = 2048,
        projection_dim: int = 512,
        num_tokens: int = 4,
        in_channels: int = 7,
        freeze_backbone: bool = True,
        num_transformer_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = resnet_backbone
        self.num_tokens = num_tokens
        self._pool_size = int(math.ceil(num_tokens ** 0.5))

        # Multi-modal input fusion (7ch -> 3ch)
        self.input_conv: Optional[InputConvBlock] = None
        if in_channels != 3:
            self.input_conv = InputConvBlock(
                in_channels=in_channels, out_channels=3
            )

        # Transformer head on ResNet spatial features
        self.transformer_head = TransformerHead(
            embed_dim=resnet_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            num_spatial_tokens=num_tokens,
            dropout=dropout,
        )

        # Final projection to cross-attention dimension
        self.projection = ProjectionMLP(
            in_dim=resnet_dim,
            projection_dim=projection_dim,
            dropout=dropout,
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(self, crop: torch.Tensor) -> torch.Tensor:
        """
        Args:
            crop: Region crop tensor, shape (B, in_channels, H, W).
                  With in_channels=7: [RGB(3) || normal(3) || depth(1)].
        Returns:
            Feature tokens, shape (B, num_tokens, projection_dim).
        """
        # Fuse modalities -> 3-channel input
        if self.input_conv is not None:
            x = self.input_conv(crop)       # (B, 7, H, W) -> (B, 3, H, W)
        else:
            x = crop

        # Extract spatial features from ResNet-50
        features = self._extract_features(x)    # (B, 2048, H', W')

        # Adaptive pool to get exactly num_tokens spatial positions
        features = F.adaptive_avg_pool2d(
            features, self._pool_size
        )                                        # (B, 2048, 2, 2)

        # Flatten spatial dims to token sequence
        tokens = features.flatten(2).transpose(1, 2)  # (B, 4, 2048)
        tokens = tokens[:, : self.num_tokens]          # trim if pool overshoots

        # Transformer self-attention over spatial tokens
        tokens = self.transformer_head(tokens)   # (B, 4, 2048)

        # Project to cross-attention dimension
        return self.projection(tokens)           # (B, 4, projection_dim)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial feature map from ResNet-50 (everything before avgpool/fc).

        Args:
            x: (B, 3, H, W) input image tensor.
        Returns:
            (B, 2048, H', W') spatial feature map from layer4.
        """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x


# ---------------------------------------------------------------------------
# Multi-Region Encoder
# ---------------------------------------------------------------------------


class FaceRegionEncoder(nn.Module):
    """
    Multi-region facial encoder with ResNet-50 + Transformer architecture.

    Produces identity feature tokens for each facial region via:
      - 4 independent frozen ResNet-50 backbones (one per region)
      - Trainable transformer layers on top of each
      - Trainable input conv blocks for multi-modal fusion
      - Trainable projection MLPs for cross-attention dimension

    Total output: (B, num_regions * num_tokens, projection_dim)
                  = (B, 4 * 4, 512) = (B, 16, 512) by default.

    Args:
        projection_dim (int): Output feature dimension per token.
        num_tokens (int): Number of output tokens per region (4 = 2x2 spatial).
        regions (List[str]): Region names to encode. Defaults to REGIONS.
        in_channels (int): Channels per region crop.
                           7 = RGB(3) + normal(3) + depth(1)  [default]
                           3 = RGB-only.
        shared_backbone (bool): Share one ResNet-50 across all regions.
                                Default False (separate per region).
        freeze_backbone (bool): Freeze ResNet-50 weights. Default True.
        pretrained (bool): Load ImageNet pretrained ResNet-50 weights.
        num_transformer_layers (int): Transformer layers per region head.
        num_heads (int): Attention heads in transformer layers.
        dropout (float): Dropout in transformer + projection.
    """

    def __init__(
        self,
        projection_dim: int = 512,
        num_tokens: int = 4,
        regions: Optional[List[str]] = None,
        in_channels: int = 7,
        shared_backbone: bool = False,
        freeze_backbone: bool = True,
        pretrained: bool = True,
        num_transformer_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if regions is None:
            regions = REGIONS
        self.regions = regions
        self.num_tokens = num_tokens
        self.projection_dim = projection_dim
        self.in_channels = in_channels

        if tv_models is None:
            raise ImportError(
                "torchvision is required for FaceRegionEncoder. "
                "Install with: pip install torchvision"
            )

        # Build ResNet-50 backbone(s) -------------------------------------------
        def _make_resnet() -> nn.Module:
            weights = (
                tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            )
            model = tv_models.resnet50(weights=weights)
            # Strip the classification head -- we only need the conv trunk
            model.fc = nn.Identity()
            model.avgpool = nn.Identity()
            return model

        if shared_backbone:
            backbone = _make_resnet()
            backbones = {r: backbone for r in regions}
        else:
            backbones = {r: _make_resnet() for r in regions}

        resnet_dim: int = 2048  # ResNet-50 layer4 output channels

        self.encoders = nn.ModuleDict(
            {
                r: SingleRegionEncoder(
                    resnet_backbone=backbones[r],
                    resnet_dim=resnet_dim,
                    projection_dim=projection_dim,
                    num_tokens=num_tokens,
                    in_channels=in_channels,
                    freeze_backbone=freeze_backbone,
                    num_transformer_layers=num_transformer_layers,
                    num_heads=num_heads,
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
            region_crops: Dict mapping region name -> crop tensor
                          (B, in_channels, H, W).  With in_channels=7 the
                          channel order is [RGB(3) || normal(3) || depth(1)].
                          Missing regions are skipped.

        Returns:
            Dict mapping region name -> feature tensor (B, num_tokens, projection_dim).
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
            region_crops: Dict of region crops (B, in_channels, H, W) per region.

        Returns:
            Concatenated features (B, num_regions * num_tokens, projection_dim).
        """
        region_feats = self.forward(region_crops)
        feats_list = [region_feats[r] for r in self.regions if r in region_feats]
        return torch.cat(feats_list, dim=1)

    @property
    def output_tokens(self) -> int:
        """Total number of output tokens (num_regions x num_tokens)."""
        return len(self.regions) * self.num_tokens

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Return only the trainable parameters (input conv + transformer + projection)."""
        params: List[nn.Parameter] = []
        for enc in self.encoders.values():
            if enc.input_conv is not None:
                params.extend(list(enc.input_conv.parameters()))
            params.extend(list(enc.transformer_head.parameters()))
            params.extend(list(enc.projection.parameters()))
        return params
