"""
models/controlnet.py
ControlNet-style geometry conditioning for face swapping.

Implements a lightweight ControlNet branch that processes:
  - Rendered depth/normal maps from DECA (spatial conditioning)
  - 3DMM parameter embeddings (global conditioning via adaLN-style injection)

The branch outputs additive residuals for U-Net down-blocks and mid-block,
mirroring the original ControlNet architecture.

Reference:
  Adding Conditional Control to Text-to-Image Diffusion Models (Zhang et al., 2023)
  https://arxiv.org/abs/2302.05543
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from diffusers import ControlNetModel
    from diffusers.models.controlnet import ControlNetOutput
except ImportError:
    ControlNetModel = None  # type: ignore[assignment, misc]
    ControlNetOutput = None  # type: ignore[assignment, misc]


# ─────────────────────────────────────────────────────────────────────────────
# Small building blocks
# ─────────────────────────────────────────────────────────────────────────────


class ConditioningBlock(nn.Module):
    """
    Simple conv block: Conv → GroupNorm → SiLU (× 2).

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        groups (int): GroupNorm groups.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 32,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ZeroConv2d(nn.Module):
    """
    1×1 conv initialised with zero weights (ControlNet trainable copy trick).

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class GeometryEmbedding(nn.Module):
    """
    Project a 1-D geometry parameter embedding to a spatial feature map
    via broadcasting.  Used for adaLN-style global conditioning.

    Args:
        param_dim (int): Dimensionality of input embedding.
        out_channels (int): Number of output feature channels.
    """

    def __init__(self, param_dim: int = 320, out_channels: int = 320) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(param_dim, out_channels * 2),
            nn.SiLU(),
            nn.Linear(out_channels * 2, out_channels * 2),
        )
        self.out_channels = out_channels

    def forward(
        self,
        param_emb: torch.Tensor,
        spatial: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            param_emb: (B, param_dim) geometry parameter embedding.
            spatial: (B, C, H, W) spatial feature map to modulate.

        Returns:
            Modulated spatial feature map, same shape as ``spatial``.
        """
        stats = self.proj(param_emb)           # (B, 2*C)
        scale, shift = stats.chunk(2, dim=-1)  # each (B, C)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return spatial * (1.0 + scale) + shift


# ─────────────────────────────────────────────────────────────────────────────
# Geometry ControlNet
# ─────────────────────────────────────────────────────────────────────────────


class GeometryControlNet(nn.Module):
    """
    Lightweight ControlNet branch for 3D geometry conditioning.

    Takes depth/normal maps and 3DMM param embeddings as input and produces
    additive residuals at multiple scales for injection into the U-Net.

    Architecture:
        input_conv      → 3-channel conditioning image → initial feature map
        down_blocks     → progressively downsampled feature maps
        zero_convs      → learnable zero-init projection per scale
        mid_block       → bottleneck features
        mid_zero_conv   → zero-init projection for mid residual

    Args:
        conditioning_channels (int): Depth map channels (3 for RGB-like).
        block_out_channels (Tuple[int]): Channel widths per down-block level.
        param_dim (int): 3DMM parameter embedding dimension (from GeometryParamEncoder).
        groups (int): GroupNorm groups.
    """

    def __init__(
        self,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        param_dim: int = 320,
        groups: int = 32,
    ) -> None:
        super().__init__()
        self.block_out_channels = block_out_channels

        # Input projection: conditioning image → first feature map
        self.input_conv = nn.Sequential(
            nn.Conv2d(conditioning_channels, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, block_out_channels[0], 3, padding=1),
            nn.SiLU(),
        )

        # Down-sampling blocks
        self.down_blocks = nn.ModuleList()
        self.geo_embs = nn.ModuleList()
        self.zero_convs = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        ch_in = block_out_channels[0]
        for i, ch_out in enumerate(block_out_channels):
            block = ConditioningBlock(ch_in, ch_out, groups=min(groups, ch_out // 4))
            geo_emb = GeometryEmbedding(param_dim, ch_out)
            zero_conv = ZeroConv2d(ch_out, ch_out)
            self.down_blocks.append(block)
            self.geo_embs.append(geo_emb)
            self.zero_convs.append(zero_conv)

            if i < len(block_out_channels) - 1:
                self.downsamplers.append(
                    nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=2, padding=1)
                )
            else:
                self.downsamplers.append(nn.Identity())

            ch_in = ch_out

        # Mid block
        mid_ch = block_out_channels[-1]
        self.mid_block = ConditioningBlock(mid_ch, mid_ch, groups=min(groups, mid_ch // 4))
        self.mid_geo_emb = GeometryEmbedding(param_dim, mid_ch)
        self.mid_zero_conv = ZeroConv2d(mid_ch, mid_ch)

    def forward(
        self,
        depth_map: torch.Tensor,
        param_embedding: torch.Tensor,
        conditioning_scale: float = 1.0,
    ) -> Dict[str, List[torch.Tensor] | torch.Tensor]:
        """
        Forward pass through the geometry ControlNet branch.

        Args:
            depth_map: Rendered depth / normal map (B, 3, H, W).
            param_embedding: 3DMM param embedding (B, param_dim).
            conditioning_scale: Global scale for all residuals (0 = disabled).

        Returns:
            Dict with:
                'down_block_res_samples': List of residual tensors per level.
                'mid_block_res_sample': Mid-block residual tensor.
        """
        x = self.input_conv(depth_map)
        down_residuals: List[torch.Tensor] = []

        for block, geo_emb, zero_conv, downsampler in zip(
            self.down_blocks, self.geo_embs, self.zero_convs, self.downsamplers
        ):
            x = block(x)
            x = geo_emb(param_embedding, x)
            down_residuals.append(zero_conv(x) * conditioning_scale)
            x = downsampler(x)

        x = self.mid_block(x)
        x = self.mid_geo_emb(param_embedding, x)
        mid_residual = self.mid_zero_conv(x) * conditioning_scale

        return {
            "down_block_res_samples": down_residuals,
            "mid_block_res_sample": mid_residual,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Optional: diffusers ControlNetModel wrapper
# ─────────────────────────────────────────────────────────────────────────────


class DiffusersControlNetWrapper(nn.Module):
    """
    Wraps diffusers' ControlNetModel pre-trained on depth conditioning,
    enabling fine-tuning on DECA rendered depth maps.

    This is an alternative to GeometryControlNet when you want to start
    from a rich pre-trained ControlNet checkpoint.

    TODO: Set DEPTH_CONTROLNET_MODEL_ID to diffusers depth-ControlNet.

    Args:
        model_id (str): HuggingFace model ID or local path.
        dtype (torch.dtype): Model dtype.
        conditioning_scale (float): ControlNet influence scale.
    """

    # TODO: Set to local path or HuggingFace depth ControlNet model ID
    DEFAULT_MODEL_ID = "diffusers/controlnet-depth-sdxl-1.0"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        dtype: torch.dtype = torch.float16,
        conditioning_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if ControlNetModel is None:
            raise ImportError("diffusers >= 0.20 is required for ControlNetModel.")
        self.controlnet: ControlNetModel = ControlNetModel.from_pretrained(
            model_id, torch_dtype=dtype
        )
        self.conditioning_scale = conditioning_scale

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        depth_map: torch.Tensor,
        added_cond_kwargs: Optional[Dict] = None,
    ) -> Dict:
        """
        Args:
            noisy_latents: (B, 4, H/8, W/8).
            timesteps: (B,).
            encoder_hidden_states: (B, 77, 2048).
            depth_map: Rendered depth image (B, 3, H, W).
            added_cond_kwargs: SDXL extra conditioning.

        Returns:
            Dict with 'down_block_res_samples' and 'mid_block_res_sample'.
        """
        kwargs = dict(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=depth_map,
            conditioning_scale=self.conditioning_scale,
            return_dict=True,
        )
        if added_cond_kwargs:
            kwargs["added_cond_kwargs"] = added_cond_kwargs
        output: ControlNetOutput = self.controlnet(**kwargs)
        return {
            "down_block_res_samples": output.down_block_res_samples,
            "mid_block_res_sample": output.mid_block_res_sample,
        }
