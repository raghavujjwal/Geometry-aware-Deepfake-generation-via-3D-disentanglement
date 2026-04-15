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
    ControlNet branch whose residuals exactly match SDXL U-Net's skip connections.

    SDXL (stable-diffusion-xl-base-1.0) down_block_types:
        ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"]
    layers_per_block = 2, block_out_channels = [320, 640, 1280]

    Expected residual list (8 down + 1 mid):
        Stage 0 — 32×32 × 2 layers + downsampler  → [B,320,32,32] ×2, [B,320,16,16]
        Stage 1 — 16×16 × 2 layers + downsampler  → [B,640,16,16] ×2, [B,640,8,8]
        Stage 2 — 8×8  × 2 layers (no downsample) → [B,1280,8,8]  ×2
        Mid     — 8×8                              → [B,1280,8,8]

    The conditioning input must be passed at latent spatial size (image_size // 8),
    e.g. 32×32 for 256×256 images. The trainer handles this resize.

    Args:
        conditioning_channels (int): Input channels (6 = depth(3) + normal(3)).
        internal_channels (int): Internal feature width (lightweight backbone).
        param_dim (int): Geometry param embedding dimension.
    """

    # SDXL channel sizes at each residual position
    _SDXL_CHANNELS = [320, 320, 320, 640, 640, 640, 1280, 1280, 1280]  # 8 down + 1 mid

    def __init__(
        self,
        conditioning_channels: int = 6,
        internal_channels: int = 128,
        param_dim: int = 320,
    ) -> None:
        super().__init__()
        C = internal_channels
        g = min(32, C // 4)  # GroupNorm groups

        # ── Input projection: conditioning → C-channel feature map ──────────
        self.input_conv = nn.Sequential(
            nn.Conv2d(conditioning_channels, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1),                    nn.SiLU(),
            nn.Conv2d(32, C, 3, padding=1),                     nn.SiLU(),
        )

        # ── Stage 0: 2 blocks at 32×32, then downsample → 16×16 ────────────
        # Produces residuals matching DownBlock2D (layers=2 + downsampler)
        self.s0_b1 = ConditioningBlock(C, C, groups=g)
        self.s0_g1 = GeometryEmbedding(param_dim, C)
        self.s0_z1 = ZeroConv2d(C, 320)          # → [B, 320, 32, 32]

        self.s0_b2 = ConditioningBlock(C, C, groups=g)
        self.s0_g2 = GeometryEmbedding(param_dim, C)
        self.s0_z2 = ZeroConv2d(C, 320)          # → [B, 320, 32, 32]

        self.s0_down = nn.Conv2d(C, C, 3, stride=2, padding=1)  # 32 → 16
        self.s0_b3 = ConditioningBlock(C, C, groups=g)
        self.s0_z3 = ZeroConv2d(C, 320)          # → [B, 320, 16, 16]

        # ── Stage 1: 2 blocks at 16×16, then downsample → 8×8 ──────────────
        # Produces residuals matching CrossAttnDownBlock2D (layers=2 + downsampler)
        self.s1_b1 = ConditioningBlock(C, C, groups=g)
        self.s1_g1 = GeometryEmbedding(param_dim, C)
        self.s1_z1 = ZeroConv2d(C, 640)          # → [B, 640, 16, 16]

        self.s1_b2 = ConditioningBlock(C, C, groups=g)
        self.s1_g2 = GeometryEmbedding(param_dim, C)
        self.s1_z2 = ZeroConv2d(C, 640)          # → [B, 640, 16, 16]

        self.s1_down = nn.Conv2d(C, C, 3, stride=2, padding=1)  # 16 → 8
        self.s1_b3 = ConditioningBlock(C, C, groups=g)
        self.s1_z3 = ZeroConv2d(C, 640)          # → [B, 640, 8, 8]

        # ── Stage 2: 2 blocks at 8×8, no downsample ─────────────────────────
        # Produces residuals matching CrossAttnDownBlock2D (layers=2, no downsampler)
        self.s2_b1 = ConditioningBlock(C, C, groups=g)
        self.s2_g1 = GeometryEmbedding(param_dim, C)
        self.s2_z1 = ZeroConv2d(C, 1280)         # → [B, 1280, 8, 8]

        self.s2_b2 = ConditioningBlock(C, C, groups=g)
        self.s2_g2 = GeometryEmbedding(param_dim, C)
        self.s2_z2 = ZeroConv2d(C, 1280)         # → [B, 1280, 8, 8]

        # ── Mid block at 8×8 ─────────────────────────────────────────────────
        self.mid_b = ConditioningBlock(C, C, groups=g)
        self.mid_g = GeometryEmbedding(param_dim, C)
        self.mid_z = ZeroConv2d(C, 1280)         # → [B, 1280, 8, 8]

    def forward(
        self,
        depth_map: torch.Tensor,
        param_embedding: torch.Tensor,
        conditioning_scale: float = 1.0,
    ) -> Dict[str, List[torch.Tensor] | torch.Tensor]:
        """
        Args:
            depth_map: (B, 6, H, W) at latent spatial size (e.g. 32×32).
            param_embedding: (B, param_dim) geometry embedding.
            conditioning_scale: Multiplier for all residuals.

        Returns:
            Dict with 8 'down_block_res_samples' and 1 'mid_block_res_sample'
            matching SDXL U-Net skip-connection shapes exactly.
        """
        s = conditioning_scale
        x = self.input_conv(depth_map)   # [B, C, 32, 32]
        p = param_embedding

        # Stage 0 at 32×32
        x = self.s0_g1(p, self.s0_b1(x));  r1 = self.s0_z1(x) * s
        x = self.s0_g2(p, self.s0_b2(x));  r2 = self.s0_z2(x) * s
        x = self.s0_b3(self.s0_down(x));   r3 = self.s0_z3(x) * s   # now 16×16

        # Stage 1 at 16×16
        x = self.s1_g1(p, self.s1_b1(x));  r4 = self.s1_z1(x) * s
        x = self.s1_g2(p, self.s1_b2(x));  r5 = self.s1_z2(x) * s
        x = self.s1_b3(self.s1_down(x));   r6 = self.s1_z3(x) * s   # now 8×8

        # Stage 2 at 8×8
        x = self.s2_g1(p, self.s2_b1(x));  r7 = self.s2_z1(x) * s
        x = self.s2_g2(p, self.s2_b2(x));  r8 = self.s2_z2(x) * s

        # Mid
        x = self.mid_g(p, self.mid_b(x));  r_mid = self.mid_z(x) * s

        return {
            "down_block_res_samples": [r1, r2, r3, r4, r5, r6, r7, r8],
            "mid_block_res_sample": r_mid,
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
