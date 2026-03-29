"""
models/discriminator.py
PatchGAN discriminator for adversarial training of the face swap model.

Implements:
  - PatchGANDiscriminator: Multi-scale patch-based discriminator with
    optional spectral normalisation.
  - MultiScaleDiscriminator: Ensemble of PatchGANs operating at different
    image resolutions for improved texture and structure discrimination.

Reference:
  Image-to-Image Translation with Conditional Adversarial Networks (Pix2Pix)
  Isola et al., 2017 — https://arxiv.org/abs/1611.07004
"""

from __future__ import annotations

from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Spectral norm shorthand
# ─────────────────────────────────────────────────────────────────────────────


def _maybe_spectral(module: nn.Module, use: bool) -> nn.Module:
    """Apply spectral normalisation if ``use=True``."""
    return nn.utils.spectral_norm(module) if use else module


# ─────────────────────────────────────────────────────────────────────────────
# PatchGAN Discriminator
# ─────────────────────────────────────────────────────────────────────────────


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator that classifies overlapping image patches.

    Architecture (n_layers=3):
        Conv(3→ndf, 4×4, stride=2)
        ConvNorm(ndf→2*ndf, 4×4, stride=2)
        ConvNorm(2*ndf→4*ndf, 4×4, stride=2)
        ConvNorm(4*ndf→8*ndf, 4×4, stride=1)
        Conv(8*ndf→1, 4×4, stride=1)   ← patch score map

    Args:
        in_channels (int): Input image channels (3 for RGB, 6 for concatenated pair).
        ndf (int): Base number of discriminator filters.
        n_layers (int): Number of conv layers with downsampling.
        use_spectral_norm (bool): Apply spectral norm to conv weights.
        norm_layer: Normalisation layer class. Default: InstanceNorm2d.
    """

    def __init__(
        self,
        in_channels: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        use_spectral_norm: bool = True,
        norm_layer: Optional[type] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.InstanceNorm2d

        use_bias = norm_layer == nn.InstanceNorm2d

        layers: List[nn.Module] = [
            _maybe_spectral(
                nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
                use_spectral_norm,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            layers += [
                _maybe_spectral(
                    nn.Conv2d(
                        ndf * nf_mult_prev,
                        ndf * nf_mult,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=use_bias,
                    ),
                    use_spectral_norm,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # Final stride=1 block
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        layers += [
            _maybe_spectral(
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=4,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                ),
                use_spectral_norm,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Output patch map (no sigmoid — used with hinge / WGAN loss)
        layers += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor (B, C, H, W), normalised [-1, 1].
        Returns:
            Patch logit map (B, 1, H', W'), un-activated.
        """
        return self.model(x)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Scale Discriminator
# ─────────────────────────────────────────────────────────────────────────────


class MultiScaleDiscriminator(nn.Module):
    """
    Ensemble of PatchGAN discriminators at multiple image scales.

    Input image is progressively downsampled by 2× per scale.
    Each discriminator operates on its own resolution.

    Args:
        num_scales (int): Number of discriminators / scales. Default 3.
        in_channels (int): Image channels.
        ndf (int): Base filter count per discriminator.
        n_layers (int): Down-sampling layers per discriminator.
        use_spectral_norm (bool): Spectral norm on conv weights.
    """

    def __init__(
        self,
        num_scales: int = 3,
        in_channels: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList(
            [
                PatchGANDiscriminator(
                    in_channels=in_channels,
                    ndf=ndf,
                    n_layers=n_layers,
                    use_spectral_norm=use_spectral_norm,
                )
                for _ in range(num_scales)
            ]
        )
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            x: Input image tensor (B, C, H, W).

        Returns:
            Tuple of (outputs, intermediate_feats) per scale.
            Each output is a patch logit map.
        """
        outputs: List[torch.Tensor] = []
        inp = x
        for disc in self.discriminators:
            outputs.append(disc(inp))
            inp = self.downsample(inp)
        return outputs


# ─────────────────────────────────────────────────────────────────────────────
# Losses compatible with discriminator
# ─────────────────────────────────────────────────────────────────────────────


def hinge_d_loss(
    real_logits: List[torch.Tensor],
    fake_logits: List[torch.Tensor],
) -> torch.Tensor:
    """
    Hinge discriminator loss aggregated across scales.

    Args:
        real_logits: Patch maps for real images (per scale).
        fake_logits: Patch maps for generated images (per scale).

    Returns:
        Scalar hinge loss for discriminator update.
    """
    d_loss = torch.tensor(0.0, device=real_logits[0].device, requires_grad=True)
    for r, f in zip(real_logits, fake_logits):
        d_loss = d_loss + F.relu(1.0 - r).mean() + F.relu(1.0 + f).mean()
    return d_loss / len(real_logits)


def hinge_g_loss(fake_logits: List[torch.Tensor]) -> torch.Tensor:
    """
    Hinge generator adversarial loss aggregated across scales.

    Args:
        fake_logits: Patch maps for generated images (per scale).

    Returns:
        Scalar adversarial loss for generator (U-Net) update.
    """
    g_loss = torch.tensor(0.0, device=fake_logits[0].device, requires_grad=True)
    for f in fake_logits:
        g_loss = g_loss - f.mean()
    return g_loss / len(fake_logits)
