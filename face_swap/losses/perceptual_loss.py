"""
losses/perceptual_loss.py
VGG-based perceptual feature matching loss.

Extracts intermediate activations from a pretrained VGG-19 network and
computes L1 distance in feature space between generated and target faces.
Encourages preservation of high-level structure and texture.

Reference:
  Perceptual Losses for Real-Time Style Transfer and Super-Resolution
  Johnson et al., 2016 — https://arxiv.org/abs/1603.08155
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


# ─────────────────────────────────────────────────────────────────────────────
# VGG feature extractor
# ─────────────────────────────────────────────────────────────────────────────


# Map human-readable layer names → VGG-19 sequential indices
VGG19_LAYER_MAP: Dict[str, int] = {
    "relu1_1": 2,
    "relu1_2": 4,
    "relu2_1": 7,
    "relu2_2": 9,
    "relu3_1": 12,
    "relu3_2": 14,
    "relu3_3": 16,
    "relu3_4": 18,
    "relu4_1": 21,
    "relu4_2": 23,
    "relu4_3": 25,
    "relu4_4": 27,
    "relu5_1": 30,
    "relu5_2": 32,
    "relu5_3": 34,
    "relu5_4": 36,
}


class VGGFeatureExtractor(nn.Module):
    """
    Extracts multi-layer feature maps from a frozen VGG-19 network.

    Args:
        layer_names (List[str]): Layer names to extract (see VGG19_LAYER_MAP).
        pretrained (bool): Use ImageNet pretrained weights. Default True.
        requires_grad (bool): Allow gradient flow through VGG. Default False.
    """

    # ImageNet normalisation (VGG expects 0-1 input normalised with these stats)
    _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(
        self,
        layer_names: Optional[List[str]] = None,
        pretrained: bool = True,
        requires_grad: bool = False,
    ) -> None:
        super().__init__()
        if layer_names is None:
            layer_names = ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"]
        self.layer_names = layer_names

        # Validate names
        for name in layer_names:
            if name not in VGG19_LAYER_MAP:
                raise ValueError(f"Unknown VGG layer: {name!r}. Valid: {list(VGG19_LAYER_MAP)}")

        max_idx = max(VGG19_LAYER_MAP[n] for n in layer_names)

        vgg = tv_models.vgg19(weights="DEFAULT" if pretrained else None)
        self.features = nn.Sequential(*list(vgg.features.children())[: max_idx + 1])

        if not requires_grad:
            for p in self.features.parameters():
                p.requires_grad_(False)

        # Sort layer indices for sequential extraction
        self._targets: List[Tuple[int, str]] = sorted(
            [(VGG19_LAYER_MAP[n], n) for n in layer_names], key=lambda x: x[0]
        )

        self.register_buffer("_mean", self._MEAN)
        self.register_buffer("_std", self._STD)

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Convert [-1, 1] → ImageNet-normalised for VGG."""
        x = (x + 1.0) / 2.0  # [-1,1] → [0,1]
        return (x - self._mean.to(x.device)) / self._std.to(x.device)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Image tensor (B, 3, H, W), normalised [-1, 1].

        Returns:
            Dict mapping layer names → feature maps.
        """
        x = self._normalize_input(x)
        activations: Dict[str, torch.Tensor] = {}
        target_iter = iter(self._targets)
        current_idx, current_name = next(target_iter)

        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == current_idx:
                activations[current_name] = x
                try:
                    current_idx, current_name = next(target_iter)
                except StopIteration:
                    break

        return activations


# ─────────────────────────────────────────────────────────────────────────────
# Perceptual Loss
# ─────────────────────────────────────────────────────────────────────────────


class PerceptualLoss(nn.Module):
    """
    Multi-layer VGG perceptual loss.

    L_perc = Σ_l  w_l * ||F_l(generated) - F_l(target)||_1

    where F_l denotes VGG-19 feature map at layer l.

    Args:
        layer_names (List[str]): VGG layers to include.
        layer_weights (Optional[Dict[str, float]]): Per-layer weights.
                                                    Uniform if None.
        weight (float): Global loss weight scalar.
        style_weight (float): Weight for Gram-matrix style loss (0 = disabled).
        pretrained (bool): Use ImageNet pretrained VGG.
    """

    def __init__(
        self,
        layer_names: Optional[List[str]] = None,
        layer_weights: Optional[Dict[str, float]] = None,
        weight: float = 0.1,
        style_weight: float = 0.0,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if layer_names is None:
            layer_names = ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"]

        self.weight = weight
        self.style_weight = style_weight
        self.layer_names = layer_names
        self.extractor = VGGFeatureExtractor(layer_names=layer_names, pretrained=pretrained)

        # Per-layer weights (uniform default)
        if layer_weights:
            self.layer_weights = layer_weights
        else:
            w = 1.0 / len(layer_names)
            self.layer_weights = {n: w for n in layer_names}

    @staticmethod
    def _gram(feat: torch.Tensor) -> torch.Tensor:
        """Gram matrix for style loss."""
        B, C, H, W = feat.shape
        f = feat.view(B, C, H * W)
        return torch.bmm(f, f.transpose(1, 2)) / (C * H * W)

    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute perceptual (and optionally style) loss.

        Args:
            generated: Generated image (B, 3, H, W), range [-1, 1].
            target: Target image (B, 3, H, W), range [-1, 1].

        Returns:
            Scalar perceptual loss.
        """
        gen_feats = self.extractor(generated)
        tgt_feats = self.extractor(target)

        perc_loss = torch.tensor(0.0, device=generated.device, requires_grad=True)
        style_loss = torch.tensor(0.0, device=generated.device, requires_grad=True)

        for name in self.layer_names:
            gen_f = gen_feats[name]
            tgt_f = tgt_feats[name].detach()  # target features are not differentiated
            w = self.layer_weights.get(name, 1.0)
            perc_loss = perc_loss + w * F.l1_loss(gen_f, tgt_f)

            if self.style_weight > 0:
                style_loss = style_loss + w * F.l1_loss(
                    self._gram(gen_f), self._gram(tgt_f).detach()
                )

        total = self.weight * perc_loss
        if self.style_weight > 0:
            total = total + self.style_weight * style_loss
        return total
