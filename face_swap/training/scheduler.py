"""
training/scheduler.py
LR scheduler utilities for geometry-aware face swap training.

Provides:
  - build_scheduler: Factory function that creates a scheduler from the
    parsed training config.
  - CosineWithWarmup: Cosine decay with linear warmup (standalone impl).
  - get_param_groups: Creates differential LR parameter groups for the AdamW
    optimiser (U-Net vs encoders vs discriminator).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


# ─────────────────────────────────────────────────────────────────────────────
# Warmup helpers
# ─────────────────────────────────────────────────────────────────────────────


def _linear_warmup(step: int, warmup_steps: int) -> float:
    """Linear ramp from 0 → 1 over warmup_steps."""
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, step / warmup_steps)


# ─────────────────────────────────────────────────────────────────────────────
# Cosine schedule with warmup and optional restarts
# ─────────────────────────────────────────────────────────────────────────────


def _cosine_with_warmup_lambda(
    step: int,
    warmup_steps: int,
    total_steps: int,
    num_cycles: float = 0.5,
) -> float:
    """
    LambdaLR schedule function: linear warmup followed by cosine decay.

    Args:
        step: Current global step.
        warmup_steps: Number of linear warmup steps.
        total_steps: Total training steps.
        num_cycles: Number of cosine half-periods. 0.5 = one half-cosine decay.
    """
    if step < warmup_steps:
        return _linear_warmup(step, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))


def _constant_with_warmup_lambda(step: int, warmup_steps: int) -> float:
    return _linear_warmup(step, warmup_steps)


def _linear_decay_lambda(
    step: int, warmup_steps: int, total_steps: int
) -> float:
    if step < warmup_steps:
        return _linear_warmup(step, warmup_steps)
    return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────


def build_scheduler(
    optimiser: Optimizer,
    scheduler_type: str,
    warmup_steps: int,
    total_steps: int,
    num_cycles: float = 1.0,
    last_epoch: int = -1,
) -> LRScheduler:
    """
    Build a LambdaLR scheduler from the config ``scheduler.type`` field.

    Args:
        optimiser: The optimiser to attach the scheduler to.
        scheduler_type: One of 'cosine_with_restarts' | 'linear' | 'constant'.
        warmup_steps: Number of warmup steps.
        total_steps: Total training steps.
        num_cycles: Cosine cycles (only for 'cosine_with_restarts').
        last_epoch: Epoch to resume from (-1 = fresh start).

    Returns:
        A configured LambdaLR scheduler.
    """
    if scheduler_type == "cosine_with_restarts":
        lr_lambda = lambda step: _cosine_with_warmup_lambda(
            step, warmup_steps, total_steps, num_cycles / 2.0
        )
    elif scheduler_type == "linear":
        lr_lambda = lambda step: _linear_decay_lambda(step, warmup_steps, total_steps)
    elif scheduler_type == "constant":
        lr_lambda = lambda step: _constant_with_warmup_lambda(step, warmup_steps)
    else:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type!r}. "
            "Valid: cosine_with_restarts | linear | constant"
        )

    return LambdaLR(optimiser, lr_lambda=lr_lambda, last_epoch=last_epoch)


# ─────────────────────────────────────────────────────────────────────────────
# Parameter group builder
# ─────────────────────────────────────────────────────────────────────────────


def get_generator_param_groups(
    backbone: nn.Module,
    region_encoder: nn.Module,
    controlnet: nn.Module,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Create AdamW parameter groups with differential learning rates.

    Groups:
        1. U-Net backbone parameters → ``unet_lr``
        2. Region encoder projection parameters → ``encoder_lr``
        3. ControlNet parameters → ``encoder_lr``

    Args:
        backbone: FaceSwapBackbone instance.
        region_encoder: FaceRegionEncoder instance.
        controlnet: GeometryControlNet instance.
        config: Parsed training config dict (``config['training']['optimizer']``).

    Returns:
        List of parameter groups for AdamW.
    """
    opt_cfg = config["training"]["optimizer"]
    unet_lr: float = opt_cfg["unet_lr"]
    encoder_lr: float = opt_cfg["encoder_lr"]
    wd: float = opt_cfg["weight_decay"]

    groups: List[Dict[str, Any]] = [
        {
            "params": [p for p in backbone.unet_parameters() if p.requires_grad],
            "lr": unet_lr,
            "weight_decay": wd,
            "name": "unet",
        },
        {
            "params": region_encoder.trainable_parameters(),
            "lr": encoder_lr,
            "weight_decay": wd,
            "name": "region_encoder",
        },
        {
            "params": list(controlnet.parameters()),
            "lr": encoder_lr,
            "weight_decay": wd,
            "name": "controlnet",
        },
    ]
    return [g for g in groups if len(g["params"]) > 0]


def build_generator_optimiser(
    backbone: nn.Module,
    region_encoder: nn.Module,
    controlnet: nn.Module,
    config: Dict[str, Any],
) -> AdamW:
    """
    Create AdamW optimiser for the generator (U-Net + encoders + ControlNet).

    Args:
        backbone: FaceSwapBackbone.
        region_encoder: FaceRegionEncoder.
        controlnet: GeometryControlNet.
        config: Full training config dict.

    Returns:
        Configured AdamW optimiser.
    """
    opt_cfg = config["training"]["optimizer"]
    param_groups = get_generator_param_groups(backbone, region_encoder, controlnet, config)
    return AdamW(
        param_groups,
        betas=tuple(opt_cfg["betas"]),
        eps=opt_cfg["eps"],
    )


def build_discriminator_optimiser(
    discriminator: nn.Module,
    config: Dict[str, Any],
) -> AdamW:
    """
    Create AdamW optimiser for the discriminator.

    Args:
        discriminator: PatchGAN or MultiScaleDiscriminator.
        config: Full training config dict.

    Returns:
        Configured AdamW optimiser.
    """
    opt_cfg = config["training"]["optimizer"]
    return AdamW(
        discriminator.parameters(),
        lr=opt_cfg["discriminator_lr"],
        betas=tuple(opt_cfg["betas"]),
        eps=opt_cfg["eps"],
        weight_decay=opt_cfg["weight_decay"],
    )
