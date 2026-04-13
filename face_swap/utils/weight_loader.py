"""
utils/weight_loader.py
Utilities for loading pre-trained region encoder weights into the pipeline.

Handles the common case where users have separately trained ResNet-50 +
transformer encoder checkpoints (one per facial region) from similarity-
based pre-training, and need to map those weights into the
FaceRegionEncoder's SingleRegionEncoder sub-modules.

Supported checkpoint formats:
  1. Full SingleRegionEncoder state dict (backbone + transformer + projection)
  2. Separate backbone + head state dicts
  3. Flat state dict with automatic key mapping

Usage:
    from utils.weight_loader import load_pretrained_region_encoders

    encoder = FaceRegionEncoder(...)
    load_pretrained_region_encoders(
        encoder,
        weight_paths={
            "eyes":  "pretrained/eyes_encoder.pth",
            "nose":  "pretrained/nose_encoder.pth",
            "mouth": "pretrained/mouth_encoder.pth",
            "ears":  "pretrained/ears_encoder.pth",
        },
        strict=False,
    )
"""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Key mapping heuristics
# ---------------------------------------------------------------------------

# Common patterns for ResNet-50 layers in various training frameworks
_RESNET_LAYER_PATTERNS = [
    # Exact match (torchvision standard)
    (r"^(conv1|bn1|layer[1-4])\.", r"backbone.\1."),
    # Prefixed with "resnet." or "encoder." or "backbone."
    (r"^(?:resnet|encoder|backbone|feature_extractor)\.(conv1|bn1|layer[1-4])\.",
     r"backbone.\1."),
    # Prefixed with "model."
    (r"^model\.(conv1|bn1|layer[1-4])\.", r"backbone.\1."),
]

# Common patterns for transformer head layers
_TRANSFORMER_PATTERNS = [
    # Exact match with our naming
    (r"^transformer_head\.", r"transformer_head."),
    # Common alternatives
    (r"^(?:transformer|attn_head|head)\.", r"transformer_head."),
    (r"^(?:transformer|attn_head|head)\.layers\.", r"transformer_head.encoder.layers."),
    (r"^(?:transformer|attn_head|head)\.norm\.", r"transformer_head.norm."),
    (r"^(?:transformer|attn_head|head)\.pos_embed", r"transformer_head.pos_embed"),
]

# Common patterns for projection MLP
_PROJECTION_PATTERNS = [
    (r"^projection\.", r"projection."),
    (r"^(?:proj|proj_mlp|output_proj|fc_head)\.", r"projection."),
    (r"^projection\.net\.", r"projection.net."),
    (r"^(?:proj|proj_mlp|output_proj|fc_head)\.net\.", r"projection.net."),
]

# Input conv patterns
_INPUT_CONV_PATTERNS = [
    (r"^input_conv\.", r"input_conv."),
    (r"^(?:input_proj|fusion|modality_fusion)\.", r"input_conv."),
]


def _try_remap_key(key: str) -> Optional[str]:
    """
    Attempt to remap a state dict key to match SingleRegionEncoder structure.

    Returns the remapped key if a pattern matches, or None.
    """
    all_patterns = (
        _RESNET_LAYER_PATTERNS
        + _TRANSFORMER_PATTERNS
        + _PROJECTION_PATTERNS
        + _INPUT_CONV_PATTERNS
    )
    for pattern, replacement in all_patterns:
        new_key = re.sub(pattern, replacement, key)
        if new_key != key:
            return new_key
    return None


def remap_state_dict(
    source_state: Dict[str, torch.Tensor],
    target_module: nn.Module,
) -> Tuple[Dict[str, torch.Tensor], List[str], List[str]]:
    """
    Remap keys from a source state dict to match a target module's parameter names.

    Tries exact match first, then pattern-based remapping.

    Args:
        source_state: State dict from a checkpoint.
        target_module: Target nn.Module whose state dict keys are the ground truth.

    Returns:
        Tuple of (remapped_state_dict, matched_keys, unmatched_keys).
    """
    target_keys = set(target_module.state_dict().keys())
    remapped: Dict[str, torch.Tensor] = {}
    matched: List[str] = []
    unmatched: List[str] = []

    for src_key, tensor in source_state.items():
        # 1) Exact match
        if src_key in target_keys:
            remapped[src_key] = tensor
            matched.append(src_key)
            continue

        # 2) Try pattern remapping
        new_key = _try_remap_key(src_key)
        if new_key is not None and new_key in target_keys:
            remapped[new_key] = tensor
            matched.append(f"{src_key} -> {new_key}")
            continue

        # 3) Unmatched
        unmatched.append(src_key)

    return remapped, matched, unmatched


# ---------------------------------------------------------------------------
# Main loading functions
# ---------------------------------------------------------------------------


def load_single_region_encoder(
    encoder: nn.Module,
    weight_path: str | Path,
    region_name: str = "",
    strict: bool = False,
    verbose: bool = True,
) -> None:
    """
    Load pre-trained weights into a SingleRegionEncoder.

    Handles three checkpoint layouts:
      A) Full SingleRegionEncoder state dict (keys start with backbone., transformer_head., etc.)
      B) Just the backbone state dict (only ResNet keys)
      C) Flat / custom-named state dict (uses heuristic key remapping)

    Args:
        encoder: A SingleRegionEncoder instance.
        weight_path: Path to the .pth / .pt checkpoint file.
        region_name: Region name for logging.
        strict: If True, require all keys to match exactly.
        verbose: Print loading summary.
    """
    weight_path = Path(weight_path)
    if not weight_path.exists():
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    checkpoint = torch.load(str(weight_path), map_location="cpu")

    # Handle checkpoints that wrap state_dict in a dict
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            checkpoint = checkpoint["model"]
        elif "encoder" in checkpoint:
            checkpoint = checkpoint["encoder"]
        elif "region_encoder" in checkpoint:
            checkpoint = checkpoint["region_encoder"]

    # If still not a state dict (e.g., the full model was saved), extract it
    if isinstance(checkpoint, nn.Module):
        checkpoint = checkpoint.state_dict()

    # Remap keys to match our architecture
    remapped, matched, unmatched = remap_state_dict(checkpoint, encoder)

    target_state = encoder.state_dict()
    compatible: Dict[str, torch.Tensor] = {}
    shape_mismatched: List[str] = []
    for key, tensor in remapped.items():
        target_tensor = target_state.get(key)
        if target_tensor is None:
            continue
        if tuple(tensor.shape) != tuple(target_tensor.shape):
            shape_mismatched.append(
                f"{key}: ckpt{tuple(tensor.shape)} != model{tuple(target_tensor.shape)}"
            )
            continue
        compatible[key] = tensor

    if verbose:
        tag = f"[{region_name}] " if region_name else ""
        print(f"  {tag}Matched {len(compatible)}/{len(checkpoint)} compatible keys from {weight_path.name}")
        if unmatched:
            print(f"  {tag}Unmatched keys ({len(unmatched)}): {unmatched[:5]}{'...' if len(unmatched) > 5 else ''}")
        if shape_mismatched:
            print(
                f"  {tag}Shape-mismatched keys skipped ({len(shape_mismatched)}): "
                f"{shape_mismatched[:5]}{'...' if len(shape_mismatched) > 5 else ''}"
            )

    missing, unexpected = encoder.load_state_dict(compatible, strict=False)

    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Strict loading failed for {region_name}.\n"
            f"  Missing: {missing}\n  Unexpected: {unexpected}"
        )

    if verbose and missing:
        # Filter out expected missing keys (frozen backbone params that were already loaded,
        # or new trainable layers that haven't been pre-trained)
        truly_missing = [k for k in missing if k not in compatible]
        if truly_missing:
            print(f"  {tag}Keys initialised from scratch ({len(truly_missing)}): "
                  f"{truly_missing[:5]}{'...' if len(truly_missing) > 5 else ''}")


def load_pretrained_region_encoders(
    face_encoder: nn.Module,
    weight_paths: Dict[str, str | Path],
    strict: bool = False,
    verbose: bool = True,
) -> None:
    """
    Load pre-trained weights into all region encoders of a FaceRegionEncoder.

    Args:
        face_encoder: FaceRegionEncoder instance.
        weight_paths: Dict mapping region name -> path to pre-trained checkpoint.
                      Example:
                        {
                            "eyes":  "pretrained/eyes_encoder.pth",
                            "nose":  "pretrained/nose_encoder.pth",
                            "mouth": "pretrained/mouth_encoder.pth",
                            "ears":  "pretrained/ears_encoder.pth",
                        }
                      Regions not in the dict are skipped (keep default init).
        strict: If True, all checkpoint keys must match.
        verbose: Print per-region loading summary.
    """
    if verbose:
        print(f"[weight_loader] Loading pre-trained region encoder weights...")

    for region_name, path in weight_paths.items():
        if region_name not in face_encoder.encoders:
            print(f"  [WARN] Region '{region_name}' not in encoder. Skipping.")
            continue

        encoder = face_encoder.encoders[region_name]
        load_single_region_encoder(
            encoder=encoder,
            weight_path=path,
            region_name=region_name,
            strict=strict,
            verbose=verbose,
        )

    if verbose:
        print(f"[weight_loader] Done. Loaded weights for: {list(weight_paths.keys())}")


def inspect_checkpoint(weight_path: str | Path) -> Dict[str, Any]:
    """
    Inspect a checkpoint file and report its structure.

    Useful for debugging weight loading issues.

    Args:
        weight_path: Path to checkpoint.

    Returns:
        Dict with 'keys', 'shapes', 'type', and 'top_level_keys' info.
    """
    weight_path = Path(weight_path)
    checkpoint = torch.load(str(weight_path), map_location="cpu")

    info: Dict[str, Any] = {"path": str(weight_path), "type": type(checkpoint).__name__}

    if isinstance(checkpoint, dict):
        info["top_level_keys"] = list(checkpoint.keys())

        # Find the state dict
        state = checkpoint
        if "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state = checkpoint["model"]

        if isinstance(state, dict) and all(isinstance(v, torch.Tensor) for v in state.values()):
            info["num_params"] = len(state)
            info["keys"] = list(state.keys())
            info["shapes"] = {k: tuple(v.shape) for k, v in state.items()}
            total_params = sum(v.numel() for v in state.values())
            info["total_parameters"] = total_params
            info["total_parameters_M"] = f"{total_params / 1e6:.1f}M"
        else:
            info["note"] = "Top-level dict does not contain a flat state dict"
    elif isinstance(checkpoint, nn.Module):
        state = checkpoint.state_dict()
        info["num_params"] = len(state)
        info["keys"] = list(state.keys())

    return info
