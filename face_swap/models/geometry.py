"""
models/geometry.py
DECA-based 3D face geometry encoder for ControlNet conditioning.

Provides:
  - DECAWrapper: Thin wrapper around the DECA library for extracting
    3DMM parameters (shape, expression, pose, camera, texture) from a
    face image.
  - GeometryEncoder: Converts DECA's parameter vectors into a spatial
    conditioning signal (rendered depth/normal map + flattened params),
    ready for injection into the ControlNet backbone.
  - GeometryConditioning: Full pipeline: DECA → render → conditioning map.

Reference:
  DECA: Detailed Expression Capture and Animation (Feng et al., 2021)
  https://arxiv.org/abs/2012.04012

TODO: Set DECA_MODEL_PATH and DECA_CFG_PATH to your local DECA installation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# DECA 3DMM parameter keys and sizes (FLAME model)
# ─────────────────────────────────────────────────────────────────────────────

DECA_PARAM_SIZES: Dict[str, int] = {
    "shape": 100,       # identity / shape coefficients (FLAME basis)
    "exp": 50,          # expression coefficients
    "pose": 6,          # global head pose (rotation + translation, in 6D)
    "cam": 3,           # weak-perspective camera (scale, tx, ty)
    "tex": 50,          # texture coefficients
    "light": 27,        # spherical harmonic lighting (9 bands × RGB)
}

DECA_TOTAL_PARAMS: int = sum(DECA_PARAM_SIZES.values())  # 236


# ─────────────────────────────────────────────────────────────────────────────
# DECA Wrapper
# ─────────────────────────────────────────────────────────────────────────────


class DECAWrapper(nn.Module):
    """
    Thin PyTorch-compatible wrapper around the DECA face reconstruction library.

    Loads DECA's encoder network and provides a differentiable forward pass
    that returns 3DMM parameter dicts as tensors on the correct device.

    Args:
        model_path (str | Path): Path to the pretrained DECA model tar file.
        cfg_path (str | Path): Path to the DECA config YAML file.
        device (str): Target device ('cuda' | 'cpu').

    TODO: Install DECA from https://github.com/YadiraF/DECA and set paths.
    """

    # TODO: Set to your DECA installation paths
    DEFAULT_MODEL_PATH: str = "pretrained/deca_model.tar"
    DEFAULT_CFG_PATH: str = "configs/deca_cfg.yml"

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        cfg_path: str | Path = DEFAULT_CFG_PATH,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.device = device
        self.model_path = Path(model_path)
        self.cfg_path = Path(cfg_path)
        self._deca = None  # Lazy-loaded
        self._flame = None

    def _load_deca(self) -> None:
        """Lazy-load DECA. Called on first forward pass."""
        try:
            # TODO: Import from local DECA installation
            from decalib.deca import DECA
            from decalib.utils.config import cfg as deca_cfg

            deca_cfg.pretrained_modelpath = str(self.model_path)
            self._deca = DECA(config=deca_cfg, device=self.device)
        except ImportError:
            raise ImportError(
                "DECA library not found. Please install it from "
                "https://github.com/YadiraF/DECA and add it to your PYTHONPATH."
            )

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract DECA 3DMM parameters from face images.

        Args:
            images: Face image tensors, shape (B, 3, 224, 224), normalised
                    to [-1, 1] (DECA expects BGR images internally — wrapper
                    handles the conversion).

        Returns:
            Dict with keys: 'shape', 'exp', 'pose', 'cam', 'tex', 'light',
            each a (B, param_size) float tensor.
        """
        if self._deca is None:
            self._load_deca()

        # DECA encodes single images; process as batch
        param_dicts = []
        for i in range(images.shape[0]):
            img = images[i]  # (3, 224, 224)
            # Convert from [-1, 1] to [0, 255] uint8 ndarray (DECA format)
            img_np = ((img.cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            # DECA encode
            codedict = self._deca.encode(
                torch.from_numpy(img_np).unsqueeze(0).permute(0, 3, 1, 2).float().to(self.device) / 255.0
            )
            param_dicts.append(codedict)

        # Collate batch
        result: Dict[str, torch.Tensor] = {}
        for key in DECA_PARAM_SIZES:
            result[key] = torch.cat([d[key] for d in param_dicts], dim=0)
        return result

    def decode_landmarks(
        self,
        codedict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Decode 68 3D facial landmarks from DECA parameters.

        Args:
            codedict: Output of ``encode()``.

        Returns:
            Landmarks tensor, shape (B, 68, 3).
        """
        if self._deca is None:
            self._load_deca()
        opdict = self._deca.decode(codedict, rendering=False, vis_lmk=True)
        return opdict["landmarks3d"]  # (B, 68, 3)

    def render_depth(
        self,
        codedict: Dict[str, torch.Tensor],
        image_size: int = 512,
    ) -> torch.Tensor:
        """
        Render a depth map from DECA 3DMM parameters.

        Args:
            codedict: Output of ``encode()``.
            image_size: Render resolution.

        Returns:
            Depth map tensor, shape (B, 1, image_size, image_size), normalised [0, 1].
        """
        if self._deca is None:
            self._load_deca()
        opdict = self._deca.decode(
            codedict, rendering=True, vis_lmk=False, return_vis=False
        )
        depth = opdict.get("depth_images")  # (B, 1, H, W) or None
        if depth is None:
            # Fall back to rendered face mask as proxy
            depth = opdict.get("shape_images", torch.zeros(codedict["shape"].shape[0], 1, image_size, image_size))
        depth = F.interpolate(depth, size=(image_size, image_size), mode="bilinear", align_corners=False)
        # Normalise
        d_min = depth.amin(dim=(2, 3), keepdim=True)
        d_max = depth.amax(dim=(2, 3), keepdim=True)
        depth = (depth - d_min) / (d_max - d_min + 1e-8)
        return depth


# ─────────────────────────────────────────────────────────────────────────────
# Geometry Encoder (parameter embedding)
# ─────────────────────────────────────────────────────────────────────────────


class GeometryParamEncoder(nn.Module):
    """
    Encodes raw DECA parameter vectors into a rich geometry embedding
    suitable for ControlNet conditioning.

    Projects [shape, exp, pose, cam] concatenation → (hidden_dim,) vector,
    which is tiled spatially to form a (hidden_dim, H/8, W/8) feature map
    for injection into ControlNet's input layer.

    Args:
        param_keys (Tuple[str, ...]): Which DECA param groups to use.
        hidden_dim (int): Output embedding dimension.
        dropout (float): Dropout in MLP.
    """

    CONDITIONING_KEYS: Tuple[str, ...] = ("shape", "exp", "pose", "cam")

    def __init__(
        self,
        param_keys: Optional[Tuple[str, ...]] = None,
        hidden_dim: int = 320,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if param_keys is None:
            param_keys = self.CONDITIONING_KEYS
        self.param_keys = param_keys
        self.hidden_dim = hidden_dim

        in_dim = sum(DECA_PARAM_SIZES[k] for k in param_keys)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, codedict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            codedict: DECA output dict with param tensors.

        Returns:
            Geometry embedding, shape (B, hidden_dim).
        """
        params = torch.cat([codedict[k] for k in self.param_keys], dim=-1)
        return self.mlp(params)


# ─────────────────────────────────────────────────────────────────────────────
# Full Geometry Conditioning Pipeline
# ─────────────────────────────────────────────────────────────────────────────


class GeometryConditioning(nn.Module):
    """
    End-to-end geometry conditioning pipeline.

    Produces two conditioning signals:
    1. ``param_embedding`` (B, hidden_dim): Global geometry embedding from
       DECA params, injected into ControlNet as a global hint.
    2. ``depth_map`` (B, 3, H, W): Rendered depth map (3-channel by tiling),
       used as the ControlNet image condition.

    Args:
        deca_model_path (str): Path to DECA model weights.
        deca_cfg_path (str): Path to DECA config YAML.
        hidden_dim (int): Param embedding dimension.
        image_size (int): Output depth map resolution.
        device (str): Torch device.
    """

    def __init__(
        self,
        deca_model_path: str = DECAWrapper.DEFAULT_MODEL_PATH,
        deca_cfg_path: str = DECAWrapper.DEFAULT_CFG_PATH,
        hidden_dim: int = 320,
        image_size: int = 512,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.deca = DECAWrapper(
            model_path=deca_model_path,
            cfg_path=deca_cfg_path,
            device=device,
        )
        self.param_encoder = GeometryParamEncoder(hidden_dim=hidden_dim)

        # 3-channel "depth image" projection (tiled grayscale → RGB-like)
        self.depth_project = nn.Conv2d(1, 3, kernel_size=1, bias=True)
        nn.init.ones_(self.depth_project.weight)
        nn.init.zeros_(self.depth_project.bias)

    def forward(
        self,
        face_images: torch.Tensor,
        return_depth: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            face_images: Normalised face tensors (B, 3, 224, 224) for DECA.
            return_depth: Whether to render and return depth maps.

        Returns:
            Dict with:
                'param_embedding': (B, hidden_dim)
                'depth_map': (B, 3, H, W)  if return_depth=True
                'codedict': raw DECA outputs
        """
        codedict = self.deca.encode(face_images)
        param_emb = self.param_encoder(codedict)

        result: Dict[str, torch.Tensor] = {
            "param_embedding": param_emb,
            "codedict": codedict,
        }

        if return_depth:
            depth = self.deca.render_depth(codedict, image_size=self.image_size)
            result["depth_map"] = self.depth_project(depth)

        return result

    def get_landmarks(self, face_images: torch.Tensor) -> torch.Tensor:
        """
        Convenience method: encode and return 68 3D landmarks.

        Args:
            face_images: (B, 3, 224, 224) normalised face tensors.

        Returns:
            Landmark tensor (B, 68, 3).
        """
        codedict = self.deca.encode(face_images)
        return self.deca.decode_landmarks(codedict)
