"""
losses/identity_loss.py
ArcFace-based identity preservation loss.

Computes the cosine similarity between ArcFace embeddings of the
generated face and the source face.  Minimising (1 - cosine_sim)
encourages identity preservation during face swap.

Reference:
  ArcFace: Additive Angular Margin Loss for Deep Face Recognition
  Deng et al., 2019 — https://arxiv.org/abs/1801.07698

TODO: Set ARCFACE_MODEL_PATH to your pretrained ArcFace checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# ArcFace backbone wrapper
# ─────────────────────────────────────────────────────────────────────────────


class ArcFaceEncoder(nn.Module):
    """
    Thin wrapper around a pretrained ArcFace ResNet-100 model.

    Loads an InsightFace-format checkpoint (.pth) and provides a forward
    method that returns L2-normalised feature embeddings.

    Args:
        model_path (str | Path): Path to pretrained ArcFace weights.
        input_size (int): Expected input size (112 for standard ArcFace).

    TODO: Set model_path to your ArcFace checkpoint path.
    """

    # TODO: Update to your local ArcFace model path
    DEFAULT_MODEL_PATH: str = "pretrained/arcface_r100.pth"

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        input_size: int = 112,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.backbone = self._load_backbone(Path(model_path))

        # Freeze ArcFace weights
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def _load_backbone(self, model_path: Path) -> nn.Module:
        """Load backbone from InsightFace-format checkpoint."""
        try:
            from insightface.model_zoo import get_model

            model = get_model("arcface_r100_v1")
            model.prepare(ctx_id=0, input_size=(self.input_size, self.input_size))
            return model.model
        except ImportError:
            pass

        try:
            # Fallback: load via iresnet from unofficial implementation
            from backbones import get_model as _get  # type: ignore[import]

            model = _get("r100")
            ckpt = torch.load(model_path, map_location="cpu")
            model.load_state_dict(ckpt, strict=False)
            return model
        except Exception:
            pass

        # Last resort: assume model_path is a TorchScript / state_dict
        import torchvision.models as tm

        model = tm.resnet50()  # placeholder — replace with real backbone
        if model_path.exists():
            ckpt = torch.load(model_path, map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            model.load_state_dict(ckpt, strict=False)
        return model

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Face image tensor (B, 3, 112, 112), normalised [-1, 1].
        Returns:
            L2-normalised embeddings (B, 512).
        """
        feat = self.backbone(x)
        return F.normalize(feat, p=2, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Identity Loss
# ─────────────────────────────────────────────────────────────────────────────


class IdentityLoss(nn.Module):
    """
    ArcFace cosine similarity identity loss.

    L_id = 1 - cos_sim(ArcFace(generated), ArcFace(source))

    Expected input images are resized to 112×112 internally.

    Args:
        model_path (str): Path to pretrained ArcFace weights.
        weight (float): Loss weight scalar.
        input_size (int): ArcFace input resolution.
    """

    def __init__(
        self,
        model_path: str = ArcFaceEncoder.DEFAULT_MODEL_PATH,
        weight: float = 1.0,
        input_size: int = 112,
    ) -> None:
        super().__init__()
        self.weight = weight
        self.input_size = input_size
        self.encoder = ArcFaceEncoder(model_path=model_path, input_size=input_size)

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Resize images to ArcFace input size."""
        return F.interpolate(
            images,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        generated: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute identity loss.

        Args:
            generated: Swapped face images (B, 3, H, W), range [-1, 1].
            source: Source identity face images (B, 3, H, W), range [-1, 1].

        Returns:
            Scalar identity loss tensor.
        """
        gen_resized = self._preprocess(generated)
        src_resized = self._preprocess(source)

        gen_emb = self.encoder(gen_resized)
        src_emb = self.encoder(src_resized)

        cos_sim = F.cosine_similarity(gen_emb, src_emb, dim=-1)  # (B,)
        loss = (1.0 - cos_sim).mean()
        return self.weight * loss

    def compute_similarity_score(
        self,
        generated: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the mean cosine similarity score (used as an evaluation metric).

        Args:
            generated: Swapped face images (B, 3, H, W).
            source: Source face images (B, 3, H, W).

        Returns:
            Mean cosine similarity scalar tensor, range [-1, 1].
        """
        gen_emb = self.encoder(self._preprocess(generated))
        src_emb = self.encoder(self._preprocess(source))
        return F.cosine_similarity(gen_emb, src_emb, dim=-1).mean()
