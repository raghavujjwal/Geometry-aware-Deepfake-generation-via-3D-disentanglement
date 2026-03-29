"""
utils/metrics.py
Evaluation metrics for the face swapping system.

Implements:
  - ArcFace cosine similarity score (identity preservation)
  - FID (Frechet Inception Distance)
  - SSIM (Structural Similarity Index)
  - PSNR (Peak Signal-to-Noise Ratio)
  - 3D landmark error vs target pose
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# ArcFace Similarity
# ─────────────────────────────────────────────────────────────────────────────


def compute_arcface_similarity(
    generated: torch.Tensor,
    source: torch.Tensor,
    arcface_model: nn.Module,
    input_size: int = 112,
) -> float:
    """
    Compute mean ArcFace cosine similarity between generated and source identities.

    Args:
        generated: Generated face images (B, 3, H, W), range [-1, 1].
        source: Source face images (B, 3, H, W), range [-1, 1].
        arcface_model: Loaded ArcFaceEncoder or equivalent.
        input_size: ArcFace input resolution.

    Returns:
        Mean cosine similarity score (float, range [-1, 1]; higher = better identity).
    """
    with torch.no_grad():
        gen_r = F.interpolate(generated, (input_size, input_size), mode="bilinear", align_corners=False)
        src_r = F.interpolate(source, (input_size, input_size), mode="bilinear", align_corners=False)
        gen_emb = arcface_model(gen_r)
        src_emb = arcface_model(src_r)
        cos_sim = F.cosine_similarity(gen_emb, src_emb, dim=-1)
    return cos_sim.mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# SSIM
# ─────────────────────────────────────────────────────────────────────────────


def compute_ssim(
    generated: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
) -> float:
    """
    Compute mean Structural Similarity Index (SSIM).

    Args:
        generated: Generated images (B, 3, H, W), normalised [0, 1] or [-1, 1].
        target: Target images (B, 3, H, W).
        data_range: Data range (1.0 for [0,1], 2.0 for [-1,1]).
        window_size: Gaussian window size.

    Returns:
        Mean SSIM score (float, range [-1, 1]; higher = better).
    """
    try:
        from torchmetrics.functional import structural_similarity_index_measure as ssim_fn

        score = ssim_fn(generated.float(), target.float(), data_range=data_range)
        return score.item()
    except ImportError:
        pass

    # Simple fallback implementation
    gen = generated.float()
    tgt = target.float()
    if data_range == 2.0:
        gen = (gen + 1.0) / 2.0
        tgt = (tgt + 1.0) / 2.0

    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_g = F.avg_pool2d(gen, window_size, 1, window_size // 2)
    mu_t = F.avg_pool2d(tgt, window_size, 1, window_size // 2)
    mu_g2, mu_t2 = mu_g ** 2, mu_t ** 2
    mu_gt = mu_g * mu_t
    sigma_g2 = F.avg_pool2d(gen ** 2, window_size, 1, window_size // 2) - mu_g2
    sigma_t2 = F.avg_pool2d(tgt ** 2, window_size, 1, window_size // 2) - mu_t2
    sigma_gt = F.avg_pool2d(gen * tgt, window_size, 1, window_size // 2) - mu_gt
    ssim_map = ((2 * mu_gt + C1) * (2 * sigma_gt + C2)) / (
        (mu_g2 + mu_t2 + C1) * (sigma_g2 + sigma_t2 + C2)
    )
    return ssim_map.mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# PSNR
# ─────────────────────────────────────────────────────────────────────────────


def compute_psnr(
    generated: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> float:
    """
    Compute mean PSNR between generated and target images.

    Args:
        generated: Generated image tensor.
        target: Target image tensor.
        data_range: Pixel value range (1.0 for [0,1]).

    Returns:
        Mean PSNR in dB (float; higher = better).
    """
    mse = F.mse_loss(generated.float(), target.float()).item()
    if mse < 1e-10:
        return float("inf")
    return 10.0 * np.log10((data_range ** 2) / mse)


# ─────────────────────────────────────────────────────────────────────────────
# FID
# ─────────────────────────────────────────────────────────────────────────────


def compute_fid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    inception_model: Optional[nn.Module] = None,
    device: str = "cuda",
) -> float:
    """
    Compute Frechet Inception Distance (FID) between real and fake image sets.

    Requires torchmetrics or a custom Inception-V3 feature extractor.

    Args:
        real_images: real image tensor (N, 3, H, W), range [0, 1].
        fake_images: generated image tensor (N, 3, H, W), range [0, 1].
        inception_model: Optional pre-loaded InceptionV3 feature extractor.
        device: Torch device.

    Returns:
        FID score (float; lower = better).
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance

        fid_metric = FrechetInceptionDistance(feature=2048).to(device)
        # torchmetrics expects uint8 [0, 255]
        real_u8 = (real_images.clamp(0, 1) * 255).byte().to(device)
        fake_u8 = (fake_images.clamp(0, 1) * 255).byte().to(device)
        fid_metric.update(real_u8, real=True)
        fid_metric.update(fake_u8, real=False)
        return fid_metric.compute().item()
    except ImportError:
        pass

    # Minimal custom implementation using torchvision InceptionV3
    import torchvision.models as tv_models

    if inception_model is None:
        inception_model = tv_models.inception_v3(weights="DEFAULT", transform_input=False)
        inception_model.fc = nn.Identity()
        inception_model = inception_model.to(device).eval()

    def _get_features(imgs: torch.Tensor) -> np.ndarray:
        feats = []
        for i in range(0, len(imgs), 32):
            batch = F.interpolate(imgs[i:i+32].to(device), (299, 299), mode="bilinear", align_corners=False)
            with torch.no_grad():
                f = inception_model(batch)
            feats.append(f.cpu().numpy())
        return np.concatenate(feats, axis=0)

    mu_r, sigma_r = _compute_statistics(_get_features(real_images))
    mu_f, sigma_f = _compute_statistics(_get_features(fake_images))
    return _frechet_distance(mu_r, sigma_r, mu_f, sigma_f)


def _compute_statistics(features: np.ndarray):
    return features.mean(0), np.cov(features, rowvar=False)


def _frechet_distance(mu1, sigma1, mu2, sigma2) -> float:
    from scipy.linalg import sqrtm

    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


# ─────────────────────────────────────────────────────────────────────────────
# 3D Landmark Error
# ─────────────────────────────────────────────────────────────────────────────


def compute_landmark_3d_error(
    generated: torch.Tensor,
    target: torch.Tensor,
    geometry_module: nn.Module,
    device: str = "cuda",
) -> float:
    """
    Compute mean 3D landmark Euclidean error between generated and target faces.

    Args:
        generated: Generated images (B, 3, H, W), range [-1, 1].
        target: Target pose images (B, 3, H, W), range [-1, 1].
        geometry_module: Loaded GeometryConditioning module with DECA.
        device: Torch device.

    Returns:
        Mean landmark L2 error (float; lower = better pose matching).
    """
    gen = F.interpolate(generated.to(device), (224, 224))
    tgt = F.interpolate(target.to(device), (224, 224))
    with torch.no_grad():
        gen_lm = geometry_module.get_landmarks(gen)  # (B, 68, 3)
        tgt_lm = geometry_module.get_landmarks(tgt)
    dist = torch.norm(gen_lm - tgt_lm, p=2, dim=-1)  # (B, 68)
    return dist.mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Aggregated metric computation
# ─────────────────────────────────────────────────────────────────────────────


def compute_metrics(
    generated: torch.Tensor,
    target: torch.Tensor,
    source: Optional[torch.Tensor] = None,
    arcface_model: Optional[nn.Module] = None,
    geometry_module: Optional[nn.Module] = None,
    compute_fid_flag: bool = False,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Compute all available metrics for a batch of generated images.

    Args:
        generated: Generated face images (B, 3, H, W), range [-1, 1].
        target: Target face images (B, 3, H, W), range [-1, 1].
        source: Source identity images (B, 3, H, W). Required for ArcFace.
        arcface_model: Loaded ArcFaceEncoder (optional).
        geometry_module: Loaded GeometryConditioning (optional).
        compute_fid_flag: Whether to compute FID (expensive).
        device: Torch device.

    Returns:
        Dict of metric_name → float value.
    """
    # Normalise to [0, 1] for SSIM/PSNR/FID
    gen_01 = (generated.clamp(-1, 1) + 1.0) / 2.0
    tgt_01 = (target.clamp(-1, 1) + 1.0) / 2.0

    metrics: Dict[str, float] = {}
    metrics["ssim"] = compute_ssim(gen_01, tgt_01, data_range=1.0)
    metrics["psnr"] = compute_psnr(gen_01, tgt_01, data_range=1.0)

    if source is not None and arcface_model is not None:
        metrics["arcface_similarity"] = compute_arcface_similarity(
            generated, source, arcface_model, input_size=112
        )

    if compute_fid_flag:
        metrics["fid"] = compute_fid(tgt_01, gen_01, device=device)

    if geometry_module is not None:
        metrics["landmark_3d_error"] = compute_landmark_3d_error(
            generated, target, geometry_module, device=device
        )

    return metrics
