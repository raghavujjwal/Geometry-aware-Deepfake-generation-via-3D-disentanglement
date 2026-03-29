"""
data/dataset.py
Multi-dataset dataloader for geometry-aware face swapping training.

Supports FFHQ, VGGFace2, and CelebA-HQ datasets, returning:
- source image tensor
- target image tensor
- source facial region crops (eyes, nose, lips, skin, hairline, ears)
- DECA 3DMM parameter paths (loaded lazily during training)
- Optional text prompts for CLIP conditioning

Usage:
    dataset = FaceSwapDataset(config, split='train')
    loader  = DataLoader(dataset, batch_size=8, num_workers=8)
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from data.augmentations import FaceAugmentationPipeline, PairedAugmentation, RegionCropAugmentation
from utils.face_crop import FaceRegionCropper


# ─────────────────────────────────────────────────────────────────────────────
# Base face dataset class
# ─────────────────────────────────────────────────────────────────────────────


class BaseFaceDataset(Dataset):
    """
    Abstract face dataset.  Sub-classes implement ``_build_index``.

    Args:
        root (str | Path): Root directory of the dataset.
        split (str): 'train' | 'val' | 'test'.
        paired_aug (PairedAugmentation): Paired source/target augmenter.
        region_aug (RegionCropAugmentation): Region crop augmenter.
        cropper (FaceRegionCropper): Region detector/cropper.
        num_samples (Optional[int]): Subsample to exactly this many samples.
    """

    # Sub-classes fill this list with absolute image paths.
    _image_paths: List[Path] = []

    def __init__(
        self,
        root: str | Path,
        split: str,
        paired_aug: PairedAugmentation,
        region_aug: RegionCropAugmentation,
        cropper: FaceRegionCropper,
        num_samples: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.paired_aug = paired_aug
        self.region_aug = region_aug
        self.cropper = cropper

        self._build_index()

        if num_samples is not None and num_samples < len(self._image_paths):
            self._image_paths = random.sample(self._image_paths, num_samples)

    def _build_index(self) -> None:
        """Populate ``self._image_paths`` from disk.  Override in sub-class."""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a dict:
            source_image   : (3, H, W) tensor, normalised [-1, 1]
            target_image   : (3, H, W) tensor, normalised [-1, 1]
            source_regions : Dict[str, (3, Hc, Wc) tensor] per region
            source_path    : str
            target_path    : str
        """
        src_path = self._image_paths[idx]
        # Random different target from same dataset
        tgt_idx = random.randint(0, len(self._image_paths) - 1)
        while tgt_idx == idx:
            tgt_idx = random.randint(0, len(self._image_paths) - 1)
        tgt_path = self._image_paths[tgt_idx]

        src_img = Image.open(src_path).convert("RGB")
        tgt_img = Image.open(tgt_path).convert("RGB")

        src_dict, tgt_dict = self.paired_aug(src_img, tgt_img)

        # Extract region crops BEFORE augmentation (from original PIL)
        src_regions = self._get_region_crops(src_img)

        return {
            "source_image": src_dict["image"],
            "target_image": tgt_dict["image"],
            "source_regions": src_regions,
            "source_path": str(src_path),
            "target_path": str(tgt_path),
        }

    def _get_region_crops(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        """
        Extract and augment facial region crops from a PIL image.

        Returns a dict with keys matching FaceRegionCropper.REGIONS.
        Missing regions (detection failure) are replaced by zero tensors.
        """
        crops = self.cropper.crop_regions(image)  # Dict[str, PIL or None]
        region_tensors: Dict[str, torch.Tensor] = {}
        for region_name, crop in crops.items():
            if crop is not None:
                region_tensors[region_name] = self.region_aug(crop)
            else:
                region_tensors[region_name] = torch.zeros(3, 64, 64)
        return region_tensors


# ─────────────────────────────────────────────────────────────────────────────
# Dataset implementations
# ─────────────────────────────────────────────────────────────────────────────


class FFHQDataset(BaseFaceDataset):
    """
    FFHQ dataset (70k face images at 1024x1024).

    Expected structure::

        <root>/
            images1024x1024/
                00000/
                    00000.png
                    ...
            ffhq-dataset-v2.json   (optional, for metadata)

    Args:
        root: Path to FFHQ root directory.
        split: Only 'train' is supported (FFHQ has no official val split).
        **kwargs: Forwarded to BaseFaceDataset.
    """

    def _build_index(self) -> None:
        img_dir = self.root / "images1024x1024"
        if not img_dir.exists():
            # Fallback: flat structure
            img_dir = self.root
        self._image_paths = sorted(img_dir.rglob("*.png"))
        # Also grab jpegs if present
        self._image_paths += sorted(img_dir.rglob("*.jpg"))
        if not self._image_paths:
            raise FileNotFoundError(f"No images found in {img_dir}")


class VGGFace2Dataset(BaseFaceDataset):
    """
    VGGFace2 dataset (~3.3M images).

    Expected structure::

        <root>/
            data/
                train/
                    n000001/
                        0001_01.jpg
                        ...
                test/
                    ...

    Args:
        root: Path to VGGFace2 root directory.
        split: 'train' | 'test'.
        **kwargs: Forwarded to BaseFaceDataset.
    """

    def _build_index(self) -> None:
        data_dir = self.root / "data" / self.split
        if not data_dir.exists():
            data_dir = self.root / self.split
        if not data_dir.exists():
            raise FileNotFoundError(f"VGGFace2 split directory not found: {data_dir}")
        self._image_paths = sorted(data_dir.rglob("*.jpg"))


class CelebAHQDataset(BaseFaceDataset):
    """
    CelebA-HQ dataset (30k face images at 1024x1024).

    Expected structure::

        <root>/
            CelebA-HQ-img/
                00000.jpg
                ...

    Args:
        root: Path to CelebA-HQ root directory.
        split: 'train' | 'val'. Uses a 28k/2k split.
        **kwargs: Forwarded to BaseFaceDataset.
    """

    # Simple fixed train/val split
    TRAIN_RATIO = 0.93

    def _build_index(self) -> None:
        img_dir = self.root / "CelebA-HQ-img"
        if not img_dir.exists():
            img_dir = self.root
        all_paths = sorted(img_dir.rglob("*.jpg")) + sorted(img_dir.rglob("*.png"))
        n_train = int(len(all_paths) * self.TRAIN_RATIO)
        if self.split == "train":
            self._image_paths = all_paths[:n_train]
        else:
            self._image_paths = all_paths[n_train:]


# ─────────────────────────────────────────────────────────────────────────────
# Factory & multi-dataset concat
# ─────────────────────────────────────────────────────────────────────────────


_DATASET_REGISTRY: Dict[str, type] = {
    "FFHQ": FFHQDataset,
    "VGGFace2": VGGFace2Dataset,
    "CelebA-HQ": CelebAHQDataset,
}


def build_dataset_from_config(
    config: Dict[str, Any],
    split: str = "train",
    cropper: Optional[FaceRegionCropper] = None,
) -> ConcatDataset:
    """
    Build a ConcatDataset from a config dict matching train_config.yaml.

    Args:
        config: Parsed yaml config dict (full root config).
        split:  'train' | 'val'.
        cropper: Pre-build FaceRegionCropper. Created lazily if None.

    Returns:
        torch ConcatDataset combining all configured datasets.
    """
    if cropper is None:
        cropper = FaceRegionCropper()

    data_cfg = config["data"]
    image_size = data_cfg["image_size"]
    paired_aug = PairedAugmentation(image_size=image_size)
    region_aug = RegionCropAugmentation(image_size=64)

    datasets: List[Dataset] = []
    for ds_cfg in data_cfg["datasets"]:
        name = ds_cfg["name"]
        if name not in _DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset: {name!r}. Available: {list(_DATASET_REGISTRY)}")
        cls = _DATASET_REGISTRY[name]
        ns = ds_cfg.get("num_samples", None) if split == "train" else None
        ds = cls(
            root=ds_cfg["root"],
            split=split,
            paired_aug=paired_aug,
            region_aug=region_aug,
            cropper=cropper,
            num_samples=ns,
        )
        datasets.append(ds)
        print(f"[Dataset] {name}: {len(ds):,} samples")

    return ConcatDataset(datasets)


def build_dataloader(
    config: Dict[str, Any],
    split: str = "train",
    cropper: Optional[FaceRegionCropper] = None,
) -> DataLoader:
    """
    Convenience wrapper: builds dataset + DataLoader from config.

    Args:
        config: Parsed yaml config dict.
        split:  'train' | 'val'.
        cropper: Pre-built FaceRegionCropper (optional).

    Returns:
        A PyTorch DataLoader.
    """
    dataset = build_dataset_from_config(config, split=split, cropper=cropper)
    data_cfg = config["data"]
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=(split == "train"),
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"],
        prefetch_factor=data_cfg.get("prefetch_factor", 2),
        drop_last=(split == "train"),
        collate_fn=_collate_fn,
    )
    return loader


def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate that handles nested dicts of tensors (region crops).

    Args:
        batch: List of sample dicts from ``BaseFaceDataset.__getitem__``.

    Returns:
        Collated dict with stacked tensors.
    """
    # Stack flat tensor keys
    collated: Dict[str, Any] = {
        "source_image": torch.stack([b["source_image"] for b in batch]),
        "target_image": torch.stack([b["target_image"] for b in batch]),
        "source_path": [b["source_path"] for b in batch],
        "target_path": [b["target_path"] for b in batch],
    }
    # Collate nested region dict
    region_keys = batch[0]["source_regions"].keys()
    collated["source_regions"] = {
        k: torch.stack([b["source_regions"][k] for b in batch]) for k in region_keys
    }
    return collated
