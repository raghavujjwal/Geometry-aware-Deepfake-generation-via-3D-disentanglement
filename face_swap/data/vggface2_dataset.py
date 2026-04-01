"""
data/vggface2_dataset.py

VGGFace2 dataset that generates same-identity pairs for face swap training.

VGGFace2 folder structure on Kaggle:
  /kaggle/input/vggface2/
    train/
      n000001/   <- identity folder
        0001_01.jpg
        0002_01.jpg
        ...
      n000002/
        ...

Each __getitem__ returns:
  source: dict with rgb, depth, normal crops per region (stacked 7ch tensor)
  target: dict with rgb, depth, normal crops per region (stacked 7ch tensor)
Both are from the SAME identity but DIFFERENT images.

Usage:
  dataset = VGGFace2PairDataset(root="/kaggle/input/vggface2/train",
                                  max_identities=1000)
  loader  = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
"""

from __future__ import annotations

import gc
import os
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

# ── Region constants ───────────────────────────────────────────────────────────

REGIONS = ["mouth", "eyes", "ears", "nose"]

# CelebAMask-HQ label keywords per region (for face parser)
REGION_KEYWORDS = {
    "mouth": ("mouth", "lip"),
    "eyes":  ("l_eye", "r_eye"),
    "ears":  ("l_ear", "r_ear"),
    "nose":  ("nose",),
}

CROP_SIZE  = 64     # each region crop is resized to 64x64
IMG_SIZE   = 224    # DECA input size


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_rgb(path: Path, size: int = IMG_SIZE) -> np.ndarray:
    """Load image as (H, W, 3) float32 in [0, 1]."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def tensor_from_np(arr: np.ndarray) -> torch.Tensor:
    """(H, W, C) numpy -> (C, H, W) tensor."""
    return torch.from_numpy(arr).permute(2, 0, 1)


def extract_region_crop(
    rgb: np.ndarray,
    seg_map: np.ndarray,
    depth: np.ndarray,
    normal: np.ndarray,
    label_indices: set,
    crop_size: int = CROP_SIZE,
) -> torch.Tensor:
    """
    Extract a region and stack RGB + normal + depth into a 7-channel tensor.

    Args:
        rgb     : (H, W, 3) float [0,1]
        seg_map : (H, W) int  label map
        depth   : (H, W, 1) float [0,1]
        normal  : (H, W, 3) float [0,1]
        label_indices: set of seg label indices for this region
        crop_size: output spatial size

    Returns:
        (7, crop_size, crop_size) tensor:  [RGB(3) | normal(3) | depth(1)]
    """
    H, W = seg_map.shape
    mask = np.zeros((H, W), dtype=bool)
    for idx in label_indices:
        mask |= (seg_map == idx)

    ys, xs = np.where(mask)
    if len(ys) == 0:
        # Region not found - return zeros
        return torch.zeros(7, crop_size, crop_size)

    pad = 8
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(H, int(ys.max()) + pad)
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(W, int(xs.max()) + pad)

    rgb_crop    = rgb[y1:y2, x1:x2]       # (h, w, 3)
    normal_crop = normal[y1:y2, x1:x2]    # (h, w, 3)
    depth_crop  = depth[y1:y2, x1:x2]     # (h, w, 1)

    stacked = np.concatenate([rgb_crop, normal_crop, depth_crop], axis=2)  # (h, w, 7)
    t = torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0)            # (1, 7, h, w)
    t = F.interpolate(t, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
    return t.squeeze(0)   # (7, crop_size, crop_size)


# ── Dataset ────────────────────────────────────────────────────────────────────

class VGGFace2PairDataset(Dataset):
    """
    VGGFace2 dataset returning same-identity image pairs with per-region
    7-channel crops (RGB + normal + depth).

    Each identity folder must have >= 2 images. A face parser and DECA
    are used to extract segmentation and geometry on-the-fly. For large
    scale training, pre-compute and cache the seg maps and geometry maps.

    Args:
        root           : Path to VGGFace2 train/ folder
        max_identities : Limit to first N identity folders (None = all)
        min_images     : Skip identities with fewer than this many images
        crop_size      : Spatial size of each region crop
        cache_dir      : If set, cache seg maps and geometry here
        transform      : Optional transform on the full RGB image
        precomputed_geo: If True, load depth/normal from <img_dir>/geo/ subfolder
                         instead of running DECA on-the-fly.
    """

    def __init__(
        self,
        root: str,
        max_identities: Optional[int] = 1000,
        min_images: int = 5,
        crop_size: int = CROP_SIZE,
        cache_dir: Optional[str] = None,
        precomputed_geo: bool = True,
    ) -> None:
        super().__init__()
        self.root        = Path(root)
        self.crop_size   = crop_size
        self.cache_dir   = Path(cache_dir) if cache_dir else None
        self.precomputed_geo = precomputed_geo

        # ── Collect identity folders ───────────────────────────────────────
        all_ids = sorted([
            d for d in self.root.iterdir()
            if d.is_dir()
        ])
        if max_identities:
            all_ids = all_ids[:max_identities]

        self.identities: List[List[Path]] = []
        for id_dir in all_ids:
            imgs = sorted(list(id_dir.glob("*.jpg")) + list(id_dir.glob("*.png")))
            if len(imgs) >= min_images:
                self.identities.append(imgs)

        print(f"[VGGFace2Dataset] {len(self.identities)} identities loaded "
              f"(max_identities={max_identities})")

        # ── Lazy-load face parser ──────────────────────────────────────────
        self._parser_processor = None
        self._parser_model     = None
        self._id2label         = None
        self._region_indices   = None

    # ── Face parser (lazy) ─────────────────────────────────────────────────

    def _load_parser(self) -> None:
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        self._parser_processor = SegformerImageProcessor.from_pretrained(
            "jonathandinu/face-parsing"
        )
        self._parser_model = SegformerForSemanticSegmentation.from_pretrained(
            "jonathandinu/face-parsing"
        ).eval()
        self._id2label = self._parser_model.config.id2label
        self._region_indices = {
            region: {
                idx for idx, lbl in self._id2label.items()
                if any(kw in lbl.lower() for kw in keywords)
            }
            for region, keywords in REGION_KEYWORDS.items()
        }

    def _get_seg_map(self, pil_img: Image.Image) -> np.ndarray:
        """Run face parser -> (H, W) int32 label map."""
        import torch
        import torch.nn.functional as F

        if self._parser_model is None:
            self._load_parser()

        inputs = self._parser_processor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            logits = self._parser_model(**inputs).logits
        H, W = pil_img.size[1], pil_img.size[0]
        up = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return up.argmax(dim=1).squeeze(0).numpy().astype(np.int32)

    # ── Geometry loading ───────────────────────────────────────────────────

    def _load_geometry(
        self, img_path: Path
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load precomputed depth (H, W, 1) and normal (H, W, 3) maps.

        Expects files:
          <img_path>.depth.png   -> grayscale
          <img_path>.normal.png  -> RGB

        If not found, returns zeros.
        """
        stem = img_path.stem
        parent = img_path.parent / "geo"

        depth_path  = parent / f"{stem}.depth.png"
        normal_path = parent / f"{stem}.normal.png"

        if depth_path.exists():
            d = np.array(Image.open(depth_path).convert("L"), dtype=np.float32) / 255.0
            depth = d[:, :, None]   # (H, W, 1)
        else:
            depth = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)

        if normal_path.exists():
            normal = np.array(
                Image.open(normal_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE)),
                dtype=np.float32,
            ) / 255.0
        else:
            normal = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

        return depth, normal

    # ── Core sample builder ────────────────────────────────────────────────

    def _build_sample(self, img_path: Path) -> Dict[str, torch.Tensor]:
        """
        Build per-region 7-channel crops for one image.

        Returns:
            Dict[region -> (7, crop_size, crop_size) tensor]
            plus 'rgb_full': (3, IMG_SIZE, IMG_SIZE) tensor
        """
        # Load RGB at 512 for parsing, then resize to IMG_SIZE
        pil_full = Image.open(img_path).convert("RGB").resize(
            (IMG_SIZE, IMG_SIZE), Image.LANCZOS
        )
        rgb = np.array(pil_full, dtype=np.float32) / 255.0

        # Segmentation map
        seg_map = self._get_seg_map(pil_full)

        # Geometry
        if self.precomputed_geo:
            depth, normal = self._load_geometry(img_path)
        else:
            depth  = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
            normal = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

        # Resize depth/normal to match rgb size
        if depth.shape[:2] != (IMG_SIZE, IMG_SIZE):
            d_pil = Image.fromarray((depth[:, :, 0] * 255).astype(np.uint8)).resize(
                (IMG_SIZE, IMG_SIZE)
            )
            depth = np.array(d_pil, dtype=np.float32)[:, :, None] / 255.0
        if normal.shape[:2] != (IMG_SIZE, IMG_SIZE):
            n_pil = Image.fromarray((normal * 255).astype(np.uint8)).resize(
                (IMG_SIZE, IMG_SIZE)
            )
            normal = np.array(n_pil, dtype=np.float32) / 255.0

        # Extract per-region 7-channel crops
        sample: Dict[str, torch.Tensor] = {}
        if self._region_indices is None:
            self._load_parser()

        for region, indices in self._region_indices.items():
            sample[region] = extract_region_crop(
                rgb, seg_map, depth, normal, indices, self.crop_size
            )

        # Full RGB for reconstruction loss
        sample["rgb_full"] = tensor_from_np(rgb)   # (3, IMG_SIZE, IMG_SIZE)
        return sample

    # ── Dataset interface ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.identities)

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Returns:
            {
              'source': {region -> (7, 64, 64), 'rgb_full': (3, 224, 224)},
              'target': {region -> (7, 64, 64), 'rgb_full': (3, 224, 224)},
              'identity_idx': int
            }
        Both source and target are from the same identity.
        """
        imgs = self.identities[idx]
        src_path, tgt_path = random.sample(imgs, 2)

        source = self._build_sample(src_path)
        target = self._build_sample(tgt_path)

        return {
            "source": source,
            "target": target,
            "identity_idx": idx,
        }


# ── Collation ──────────────────────────────────────────────────────────────────

def collate_fn(
    batch: List[Dict],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Custom collate that stacks per-region tensors across the batch.
    """
    keys = ["source", "target"]
    region_keys = REGIONS + ["rgb_full"]

    out = {}
    for key in keys:
        out[key] = {
            rk: torch.stack([b[key][rk] for b in batch], dim=0)
            for rk in region_keys
            if rk in batch[0][key]
        }
    out["identity_idx"] = torch.tensor([b["identity_idx"] for b in batch])
    return out
