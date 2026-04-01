"""
data/vggface2_dataset.py

VGGFace2 dataset for contrastive training of per-region facial encoders.

VGGFace2 folder structure on Kaggle:
  /kaggle/input/vggface2/data/vggface2_train/train/
    n000001/
      0001_01.jpg
      0002_01.jpg
      ...
    n000002/
      ...

Each __getitem__ returns a TRIPLET per region:
  anchor   : region crop, Person A, random pose 1
  positive : region crop, Person A, random pose 2  (same identity, different pose)
  negative : region crop, Person B, random pose    (different identity)

Used for NT-Xent contrastive loss:
  anchor token ≈ positive token  (same person)
  anchor token ≠ negative token  (different person)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


# ── Constants ──────────────────────────────────────────────────────────────────

REGIONS = ["mouth", "eyes", "ears", "nose"]

REGION_KEYWORDS = {
    "mouth": ("mouth", "lip"),
    "eyes":  ("l_eye", "r_eye"),
    "ears":  ("l_ear", "r_ear"),
    "nose":  ("nose",),
}

CROP_SIZE = 64
IMG_SIZE  = 224


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_rgb(path: Path, size: int = IMG_SIZE) -> np.ndarray:
    """Load image -> (H, W, 3) float32 [0, 1]."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def extract_region_crop(
    rgb: np.ndarray,
    seg_map: np.ndarray,
    depth: np.ndarray,
    normal: np.ndarray,
    label_indices: set,
    crop_size: int = CROP_SIZE,
) -> torch.Tensor:
    """
    Crop a region and stack RGB + normal + depth into (7, crop_size, crop_size).

    If region not found in seg map, returns zeros.
    """
    H, W = seg_map.shape
    mask = np.zeros((H, W), dtype=bool)
    for idx in label_indices:
        mask |= (seg_map == idx)

    ys, xs = np.where(mask)
    if len(ys) == 0:
        return torch.zeros(7, crop_size, crop_size)

    pad = 8
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(H, int(ys.max()) + pad)
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(W, int(xs.max()) + pad)

    rgb_c    = rgb[y1:y2, x1:x2]       # (h, w, 3)
    normal_c = normal[y1:y2, x1:x2]    # (h, w, 3)
    depth_c  = depth[y1:y2, x1:x2]     # (h, w, 1)

    stacked = np.concatenate([rgb_c, normal_c, depth_c], axis=2)   # (h, w, 7)
    t = torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0)    # (1, 7, h, w)
    t = F.interpolate(t, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
    return t.squeeze(0)   # (7, crop_size, crop_size)


# ── Dataset ────────────────────────────────────────────────────────────────────

class VGGFace2TripletDataset(Dataset):
    """
    VGGFace2 dataset returning per-region anchor/positive/negative triplets.

    Each sample:
      anchor[region]   : (7, 64, 64) — Person A, pose 1
      positive[region] : (7, 64, 64) — Person A, pose 2
      negative[region] : (7, 64, 64) — Person B, pose 1

    Args:
        root           : Path to VGGFace2 train/ folder
        target_region  : Which region to return triplets for.
                         If None, returns all regions.
        max_identities : Limit number of identities (None = all)
        min_images     : Skip identities with fewer images
        crop_size      : Region crop spatial size
        precomputed_geo: Load depth/normal from <img_dir>/geo/ subfolders.
                         If False, depth and normal are zeros (RGB-only mode).
    """

    def __init__(
        self,
        root: str,
        target_region: Optional[str] = None,
        max_identities: Optional[int] = 1000,
        min_images: int = 5,
        crop_size: int = CROP_SIZE,
        precomputed_geo: bool = False,
    ) -> None:
        super().__init__()
        self.root           = Path(root)
        self.target_region  = target_region
        self.crop_size      = crop_size
        self.precomputed_geo = precomputed_geo

        # Active regions
        self.regions = [target_region] if target_region else REGIONS

        # ── Collect identity folders ───────────────────────────────────────
        all_ids = sorted([d for d in self.root.iterdir() if d.is_dir()])
        if max_identities:
            all_ids = all_ids[:max_identities]

        self.identities: List[List[Path]] = []
        for id_dir in all_ids:
            imgs = sorted(
                list(id_dir.glob("*.jpg")) + list(id_dir.glob("*.png"))
            )
            if len(imgs) >= min_images:
                self.identities.append(imgs)

        print(f"[VGGFace2Dataset] {len(self.identities)} identities "
              f"| region={target_region or 'all'} "
              f"| geo={'precomputed' if precomputed_geo else 'zeros'}")

        # ── Lazy face parser ───────────────────────────────────────────────
        self._parser      = None
        self._processor   = None
        self._id2label    = None
        self._region_idx  = None

    # ── Face parser ────────────────────────────────────────────────────────

    def _load_parser(self) -> None:
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        self._processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
        self._parser    = SegformerForSemanticSegmentation.from_pretrained(
            "jonathandinu/face-parsing"
        ).eval()
        self._id2label   = self._parser.config.id2label
        self._region_idx = {
            r: {
                idx for idx, lbl in self._id2label.items()
                if any(kw in lbl.lower() for kw in kws)
            }
            for r, kws in REGION_KEYWORDS.items()
        }

    def _get_seg_map(self, pil_img: Image.Image) -> np.ndarray:
        if self._parser is None:
            self._load_parser()
        inputs = self._processor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            logits = self._parser(**inputs).logits
        H, W = pil_img.size[1], pil_img.size[0]
        up = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return up.argmax(dim=1).squeeze(0).numpy().astype(np.int32)

    # ── Geometry loading ───────────────────────────────────────────────────

    def _load_geo(self, img_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load precomputed depth (H,W,1) and normal (H,W,3) or return zeros."""
        if not self.precomputed_geo:
            return (
                np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.float32),
                np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32),
            )

        geo_dir     = img_path.parent / "geo"
        depth_path  = geo_dir / f"{img_path.stem}.depth.png"
        normal_path = geo_dir / f"{img_path.stem}.normal.png"

        if depth_path.exists():
            d = np.array(Image.open(depth_path).convert("L").resize(
                (IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
            depth = d[:, :, None]
        else:
            depth = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)

        if normal_path.exists():
            normal = np.array(Image.open(normal_path).convert("RGB").resize(
                (IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
        else:
            normal = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

        return depth, normal

    # ── Sample builder ─────────────────────────────────────────────────────

    def _build_sample(self, img_path: Path) -> Dict[str, torch.Tensor]:
        """
        Build region crops for one image.

        Returns:
            Dict[region -> (7, CROP_SIZE, CROP_SIZE)]
        """
        pil_img = Image.open(img_path).convert("RGB").resize(
            (IMG_SIZE, IMG_SIZE), Image.LANCZOS
        )
        rgb     = np.array(pil_img, dtype=np.float32) / 255.0
        seg_map = self._get_seg_map(pil_img)
        depth, normal = self._load_geo(img_path)

        if self._region_idx is None:
            self._load_parser()

        return {
            r: extract_region_crop(
                rgb, seg_map, depth, normal,
                self._region_idx[r], self.crop_size
            )
            for r in self.regions
        }

    # ── Dataset interface ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.identities)

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Returns triplet:
        {
          'anchor':   {region: (7, 64, 64)},
          'positive': {region: (7, 64, 64)},   <- same identity as anchor
          'negative': {region: (7, 64, 64)},   <- different identity
          'identity_idx': int
        }
        """
        # Anchor + positive: same identity, 2 different images
        anchor_path, positive_path = random.sample(self.identities[idx], 2)

        # Negative: different identity
        neg_idx = random.choice(
            [i for i in range(len(self.identities)) if i != idx]
        )
        negative_path = random.choice(self.identities[neg_idx])

        return {
            "anchor":       self._build_sample(anchor_path),
            "positive":     self._build_sample(positive_path),
            "negative":     self._build_sample(negative_path),
            "identity_idx": idx,
        }


# ── Collation ──────────────────────────────────────────────────────────────────

def collate_triplets(batch: List[Dict]) -> Dict:
    """Stack triplets across batch dimension."""
    regions = list(batch[0]["anchor"].keys())
    out = {}
    for split in ("anchor", "positive", "negative"):
        out[split] = {
            r: torch.stack([b[split][r] for b in batch])
            for r in regions
        }
    out["identity_idx"] = torch.tensor([b["identity_idx"] for b in batch])
    return out
