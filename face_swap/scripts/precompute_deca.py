"""
scripts/precompute_deca.py
Pre-compute DECA geometry features for all dataset images.

Runs DECA once over every image and saves per-image cache files:
    <image_path>.deca.pt  →  {'depth_map': ..., 'normal_map': ...,
                               'depth_map_raw': ..., 'param_embedding': ...}

This eliminates live DECA inference during training, cutting step time
from ~22s to ~4-6s on an H100.

Usage (run from face_swap_3d/):
    PYTHONPATH=/home/ubuntu/face_swap_3d python3 face_swap/scripts/precompute_deca.py \\
        --config face_swap/configs/train_config.yaml \\
        --split train \\
        --batch_size 4 \\
        --num_workers 4

Estimated time: ~1-2 hours for 10k CelebA images on H100.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# ── Resolve paths ────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
FACE_SWAP_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(FACE_SWAP_DIR))


# ── Image transform (must match trainer) ────────────────────────────────────

_TO_TENSOR = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def _collect_image_paths(cfg: dict, split: str) -> list[Path]:
    """Return image paths for every configured dataset."""
    paths: list[Path] = []
    for ds_cfg in cfg["data"]["datasets"]:
        name = ds_cfg["name"]
        root = Path(ds_cfg["root"])
        num = ds_cfg.get("num_samples", None)

        if name == "CelebA":
            img_dir = root / "img_align_celeba"
            if not img_dir.exists():
                img_dir = root  # flat layout fallback
            # Use rglob to handle double-nested layouts (e.g. img_align_celeba/img_align_celeba/)
            found = sorted(img_dir.rglob("*.jpg")) + sorted(img_dir.rglob("*.png"))
        elif name in ("FFHQ", "VGGFace2"):
            found = sorted(root.rglob("*.png")) + sorted(root.rglob("*.jpg"))
        else:
            found = sorted(root.rglob("*.jpg")) + sorted(root.rglob("*.png"))

        if num is not None:
            found = found[:num]
        paths.extend(found)
        print(f"  [{name}] {len(found)} images from {root}")

    return paths


def _load_geometry_module(cfg: dict, device: str):
    """Load GeometryConditioning (lazy DECA init)."""
    from models.geometry import GeometryConditioning
    geo = GeometryConditioning(
        deca_model_path=cfg["model"]["deca_model_path"],
        deca_cfg_path=cfg["model"]["deca_cfg_path"],
        device=device,
        image_size=cfg["data"]["image_size"],
    )
    return geo


@torch.no_grad()
def precompute(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("Collecting image paths...")
    paths = _collect_image_paths(cfg, args.split)
    print(f"Total images: {len(paths)}")

    # Filter already-cached images
    if not args.overwrite:
        paths = [p for p in paths if not Path(str(p) + ".deca.pt").exists()]
        print(f"Images needing processing: {len(paths)}")

    if not paths:
        print("All images already cached. Done.")
        return

    print("Loading DECA geometry module...")
    geo = _load_geometry_module(cfg, device)

    t0 = time.time()
    errors = 0

    for i in tqdm(range(0, len(paths), args.batch_size), desc="Pre-computing DECA"):
        batch_paths = paths[i: i + args.batch_size]

        # Load and preprocess images
        imgs = []
        valid_paths = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(_TO_TENSOR(img))
                valid_paths.append(p)
            except Exception as e:
                print(f"  [WARN] Could not load {p}: {e}")
                errors += 1

        if not imgs:
            continue

        batch = torch.stack(imgs).to(device)  # (B, 3, 256, 256)

        # DECA expects 224×224
        batch_224 = F.interpolate(batch, (224, 224), mode="bilinear", align_corners=False)

        try:
            result = geo(
                batch_224,
                return_depth=True,
                return_normal=True,
            )
        except Exception as e:
            print(f"  [WARN] DECA failed on batch {i//args.batch_size}: {e}")
            errors += args.batch_size
            continue

        # Save per-image cache files
        for j, p in enumerate(valid_paths):
            cache = {
                "depth_map":       result["depth_map"][j].cpu(),       # (3, H, W)
                "normal_map":      result["normal_map"][j].cpu(),       # (3, H, W)
                "depth_map_raw":   result["depth_map_raw"][j].cpu(),    # (1, H, W)
                "param_embedding": result["param_embedding"][j].cpu(),  # (D,)
            }
            torch.save(cache, str(p) + ".deca.pt")

    elapsed = time.time() - t0
    print(f"\nDone. Processed {len(paths)} images in {elapsed/60:.1f} min.")
    if errors:
        print(f"  {errors} errors encountered (images skipped).")
    print(f"Cache files saved as <image>.deca.pt alongside each image.")


def main():
    parser = argparse.ArgumentParser(description="Pre-compute DECA features for training images.")
    parser.add_argument("--config", default="face_swap/configs/train_config.yaml")
    parser.add_argument("--split",  default="train", choices=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-compute even if cache file already exists.")
    args = parser.parse_args()
    precompute(args)


if __name__ == "__main__":
    main()
