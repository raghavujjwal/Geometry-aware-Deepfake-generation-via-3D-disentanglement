"""
scripts/precompute_deca.py
Pre-compute lightweight MediaPipe geometry cache files for training.

The cache tensor names and .deca.pt extension are kept for compatibility with
the existing trainer:
    <image_path>.deca.pt -> {
        "depth_map": (3, H, W),
        "normal_map": (3, H, W),
        "depth_map_raw": (1, H, W),
        "param_embedding": (D,),
    }

When data.geometry_cache_dir is set, cache files are written there instead of
beside the source images. This is required on Kaggle because /kaggle/input is
read-only.

Usage from DECA/:
    python face_swap/scripts/precompute_deca.py \
        --config face_swap/configs/train_config.yaml \
        --split train \
        --batch_size 32
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import torch
import yaml
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
FACE_SWAP_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(FACE_SWAP_DIR))


def _build_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def _split_paths(paths: List[Path], name: str, split: str) -> List[Path]:
    if name == "CelebA":
        n_train = int(len(paths) * 0.95)
    elif name == "CelebA-HQ":
        n_train = int(len(paths) * 0.93)
    else:
        return paths

    return paths[:n_train] if split == "train" else paths[n_train:]


def _collect_image_paths(cfg: dict, split: str) -> list[Path]:
    """Return image paths for every configured dataset."""
    paths: list[Path] = []
    for ds_cfg in cfg["data"]["datasets"]:
        name = ds_cfg["name"]
        root = Path(ds_cfg["root"])
        num = ds_cfg.get("num_samples", None)

        if name == "VGGFace2":
            data_dir = root / "data" / split
            if not data_dir.exists():
                data_dir = root / split
            if not data_dir.exists():
                data_dir = root
            found = sorted(data_dir.rglob("*.jpg")) + sorted(data_dir.rglob("*.png"))
        else:
            img_dir = root / "img_align_celeba"
            if name == "CelebA-HQ":
                img_dir = root / "CelebA-HQ-img"
            if not img_dir.exists():
                img_dir = root
            found = sorted(img_dir.rglob("*.jpg")) + sorted(img_dir.rglob("*.png"))
            found = _split_paths(found, name, split)

        if split == "train" and num is not None:
            found = found[:num]

        paths.extend(found)
        print(f"  [{name}:{split}] {len(found)} images from {root}")

    return paths


def _load_geometry_module(cfg: dict, device: str):
    from models.geometry import GeometryConditioning

    return GeometryConditioning(
        deca_model_path=cfg["model"].get("deca_model_path", ""),
        deca_cfg_path=cfg["model"].get("deca_cfg_path", ""),
        device=device,
        image_size=cfg["data"]["image_size"],
        cache_dir=cfg["data"].get("geometry_cache_dir"),
        cache_key_root=cfg["data"].get(
            "geometry_cache_key_root",
            cfg["data"]["datasets"][0]["root"],
        ),
    )


@torch.no_grad()
def precompute(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    geo = _load_geometry_module(cfg, device)
    if geo.cache_dir is not None:
        geo.cache_dir.mkdir(parents=True, exist_ok=True)

    paths = _collect_image_paths(cfg, args.split)
    if args.max_images is not None:
        paths = paths[: args.max_images]
    if not args.overwrite:
        paths = [p for p in paths if not geo.cache_path_for(p).exists()]
    print(f"Images needing cache: {len(paths)}")

    if not paths:
        print("All requested images are already cached.")
        return

    to_tensor = _build_transform(cfg["data"]["image_size"])

    t0 = time.time()
    errors = 0
    processed = 0

    for i in tqdm(range(0, len(paths), args.batch_size), desc="Pre-computing MediaPipe geometry"):
        batch_paths = paths[i: i + args.batch_size]
        imgs = []
        valid_paths = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(to_tensor(img))
                valid_paths.append(p)
            except Exception as exc:
                print(f"  [WARN] Could not load {p}: {exc}")
                errors += 1

        if not imgs:
            continue

        batch = torch.stack(imgs).to(device)

        try:
            result = geo(batch, return_depth=True, return_normal=True)
        except Exception as exc:
            print(f"  [WARN] Geometry failed on batch {i // args.batch_size}: {exc}")
            errors += len(valid_paths)
            continue

        for j, p in enumerate(valid_paths):
            cache = {
                "depth_map": result["depth_map"][j].cpu(),
                "normal_map": result["normal_map"][j].cpu(),
                "depth_map_raw": result["depth_map_raw"][j].cpu(),
                "param_embedding": result["param_embedding"][j].cpu(),
            }
            torch.save(cache, geo.cache_path_for(p))
            processed += 1

    elapsed = time.time() - t0
    print(f"Done. Cached {processed} images in {elapsed / 60:.1f} min.")
    if errors:
        print(f"Encountered {errors} skipped/failed images.")
    if geo.cache_dir is None:
        print("Cache files saved as <image>.deca.pt alongside each image.")
    else:
        print(f"Cache files saved under {geo.cache_dir}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute MediaPipe geometry cache files.")
    parser.add_argument("--config", default="face_swap/configs/train_config.yaml")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0, help="Reserved for CLI compatibility.")
    parser.add_argument("--max_images", type=int, default=None, help="Optional cap for smoke tests/subsets.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing cache files.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU preprocessing.")
    args = parser.parse_args()
    precompute(args)


if __name__ == "__main__":
    main()
