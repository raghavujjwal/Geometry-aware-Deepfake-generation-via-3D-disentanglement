"""
pipeline/face_parse.py

Parses facial regions from the source image using a SegFormer model
trained on CelebAMask-HQ (jonathandinu/face-parsing).

Flow:
  1. Load source_512.jpg (pre-resized by preprocess.py)
  2. Run face parsing -> 19-class segmentation map
  3. Extract mouth, eyes, ears, nose regions
  4. Save outputs to pipeline/output/source/

Outputs per region (e.g. mouth):
  - mouth_mask.png    : binary mask (0/255)
  - mouth_overlay.png : source image with region highlighted
  - mouth_crop.png    : tight crop of the region

Usage:
  python pipeline/face_parse.py
  python pipeline/face_parse.py --source pipeline/input/source_512.jpg
"""

from __future__ import annotations

# ── Path bootstrap ─────────────────────────────────────────────────────────────
import sys as _sys
import os as _os
_PKG = "e:/High fidelity face swap using 3D disintanglement/packages"
_HF_CACHE = "e:/High fidelity face swap using 3D disintanglement/hf_cache"
_sys.path = [p for p in _sys.path if "site-packages" not in p]
_sys.path.insert(0, _PKG)
_os.environ["HF_HOME"] = _HF_CACHE
_os.environ["HUGGINGFACE_HUB_CACHE"] = _HF_CACHE
# ──────────────────────────────────────────────────────────────────────────────

import argparse
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_ID   = "jonathandinu/face-parsing"
RESOLUTION = 512

# Keywords to match against model's id2label for each region
# CelebAMask-HQ labels: skin, l_brow, r_brow, l_eye, r_eye, eye_g,
#   l_ear, r_ear, ear_r, nose, mouth, u_lip, l_lip, neck, ...
REGION_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "mouth": ("mouth", "lip"),
    "eyes":  ("l_eye", "r_eye"),
    "ears":  ("l_ear", "r_ear"),
    "nose":  ("nose",),
}

# Overlay highlight colours per region (R, G, B)
REGION_COLORS: Dict[str, Tuple[int, int, int]] = {
    "mouth": (220,  30,  30),   # red
    "eyes":  ( 30, 144, 220),   # blue
    "ears":  ( 30, 220,  80),   # green
    "nose":  (220, 160,  30),   # orange
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def run_face_parser(
    image: Image.Image,
    processor: SegformerImageProcessor,
    model: SegformerForSemanticSegmentation,
    device: str,
) -> np.ndarray:
    """Run face parsing; returns (H, W) int32 label map."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits   # (1, C, H/4, W/4)
    H, W = image.size[1], image.size[0]
    upsampled = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    seg_map = upsampled.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)
    return seg_map


def extract_region(
    image: Image.Image,
    seg_map: np.ndarray,
    label_indices: Set[int],
    color: Tuple[int, int, int],
    name: str,
) -> Tuple[np.ndarray, Image.Image, Image.Image]:
    """
    Extract binary mask, colour overlay, and tight crop for a set of label indices.

    Returns:
        mask    : (H, W) uint8 binary mask (0 / 255)
        overlay : PIL image with region highlighted
        crop    : PIL image tightly cropped to region bounding box
    """
    img_np = np.array(image)

    mask = np.zeros(seg_map.shape, dtype=np.uint8)
    for idx in label_indices:
        mask[seg_map == idx] = 255

    overlay = img_np.copy()
    region_px = mask > 0
    if region_px.any():
        overlay[region_px] = (
            overlay[region_px] * 0.4 + np.array(color) * 0.6
        ).astype(np.uint8)

    ys, xs = np.where(region_px)
    if len(ys) == 0:
        print(f"[face_parse] WARNING: no pixels found for region '{name}'.")
        crop = image.copy()
    else:
        pad = 10
        y1 = max(0, int(ys.min()) - pad)
        y2 = min(img_np.shape[0], int(ys.max()) + pad)
        x1 = max(0, int(xs.min()) - pad)
        x2 = min(img_np.shape[1], int(xs.max()) + pad)
        crop = image.crop((x1, y1, x2, y2))

    return mask, Image.fromarray(overlay), crop


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[face_parse] device={device}")

    root       = Path(__file__).parent
    source_img = Path(args.source)
    out_dir    = root / "output" / "source"
    out_dir.mkdir(parents=True, exist_ok=True)

    assert source_img.exists(), f"Source image not found: {source_img}"
    print(f"[face_parse] source image : {source_img}")
    print(f"[face_parse] output dir   : {out_dir}")

    # ── Load image ────────────────────────────────────────────────────────────
    image = Image.open(source_img).convert("RGB")
    print(f"[face_parse] image size   : {image.size}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[face_parse] loading model : {MODEL_ID}")
    processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
    model     = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID).to(device)
    model.eval()

    id2label = model.config.id2label
    print(f"[face_parse] label map : {id2label}")

    # ── Run segmentation ──────────────────────────────────────────────────────
    seg_map = run_face_parser(image, processor, model, device)
    unique = np.unique(seg_map)
    print(f"[face_parse] detected labels : {[id2label.get(int(l), l) for l in unique]}")

    # ── Extract and save each region ──────────────────────────────────────────
    for region_name, keywords in REGION_KEYWORDS.items():
        indices = {
            idx for idx, lbl in id2label.items()
            if any(kw in lbl.lower() for kw in keywords)
        }
        color = REGION_COLORS[region_name]
        print(f"[face_parse] {region_name:6s} indices : {indices} -> "
              f"{[id2label[i] for i in indices]}")

        mask, overlay, crop = extract_region(image, seg_map, indices, color, region_name)

        Image.fromarray(mask).save(out_dir / f"{region_name}_mask.png")
        overlay.save(out_dir / f"{region_name}_overlay.png")
        crop.save(out_dir / f"{region_name}_crop.png")
        print(f"[face_parse] saved {region_name}_mask/overlay/crop -> {out_dir}")

    print("[face_parse] done.")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = Path(__file__).parent

    parser = argparse.ArgumentParser(description="Extract facial regions via face parsing")
    parser.add_argument(
        "--source",
        default=str(root / "input" / "source_512.jpg"),
        help="Path to source image (default: pipeline/input/source_512.jpg)",
    )
    main(parser.parse_args())
