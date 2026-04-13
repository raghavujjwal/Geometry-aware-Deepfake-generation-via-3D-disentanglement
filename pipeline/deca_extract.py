"""
pipeline/deca_extract.py

Passes source and target images through DECA to extract 3D geometry.
Saves depth maps, normal maps, and FLAME parameters for both images.

These geometry outputs feed into two downstream steps:
  - depth/normal maps -> ControlNet conditioning for the U-Net
  - same maps + face-parse masks -> per-region geometry crops (mouth, eyes, etc.)

Outputs saved to pipeline/output/source/ and pipeline/output/target/:
  - depth_map.png      : rendered depth map (grayscale, 512x512)
  - normal_map.png     : rendered surface normal map (RGB, 512x512)
  - flame_params.npz   : FLAME parameters (shape, exp, pose, cam, light)

Usage:
  python pipeline/deca_extract.py
  python pipeline/deca_extract.py --source pipeline/input/source_512.jpg
  python pipeline/deca_extract.py --source pipeline/input/source_512.jpg --target pipeline/input/target_512.jpg
"""

from __future__ import annotations

# ── Path bootstrap ─────────────────────────────────────────────────────────────
import sys as _sys
import os as _os
_ROOT   = "e:/High fidelity face swap using 3D disintanglement/DECA"
_PKG    = "e:/High fidelity face swap using 3D disintanglement/packages"
_HF     = "e:/High fidelity face swap using 3D disintanglement/hf_cache"
_sys.path = [p for p in _sys.path if "site-packages" not in p]
_sys.path.insert(0, _PKG)
_sys.path.insert(0, _ROOT)          # gives access to decalib.*
_os.environ["HF_HOME"]              = _HF
_os.environ["HUGGINGFACE_HUB_CACHE"] = _HF
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import gc
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

warnings.filterwarnings("ignore")
torch.set_num_threads(2)   # fewer threads = less per-thread memory overhead on CPU

from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg


# ── Constants ─────────────────────────────────────────────────────────────────

DECA_DATA_DIR = str(Path(_ROOT) / "data")
OUT_SIZE      = 512     # upsample geometry maps to match pipeline resolution


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_deca(device: str) -> DECA:
    """Initialise DECA with the standard config (no texture model needed)."""
    deca_cfg.model.use_tex     = False
    deca_cfg.model.extract_tex = False
    deca_cfg.rasterizer_type   = "standard"   # CPU-compatible; no pytorch3d needed
    deca_cfg.model.cfg_path    = str(Path(_ROOT) / "configs" / "release_version" / "deca_coarse.yml")
    deca_cfg.pretrained_modelpath = str(Path(DECA_DATA_DIR) / "deca_model.tar")

    deca_cfg.model.flame_model_path  = str(Path(DECA_DATA_DIR) / "generic_model.pkl")
    deca_cfg.model.static_landmark_embedding_path = str(
        Path(DECA_DATA_DIR) / "landmark_embedding.npy"
    )
    deca_cfg.model.flame_lmk_embedding_path = str(
        Path(DECA_DATA_DIR) / "landmark_embedding.npy"
    )

    print(f"[deca_extract] loading DECA checkpoint: {deca_cfg.pretrained_modelpath}")
    model = DECA(config=deca_cfg, device=device)
    model.eval()
    gc.collect()   # free any leftover allocations from model loading
    return model


def load_image_for_deca(image_path: Path) -> torch.Tensor:
    """
    Load a face image and prepare it for DECA encoding.
    Resizes to 224x224 and normalises to [0, 1].
    Returns (1, 3, 224, 224) float tensor.
    """
    img = Image.open(image_path).convert("RGB").resize((224, 224), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0   # [0, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 224, 224)
    return tensor


def run_deca(image_path: Path, deca: DECA, device: str) -> dict:
    """
    Load one image and run full DECA encode+decode.
    Returns a flat dict with tensors on CPU.
    """
    images = load_image_for_deca(image_path).to(device)   # (1, 3, 224, 224)

    with torch.no_grad():
        codedict = deca.encode(images)
        opdict, _visdict = deca.decode(codedict)

        # Depth map: render_depth returns (1, 1, H, W), values in [0, 1]
        depth = deca.render.render_depth(opdict["trans_verts"])   # (1, 1, 224, 224)

    return {
        "normal": opdict["normal_images"].cpu(),    # (1, 3, 224, 224) in [0, 1]
        "depth":  depth.cpu(),                      # (1, 1, 224, 224) in [0, 1]
        "shape":  codedict["shape"].cpu(),
        "exp":    codedict["exp"].cpu(),
        "pose":   codedict["pose"].cpu(),
        "cam":    codedict["cam"].cpu(),
        "light":  codedict["light"].cpu(),
    }


def save_outputs(result: dict, out_dir: Path, label: str) -> None:
    """
    Save depth map, normal map, and FLAME params to out_dir.

    Args:
        result: dict returned by run_deca()
        out_dir: output directory (created if needed)
        label: 'source' or 'target' (for log messages)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Normal map ────────────────────────────────────────────────────────────
    # Upsample (1, 3, 224, 224) -> (1, 3, 512, 512), then save as RGB PNG
    normal_up = F.interpolate(
        result["normal"], size=(OUT_SIZE, OUT_SIZE), mode="bilinear", align_corners=False
    )
    normal_np = (normal_up[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(normal_np).save(out_dir / "normal_map.png")
    print(f"[deca_extract] {label}: saved normal_map.png  shape={normal_np.shape}")

    # ── Depth map ─────────────────────────────────────────────────────────────
    # Upsample (1, 1, 224, 224) -> (1, 1, 512, 512), save as grayscale PNG
    depth_up = F.interpolate(
        result["depth"], size=(OUT_SIZE, OUT_SIZE), mode="bilinear", align_corners=False
    )
    depth_np = (depth_up[0, 0].numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(depth_np, mode="L").save(out_dir / "depth_map.png")
    print(f"[deca_extract] {label}: saved depth_map.png   shape={depth_np.shape}")

    # ── FLAME parameters ──────────────────────────────────────────────────────
    np.savez(
        out_dir / "flame_params.npz",
        shape=result["shape"].numpy(),
        exp=result["exp"].numpy(),
        pose=result["pose"].numpy(),
        cam=result["cam"].numpy(),
        light=result["light"].numpy(),
    )
    print(f"[deca_extract] {label}: saved flame_params.npz")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[deca_extract] device={device}")

    pipeline_root = Path(__file__).parent

    # Resolve image paths
    source_img = Path(args.source)
    target_img = Path(args.target) if args.target else None

    assert source_img.exists(), f"Source image not found: {source_img}"
    if target_img is not None:
        assert target_img.exists(), f"Target image not found: {target_img}"

    # Load DECA once, reuse for both images
    deca = load_deca(device)

    # ── Process source ─────────────────────────────────────────────────────────
    print(f"\n[deca_extract] processing source: {source_img}")
    src_result = run_deca(source_img, deca, device)
    save_outputs(src_result, pipeline_root / "output" / "source", label="source")

    # ── Process target (optional) ──────────────────────────────────────────────
    if target_img is not None:
        print(f"\n[deca_extract] processing target: {target_img}")
        tgt_result = run_deca(target_img, deca, device)
        save_outputs(tgt_result, pipeline_root / "output" / "target", label="target")

    print("\n[deca_extract] done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Extract DECA depth/normal maps and FLAME params from face images"
    )
    parser.add_argument(
        "--source",
        default=str(root / "input" / "source_512.jpg"),
        help="Path to source image (default: pipeline/input/source_512.jpg)",
    )
    parser.add_argument(
        "--target",
        default=str(root / "input" / "target_512.jpg"),
        help="Path to target image (default: pipeline/input/target_512.jpg). "
             "Pass an empty string to skip target.",
    )
    main(parser.parse_args())
