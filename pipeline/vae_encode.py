"""
pipeline/vae_encode.py

Encodes pipeline/input/target image through the SDXL VAE and optionally
adds noise at a given diffusion timestep T.

Outputs saved to pipeline/output/target/:
  - clean_latent.pt      : VAE-encoded latent, shape (1, 4, 64, 64)
  - noisy_latent.pt      : Latent with DDPM noise added at timestep T
  - clean_decoded.png    : Decoded clean latent (reconstruction sanity check)
  - noisy_decoded.png    : Decoded noisy latent (visual of noise level)

Usage:
  python pipeline/vae_encode.py
  python pipeline/vae_encode.py --timestep 600 --no-decode-viz
  python pipeline/vae_encode.py --target pipeline/input/target.png
"""

from __future__ import annotations

# ── Path bootstrap (must be before third-party imports) ───────────────────────
import sys as _sys
_PKG = "e:/High fidelity face swap using 3D disintanglement/packages"
# Remove C drive site-packages so our E drive torch/diffusers win
_sys.path = [p for p in _sys.path if "site-packages" not in p]
_sys.path.insert(0, _PKG)
# ─────────────────────────────────────────────────────────────────────────────

import argparse
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, DDPMScheduler
from PIL import Image


# ── Constants ─────────────────────────────────────────────────────────────────

VAE_MODEL_ID  = "madebyollin/sdxl-vae-fp16-fix"
SCHED_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
RESOLUTION    = 512          # must match SDXL latent space (512 -> 64×64 latents)
DEFAULT_T     = 800          # noise level: 0 = clean, 999 = pure noise


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_image(path: Path) -> torch.Tensor:
    """Load image from disk -> (1, 3, H, W) float tensor in [-1, 1]."""
    img = Image.open(path).convert("RGB").resize((RESOLUTION, RESOLUTION))
    t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0  # [0, 1]
    t = t * 2.0 - 1.0                                                       # [-1, 1]
    return t.unsqueeze(0)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """(3, H, W) tensor in [-1, 1] -> PIL image."""
    tensor = (tensor.clamp(-1.0, 1.0) + 1.0) / 2.0
    arr = (tensor.cpu().float().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    print(f"[vae_encode] device={device}  dtype={dtype}")

    # ── Paths ─────────────────────────────────────────────────────────────────
    root       = Path(__file__).parent          # DECA/pipeline/
    target_img = Path(args.target)
    out_dir    = root / "output" / "target"
    out_dir.mkdir(parents=True, exist_ok=True)

    assert target_img.exists(), f"Target image not found: {target_img}"
    print(f"[vae_encode] target image : {target_img}")
    print(f"[vae_encode] output dir   : {out_dir}")

    # ── Load VAE ──────────────────────────────────────────────────────────────
    print(f"[vae_encode] loading VAE  : {VAE_MODEL_ID}")
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        VAE_MODEL_ID,
        torch_dtype=dtype,
    ).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # ── Load DDPM scheduler (for noise addition) ──────────────────────────────
    print(f"[vae_encode] loading scheduler from : {SCHED_MODEL_ID}")
    scheduler = DDPMScheduler.from_pretrained(SCHED_MODEL_ID, subfolder="scheduler")

    # ── Encode image -> clean latent ───────────────────────────────────────────
    image_tensor = load_image(target_img).to(device, dtype)

    with torch.no_grad():
        posterior = vae.encode(image_tensor).latent_dist
        clean_latent = posterior.sample() * vae.config.scaling_factor
        # shape: (1, 4, 64, 64)

    print(f"[vae_encode] clean latent : shape={tuple(clean_latent.shape)}"
          f"  min={clean_latent.min():.3f}  max={clean_latent.max():.3f}")

    torch.save(clean_latent.cpu(), out_dir / "clean_latent.pt")
    print(f"[vae_encode] saved -> {out_dir / 'clean_latent.pt'}")

    # ── Add DDPM noise at timestep T ──────────────────────────────────────────
    t = torch.tensor([args.timestep], device=device)
    noise = torch.randn_like(clean_latent)
    noisy_latent = scheduler.add_noise(clean_latent, noise, t)

    print(f"[vae_encode] noisy latent : timestep={args.timestep}"
          f"  min={noisy_latent.min():.3f}  max={noisy_latent.max():.3f}")

    torch.save(noisy_latent.cpu(), out_dir / "noisy_latent.pt")
    print(f"[vae_encode] saved -> {out_dir / 'noisy_latent.pt'}")

    # ── Decode back to pixel space for visual inspection ──────────────────────
    if not args.no_decode_viz:
        with torch.no_grad():
            clean_decoded = vae.decode(clean_latent / vae.config.scaling_factor).sample
            noisy_decoded = vae.decode(noisy_latent / vae.config.scaling_factor).sample

        tensor_to_pil(clean_decoded[0]).save(out_dir / "clean_decoded.png")
        tensor_to_pil(noisy_decoded[0]).save(out_dir / "noisy_decoded.png")
        print(f"[vae_encode] saved -> {out_dir / 'clean_decoded.png'}")
        print(f"[vae_encode] saved -> {out_dir / 'noisy_decoded.png'}")

    print("[vae_encode] done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = Path(__file__).parent

    parser = argparse.ArgumentParser(description="Encode target image through SDXL VAE")
    parser.add_argument(
        "--target",
        default=str(root / "input" / "target_512.jpg"),
        help="Path to target image (default: pipeline/input/target_512.jpg)",
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=DEFAULT_T,
        help=f"DDPM timestep for noise addition, 0–999 (default: {DEFAULT_T})",
    )
    parser.add_argument(
        "--no-decode-viz",
        action="store_true",
        help="Skip decoding latents back to images (faster, no viz output)",
    )
    main(parser.parse_args())
