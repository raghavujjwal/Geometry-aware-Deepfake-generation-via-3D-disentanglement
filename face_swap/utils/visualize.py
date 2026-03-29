"""
utils/visualize.py
Visualization utilities for training and evaluation of the face swap model.

Provides:
  - save_comparison_grid: Side-by-side source / generated / target comparison grid.
  - plot_training_curves: Loss / metric plotting from TensorBoard event files.
  - visualize_region_crops: Display per-region crops in a grid.
  - visualize_depth_map: Colourised depth map display.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image, ImageDraw, ImageFont


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _tensor_to_pil_list(tensor: torch.Tensor) -> List[Image.Image]:
    """
    Convert (B, 3, H, W) float tensor in [-1, 1] to list of PIL images.

    Args:
        tensor: Batch of images in [-1, 1] or [0, 1].
    """
    tensor = (tensor.detach().cpu().clamp(-1, 1) + 1.0) / 2.0
    images = []
    for i in range(tensor.shape[0]):
        img_np = (tensor[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        images.append(Image.fromarray(img_np))
    return images


def _add_label(
    img: Image.Image,
    text: str,
    font_size: int = 18,
    bg_color: Tuple = (0, 0, 0),
    text_color: Tuple = (255, 255, 255),
) -> Image.Image:
    """Draw a label bar at the bottom of an image."""
    label_h = font_size + 8
    new_img = Image.new("RGB", (img.width, img.height + label_h), bg_color)
    new_img.paste(img, (0, 0))
    draw = ImageDraw.Draw(new_img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    x = (img.width - text_w) // 2
    draw.text((x, img.height + 4), text, fill=text_color, font=font)
    return new_img


# ─────────────────────────────────────────────────────────────────────────────
# Comparison grid
# ─────────────────────────────────────────────────────────────────────────────


def save_comparison_grid(
    source: torch.Tensor,
    generated: torch.Tensor,
    target: torch.Tensor,
    save_path: Union[str, Path],
    nrow: int = 8,
    max_images: int = 8,
) -> None:
    """
    Save a side-by-side comparison grid: Source | Generated | Target.

    Each row corresponds to one sample. Saves as PNG.

    Args:
        source: Source identity images (B, 3, H, W), range [-1, 1].
        generated: Swapped images (B, 3, H, W).
        target: Target pose images (B, 3, H, W).
        save_path: Output file path.
        nrow: Number of triplets per row.
        max_images: Maximum number of samples to display.
    """
    n = min(max_images, source.shape[0])
    src_list = _tensor_to_pil_list(source[:n])
    gen_list = _tensor_to_pil_list(generated[:n])
    tgt_list = _tensor_to_pil_list(target[:n])

    # Add labels
    src_list = [_add_label(img, "Source") for img in src_list]
    gen_list = [_add_label(img, "Generated") for img in gen_list]
    tgt_list = [_add_label(img, "Target") for img in tgt_list]

    # Interleave: [src0, gen0, tgt0, src1, gen1, tgt1, ...]
    interleaved: List[Image.Image] = []
    for s, g, t in zip(src_list, gen_list, tgt_list):
        interleaved.extend([s, g, t])

    # Convert to tensors
    def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
        arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    tensors = torch.stack([_pil_to_tensor(im) for im in interleaved])
    grid = vutils.make_grid(tensors, nrow=nrow * 3, padding=4, pad_value=0.5)
    grid_pil = Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    grid_pil.save(str(save_path))


# ─────────────────────────────────────────────────────────────────────────────
# Region crop grid
# ─────────────────────────────────────────────────────────────────────────────


def visualize_region_crops(
    region_crops: Dict[str, torch.Tensor],
    save_path: Optional[Union[str, Path]] = None,
) -> Image.Image:
    """
    Visualise per-region crops in a horizontal strip with region labels.

    Args:
        region_crops: Dict[region_name → (B, 3, H, W) or (3, H, W) tensor].
        save_path: Optional path to save the resulting image.

    Returns:
        PIL image of the crop grid.
    """
    images: List[Image.Image] = []
    for region, crop_tensor in region_crops.items():
        if crop_tensor.ndim == 4:
            crop_tensor = crop_tensor[0]  # take first in batch
        pil_list = _tensor_to_pil_list(crop_tensor.unsqueeze(0))
        images.append(_add_label(pil_list[0].resize((96, 96)), region, font_size=12))

    if not images:
        return Image.new("RGB", (100, 100))

    total_w = sum(img.width for img in images) + (len(images) - 1) * 4
    max_h = max(img.height for img in images)
    canvas = Image.new("RGB", (total_w, max_h), color=(50, 50, 50))
    x_off = 0
    for img in images:
        canvas.paste(img, (x_off, 0))
        x_off += img.width + 4

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        canvas.save(str(save_path))
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Depth map visualisation
# ─────────────────────────────────────────────────────────────────────────────


def visualize_depth_map(
    depth: torch.Tensor,
    colormap: str = "plasma",
    save_path: Optional[Union[str, Path]] = None,
) -> Image.Image:
    """
    Render a 1-channel or 3-channel depth map with a colormap.

    Args:
        depth: Depth tensor (B, 1, H, W) or (B, 3, H, W) or (1, H, W),
               values in [0, 1].
        colormap: Matplotlib colormap name.
        save_path: Optional save path.

    Returns:
        PIL image with false-colour depth visualisation.
    """
    import matplotlib.pyplot as plt

    if depth.ndim == 4:
        depth = depth[0]
    if depth.shape[0] == 3:
        depth = depth.mean(0, keepdim=True)

    depth_np = depth[0].cpu().float().numpy()
    depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)

    cmap = plt.get_cmap(colormap)
    colored = (cmap(depth_np)[:, :, :3] * 255).astype(np.uint8)
    result = Image.fromarray(colored)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        result.save(str(save_path))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Training curve plotting
# ─────────────────────────────────────────────────────────────────────────────


def plot_training_curves(
    log_dir: Union[str, Path],
    tags: List[str],
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot training curves from TensorBoard event files.

    Args:
        log_dir: Path to TensorBoard log directory.
        tags: List of scalar tag names to plot.
        save_path: Path to save the generated figure. Displays if None.
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
        import matplotlib.pyplot as plt

        ea = event_accumulator.EventAccumulator(str(log_dir))
        ea.Reload()

        fig, axes = plt.subplots(
            len(tags), 1, figsize=(12, 4 * len(tags)), squeeze=False
        )
        for i, tag in enumerate(tags):
            if tag in ea.scalars.Keys():
                events = ea.scalars.Items(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                axes[i][0].plot(steps, values, linewidth=1.5)
                axes[i][0].set_title(tag)
                axes[i][0].set_xlabel("Step")
                axes[i][0].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        else:
            plt.show()
        plt.close()
    except ImportError:
        print("tensorboard package required for plot_training_curves.")
