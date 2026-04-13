"""
pipeline/preprocess.py

Resizes raw source and target images to 512x512 and saves them as
pipeline-ready inputs. Run this once before any other pipeline script.

Outputs saved to pipeline/input/:
  - source_512.jpg
  - target_512.jpg

Usage:
  python pipeline/preprocess.py
  python pipeline/preprocess.py --source pipeline/input/source.jpg --target pipeline/input/target.jpg
  python pipeline/preprocess.py --size 512
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


RESOLUTION = 512


def resize_and_save(src_path: Path, out_path: Path, size: int) -> None:
    img = Image.open(src_path).convert("RGB")
    original = img.size
    img = img.resize((size, size), Image.LANCZOS)
    img.save(out_path, quality=95)
    print(f"[preprocess] {src_path.name} : {original[0]}x{original[1]} -> {size}x{size} -> {out_path.name}")


def main(args: argparse.Namespace) -> None:
    source = Path(args.source)
    target = Path(args.target)
    size   = args.size
    out    = Path(args.source).parent   # save back into pipeline/input/

    assert source.exists(), f"Source image not found: {source}"
    assert target.exists(), f"Target image not found: {target}"

    resize_and_save(source, out / f"source_{size}.jpg", size)
    resize_and_save(target, out / f"target_{size}.jpg", size)

    print("[preprocess] done. Use source_512.jpg and target_512.jpg in all pipeline scripts.")


if __name__ == "__main__":
    root = Path(__file__).parent

    parser = argparse.ArgumentParser(description="Resize source and target images for pipeline")
    parser.add_argument("--source", default=str(root / "input" / "source.jpg"))
    parser.add_argument("--target", default=str(root / "input" / "target.jpg"))
    parser.add_argument("--size",   type=int, default=RESOLUTION)
    main(parser.parse_args())
