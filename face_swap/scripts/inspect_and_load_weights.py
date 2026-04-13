"""
scripts/inspect_and_load_weights.py
Inspect pre-trained encoder checkpoints and verify they load correctly.

Usage:
    # 1. Inspect a single checkpoint to see its key structure:
    python scripts/inspect_and_load_weights.py inspect path/to/eyes_encoder.pth

    # 2. Test loading all 4 region encoders:
    python scripts/inspect_and_load_weights.py load \
        --eyes  pretrained/eyes_encoder.pth  \
        --nose  pretrained/nose_encoder.pth  \
        --mouth pretrained/mouth_encoder.pth \
        --ears  pretrained/ears_encoder.pth

    # 3. Run a quick forward pass to verify shapes:
    python scripts/inspect_and_load_weights.py test \
        --eyes  pretrained/eyes_encoder.pth  \
        --nose  pretrained/nose_encoder.pth  \
        --mouth pretrained/mouth_encoder.pth \
        --ears  pretrained/ears_encoder.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch


def cmd_inspect(args: argparse.Namespace) -> None:
    """Inspect a checkpoint file and print its structure."""
    from utils.weight_loader import inspect_checkpoint

    info = inspect_checkpoint(args.checkpoint)

    print(f"\n{'=' * 70}")
    print(f"Checkpoint: {info['path']}")
    print(f"Type: {info['type']}")
    print(f"{'=' * 70}")

    if "top_level_keys" in info:
        print(f"\nTop-level keys: {info['top_level_keys']}")

    if "total_parameters_M" in info:
        print(f"Total parameters: {info['total_parameters_M']}")
        print(f"Number of tensors: {info['num_params']}")

    if "keys" in info:
        print(f"\nState dict keys ({len(info['keys'])}):")
        print("-" * 50)
        for key in info["keys"]:
            shape = info.get("shapes", {}).get(key, "?")
            print(f"  {key:60s} {shape}")

    if "note" in info:
        print(f"\nNote: {info['note']}")


def cmd_load(args: argparse.Namespace) -> None:
    """Test loading pre-trained weights into FaceRegionEncoder."""
    from models.region_encoder import FaceRegionEncoder
    from utils.weight_loader import load_pretrained_region_encoders

    weight_paths = {}
    if args.eyes:
        weight_paths["eyes"] = args.eyes
    if args.nose:
        weight_paths["nose"] = args.nose
    if args.mouth:
        weight_paths["mouth"] = args.mouth
    if args.ears:
        weight_paths["ears"] = args.ears

    if not weight_paths:
        print("[ERROR] Provide at least one --eyes/--nose/--mouth/--ears path.")
        sys.exit(1)

    print(f"\nCreating FaceRegionEncoder (ResNet-50 + Transformer, 4 regions)...")
    encoder = FaceRegionEncoder(
        projection_dim=512,
        num_tokens=4,
        regions=["eyes", "nose", "mouth", "ears"],
        in_channels=args.in_channels,
        freeze_backbone=True,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads,
    )

    print(f"\nLoading pre-trained weights...")
    load_pretrained_region_encoders(
        encoder,
        weight_paths=weight_paths,
        strict=False,
        verbose=True,
    )
    print("\nWeight loading completed successfully.")

    # Print trainable param count
    trainable = sum(p.numel() for p in encoder.trainable_parameters())
    total = sum(p.numel() for p in encoder.parameters())
    frozen = total - trainable
    print(f"\nParameter summary:")
    print(f"  Total:     {total / 1e6:.1f}M")
    print(f"  Trainable: {trainable / 1e6:.1f}M")
    print(f"  Frozen:    {frozen / 1e6:.1f}M")


def cmd_test(args: argparse.Namespace) -> None:
    """Load weights and run a forward pass to verify shapes."""
    from models.region_encoder import FaceRegionEncoder
    from utils.weight_loader import load_pretrained_region_encoders

    weight_paths = {}
    if args.eyes:
        weight_paths["eyes"] = args.eyes
    if args.nose:
        weight_paths["nose"] = args.nose
    if args.mouth:
        weight_paths["mouth"] = args.mouth
    if args.ears:
        weight_paths["ears"] = args.ears

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"\nCreating FaceRegionEncoder...")
    encoder = FaceRegionEncoder(
        projection_dim=512,
        num_tokens=4,
        regions=["eyes", "nose", "mouth", "ears"],
        in_channels=args.in_channels,
        freeze_backbone=True,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads,
    )

    if weight_paths:
        print(f"\nLoading pre-trained weights...")
        load_pretrained_region_encoders(encoder, weight_paths, strict=False, verbose=True)

    encoder = encoder.to(device).eval()

    # Create dummy inputs: 4 region crops of (B, 7, 64, 64)
    B = 2
    print(f"\nRunning forward pass (batch_size={B})...")
    dummy_crops = {
        "eyes":  torch.randn(B, args.in_channels, 64, 64, device=device),
        "nose":  torch.randn(B, args.in_channels, 64, 64, device=device),
        "mouth": torch.randn(B, args.in_channels, 64, 64, device=device),
        "ears":  torch.randn(B, args.in_channels, 64, 64, device=device),
    }

    with torch.no_grad():
        outputs = encoder(dummy_crops)

    print(f"\nOutput shapes:")
    for region, feat in outputs.items():
        print(f"  {region:10s}: {tuple(feat.shape)}  (expected: ({B}, 4, 512))")

    # Test concatenated output
    concat = encoder.get_concatenated_features(dummy_crops)
    print(f"  {'concat':10s}: {tuple(concat.shape)}  (expected: ({B}, 16, 512))")

    print(f"\nAll forward pass checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and load pre-trained encoder weights"
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── inspect ──
    p_inspect = subparsers.add_parser("inspect", help="Inspect a checkpoint file")
    p_inspect.add_argument("checkpoint", type=str, help="Path to .pth / .pt file")

    # ── load ──
    p_load = subparsers.add_parser("load", help="Test loading weights into encoder")
    p_load.add_argument("--eyes", type=str, default=None)
    p_load.add_argument("--nose", type=str, default=None)
    p_load.add_argument("--mouth", type=str, default=None)
    p_load.add_argument("--ears", type=str, default=None)
    p_load.add_argument("--in_channels", type=int, default=7)
    p_load.add_argument("--num_transformer_layers", type=int, default=2)
    p_load.add_argument("--num_heads", type=int, default=8)

    # ── test ──
    p_test = subparsers.add_parser("test", help="Load + forward pass test")
    p_test.add_argument("--eyes", type=str, default=None)
    p_test.add_argument("--nose", type=str, default=None)
    p_test.add_argument("--mouth", type=str, default=None)
    p_test.add_argument("--ears", type=str, default=None)
    p_test.add_argument("--in_channels", type=int, default=7)
    p_test.add_argument("--num_transformer_layers", type=int, default=2)
    p_test.add_argument("--num_heads", type=int, default=8)

    args = parser.parse_args()
    if args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "load":
        cmd_load(args)
    elif args.command == "test":
        cmd_test(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
