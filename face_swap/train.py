"""
train.py
Entry point for geometry-aware face swapping model training.

Usage:
    python train.py --config configs/train_config.yaml
    python train.py --config configs/train_config.yaml --output_dir runs/exp1

Multi-GPU (Accelerate):
    accelerate launch --multi_gpu train.py --config configs/train_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Geometry-Aware Face Swapping — Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to train_config.yaml",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (overrides config)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override total training steps from config",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size from config",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with fast data (100 samples, 10 steps)",
    )
    return parser.parse_args()


def main() -> None:
    """Main training entry point."""
    args = parse_args()

    # Validate config path
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    # Apply CLI overrides to config before loading trainer
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if args.output_dir:
        cfg["experiment"]["output_dir"] = args.output_dir
    if args.resume:
        cfg["experiment"]["resume_from_checkpoint"] = args.resume
    if args.steps:
        cfg["training"]["total_steps"] = args.steps
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size

    if args.debug:
        print("[DEBUG] Running in debug mode with reduced data and steps.")
        cfg["data"]["datasets"] = [
            {
                "name": cfg["data"]["datasets"][0]["name"],
                "root": cfg["data"]["datasets"][0]["root"],
                "split": "train",
                "num_samples": 100,
            }
        ]
        cfg["training"]["total_steps"] = 10
        cfg["training"]["save_every_steps"] = 5
        cfg["training"]["validate_every_steps"] = 5
        cfg["training"]["log_every_steps"] = 1
        cfg["training"]["batch_size"] = 2

    # Write effective config to output dir for reproducibility
    import os
    from pathlib import Path as _Path

    out_dir = _Path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    eff_cfg_path = out_dir / "effective_config.yaml"
    with open(eff_cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"[train.py] Effective config saved to: {eff_cfg_path}")

    # Also write the patched config back temporarily so trainer can load it
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as tmp:
        yaml.dump(cfg, tmp, default_flow_style=False)
        tmp_path = tmp.name

    try:
        from training.trainer import FaceSwapTrainer

        trainer = FaceSwapTrainer(
            config_path=tmp_path,
            output_dir=args.output_dir,
        )
        trainer.train()
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    main()
