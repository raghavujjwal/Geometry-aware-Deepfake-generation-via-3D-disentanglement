"""
training/train.py
Entry point for launching face swap training.

Usage:
    PYTHONPATH=/home/ubuntu/face_swap_3d:/home/ubuntu/face_swap_3d/face_swap \
        python3 face_swap/training/train.py \
        --config face_swap/configs/train_config.yaml
"""

import argparse
from pathlib import Path

from trainer import FaceSwapTrainer


def main():
    parser = argparse.ArgumentParser(description="Train geometry-aware face swap model.")
    parser.add_argument("--config", required=True, help="Path to train_config.yaml")
    parser.add_argument("--output_dir", default=None, help="Override output directory")
    args = parser.parse_args()

    trainer = FaceSwapTrainer(
        config_path=args.config,
        output_dir=args.output_dir,
    )
    trainer.train()


if __name__ == "__main__":
    main()
