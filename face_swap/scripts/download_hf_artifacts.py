"""
Download training artifacts from the Hugging Face artifact repository.

The model checkpoints are intentionally kept on Hugging Face instead of being
committed to GitHub because the trainable checkpoint files are multi-GB.

Examples:
    python face_swap/scripts/download_hf_artifacts.py --run 5000
    python face_swap/scripts/download_hf_artifacts.py --run 10000 --token-env HF_TOKEN
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional


RUN_DIRS = {
    "5000": "face_swap_sd15_mediapipe_kaggle_t4_5000",
    "10000": "face_swap_sd15_mediapipe_kaggle_t4_10000",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download face-swap checkpoints and run artifacts from Hugging Face."
    )
    parser.add_argument(
        "--repo-id",
        default="RaghavUjjwal/face-encoders",
        help="Hugging Face repo containing the artifacts.",
    )
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=["model", "dataset", "space"],
        help="Hugging Face repository type.",
    )
    parser.add_argument(
        "--run",
        default="10000",
        choices=sorted(RUN_DIRS),
        help="Run/checkpoint bundle to download.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/hf_face_encoders",
        help="Local destination directory.",
    )
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable containing a Hugging Face token for private repos.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Do not use a token. Use only if the Hugging Face repo is public.",
    )
    return parser.parse_args()


def get_token(args: argparse.Namespace) -> Optional[str]:
    if args.public:
        return None
    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(
            f"Missing Hugging Face token. Set {args.token_env} or pass --public "
            "if the repo is public."
        )
    return token


def main() -> None:
    args = parse_args()
    token = get_token(args)
    run_dir = RUN_DIRS[args.run]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from exc

    local_dir = snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        token=token,
        allow_patterns=[f"{run_dir}/**"],
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
    )

    checkpoint_dir = output_dir / run_dir / f"checkpoint-{args.run}"
    print(f"Downloaded to: {local_dir}")
    print(f"Run directory: {output_dir / run_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print()
    print("Resume example:")
    print(
        "CUDA_VISIBLE_DEVICES=0 python face_swap/train.py "
        "--config face_swap/configs/train_config_kaggle_t4.yaml "
        f"--steps 15000 --resume {checkpoint_dir}"
    )


if __name__ == "__main__":
    main()
