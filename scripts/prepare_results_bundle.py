from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


BASELINE_DEFAULTS = {
    "dataset": "CelebA-HQ-256",
    "num_pairs": 100,
    "checkpoint": "checkpoint-5000",
    "num_inference_steps": 20,
    "guidance_scale": 1.0,
    "denoise_strength": 0.25,
    "blend_alpha": 0.65,
    "region_attn_scale": 1.0,
    "csim": 0.024533,
    "id_retrieval_top1": 0.0,
    "expression_error": 0.222982,
}

MAIN_COLUMNS = [
    "dataset",
    "num_pairs",
    "checkpoint",
    "num_inference_steps",
    "guidance_scale",
    "denoise_strength",
    "blend_alpha",
    "region_attn_scale",
    "csim",
    "id_retrieval_top1",
    "expression_error",
]

ABLATION_COLUMNS = [
    "run_index",
    "denoise_strength",
    "blend_alpha",
    "region_attn_scale",
    "guidance_scale",
    "num_inference_steps",
    "csim",
    "id_retrieval_top1",
    "expression_error",
]


def ensure_dirs(results_dir: Path) -> None:
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)
    (results_dir / "logs").mkdir(parents=True, exist_ok=True)
    (results_dir / "logs" / "raw_kaggle_exports").mkdir(parents=True, exist_ok=True)


def normalize_main_results(summary_path: Path | None) -> pd.DataFrame:
    if summary_path and summary_path.exists():
        df = pd.read_csv(summary_path)
        if df.empty:
            return pd.DataFrame([BASELINE_DEFAULTS], columns=MAIN_COLUMNS)

        row = df.iloc[0].to_dict()
        normalized = {key: row.get(key, BASELINE_DEFAULTS[key]) for key in MAIN_COLUMNS}
        if "checkpoint" not in row and "checkpoint" in BASELINE_DEFAULTS:
            normalized["checkpoint"] = BASELINE_DEFAULTS["checkpoint"]
        return pd.DataFrame([normalized], columns=MAIN_COLUMNS)

    return pd.DataFrame([BASELINE_DEFAULTS], columns=MAIN_COLUMNS)


def normalize_ablations(ablation_path: Path | None) -> pd.DataFrame:
    if ablation_path and ablation_path.exists():
        df = pd.read_csv(ablation_path)
        if df.empty:
            return pd.DataFrame(columns=ABLATION_COLUMNS)

        for col in ABLATION_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA

        return df[ABLATION_COLUMNS].copy()

    return pd.DataFrame(columns=ABLATION_COLUMNS)


def copy_raw_exports(raw_files: list[Path], logs_dir: Path) -> list[str]:
    copied = []
    for raw_file in raw_files:
        if raw_file.exists():
            target = logs_dir / "raw_kaggle_exports" / raw_file.name
            shutil.copy2(raw_file, target)
            copied.append(raw_file.name)
    return copied


def write_inventory(results_dir: Path, copied_files: list[str]) -> None:
    inventory_path = results_dir / "results_inventory.md"
    text = inventory_path.read_text(encoding="utf-8")
    if copied_files:
        extra = "\n## Copied raw Kaggle exports\n\n" + "\n".join(f"- `{name}`" for name in copied_files) + "\n"
    else:
        extra = "\n## Copied raw Kaggle exports\n\n- No raw Kaggle files were supplied during the latest normalization run.\n"

    if "## Copied raw Kaggle exports" in text:
        text = text.split("## Copied raw Kaggle exports")[0].rstrip() + extra
    else:
        text = text.rstrip() + extra
    inventory_path.write_text(text + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize Kaggle benchmark outputs into 05_results/.")
    parser.add_argument("--results-dir", type=Path, default=Path("05_results"))
    parser.add_argument("--main-summary", type=Path)
    parser.add_argument("--pairwise", type=Path)
    parser.add_argument("--retrieval", type=Path)
    parser.add_argument("--csim", type=Path)
    parser.add_argument("--ablations", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    ensure_dirs(results_dir)

    main_df = normalize_main_results(args.main_summary)
    ablation_df = normalize_ablations(args.ablations)

    main_df.to_csv(results_dir / "main_results.csv", index=False)
    ablation_df.to_csv(results_dir / "ablations.csv", index=False)

    copied = copy_raw_exports(
        [p for p in [args.main_summary, args.pairwise, args.retrieval, args.csim, args.ablations] if p],
        results_dir / "logs",
    )
    write_inventory(results_dir, copied)

    print(f"Wrote {results_dir / 'main_results.csv'}")
    print(f"Wrote {results_dir / 'ablations.csv'}")
    if copied:
        print("Copied raw exports:", ", ".join(copied))
    else:
        print("No raw exports copied.")


if __name__ == "__main__":
    main()
