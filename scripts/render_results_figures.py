from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def render_table(df: pd.DataFrame, title: str, output_path: Path, max_rows: int = 12) -> None:
    display_df = df.head(max_rows).copy()
    rows = [list(display_df.columns)] + display_df.astype(str).values.tolist()
    font = ImageFont.load_default()
    padding_x = 12
    padding_y = 10
    row_height = 28
    title_height = 44

    col_widths = []
    for col_idx in range(len(rows[0])):
        max_width = 0
        for row in rows:
            bbox = font.getbbox(str(row[col_idx]))
            max_width = max(max_width, bbox[2] - bbox[0])
        col_widths.append(max_width + 2 * padding_x)

    image_width = sum(col_widths) + 2
    image_height = title_height + row_height * len(rows) + 2
    image = Image.new("RGB", (image_width, image_height), "white")
    draw = ImageDraw.Draw(image)

    draw.text((12, 12), title, fill="black", font=font)

    y = title_height
    for row_idx, row in enumerate(rows):
        x = 0
        fill = "#E8EEF8" if row_idx == 0 else "white"
        for col_idx, cell in enumerate(row):
            width = col_widths[col_idx]
            draw.rectangle([x, y, x + width, y + row_height], outline="black", fill=fill)
            draw.text((x + padding_x, y + padding_y), str(cell), fill="black", font=font)
            x += width
        y += row_height

    image.save(output_path)


def _open_rgb(path_str: str) -> Image.Image:
    return Image.open(path_str).convert("RGB")


def render_triplet_grid(manifest_path: Path, output_path: Path, title: str) -> None:
    if not manifest_path.exists():
        return

    df = pd.read_csv(manifest_path)
    if df.empty:
        return

    font = ImageFont.load_default()
    sample_images = [
        _open_rgb(df.iloc[0]["source_path"]),
        _open_rgb(df.iloc[0]["target_path"]),
        _open_rgb(df.iloc[0]["output_path"]),
    ]
    cell_w, cell_h = sample_images[0].size
    caption_width = 180
    header_height = 28
    row_gap = 8
    rows = len(df)
    total_width = caption_width + 3 * cell_w
    total_height = header_height + rows * (cell_h + row_gap)
    canvas = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(canvas)

    headers = ["Source", "Target", "Output"]
    for idx, header in enumerate(headers):
        x = caption_width + idx * cell_w + 10
        draw.text((x, 6), header, fill="black", font=font)

    for row_idx, (_, row) in enumerate(df.iterrows()):
        y = header_height + row_idx * (cell_h + row_gap)
        caption = str(row.get("caption", row.get("category", "")))
        draw.text((10, y + 10), caption, fill="black", font=font)
        images = [
            _open_rgb(row["source_path"]),
            _open_rgb(row["target_path"]),
            _open_rgb(row["output_path"]),
        ]
        for col_idx, img in enumerate(images):
            x = caption_width + col_idx * cell_w
            canvas.paste(img.resize((cell_w, cell_h)), (x, y))

    draw.text((10, 6), title, fill="black", font=font)
    canvas.save(output_path)


def render_ablation_grid(manifest_path: Path, output_path: Path) -> None:
    if not manifest_path.exists():
        return

    df = pd.read_csv(manifest_path)
    if df.empty:
        return

    pair_groups = list(df.groupby("pair_index"))
    font = ImageFont.load_default()
    first = pair_groups[0][1].iloc[0]
    sample = _open_rgb(first["source_path"])
    cell_w, cell_h = sample.size
    max_variants = max(len(group) for _, group in pair_groups)
    cols = 3 + max_variants
    rows = len(pair_groups)
    label_width = 120
    header_height = 28
    row_gap = 8
    total_width = label_width + cols * cell_w
    total_height = header_height + rows * (cell_h + row_gap)
    canvas = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(canvas)

    for row_idx, (pair_index, group) in enumerate(pair_groups):
        y = header_height + row_idx * (cell_h + row_gap)
        draw.text((10, y + 10), f"Pair {pair_index}", fill="black", font=font)
        panels = [
            ("Source", _open_rgb(group.iloc[0]["source_path"])),
            ("Target", _open_rgb(group.iloc[0]["target_path"])),
            ("Baseline", _open_rgb(group.iloc[0]["baseline_output_path"])),
        ]
        for _, item in group.iterrows():
            panels.append((str(item["variant_label"]), _open_rgb(item["variant_output_path"])))

        for col_idx, (label, image) in enumerate(panels):
            x = label_width + col_idx * cell_w
            if row_idx == 0:
                draw.text((x + 10, 6), label, fill="black", font=font)
            canvas.paste(image.resize((cell_w, cell_h)), (x, y))

    draw.text((10, 6), "Ablation Grid", fill="black", font=font)
    canvas.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render result figures from CSV manifests.")
    parser.add_argument("--results-dir", type=Path, default=Path("05_results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    figures_dir = results_dir / "figures"

    main_results = pd.read_csv(results_dir / "main_results.csv")
    render_table(main_results, "Main Benchmark Results", figures_dir / "main_results_table.png", max_rows=5)

    ablations_path = results_dir / "ablations.csv"
    ablations = pd.read_csv(ablations_path)
    if ablations.empty:
        placeholder = pd.DataFrame(
            [{"status": "Pending ablation sweep", "note": "Populate 05_results/ablations.csv from Kaggle before export."}]
        )
        render_table(placeholder, "Ablation Results", figures_dir / "ablation_table.png", max_rows=5)
    else:
        render_table(ablations, "Ablation Results", figures_dir / "ablation_table.png")

    render_triplet_grid(
        figures_dir / "benchmark_triplets_manifest.csv",
        figures_dir / "benchmark_triplets_grid.png",
        "Benchmark Triplets",
    )
    render_triplet_grid(
        figures_dir / "failure_cases_manifest.csv",
        figures_dir / "failure_cases_grid.png",
        "Failure Cases",
    )
    render_ablation_grid(figures_dir / "ablation_grid_manifest.csv", figures_dir / "ablation_grid.png")

    print(f"Rendered figures into {figures_dir}")


if __name__ == "__main__":
    main()
