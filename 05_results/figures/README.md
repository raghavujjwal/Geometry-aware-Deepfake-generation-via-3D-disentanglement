# Figures Folder

This folder should contain the exact figures, qualitative outputs, and exportable tables used in the report or presentation.

## Expected outputs

- `main_results_table.png`
- `ablation_table.png`
- `benchmark_triplets_grid.png`
- `failure_cases_grid.png`
- `ablation_grid.png`
- `checkpoint_comparison_grid.png`

Recommended additions for the final checkpoint-comparison package:

- `benchmark_triplets_grid_5000.png`
- `benchmark_triplets_grid_10000.png`
- `failure_cases_grid_5000.png`
- `failure_cases_grid_10000.png`
- `ablation_grid_5000.png`
- `ablation_grid_10000.png`
- `checkpoint_metric_comparison.png`

## Deterministic regeneration

Use:

- [scripts/render_results_figures.py](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/scripts/render_results_figures.py)

with the manifests in this folder:

- [benchmark_triplets_manifest.csv](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/figures/benchmark_triplets_manifest.csv)
- [failure_cases_manifest.csv](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/figures/failure_cases_manifest.csv)
- [ablation_grid_manifest.csv](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/figures/ablation_grid_manifest.csv)
- [checkpoint_comparison_manifest.csv](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/figures/checkpoint_comparison_manifest.csv)

Selection guidance:

- `benchmark_triplets_manifest.csv`
  - 6–10 representative cases
  - include best-expression, typical, and visually clean outputs
- `failure_cases_manifest.csv`
  - 2–4 failure cases
  - prefer cases aligned with poor identity retrieval or visible expression drift
- `ablation_grid_manifest.csv`
  - one source-target pair shown under multiple ablation settings
- `checkpoint_comparison_manifest.csv`
  - same source-target pairs shown for `checkpoint-5000` and `checkpoint-10000`
