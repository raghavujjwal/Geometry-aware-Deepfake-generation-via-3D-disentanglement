# Results Inventory

This folder packages the current benchmark work into the submission-ready `05_results/` layout required by the course guide.

## Current files

- [main_results.csv](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/main_results.csv)
  - Canonical baseline benchmark row.
  - Based on the fixed 100-pair CelebA-HQ-256 benchmark.
- [ablations.csv](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/ablations.csv)
  - Normalized ablation sweep output imported from Kaggle.
  - Contains one row per inference-time ablation setting.
- [figures](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/figures)
  - Qualitative grids, exported table figures, and manifests for deterministic regeneration.
- [logs](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/logs)
  - Raw Kaggle exports and provenance notes.

## Metric notes

- `id_retrieval_top1` is the gallery-based Top-1 identity retrieval score computed over the fixed 100-source benchmark gallery.
- `expression_error` is the MediaPipe-based expression proxy used by the current repo pipeline rather than a paper-matched 3DMM expression coefficient metric.

## Source-of-truth Kaggle exports

Expected raw Kaggle files:

- `/kaggle/working/main_results_partial.csv`
- `/kaggle/working/benchmark_pairwise_results.csv`
- `/kaggle/working/id_retrieval_results.csv`
- `/kaggle/working/csim_results.csv`
- optional ablation summary CSV from the sweep

Use [scripts/prepare_results_bundle.py](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/scripts/prepare_results_bundle.py) to normalize those exports into this folder.

## Current ablation highlights

- Best `CSIM` tradeoff in the current sweep:
  - `run_index=15`
  - `denoise_strength=0.25`
  - `blend_alpha=0.65`
  - `region_attn_scale=1.2`
  - `csim=0.0315298047335818`
  - `id_retrieval_top1=0.01`
  - `expression_error=0.225728297829628`
- Best `Expression Error` in the current sweep:
  - `run_index=2`
  - `denoise_strength=0.2`
  - `blend_alpha=0.55`
  - `region_attn_scale=1.0`
  - `expression_error=0.1622680136561393`
  - with weaker identity metrics than the best-CSIM setting
## Copied raw Kaggle exports

- `main_results_partial.csv`
- `benchmark_pairwise_results.csv`
- `id_retrieval_results.csv`
- `csim_results.csv`
- `ablation_summary.csv`

