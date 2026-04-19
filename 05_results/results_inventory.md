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
- [checkpoint_10000](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/checkpoint_10000)
  - Packaged raw benchmark outputs, ablation outputs, and merged pair-level analysis for `checkpoint-10000`.
- [checkpoint_comparison_template.csv](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/checkpoint_comparison_template.csv)
  - Two-row comparison template for presenting `checkpoint-5000` and `checkpoint-10000` under the same baseline settings.

## Metric notes

- `id_retrieval_top1` is the gallery-based Top-1 identity retrieval score computed over the fixed 100-source benchmark gallery.
- `expression_error` is the MediaPipe-based expression proxy used by the current repo pipeline rather than a paper-matched 3DMM expression coefficient metric.

## Source-of-truth Kaggle exports

Expected raw Kaggle files:

- `/kaggle/working/main_results_partial.csv`
- `/kaggle/working/benchmark_pairwise_results.csv`
- `/kaggle/working/id_retrieval_results.csv`
- `/kaggle/working/csim_results.csv`
- `/kaggle/working/fixed_pairs_100.csv`
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
- `checkpoint-10000` best `CSIM` tradeoff:
  - `run_index=3`
  - `denoise_strength=0.2`
  - `blend_alpha=0.55`
  - `region_attn_scale=1.2`
  - `csim=0.020833`
  - `id_retrieval_top1=0.0`
  - `expression_error=0.261640`
- `checkpoint-10000` lowest `Expression Error`:
  - `run_index=3`
  - `denoise_strength=0.2`
  - `blend_alpha=0.55`
  - `region_attn_scale=1.2`
  - `expression_error=0.261640`
  - also the strongest `CSIM` row in the `checkpoint-10000` sweep
## Copied raw Kaggle exports

- `main_results_partial.csv`
- `benchmark_pairwise_results.csv`
- `id_retrieval_results.csv`
- `csim_results.csv`
- `fixed_pairs_100.csv`
- `ablation_summary.csv`

## Checkpoint-10000 package contents

- `main_results_partial_checkpoint_10000.csv`
- `benchmark_pairwise_results_checkpoint_10000.csv`
- `csim_results_checkpoint_10000.csv`
- `id_retrieval_results_checkpoint_10000.csv`
- `ablation_summary_checkpoint_10000.csv`
- `benchmark_pairwise_merged_checkpoint_10000.csv`

