# Benchmark and Ablation Results

This directory contains the quantitative results, figure assets, and supporting files used for the benchmark and ablation sections of the project report.

## Contents

- [main_results.csv](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/main_results.csv)
  - baseline benchmark summary table
- [ablations.csv](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/ablations.csv)
  - inference-time ablation sweep results
- [report_ready_results.md](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/report_ready_results.md)
  - report-ready methodology, result interpretation, and limitations text
- [methodology_notes.md](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/methodology_notes.md)
  - detailed description of the evaluation methodology and interpretation of the reported results
- [submission_instructions.md](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/submission_instructions.md)
  - instructions for using the packaged data in the final report, plus the pending `checkpoint-10000` comparison and qualitative deliverables
- [results_inventory.md](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/results_inventory.md)
  - provenance and artifact mapping
- [figures](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/figures)
  - exported tables and qualitative figure manifests
- [logs](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/logs)
  - raw Kaggle exports and merged pair-level analysis

## Benchmark Protocol

- Dataset: `CelebA-HQ-256`
- Evaluation set: fixed `100` cross-identity source-target pairs
- Checkpoints:
  - `checkpoint-5000`
  - `checkpoint-10000`
- Baseline inference settings:
  - `num_inference_steps=20`
  - `guidance_scale=1.0`
  - `denoise_strength=0.25`
  - `blend_alpha=0.65`
  - `region_attn_scale=1.0`

## Baseline Results

- `checkpoint-5000`
  - `CSIM = 0.024533`
  - `ID Retrieval Top-1 = 0.0`
  - `Expression Error = 0.222982`
- `checkpoint-10000`
  - `CSIM = 0.009938`
  - `ID Retrieval Top-1 = 0.0`
  - `Expression Error = 0.445716`

These values correspond to the fixed 100-pair CelebA-HQ-256 benchmark reported in [main_results.csv](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/main_results.csv) for `checkpoint-5000` and in the `checkpoint_10000` folder for `checkpoint-10000`.

## Ablation Highlights

- `checkpoint-5000` best identity-oriented setting:
  - `denoise_strength=0.25`
  - `blend_alpha=0.65`
  - `region_attn_scale=1.2`
  - `CSIM = 0.03153`
  - `ID Retrieval Top-1 = 0.01`
  - `Expression Error = 0.22573`
- `checkpoint-5000` best expression-oriented setting:
  - `denoise_strength=0.2`
  - `blend_alpha=0.55`
  - `region_attn_scale=1.0`
  - `Expression Error = 0.16227`
- `checkpoint-10000` best identity-oriented setting:
  - `denoise_strength=0.2`
  - `blend_alpha=0.55`
  - `region_attn_scale=1.2`
  - `CSIM = 0.020833`
  - `ID Retrieval Top-1 = 0.0`
  - `Expression Error = 0.261640`
- `checkpoint-10000` best expression-oriented setting:
  - `denoise_strength=0.2`
  - `blend_alpha=0.55`
  - `region_attn_scale=1.2`
  - `Expression Error = 0.261640`

The current checkpoint comparison and ablation results suggest that `checkpoint-5000` is the stronger checkpoint under this internal benchmark protocol.

## Metric Notes

- `id_retrieval_top1` is gallery-based over the fixed 100-source benchmark gallery.
- `expression_error` is computed using the current repo's MediaPipe-based geometry proxy rather than paper-matched 3DMM expression coefficients.

## Figure Assets

Currently available:

- [figures/main_results_table.png](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/figures/main_results_table.png)
- [figures/ablation_table.png](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/figures/ablation_table.png)
- [figures/selection_notes.md](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/figures/selection_notes.md)

The benchmark triplet, failure-case, and ablation grid manifests are already prepared. They only require the final generated image paths to render the full qualitative figures.

## Supporting Data

- [logs/benchmark_pairwise_merged.csv](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/logs/benchmark_pairwise_merged.csv)
  - merged pair-level sheet for benchmark analysis and figure selection
- [checkpoint_10000](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/checkpoint_10000)
  - raw benchmark exports and merged pair-level analysis for `checkpoint-10000`
- [logs/raw_kaggle_exports](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/logs/raw_kaggle_exports)
  - preserved raw CSV exports from Kaggle
