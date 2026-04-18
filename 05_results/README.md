# Benchmark and Ablation Handoff

This folder contains the benchmark and ablation artifacts prepared for the final report.

## What to use first

- [main_results.csv](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/main_results.csv)
  - baseline benchmark row for the report table
- [ablations.csv](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/ablations.csv)
  - full inference-time ablation sweep
- [report_ready_results.md](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/report_ready_results.md)
  - report-ready methodology and interpretation text
- [results_inventory.md](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/results_inventory.md)
  - provenance and artifact map

## Main benchmark summary

- Dataset: `CelebA-HQ-256`
- Evaluation protocol: fixed `100` source-target pairs
- Checkpoint: `checkpoint-5000`
- Baseline settings:
  - `num_inference_steps=20`
  - `guidance_scale=1.0`
  - `denoise_strength=0.25`
  - `blend_alpha=0.65`
  - `region_attn_scale=1.0`
- Baseline metrics:
  - `CSIM = 0.024533`
  - `ID Retrieval Top-1 = 0.0`
  - `Expression Error = 0.222982`

## Main ablation takeaways

- Best identity-oriented setting:
  - `denoise_strength=0.25`
  - `blend_alpha=0.65`
  - `region_attn_scale=1.2`
  - `CSIM = 0.03153`
  - `ID Retrieval Top-1 = 0.01`
  - `Expression Error = 0.22573`
- Best expression-oriented setting:
  - `denoise_strength=0.2`
  - `blend_alpha=0.55`
  - `region_attn_scale=1.0`
  - `Expression Error = 0.16227`

## Important interpretation notes

- This is an internal fixed-pair benchmark, not a direct apples-to-apples reproduction of FFHQ/FF++ paper protocols.
- `id_retrieval_top1` is gallery-based over the fixed 100-source benchmark gallery.
- `expression_error` is a MediaPipe-based proxy, not a paper-matched 3DMM expression coefficient metric.
- The current results show measurable expression preservation but weak identity preservation under the strict retrieval setup.

## Figures

Available now:

- [figures/main_results_table.png](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/figures/main_results_table.png)
- [figures/ablation_table.png](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/figures/ablation_table.png)
- [figures/selection_notes.md](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/figures/selection_notes.md)

Still pending:

- qualitative triplet/failure/ablation image grids
- those manifests are already prepared and only need real Kaggle output image paths

## Raw verification files

See [logs/raw_kaggle_exports](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/logs/raw_kaggle_exports) and the merged per-pair sheet:

- [logs/benchmark_pairwise_merged.csv](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/logs/benchmark_pairwise_merged.csv)
