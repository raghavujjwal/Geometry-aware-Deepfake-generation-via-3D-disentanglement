# Figure Selection Notes

These selections were chosen from [benchmark_pairwise_merged.csv](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/logs/benchmark_pairwise_merged.csv) to keep figure choice consistent with the benchmark metrics.

## Benchmark Triplets

Chosen pair indices:

- `36`: strongest balanced case (`CSIM` high, `expression_error` low)
- `81`: best expression-retention case with strong identity similarity
- `69`: high-identity showcase
- `21`: representative typical success case
- `12`: representative typical success case
- `15`: stable mid-range case
- `65`: strong expression-fidelity case
- `9`: moderate benchmark case

## Failure Cases

Chosen pair indices:

- `61`: worst expression drift in the benchmark
- `5`: very high expression inconsistency
- `80`: worst identity similarity (`CSIM`)
- `95`: identity mismatch under strict gallery retrieval

## Ablation Showcase Pair

Chosen pair index:

- `36`

Reason:

- It is the strongest balanced benchmark case, so changes across ablation settings should be easier to interpret visually.

## Required next input from Kaggle

The manifests currently use placeholder output paths:

- `<fill_from_kaggle>/pair_XXX_*.png`

Replace those with the actual saved generated-image paths from Kaggle before running [render_results_figures.py](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/scripts/render_results_figures.py) for qualitative grids.
