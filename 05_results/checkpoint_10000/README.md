# Checkpoint-10000 Results Package

This folder contains the raw benchmark exports and derived pair-level analysis for the `checkpoint-10000` evaluation.

## Included files

- `main_results_partial_checkpoint_10000.csv`
- `benchmark_pairwise_results_checkpoint_10000.csv`
- `csim_results_checkpoint_10000.csv`
- `id_retrieval_results_checkpoint_10000.csv`
- `ablation_summary_checkpoint_10000.csv`
- `benchmark_pairwise_merged_checkpoint_10000.csv`

If qualitative outputs are exported, place them in:

```text
images/
```

Recommended naming:

- `pair_036_baseline.png`
- `pair_036_best_identity.png`
- `pair_036_best_expression.png`
- `pair_036_checkpoint_10000_baseline.png`

## Baseline benchmark summary

- `CSIM = 0.009938`
- `ID Retrieval Top-1 = 0.0`
- `Expression Error = 0.445716`

## Ablation highlights

- Best identity-oriented setting:
  - `denoise_strength = 0.2`
  - `blend_alpha = 0.55`
  - `region_attn_scale = 1.2`
  - `CSIM = 0.020833`
  - `ID Retrieval Top-1 = 0.0`
  - `Expression Error = 0.261640`
- Lowest `Expression Error` setting:
  - `denoise_strength = 0.2`
  - `blend_alpha = 0.55`
  - `region_attn_scale = 1.2`
  - `Expression Error = 0.261640`

## Notes

- Use the same fixed 100-pair benchmark protocol already used for `checkpoint-5000`.
- Keep the same baseline inference settings for the direct checkpoint comparison.
- Reuse the same selected pair indices for representative triplets and failure cases wherever possible.
- Use `benchmark_pairwise_merged_checkpoint_10000.csv` for pair-level analysis and visual selection.
