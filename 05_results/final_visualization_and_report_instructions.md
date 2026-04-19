# Final Visualization and Report Instructions

This document is for the person responsible for using the benchmark and ablation data in the repository to prepare the final quantitative tables, qualitative figures, and report-ready results section in the format required by the course submission guide.

It is written against the actual submission guide requirements:

- `05_results/main_results.csv` should summarize the main quantitative results reported in the paper
- `05_results/ablations.csv` must include at least one ablation study
- `05_results/figures/` should contain the exact plots, qualitative outputs, and tables used in the report or presentation
- `05_results/logs/` may contain raw exports, evaluation logs, or notebook outputs used for verification
- the report must include:
  - experimental setup
  - results
  - ablations
  - failure cases
  - limitations

## 1. What was benchmarked

The evaluation uses a controlled internal benchmark:

- dataset: `CelebA-HQ-256`
- evaluation set: fixed `100` cross-identity source-target pairs
- pair file: `fixed_pairs_100.csv`
- metrics:
  - `CSIM`
  - `ID Retrieval Top-1`
  - `Expression Error`

The same fixed pair list is reused across:

- baseline benchmark runs
- ablation runs
- checkpoint-to-checkpoint comparison

This is important because it makes all comparisons inside the project fair and reproducible.

## 2. What the metrics mean

### `CSIM`

ArcFace cosine similarity between the generated image and the source identity image.

Use:

- identity preservation signal

Interpretation:

- higher is better

### `ID Retrieval Top-1`

Strict gallery-based identity retrieval over the fixed 100-source gallery.

Use:

- identity discriminability
- checks whether the generated output is recognized as the correct source among all benchmark identities

Interpretation:

- higher is better
- this is stricter than simple one-to-one similarity

Important note:

- the `id_retrieval_top1` column inside `benchmark_pairwise_results.csv` is from an earlier simplified single-source setup and should not be used for final reporting
- the correct final retrieval values come from `id_retrieval_results.csv` and the merged pairwise sheet

### `Expression Error`

Target-side expression consistency using the current MediaPipe-based geometry proxy.

Use:

- measures how closely the generated result preserves the target expression or geometry behavior

Interpretation:

- lower is better

Important note:

- this is a consistent project-side proxy, not a paper-matched 3DMM expression metric

## 3. Current benchmark summary

### Checkpoint-5000 baseline

Baseline settings:

- `num_inference_steps = 20`
- `guidance_scale = 1.0`
- `denoise_strength = 0.25`
- `blend_alpha = 0.65`
- `region_attn_scale = 1.0`

Results:

- `CSIM = 0.024533`
- `ID Retrieval Top-1 = 0.0`
- `Expression Error = 0.222982`

### Checkpoint-10000 baseline

Baseline settings:

- `num_inference_steps = 20`
- `guidance_scale = 1.0`
- `denoise_strength = 0.25`
- `blend_alpha = 0.65`
- `region_attn_scale = 1.0`

Results:

- `CSIM = 0.009938`
- `ID Retrieval Top-1 = 0.0`
- `Expression Error = 0.445716`

### Immediate comparison between checkpoints

Under the same baseline settings:

- `checkpoint-5000` performs better than `checkpoint-10000` on identity similarity
- `checkpoint-5000` also performs better than `checkpoint-10000` on expression preservation
- both checkpoints currently have `ID Retrieval Top-1 = 0.0` under strict gallery-based retrieval

This makes the direct checkpoint comparison straightforward and important for the report.

## 4. Current ablation summary

### Checkpoint-5000 ablation

Already available in the repository.

Best identity-oriented setting:

- `denoise_strength = 0.25`
- `blend_alpha = 0.65`
- `region_attn_scale = 1.2`
- `CSIM = 0.03153`
- `ID Retrieval Top-1 = 0.01`
- `Expression Error = 0.22573`

Best expression-oriented setting:

- `denoise_strength = 0.2`
- `blend_alpha = 0.55`
- `region_attn_scale = 1.0`
- `Expression Error = 0.16227`

### Checkpoint-10000 ablation

Best identity-oriented setting:

- `denoise_strength = 0.2`
- `blend_alpha = 0.55`
- `region_attn_scale = 1.2`
- `CSIM = 0.020833`
- `ID Retrieval Top-1 = 0.0`
- `Expression Error = 0.261640`

Best expression-oriented setting:

- `denoise_strength = 0.2`
- `blend_alpha = 0.55`
- `region_attn_scale = 1.2`
- `Expression Error = 0.261640`

## 5. Which files to use

### Main quantitative tables

Use:

- [main_results.csv](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/main_results.csv)
- [checkpoint_10000/main_results_partial_checkpoint_10000.csv](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/checkpoint_10000/main_results_partial_checkpoint_10000.csv)
- [checkpoint_comparison_template.csv](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/checkpoint_comparison_template.csv)

### Ablation tables

Use:

- [ablations.csv](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/ablations.csv)
- [checkpoint_10000/ablation_summary_checkpoint_10000.csv](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/checkpoint_10000/ablation_summary_checkpoint_10000.csv)

### Pair-level analysis and visual selection

Use:

- [logs/benchmark_pairwise_merged.csv](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/logs/benchmark_pairwise_merged.csv)
- [checkpoint_10000/benchmark_pairwise_merged_checkpoint_10000.csv](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/checkpoint_10000/benchmark_pairwise_merged_checkpoint_10000.csv)

These pair-level sheets should drive:

- representative example selection
- failure case selection
- checkpoint comparison visual selection

### Raw verification sources

Use the raw Kaggle export CSVs in:

- [logs/raw_kaggle_exports](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/logs/raw_kaggle_exports)

These are for provenance and verification, not for direct report tables.

## 6. Required report tables

The final report should contain at least these tables.

### Table A: Baseline checkpoint comparison

One row per checkpoint.

Recommended columns:

- checkpoint
- dataset
- num_pairs
- num_inference_steps
- guidance_scale
- denoise_strength
- blend_alpha
- region_attn_scale
- `CSIM`
- `ID Retrieval Top-1`
- `Expression Error`

This should be the main benchmark comparison table.

### Table B: Ablation summary for checkpoint-5000

Use the existing ablation output and report:

- either the full table if space allows
- or the top-performing and most informative settings

### Table C: Ablation summary for checkpoint-10000

Same as above once the ablation file is available.

### Table D: Compact best-setting comparison

One row for:

- `checkpoint-5000` baseline
- `checkpoint-5000` best identity-oriented ablation
- `checkpoint-5000` best expression-oriented ablation
- `checkpoint-10000` baseline
- `checkpoint-10000` best identity-oriented ablation
- `checkpoint-10000` best expression-oriented ablation

This table is useful if the report wants one compact summary across all key conditions.

## 7. Required figures and plots

The submission guide explicitly expects the exact plots, qualitative outputs, and tables used in the report or presentation to be stored under `05_results/figures/`.

The following figure set is recommended.

### Figure 1: Benchmark triplets grid

Format:

- `Source | Target | Output`

Content:

- `6` to `8` representative examples

Recommended strategy:

- one grid for `checkpoint-5000`
- one matching grid for `checkpoint-10000`

If the report has limited space, use the same pair indices and show a reduced side-by-side checkpoint comparison grid instead.

### Figure 2: Failure cases grid

Format:

- `Source | Target | Output`

Content:

- `2` to `4` failure cases for `checkpoint-5000`
- `2` to `4` failure cases for `checkpoint-10000`

If space is limited:

- one combined failure-case figure with rows labeled by checkpoint

### Figure 3: Ablation comparison grid

Format:

- one source-target pair
- multiple outputs under different ablation settings

Recommended settings shown:

- baseline
- best identity-oriented setting
- best expression-oriented setting

This should be made for at least `checkpoint-5000`.

If `checkpoint-10000` ablation results are available, the same type of grid should be made there too.

### Figure 4: Checkpoint comparison grid

Format:

- `Source | Target | checkpoint-5000 | checkpoint-10000`

This is one of the most important figures for the report because it directly connects the checkpoint comparison table to actual outputs.

### Figure 5: Ablation heatmaps or parameter-performance plots

Recommended quantitative visuals:

- heatmap of `CSIM` across `denoise_strength × blend_alpha`, with separate panels for `region_attn_scale`
- heatmap of `Expression Error` across the same settings

Alternative if heatmaps are too much work:

- grouped bar charts of the top 5 ablation settings
- one chart for `CSIM`
- one chart for `Expression Error`

### Figure 6: Metric comparison bar chart across checkpoints

Recommended:

- grouped bar chart comparing `checkpoint-5000` and `checkpoint-10000` baseline values for:
  - `CSIM`
  - `ID Retrieval Top-1`
  - `Expression Error`

Because the scales differ, either:

- use three separate small charts, or
- normalize values for presentation and clearly label that they are normalized

Three separate small charts are safer.

## 8. Recommended qualitative selection strategy

Do not select examples arbitrarily.

Use pair-level CSVs to choose:

- visually representative examples
- strongest expression-preservation examples
- clear failure cases
- checkpoint comparison examples

For `checkpoint-5000`, the current selected indices already recorded are:

- benchmark triplets: `36, 81, 69, 21, 12, 15, 65, 9`
- failure cases: `61, 5, 80, 95`
- ablation showcase pair: `36`

For `checkpoint-10000`, reuse the same pair indices wherever possible so visual comparisons are fair.

## 9. Suggested report subsection content

### Experimental setup

State:

- fixed 100-pair internal benchmark
- same pair list across all runs
- same baseline settings across both checkpoints
- metric definitions at a high level
- any compute or scope constraints

### Benchmark results

State:

- checkpoint comparison under fixed baseline settings
- `checkpoint-5000` is stronger than `checkpoint-10000` on the current baseline metrics
- strict retrieval remains weak for both checkpoints

### Ablation study

State:

- the ablation is inference-time only
- only `denoise_strength`, `blend_alpha`, and `region_attn_scale` change
- checkpoint, dataset, pair list, and baseline protocol remain fixed inside each sweep

### Failure cases and limitations

State:

- retrieval is computed using strict gallery-based Top-1
- this is an internal benchmark, not an official FFHQ/FF++ protocol
- expression metric is a MediaPipe-based proxy
- cross-paper numeric comparisons should be described conservatively

## 10. Safe claims

Safe:

- this is a controlled internal benchmark
- both checkpoints are compared under the same protocol
- the project includes at least one real ablation study
- later training did not automatically improve performance in the current baseline comparison

Avoid:

- claiming direct apples-to-apples superiority over published benchmarks
- claiming exact equivalence to paper-reported metrics
- making efficiency claims without a separate timing study

## 11. Immediate action items once checkpoint-10000 ablation finishes

1. Add the `checkpoint-10000` ablation CSV to the repo results structure.
2. Create a merged pairwise analysis sheet for `checkpoint-10000`, matching the structure used for `checkpoint-5000`.
3. Fill `checkpoint_comparison_template.csv` with the `checkpoint-10000` baseline values.
4. Export selected swapped output images for both checkpoints.
5. Render:
   - benchmark triplets
   - failure cases
   - ablation grid
   - checkpoint comparison grid
   - checkpoint metric comparison chart
6. Place the exact final tables and figures used by the report into `05_results/figures/`.
