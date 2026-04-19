# Benchmark and Ablation Methodology Notes

This note explains the benchmark and ablation procedure, the motivation behind the major evaluation choices, and the intended interpretation of the packaged results.

## What was done

An internal benchmark and ablation pipeline was built around the current face swapping model using:

- the downloaded `checkpoint-5000` model
- a fixed benchmark of `100` cross-identity source-target pairs
- the `CelebA-HQ-256` dataset
- a fixed baseline inference configuration for the main benchmark

The benchmark outputs were packaged into:

- [main_results.csv](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/main_results.csv)
- [ablations.csv](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/ablations.csv)
- [report_ready_results.md](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/report_ready_results.md)
- [benchmark_pairwise_merged.csv](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/logs/benchmark_pairwise_merged.csv)

## Why this benchmark protocol was used

The project needed an evaluation setup that was:

- reproducible
- feasible under limited compute
- stable across benchmark and ablation runs
- suitable for both tables and qualitative figure selection

For that reason, the evaluation was fixed to:

- one dataset protocol: `CelebA-HQ-256`
- one checkpoint: `checkpoint-5000`
- one benchmark set: `100` fixed source-target pairs
- one baseline inference setting:
  - `num_inference_steps=20`
  - `guidance_scale=1.0`
  - `denoise_strength=0.25`
  - `blend_alpha=0.65`
  - `region_attn_scale=1.0`

This makes the benchmark consistent inside the project, even though it is not a direct reproduction of the exact evaluation protocols used in some published FFHQ/FF++ face-swapping papers.

## Why these metrics were used

Three metrics were used because they cover different aspects of output quality:

- `CSIM`
  - measures identity similarity between generated output and source image using ArcFace cosine similarity
- `ID Retrieval Top-1`
  - measures identity discriminability using a gallery-based retrieval protocol over the fixed 100-source benchmark gallery
- `Expression Error`
  - measures target-side expression consistency using the repo's MediaPipe-based geometry representation

These were chosen to evaluate:

- identity preservation
- identity discriminability
- expression preservation

The current repo does not expose a paper-matched 3DMM expression representation, so the expression metric should be treated as a consistent project-side proxy rather than a strict cross-paper equivalent.

## Baseline benchmark interpretation

Baseline benchmark results:

- `CSIM = 0.024533`
- `ID Retrieval Top-1 = 0.0`
- `Expression Error = 0.222982`

Interpretation:

- expression preservation is measurable and non-trivial
- identity preservation is weak under the strict gallery-based evaluation
- the strict retrieval protocol is much harsher than simple one-to-one identity matching

These results are useful for internal comparison and ablation analysis, but they should be reported conservatively.

## Why the ablation study was designed this way

The ablation is an inference-time ablation, not a training ablation.

That means the following stayed fixed:

- checkpoint
- dataset
- fixed pair list
- guidance scale
- number of inference steps

Only the following inference controls were varied:

- `denoise_strength`
- `blend_alpha`
- `region_attn_scale`

This isolates how runtime generation settings affect identity and expression trade-offs without introducing training-side confounds.

## What the ablation showed

Two useful settings emerged:

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

Main takeaway:

- increasing `region_attn_scale` helped identity-oriented metrics slightly
- lower denoising and blending improved expression preservation more than identity preservation
- the current model shows a real inference-time trade-off between identity strength and expression fidelity

## Why the qualitative examples were selected

Qualitative examples were selected from the merged pair-level benchmark sheet rather than chosen arbitrarily. This was done so that the visual examples remain consistent with the quantitative analysis.

Selections were made for three figure types:

- benchmark triplets
- failure cases
- ablation showcase pair

The chosen indices and reasoning are recorded in:

- [selection_notes.md](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/figures/selection_notes.md)

## Why raw Kaggle CSVs were preserved

The raw Kaggle exports were copied into:

- [logs/raw_kaggle_exports](/C:/Users/DELL/Geometry-aware-Deepfake-generation-via-3D-disentanglement/05_results/logs/raw_kaggle_exports)

This was done to preserve provenance and allow later verification of:

- the summary tables
- the ablation table
- the merged per-pair analysis
- the figure-selection process

## What is still incomplete

The quantitative package is ready. The main remaining task is qualitative figure rendering.

The following are still pending:

- benchmark triplet image grid
- failure-case grid
- ablation comparison grid

Those manifests already exist, but they still need the actual saved generated-image paths from Kaggle.

## Recommended report framing

Recommended report framing:

- this is a controlled internal benchmark on a fixed 100-pair CelebA-HQ-256 subset
- the benchmark supports comparison across this project's own settings and ablations
- cross-paper comparison should be described as approximate, not directly equivalent
- efficiency should be reported as setup context unless a dedicated timing study is added
