# Report-Ready Results Section

## Experimental Setup

We evaluate the reproduced/adapted face swapping pipeline on a fixed benchmark of 100 cross-identity source-target pairs sampled from CelebA-HQ-256. The same baseline inference configuration is used for both checkpoints: `num_inference_steps=20`, `guidance_scale=1.0`, `denoise_strength=0.25`, `blend_alpha=0.65`, and `region_attn_scale=1.0`. Identity preservation is measured using ArcFace cosine similarity (CSIM) and gallery-based ID Retrieval Top-1, while expression consistency is measured using a MediaPipe-based expression proxy (`expression_error`).

Because of compute, memory, and dataset-access constraints, we use an internal fixed 100-pair protocol rather than reproducing the larger FFHQ/FF++ evaluation settings used in some prior papers. This should be interpreted as a controlled internal benchmark rather than a fully apples-to-apples reproduction of published evaluation protocols.

## Benchmark Results

Under the fixed 100-pair CelebA-HQ-256 protocol, the baseline checkpoint comparison produced:

- `checkpoint-5000`
  - `CSIM = 0.024533`
  - `ID Retrieval Top-1 = 0.0`
  - `Expression Error = 0.222982`
- `checkpoint-10000`
  - `CSIM = 0.009938`
  - `ID Retrieval Top-1 = 0.0`
  - `Expression Error = 0.445716`

These results indicate that the current setup produces a measurable expression-preservation signal, but identity preservation remains weak under the stricter gallery-based evaluation. In particular, the retrieval score should be interpreted carefully: it is computed against a 100-image source gallery, making it substantially stricter than single-source matching. Under this fixed protocol, `checkpoint-5000` performs better than `checkpoint-10000` on both identity similarity and expression preservation, while both checkpoints remain at `Top-1 = 0.0` under strict gallery retrieval.

## Ablation Study

The ablation study is designed as an inference-time ablation, not a training ablation. The checkpoint, dataset, and fixed pair list remain unchanged across all runs. Only three inference controls are varied:

- `denoise_strength`
- `blend_alpha`
- `region_attn_scale`

This isolates the effect of runtime generation choices on identity preservation and expression consistency.

For `checkpoint-5000`, the strongest identity-oriented configuration in the sweep was `denoise_strength=0.25`, `blend_alpha=0.65`, and `region_attn_scale=1.2`, which achieved `CSIM = 0.03153`, `ID Retrieval Top-1 = 0.01`, and `Expression Error = 0.22573`. In contrast, the best expression-oriented setting was `denoise_strength=0.2`, `blend_alpha=0.55`, and `region_attn_scale=1.0`, which achieved the lowest `Expression Error = 0.16227` but with weaker identity preservation.

For `checkpoint-10000`, the strongest visible ablation setting was `denoise_strength=0.2`, `blend_alpha=0.55`, and `region_attn_scale=1.2`, which achieved `CSIM = 0.020833`, `ID Retrieval Top-1 = 0.0`, and `Expression Error = 0.261640`. Even after ablation, this checkpoint does not surpass the stronger `checkpoint-5000` results under the current benchmark protocol.

Taken together, these ablations suggest a clear inference-time trade-off between identity strength and expression fidelity, while also indicating that later training did not automatically improve performance on this evaluation setup.

## Failure Cases and Limitations

The current benchmark reveals several limitations. First, strict gallery-based identity retrieval produced `Top-1 = 0.0` for both checkpoints, suggesting weak identity discriminability under this evaluation protocol. Second, the CelebA-HQ-256 benchmark differs from the FFHQ/FF++ style protocols used in competing papers, so cross-paper numeric comparisons should be presented only as approximate references. Third, the expression metric is based on the current repo's MediaPipe geometry proxy rather than a paper-matched 3DMM expression coefficient metric.

Efficiency is reported only as setup context in this project. We document hardware/runtime settings, image resolution, and inference steps, but we do not claim comparative efficiency unless a dedicated timing study is added.
