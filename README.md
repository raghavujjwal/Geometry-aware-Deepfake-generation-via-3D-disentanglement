# Geometry-Aware Deepfake Generation via 3D-Guided Disentanglement

A geometry-aware face swapping system built on latent diffusion models with region-disentangled identity encoding and explicit 3D geometric conditioning.

> **Paper:** *Geometry-Aware Deepfake Generation via 3D-Guided Disentanglement* (Raghav Ujjwal, BITS Pilani, 2026)

---

## Overview

Given a **source image** (identity to transfer) and a **target image** (pose, expression, and background to preserve), this system synthesises a photorealistic output whose identity matches the source while all structural attributes match the target.

The framework addresses three core challenges in face swapping: pose-induced identity distortion, feature entanglement between identity and geometry, and generative quality under large viewpoint changes.

### Key Ideas

- **Region-disentangled identity encoding** — Instead of a single global face embedding, the source face is decomposed into four anatomical regions (eyes, nose, mouth, ears). Each region is encoded by a frozen ResNet-50 backbone followed by a trainable Transformer head and projection MLP, producing multiple spatial identity tokens per region. This prevents identity cues from being averaged away.

- **Explicit 3D geometry conditioning** — MediaPipe Face Mesh extracts 468 3D facial landmarks from the target face. These are interpolated into a dense depth map and surface normals, which feed a ControlNet branch that injects geometry signals at every U-Net decoder resolution.

- **Latent diffusion backbone** — Built on Stable Diffusion 1.5, the system operates in compressed VAE latent space for efficient, high-fidelity generation via DDIM denoising.

## Architecture

```
Source Image                          Target Image
     │                                     │
     ▼                                     ▼
┌─────────────┐                   ┌──────────────────┐
│ MediaPipe   │                   │ MediaPipe        │
│ Face Mesh   │                   │ Face Mesh        │
│ (468 lmks)  │                   │ (468 lmks)       │
└──────┬──────┘                   └────────┬─────────┘
       │                                   │
       ▼                                   ▼
┌──────────────────┐             ┌───────────────────┐
│ Region Cropping  │             │ Depth Map +       │
│ (eyes, nose,     │             │ Normal Map        │
│  mouth, ears)    │             │ (256×256)         │
└──────┬───────────┘             └────────┬──────────┘
       │                                  │
       ▼                                  ▼
┌──────────────────┐             ┌───────────────────┐
│ 4× ResNet-50     │             │ ControlNet        │
│ (frozen)         │             │ Branch            │
│    ↓             │             │ (zero-init)       │
│ 4× Transformer   │             └────────┬──────────┘
│ Head (trainable) │                      │
│    ↓             │                      │
│ 4× Projection   │                      │
│ MLP (trainable)  │                      │
└──────┬───────────┘                      │
       │                                  │
       ▼                                  ▼
  Q ∈ R^{16×512}              Geometry residuals
       │                           at all scales
       │                                  │
       └──────────┐    ┌──────────────────┘
                  ▼    ▼
          ┌──────────────────┐
          │  SD 1.5 U-Net    │
          │  (frozen)        │
          │  + IP-Adapter    │
          │  cross-attention │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │  VAE Decoder     │
          │  (frozen)        │
          └────────┬─────────┘
                   │
                   ▼
             Swapped Face
```

### Component Summary

| Component | Description |
|---|---|
| **Backbone** | Stable Diffusion 1.5 U-Net (frozen), VAE, single CLIP text encoder |
| **Region Encoders** | 4× frozen ResNet-50 + trainable Transformer heads + projection MLPs |
| **Identity Injection** | IP-Adapter style per-region cross-attention (16 tokens × 512-dim) |
| **Geometry** | MediaPipe Face Mesh → depth/normal maps → ControlNet branch |
| **Discriminator** | Multi-scale PatchGAN (3 scales) with spectral normalisation |
| **Losses** | ArcFace identity (1.0) · VGG-19 perceptual L1 (0.1) · Pixel L1 (1.0) · Hinge adversarial (0.01) |

## Project Structure

```
face_swap/
├── configs/
│   ├── train_config.yaml              # Default config (T4/Kaggle)
│   ├── train_config_a100.yaml         # A100 config
│   ├── train_config_h100.yaml         # H100 config (512px, bf16)
│   ├── train_config_kaggle_t4.yaml    # Kaggle T4 config
│   └── train_config_l40s.yaml         # L40S config
├── data/
│   ├── dataset.py                     # CelebA, FFHQ, VGGFace2, CelebA-HQ loaders
│   └── augmentations.py               # Paired face augmentation pipeline
├── models/
│   ├── backbone.py                    # SD 1.5 U-Net wrapper (frozen)
│   ├── region_encoder.py              # ResNet-50 + Transformer region encoders
│   ├── cross_attention.py             # IP-Adapter attention injection
│   ├── geometry.py                    # MediaPipe depth/normal extraction
│   ├── controlnet.py                  # Geometry ControlNet branch
│   └── discriminator.py              # Multi-scale PatchGAN
├── losses/
│   ├── identity_loss.py               # ArcFace cosine similarity
│   ├── perceptual_loss.py             # VGG-19 multi-layer L1
│   └── geometry_loss.py               # 3D landmark consistency
├── training/
│   ├── trainer.py                     # Main training loop (Accelerate)
│   └── scheduler.py                   # LR schedulers + param groups
├── inference/
│   ├── pipeline.py                    # DDIM inference pipeline
│   └── postprocess.py                 # Blending + colour correction
├── utils/
│   ├── face_crop.py                   # MediaPipe region detector/cropper
│   ├── metrics.py                     # FID, SSIM, PSNR, ArcFace-sim, 3D-LM
│   ├── visualize.py                   # Grids, depth maps, training curves
│   └── weight_loader.py              # Checkpoint loading utilities
├── scripts/
│   ├── precompute_deca.py             # Precompute geometry caches
│   └── inspect_and_load_weights.py    # Weight inspection tool
└── train.py                           # CLI entry point

pipeline/                              # Standalone preprocessing scripts
├── deca_extract.py                    # DECA feature extraction
├── face_parse.py                      # Face parsing
├── vae_encode.py                      # VAE latent pre-encoding
└── preprocess.py                      # Full preprocessing pipeline

decalib/                               # DECA library (for target design)
kaggle/                                # Kaggle-specific training notebooks
```

## Setup

### 1. Install Dependencies

```bash
# Install PyTorch first (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

Or install core packages individually:

```bash
pip install diffusers transformers accelerate
pip install mediapipe torchmetrics safetensors scipy
pip install matplotlib opencv-python-headless Pillow tqdm pyyaml
```

### 2. Configure Pretrained Model Paths

Edit `face_swap/configs/train_config.yaml`:

```yaml
model:
  sdxl_model_id: "runwayml/stable-diffusion-v1-5"    # SD 1.5 (auto-downloaded)
  arcface_model_path: "pretrained/arcface_r100.onnx"  # Download ArcFace R100
  deca_model_path: "data/deca_model.tar"              # DECA weights (optional)
```

### 3. Prepare Data

The default configuration uses **CelebA** (aligned faces):

```bash
# Download CelebA aligned images
# Place in data/celeba/img_align_celeba/

# Structure:
# data/celeba/
#   img_align_celeba/
#     000001.jpg
#     000002.jpg
#     ...
```

Other supported datasets (for full-scale training):
- **FFHQ** — 70k images from the [FFHQ repository](https://github.com/NVlabs/ffhq-dataset)
- **VGGFace2** — 3.3M images from [VGGFace2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)
- **CelebA-HQ** — 30k images from [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)

### 4. Precompute Geometry Caches (Optional)

Pre-extract MediaPipe depth/normal maps to speed up training:

```bash
python face_swap/scripts/precompute_deca.py \
    --image_dir data/celeba/img_align_celeba \
    --output_dir data/celeba/geometry_cache
```

## Training

### Single GPU

```bash
# Default config (T4/Kaggle, 256px, 10k steps)
python face_swap/train.py --config face_swap/configs/train_config.yaml

# H100 config (512px, bf16, larger batch)
python face_swap/train.py --config face_swap/configs/train_config_h100.yaml

# Debug mode (100 samples, 10 steps)
python face_swap/train.py --config face_swap/configs/train_config.yaml --debug
```

### Multi-GPU with Accelerate

```bash
accelerate config                     # Configure multi-GPU settings
accelerate launch --multi_gpu \
    face_swap/train.py --config face_swap/configs/train_config.yaml
```

### Training Configuration

| Parameter | Default | H100 |
|---|---|---|
| Resolution | 256×256 | 512×512 |
| Batch size | 4 (eff. 8) | 12 (eff. 36) |
| U-Net | Frozen | Frozen |
| Encoder LR | 1e-4 | 1e-4 |
| Training steps | 10,000 | 10,000 |
| Mixed precision | fp16 | bf16 |
| Noise schedule | scaled linear | scaled linear |
| Geometry loss | disabled | λ=0.5 |

**Loss weights:** Identity (ArcFace) `1.0` · Perceptual (VGG-19 L1) `0.1` · Pixel L1 `1.0` · Adversarial (hinge) `0.01`

**VGG-19 layers:** `relu1_1`, `relu2_1`, `relu3_1`, `relu4_1`, `relu5_1`

## Inference

```python
from face_swap.inference.pipeline import FaceSwapPipeline
from PIL import Image

pipeline = FaceSwapPipeline.from_pretrained("outputs/checkpoint-10000")
result = pipeline(
    source_image=Image.open("source.jpg"),
    target_image=Image.open("target.jpg"),
    num_inference_steps=50,
    guidance_scale=7.5,
)
result.image.save("swapped.jpg")
```

## Results

Evaluated on a 500-pair held-out CelebA test set with >30° yaw variation:

| Method | FID ↓ | SSIM ↑ | PSNR ↑ | ArcFace ↑ | 3D-LM ↓ |
|---|---|---|---|---|---|
| HifiFace | 28.4 | 0.741 | 28.1 | 0.762 | 4.31 |
| DiffSwap | 21.7 | 0.763 | 29.4 | 0.801 | 3.12 |
| **Ours (10k steps)** | **19.2** | **0.779** | **30.1** | **0.823** | **2.74** |

### Ablation Results

| Variant | ArcFace ↑ | 3D-LM ↓ | FID ↓ |
|---|---|---|---|
| Full model | 0.823 | 2.74 | 19.2 |
| Global identity (no region split) | 0.791 | 3.18 | 22.4 |
| No ControlNet (2D depth only) | 0.818 | 4.02 | 20.1 |
| No geometry loss | 0.819 | 3.61 | 19.9 |
| No adversarial loss | 0.809 | 2.81 | 28.7 |

## Technical Details

### Region Encoder Pipeline

Each of the 4 facial regions goes through:

1. **InputConvBlock** — Fuses 7-channel input (RGB + normal + depth) → 3 channels via zero-initialised residual convolution
2. **ResNet-50** — Frozen ImageNet-1K backbone, produces 2048×2×2 spatial features from 64×64 crops
3. **TransformerHead** — 2-layer trainable transformer encoder with learned positional embeddings, pre-norm, GELU activation
4. **ProjectionMLP** — Linear(2048→1024) → GELU → Dropout → Linear(1024→512), producing 4 tokens of 512-dim per region

Total identity context: 4 regions × 4 tokens = **16 tokens of 512 dimensions**.

### Geometry Pipeline

MediaPipe Face Mesh → 468 3D landmarks → sparse depth interpolation (OpenCV Telea inpainting) → Gaussian smoothing → 256×256 depth map → finite-difference surface normals → 6-channel ControlNet input (depth×3 + normal×3).

### What's Frozen vs. Trainable

| Component | Status |
|---|---|
| SD 1.5 U-Net | Frozen |
| VAE encoder/decoder | Frozen |
| CLIP text encoder | Frozen (null embeddings pre-computed, then freed from memory) |
| ResNet-50 backbones (×4) | Frozen |
| ArcFace R100 | Frozen (loss only) |
| Transformer heads (×4) | **Trainable** |
| Projection MLPs (×4) | **Trainable** |
| InputConvBlocks (×4) | **Trainable** |
| IP-Adapter cross-attention processors | **Trainable** |
| ControlNet branch | **Trainable** |
| PatchGAN discriminator | **Trainable** |

## Known Limitations

- **Compute scale** — Trained on 10k CelebA images at 256×256 for 10k steps; the full target is 3.4M images at 512×512 for 100k steps.
- **Geometry proxy** — Uses MediaPipe Face Mesh instead of the target DECA/FLAME 3DMM pipeline. Lacks parametric shape/expression factorisation.
- **SD 1.5 vs. SDXL** — Uses the smaller SD 1.5 backbone (~860M params) rather than SDXL (~2.6B params).
- **Profile views** — MediaPipe accuracy degrades beyond ±60° yaw, producing incomplete depth maps.
- **Occlusion** — Faces >30% occluded cause landmark detection failures (~8% of test pairs).
- **Geometry loss disabled** — The 3D landmark consistency loss is disabled in the default config to save compute.

## Key References

- [Stable Diffusion](https://arxiv.org/abs/2112.10752) — Latent diffusion backbone
- [IP-Adapter](https://arxiv.org/abs/2308.06721) — Cross-attention injection pattern
- [ControlNet](https://arxiv.org/abs/2302.05543) — Geometry conditioning pattern
- [DECA](https://arxiv.org/abs/2012.04012) — 3D face reconstruction (target design)
- [MediaPipe](https://arxiv.org/abs/1906.08172) — Face mesh landmark extraction (implemented)
- [DiffSwap](https://arxiv.org/abs/2305.18078) — Diffusion face swap baseline
- [HifiFace](https://arxiv.org/abs/2106.09965) — 3D shape + semantic prior baseline
- [ArcFace](https://arxiv.org/abs/1801.07698) — Identity embedding for loss computation

## License

This project is for **non-commercial scientific research purposes only**. See [LICENSE](LICENSE) for full terms. Please ensure compliance with the licenses of all pretrained models and datasets used.

## Ethical Statement

Deepfake technology carries significant potential for misuse. This work is released strictly for research purposes. We advocate for digital watermarking, provenance tracking, and regulatory frameworks governing synthetic face media. We encourage the deepfake detection community to use systems such as this one to strengthen detection robustness.
