# Geometry-Aware Deepfake Generation via 3D-Guided Disentanglement

A complete research implementation of a geometry-aware face swapping system using latent diffusion models.

## Architecture

| Component | Description |
|-----------|-------------|
| **Backbone** | Stable Diffusion XL U-Net operating in VAE latent space |
| **Region Encoders** | Shared ViT-B/16 + 6 independent projection MLPs (eyes, nose, lips, skin, hairline, ears) |
| **Cross-Attention Injection** | IP-Adapter style per-region identity token injection |
| **Geometry Conditioning** | DECA → 3DMM params → rendered depth map → ControlNet branch |
| **Text Control** | CLIP dual-encoder for optional semantic/pose prompts |
| **Discriminator** | Multi-scale PatchGAN with spectral normalisation |

## Project Structure

```
face_swap/
├── configs/train_config.yaml     ← All hyperparameters
├── data/
│   ├── dataset.py                ← FFHQ, VGGFace2, CelebA-HQ loaders
│   └── augmentations.py          ← Paired face augmentation pipeline
├── models/
│   ├── backbone.py               ← SDXL U-Net wrapper
│   ├── region_encoder.py         ← ViT region feature encoders
│   ├── cross_attention.py        ← IP-Adapter attention injection
│   ├── geometry.py               ← DECA 3DMM encoder + depth renderer
│   ├── controlnet.py             ← Geometry ControlNet branch
│   └── discriminator.py          ← Multi-scale PatchGAN
├── losses/
│   ├── identity_loss.py          ← ArcFace cosine similarity loss
│   ├── perceptual_loss.py        ← VGG-19 multi-layer perceptual loss
│   └── geometry_loss.py          ← 3D landmark consistency + pixel L1
├── training/
│   ├── trainer.py                ← Main training loop (Accelerate)
│   └── scheduler.py              ← LR schedulers + param groups
├── inference/
│   ├── pipeline.py               ← DDIM inference pipeline
│   └── postprocess.py            ← Blending + color correction
├── utils/
│   ├── face_crop.py              ← MediaPipe region detector/cropper
│   ├── metrics.py                ← FID, SSIM, PSNR, ArcFace, 3D LM error
│   └── visualize.py              ← Grids, depth maps, training curves
└── train.py                      ← CLI entry point
```

## Setup

### 1. Install Dependencies

```bash
pip install torch torchvision diffusers transformers accelerate
pip install timm mediapipe torchmetrics safetensors scipy
pip install matplotlib opencv-python Pillow tqdm pyyaml insightface
```

### 2. Install DECA

```bash
git clone https://github.com/YadiraF/DECA.git
cd DECA && pip install -r requirements.txt
# Add DECA to your PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/DECA
```

### 3. Configure Pretrained Paths

Edit `configs/train_config.yaml` and update all `TODO` paths:

```yaml
model:
  sdxl_model_id: "stabilityai/stable-diffusion-xl-base-1.0"  
  arcface_model_path: "pretrained/arcface_r100.pth"           # ArcFace R100
  deca_model_path:    "pretrained/deca_model.tar"             # DECA weights
data:
  datasets:
    - name: FFHQ
      root: "/path/to/ffhq"
```

### 4. Prepare Data

Supported datasets:
- **FFHQ** — 70k images, download from [FFHQ repository](https://github.com/NVlabs/ffhq-dataset)
- **VGGFace2** — 3.3M images, download from [VGGFace2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)
- **CelebA-HQ** — 30k images, download from [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)

## Training

```bash
# Single GPU (A100 80GB recommended)
python face_swap/train.py --config face_swap/configs/train_config.yaml

# Multi-GPU with Accelerate
accelerate config  # configure multi-GPU settings first
accelerate launch --multi_gpu face_swap/train.py --config face_swap/configs/train_config.yaml

# Debug mode (100 samples, 10 steps)
python face_swap/train.py --config face_swap/configs/train_config.yaml --debug
```

## Inference

```python
from face_swap.inference.pipeline import FaceSwapPipeline
from PIL import Image

pipeline = FaceSwapPipeline.from_pretrained("outputs/checkpoint-100000")
result = pipeline(
    source_image=Image.open("source.jpg"),
    target_image=Image.open("target.jpg"),
    num_inference_steps=50,
    guidance_scale=7.5,
)
result.image.save("swapped.jpg")
```

## Training Config Highlights

| Parameter | Value |
|-----------|-------|
| Resolution | 512 × 512 |
| Batch size | 8 (grad accum × 4 = eff. 32) |
| U-Net LR | 1e-5 |
| Encoder LR | 1e-4 |
| Total steps | 100,000 |
| Mixed precision | fp16 |
| Hardware | A100 80 GB |

**Loss weights:** Identity (ArcFace) `1.0` · Perceptual (VGG) `0.1` · Pixel L1 `1.0` · Adversarial `0.01` · Geometry `0.5`

## Key References

- [IP-Adapter](https://arxiv.org/abs/2308.06721) — cross-attention injection pattern
- [ControlNet](https://arxiv.org/abs/2302.05543) — geometry conditioning pattern
- [DECA](https://arxiv.org/abs/2012.04012) — 3D face reconstruction
- [DiffSwap](https://arxiv.org/abs/2305.18078) — diffusion face swap pipeline
- [HifiFace](https://arxiv.org/abs/2106.09965) — 3D shape + semantic prior guidance

## License

This project is for research purposes only. Please ensure compliance with the licenses of all pretrained models and datasets used.
