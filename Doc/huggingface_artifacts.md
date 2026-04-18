# Hugging Face Training Artifacts

The large training artifacts for this project are stored in Hugging Face rather
than committed directly to GitHub.

Repository:

```text
https://huggingface.co/RaghavUjjwal/face-encoders
```

Current artifact folders:

```text
face_swap_sd15_mediapipe_kaggle_t4_5000/
face_swap_sd15_mediapipe_kaggle_t4_10000/
```

The Hugging Face repository is currently private, so downloads require a token
from an account with access.

## Why Artifacts Are Not Committed To GitHub

The checkpoint file `trainable_state.pt` is approximately 2 GB. This exceeds
normal GitHub file-size limits and would make the repository hard to clone and
maintain. GitHub should store source code, configs, docs, and lightweight
manifests. Hugging Face should store model checkpoints and generated artifacts.

## Download Artifacts

Install dependencies:

```bash
pip install huggingface_hub
```

Set a token for private repo access:

```bash
export HF_TOKEN=hf_xxx
```

PowerShell:

```powershell
$env:HF_TOKEN="hf_xxx"
```

Download the 10k-step artifact bundle:

```bash
python face_swap/scripts/download_hf_artifacts.py --run 10000
```

Download the 5k-step artifact bundle:

```bash
python face_swap/scripts/download_hf_artifacts.py --run 5000
```

Default local layout:

```text
artifacts/hf_face_encoders/
  face_swap_sd15_mediapipe_kaggle_t4_5000/
    checkpoint-5000/
      trainable_state.pt
    train_config_kaggle_t4_final.yaml
    inference_grid_checkpoint_5000.png
    inference_samples/
  face_swap_sd15_mediapipe_kaggle_t4_10000/
    checkpoint-10000/
      trainable_state.pt
    train_config_kaggle_t4_continued.yaml
    inference_grid_continued.png
    inference_samples_continued/
```

## Resume Training From 10k

After downloading the 10k artifact bundle:

```bash
CUDA_VISIBLE_DEVICES=0 python face_swap/train.py \
  --config face_swap/configs/train_config_kaggle_t4.yaml \
  --steps 15000 \
  --resume artifacts/hf_face_encoders/face_swap_sd15_mediapipe_kaggle_t4_10000/checkpoint-10000
```

`--steps 15000` means train until global step 15000. If resuming from
`checkpoint-10000`, this adds 5000 more steps.

## Resume Training On Kaggle

In Kaggle, add `HF_TOKEN` as a secret and run:

```python
from huggingface_hub import snapshot_download
from kaggle_secrets import UserSecretsClient

token = UserSecretsClient().get_secret("HF_TOKEN")

snapshot_download(
    repo_id="RaghavUjjwal/face-encoders",
    repo_type="model",
    token=token,
    allow_patterns=["face_swap_sd15_mediapipe_kaggle_t4_10000/**"],
    local_dir="/kaggle/working/hf_face_encoders",
    local_dir_use_symlinks=False,
)
```

Resume:

```bash
CUDA_VISIBLE_DEVICES=0 python face_swap/train.py \
  --config face_swap/configs/train_config_kaggle_t4.yaml \
  --steps 15000 \
  --resume /kaggle/working/hf_face_encoders/face_swap_sd15_mediapipe_kaggle_t4_10000/checkpoint-10000
```

Before resuming on a new Kaggle account, patch these config values to match the
mounted input paths:

```text
data.datasets[0].root
data.geometry_cache_dir
data.geometry_cache_key_root
model.region_encoder.pretrained_weights.eyes
model.region_encoder.pretrained_weights.nose
model.region_encoder.pretrained_weights.mouth
model.region_encoder.pretrained_weights.ears
```

If the geometry cache is not available in the new session, recompute it:

```bash
python face_swap/scripts/precompute_deca.py \
  --config face_swap/configs/train_config_kaggle_t4.yaml \
  --split train \
  --batch_size 8
```

## Artifact Manifest

| Run | Hugging Face folder | Checkpoint | Notes |
|---|---|---|---|
| 5000 steps | `face_swap_sd15_mediapipe_kaggle_t4_5000` | `checkpoint-5000/trainable_state.pt` | Initial Kaggle T4 run |
| 10000 steps | `face_swap_sd15_mediapipe_kaggle_t4_10000` | `checkpoint-10000/trainable_state.pt` | Continued run from 5k |

## Recommended GitHub Policy

Do commit:

- Source code
- Config files
- Documentation
- Lightweight manifests
- Download scripts

Do not commit:

- Multi-GB checkpoint files
- Large geometry caches
- Generated image archives

Store those large artifacts in Hugging Face and reference them from GitHub.
