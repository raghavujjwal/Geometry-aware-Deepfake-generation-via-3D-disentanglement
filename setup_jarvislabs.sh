#!/bin/bash
# ============================================================
# Jarvis Labs H100 Setup Script
# Geometry-Aware Deepfake Generation via 3D Disentanglement
# ============================================================
# Usage:
#   chmod +x setup_jarvislabs.sh
#   bash setup_jarvislabs.sh
#
# Run this ONCE after launching your H100 instance.
# Everything is installed inside /home/ so it persists
# across pause/resume cycles on Jarvis Labs.
# ============================================================

set -e  # Exit on any error

echo "============================================================"
echo " Jarvis Labs H100 Setup — Face Swap Training Pipeline"
echo "============================================================"

# ── 0. System checks ────────────────────────────────────────────
echo ""
echo "[0/9] Checking system..."
nvidia-smi || { echo "ERROR: nvidia-smi failed. Is GPU attached?"; exit 1; }
echo ""

CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
echo "Detected CUDA Version: ${CUDA_VERSION}"

# ── 1. Create persistent project directory ──────────────────────
echo ""
echo "[1/9] Setting up project directory..."
PROJECT_DIR="/home/face_swap_3d"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# ── 2. Clone / copy your repository ────────────────────────────
echo ""
echo "[2/9] Setting up repository..."
if [ ! -d "$PROJECT_DIR/DECA" ]; then
    echo "Please upload your code. Options:"
    echo "  A) git clone <your-repo-url> $PROJECT_DIR/DECA"
    echo "  B) scp -r -P <port> ./DECA/ <user>@<host>:$PROJECT_DIR/DECA"
    echo ""
    echo "For now, creating empty directory structure..."
    mkdir -p "$PROJECT_DIR/DECA"
fi

# If you've already uploaded, skip this section.
# The script below assumes DECA/ is available at $PROJECT_DIR/DECA.

# ── 3. Create persistent virtual environment ───────────────────
echo ""
echo "[3/9] Creating Python virtual environment (persists in /home)..."
VENV_DIR="/home/face_swap_venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created at $VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
echo "Activated venv: $(which python3)"
python3 --version

# ── 4. Install PyTorch with CUDA support ────────────────────────
echo ""
echo "[4/9] Installing PyTorch with CUDA support..."
# H100 requires CUDA 12.x and sm_90 support
# Using PyTorch nightly / stable with CUDA 12.4 (latest for H100)
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.4 (H100 compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify CUDA
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    echo 'WARNING: CUDA not available!'
    exit 1
"

# ── 5. Install core dependencies ────────────────────────────────
echo ""
echo "[5/9] Installing core Python dependencies..."

pip install \
    diffusers>=0.27.0 \
    transformers>=4.40.0 \
    accelerate>=0.30.0 \
    safetensors \
    huggingface-hub \
    xformers

pip install \
    kornia \
    scipy \
    numpy \
    Pillow \
    scikit-image \
    imageio \
    opencv-python-headless

pip install \
    PyYAML \
    yacs \
    tqdm \
    psutil \
    tensorboard \
    matplotlib

# chumpy (DECA dependency, needs numpy compat patch)
pip install chumpy --no-build-isolation --no-deps 2>/dev/null || true

# MediaPipe (for face region detection)
pip install mediapipe 2>/dev/null || true

# InsightFace (for ArcFace identity loss)
pip install insightface onnxruntime-gpu 2>/dev/null || true

# ── 6. Patch chumpy for modern numpy ────────────────────────────
echo ""
echo "[6/9] Patching chumpy for numpy compatibility..."
CHUMPY_INIT=$(python3 -c "import chumpy; print(chumpy.__file__)" 2>/dev/null) || true
if [ -n "$CHUMPY_INIT" ] && [ -f "$CHUMPY_INIT" ]; then
    sed -i 's/from numpy import bool, int, float, complex, object, unicode, str, nan, inf/from numpy import nan, inf/' "$CHUMPY_INIT"
    echo "chumpy patched successfully."
else
    echo "chumpy not found or already patched."
fi

# ── 7. Build DECA CUDA rasterizer ──────────────────────────────
echo ""
echo "[7/9] Building DECA standard rasterizer CUDA extension..."
RASTERIZER_DIR="$PROJECT_DIR/DECA/decalib/utils/rasterizer"
if [ -d "$RASTERIZER_DIR" ]; then
    cd "$RASTERIZER_DIR"
    
    # Fix the hardcoded gcc-7 — use system default gcc
    sed -i 's/os.environ\["CC"\] = "gcc-7"/# os.environ["CC"] = "gcc-7"/' setup.py
    sed -i 's/os.environ\["CXX"\] = "gcc-7"/# os.environ["CXX"] = "gcc-7"/' setup.py
    
    pip install ninja
    python3 setup.py build_ext --inplace
    echo "Standard rasterizer built successfully."
    cd "$PROJECT_DIR"
else
    echo "WARNING: Rasterizer directory not found. Will use pytorch3d fallback."
fi

# ── 8. Download CelebA dataset (if not already present) ─────────
echo ""
echo "[8/9] Setting up dataset..."
DATA_DIR="$PROJECT_DIR/DECA/data/celeba"
mkdir -p "$DATA_DIR"

echo ""
echo "=================================================="
echo " DATASET SETUP INSTRUCTIONS"
echo "=================================================="
echo ""
echo "You need CelebA aligned dataset images."
echo "Options to get the data:"
echo ""
echo "  Option A — Kaggle CLI (recommended):"
echo "    pip install kaggle"
echo "    kaggle datasets download -d jessicali9530/celeba-dataset"
echo '    unzip celeba-dataset.zip -d /home/face_swap_3d/DECA/data/celeba/'
echo ""
echo "  Option B — Upload via SCP:"
echo '    scp -r -P <port> ./img_align_celeba/ <user>@<host>:/home/face_swap_3d/DECA/data/celeba/'
echo ""
echo "  Option C — gdown from Google Drive:"
echo "    pip install gdown"
echo "    gdown <drive-id> -O celeba.zip"
echo '    unzip celeba.zip -d /home/face_swap_3d/DECA/data/celeba/'
echo ""
echo "Expected structure:"
echo '  /home/face_swap_3d/DECA/data/celeba/img_align_celeba/'
echo "    000001.jpg"
echo "    000002.jpg"
echo "    ..."
echo "=================================================="

# ── 9. Pre-download HuggingFace models ──────────────────────────
echo ""
echo "[9/9] Pre-downloading HuggingFace models (this takes a while)..."
export HF_HOME="/home/hf_cache"
mkdir -p "$HF_HOME"

python3 -c "
from huggingface_hub import snapshot_download
import os

os.environ['HF_HOME'] = '/home/hf_cache'

# Download SDXL base 1.0 (for U-Net, VAE, text encoders)
print('Downloading SDXL base 1.0...')
snapshot_download('stabilityai/stable-diffusion-xl-base-1.0', cache_dir='/home/hf_cache')

# Download SDXL VAE fp16 fix
print('Downloading SDXL VAE fp16 fix...')
snapshot_download('madebyollin/sdxl-vae-fp16-fix', cache_dir='/home/hf_cache')

# Download CLIP ViT-L/14 
print('Downloading CLIP ViT-L/14...')
snapshot_download('openai/clip-vit-large-patch14', cache_dir='/home/hf_cache')

print('All HuggingFace models downloaded successfully.')
" || echo "WARNING: Some HuggingFace models failed to download. They will be cached on first training run."

echo ""
echo "============================================================"
echo " SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "Activate the environment with:"
echo "  source /home/face_swap_venv/bin/activate"
echo ""
echo "Set environment variables with:"
echo "  export PYTHONPATH=/home/face_swap_3d/DECA:/home/face_swap_3d/DECA/face_swap:\$PYTHONPATH"
echo "  export HF_HOME=/home/hf_cache"
echo "  export CUDA_VISIBLE_DEVICES=0"
echo ""
echo "Run training with:"
echo "  cd /home/face_swap_3d/DECA"
echo "  python face_swap/train.py --config face_swap/configs/train_config.yaml"
echo ""
echo "Or run in debug mode first:"
echo "  python face_swap/train.py --config face_swap/configs/train_config.yaml --debug"
echo ""
echo "============================================================"
