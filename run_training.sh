#!/bin/bash
# ============================================================
# run_training.sh  — Launch face swap training on Jarvis Labs H100
# ============================================================
# Usage:
#   bash run_training.sh            # Full training
#   bash run_training.sh --debug    # Debug mode (10 steps, tiny data)
# ============================================================

set -e

# ── Activate environment ─────────────────────────────────────
source /home/face_swap_venv/bin/activate

# ── Set environment variables ────────────────────────────────
export PYTHONPATH="/home/face_swap_3d/DECA:/home/face_swap_3d/DECA/face_swap:$PYTHONPATH"
export HF_HOME="/home/hf_cache"
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="9.0"    # H100 SM90
export TOKENIZERS_PARALLELISM=false

# ── Verify GPU ───────────────────────────────────────────────
echo "=== GPU Status ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print(f'bf16 support: {torch.cuda.is_bf16_supported()}')
"
echo ""

# ── Navigate to project ─────────────────────────────────────
cd /home/face_swap_3d/DECA

# ── Determine config file ───────────────────────────────────
CONFIG="face_swap/configs/train_config_h100.yaml"

# ── Check if debug mode ─────────────────────────────────────
EXTRA_ARGS=""
if [ "$1" == "--debug" ]; then
    echo ">>> Running in DEBUG mode (10 steps, 100 samples)"
    EXTRA_ARGS="--debug"
fi

# ── Launch training ─────────────────────────────────────────
echo "=== Starting Training ==="
echo "Config: $CONFIG"
echo "Extra args: $EXTRA_ARGS"
echo ""

python3 face_swap/train.py \
    --config "$CONFIG" \
    $EXTRA_ARGS \
    2>&1 | tee /home/face_swap_3d/training_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=== Training Complete ==="
echo "Logs saved to: /home/face_swap_3d/training_*.log"
echo "Checkpoints at: /home/face_swap_3d/outputs/"
