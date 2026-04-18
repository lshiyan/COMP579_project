#!/bin/bash
#SBATCH --job-name=cham-qwen7b
#SBATCH --account=winter2026-comp579
#SBATCH --partition=all
#SBATCH --qos=comp579-1gpu-12h
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=3
#SBATCH --time=0-12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e

# ── Modules ──
module load cuda/cuda-12.6

# ── Environment ──
source ~/envs/comp579/bin/activate

# ── Project ──
cd ~/COMP579_project

# ── Model cache in shared memory (fast, avoids home quota + disk limits) ──
export HF_HOME=/dev/shm/${USER}_hf_${SLURM_JOB_ID}
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

# ── Download models to RAM ──
echo "Downloading models to $HF_HOME (tmpfs/RAM) ..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct')
snapshot_download('sentence-transformers/all-MiniLM-L6-v2')
print('Models ready.')
"

# ── Diagnostics ──
echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
echo "Python: $(which python) $(python --version 2>&1)"
date
nvidia-smi
echo "================"

# ── Parameters (override via environment) ──
NUM_RUNS=${NUM_RUNS:-10}

# ── Run experiment ──
python run_experiment.py grpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --device 0 \
    --num-runs "$NUM_RUNS" \
    --experiment-id "qwen7b-${SLURM_JOB_ID}" \
    --save-transcript

# ── Cleanup shared memory ──
rm -rf "$HF_HOME"

echo "=== Done ==="
date
