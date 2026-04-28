#!/bin/bash
#SBATCH --job-name=cham-grpo
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

# ── Parameters (override via environment) ──
NUM_RUNS=${NUM_RUNS:-10}
EVAL_RUNS=${EVAL_RUNS:-0}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
SCORER_MODEL=${SCORER_MODEL:-BAAI/bge-reranker-v2-m3}
CLUE_NUMBER=${CLUE_NUMBER:-8}

# ── Model cache in shared memory ──
export HF_HOME=/dev/shm/${USER}_hf_${SLURM_JOB_ID}
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

echo "Downloading models to $HF_HOME (tmpfs/RAM) ..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL')
snapshot_download('$SCORER_MODEL')
print('Models ready.')
"

# ── Memory tuning ──
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Diagnostics ──
echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Model:  $MODEL"
echo "Node:   $SLURM_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
echo "Python: $(which python) $(python --version 2>&1)"
date
nvidia-smi
echo "================"

# ── Run experiment ──
python run_experiment.py grpo \
    --model "$MODEL" \
    --device 0 \
    --num-runs "$NUM_RUNS" \
    --eval-runs "$EVAL_RUNS" \
    --clue-number "$CLUE_NUMBER" \
    --experiment-id "grpo-${SLURM_JOB_ID}" \
    --save-transcript

# ── Cleanup ──
rm -rf "$HF_HOME"

echo "=== Done ==="
date
