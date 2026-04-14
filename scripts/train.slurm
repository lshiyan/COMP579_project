#!/bin/bash
#SBATCH --job-name=comp579-train
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00:00:00
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
snapshot_download('Qwen/Qwen2.5-1.5B-Instruct')
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

# ── Run experiment ──
python -c "
from chatarena.experiments.os_experiment import OpenSourceExperiment
exp = OpenSourceExperiment(
    'chatarena/examples/chameleon_closed_3p.json',
    model='Qwen/Qwen2.5-1.5B-Instruct',
    device=0,
    experiment_id='qwen-mimi-test',
)
exp.run()
"

# ── Cleanup shared memory ──
rm -rf "$HF_HOME"

echo "=== Done ==="
date
