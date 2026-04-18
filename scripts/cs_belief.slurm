#!/bin/bash
#SBATCH --job-name=cham-csb
#SBATCH --account=winter2026-comp579
#SBATCH --partition=all
#SBATCH --qos=comp579-1gpu-12h
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=0-6:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e

# ── Environment ──
source ~/envs/comp579/bin/activate

# ── Project ──
cd ~/COMP579_project

# ── API keys ──
if [[ -f ~/COMP579_project/.env ]]; then
    set -a
    source ~/COMP579_project/.env
    set +a
fi

# ── Cache sentence encoder model ──
python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print('Sentence encoder ready.')
"

# ── Parameters (override via environment) ──
BACKEND=${BACKEND:-openai}
NUM_RUNS=${NUM_RUNS:-100}

# ── Diagnostics ──
echo "=== Job Info ==="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURM_NODELIST"
echo "Backend:  $BACKEND"
echo "Num runs: $NUM_RUNS"
echo "Python:   $(which python) $(python --version 2>&1)"
date
echo "================"

# ── Run experiment ──
python run_experiment.py cs-belief \
    --backend "$BACKEND" \
    --num-runs "$NUM_RUNS" \
    --experiment-id "${BACKEND}-belief-${SLURM_JOB_ID}" \
    --save-transcript

echo "=== Done ==="
date
