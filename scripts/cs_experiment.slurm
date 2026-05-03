#!/bin/bash
#SBATCH --job-name=cham-cs
#SBATCH --account=winter2026-comp579
#SBATCH --partition=all
#SBATCH --qos=comp579-1gpu-12h
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time=0-4:00:00
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

# ── Parameters (override via environment) ──
BACKEND=${BACKEND:-openai}
NUM_RUNS=${NUM_RUNS:-10}
CONFIG=${CONFIG:-chatarena/examples/chameleon_closed_3p.json}
EXPERIMENT_TAG=${EXPERIMENT_TAG:-$BACKEND}

# ── Diagnostics ──
echo "=== Job Info ==="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURM_NODELIST"
echo "Backend:  $BACKEND"
echo "Config:   $CONFIG"
echo "Num runs: $NUM_RUNS"
echo "Python:   $(which python) $(python --version 2>&1)"
date
echo "================"

# ── Run experiment ──
python run_experiment.py cs \
    --backend "$BACKEND" \
    --config "$CONFIG" \
    --num-runs "$NUM_RUNS" \
    --experiment-id "${EXPERIMENT_TAG}-${SLURM_JOB_ID}" \
    --save-transcript

echo "=== Done ==="
date
