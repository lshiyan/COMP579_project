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
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-16}
TEMPERATURE=${TEMPERATURE:-1.0}
REWARD_ALPHA=${REWARD_ALPHA:-0.5}
REWARD_GAMMA=${REWARD_GAMMA:-2.0}
REWARD_WORD_LEAK_THRESHOLD=${REWARD_WORD_LEAK_THRESHOLD:-0.15}
REWARD_MAX_TOKENS=${REWARD_MAX_TOKENS:-12}
REWARD_ZETA=${REWARD_ZETA:-0.1}
REWARD_LENGTH_CAP=${REWARD_LENGTH_CAP:-2.0}
POLICY_LR=${POLICY_LR:-2e-5}
GRPO_BETA=${GRPO_BETA:-0.3}
EXPERIMENT_TAG=${EXPERIMENT_TAG:-}

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
EXPERIMENT_ID="grpo-${SLURM_JOB_ID}"
if [ -n "$EXPERIMENT_TAG" ]; then
    EXPERIMENT_ID="grpo-${EXPERIMENT_TAG}-${SLURM_JOB_ID}"
fi

echo "=== Reward weights ==="
echo "alpha=$REWARD_ALPHA  gamma=$REWARD_GAMMA  thresh=$REWARD_WORD_LEAK_THRESHOLD  max_tokens=$REWARD_MAX_TOKENS  zeta=$REWARD_ZETA  length_cap=$REWARD_LENGTH_CAP"
echo "max_new_tokens=$MAX_NEW_TOKENS  temperature=$TEMPERATURE  policy_lr=$POLICY_LR  grpo_beta=$GRPO_BETA"
echo "experiment_id=$EXPERIMENT_ID"
echo "======================"

python run_experiment.py grpo \
    --model "$MODEL" \
    --device 0 \
    --num-runs "$NUM_RUNS" \
    --eval-runs "$EVAL_RUNS" \
    --clue-number "$CLUE_NUMBER" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --reward-alpha "$REWARD_ALPHA" \
    --reward-gamma "$REWARD_GAMMA" \
    --reward-word-leak-threshold "$REWARD_WORD_LEAK_THRESHOLD" \
    --reward-max-tokens "$REWARD_MAX_TOKENS" \
    --reward-zeta "$REWARD_ZETA" \
    --reward-length-cap "$REWARD_LENGTH_CAP" \
    --policy-lr "$POLICY_LR" \
    --grpo-beta "$GRPO_BETA" \
    --experiment-id "$EXPERIMENT_ID" \
    --save-transcript

# ── Cleanup ──
rm -rf "$HF_HOME"

echo "=== Done ==="
date
