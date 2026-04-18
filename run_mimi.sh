#!/bin/bash
#SBATCH --job-name=chameleon_arena
#SBATCH --account=clark-2026
#SBATCH -p all
#SBATCH --qos=gpu_1_mem_32_cpu_3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH -t 0-4:00
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

# Load modules
module load cuda/cuda-12.6

# Activate your virtual environment
source $HOME/envs/comp579/bin/activate

# Move to project directory
cd $SLURM_SUBMIT_DIR

# Run experiment — defaults to OS Qwen 7B via train.slurm
# Submit with: sbatch scripts/train.slurm
# Or for closed-source: BACKEND=claude sbatch scripts/cs_experiment.slurm
echo "Use 'sbatch scripts/train.slurm' or 'sbatch scripts/cs_experiment.slurm' instead."
echo "See scripts/ for available SLURM jobs."
