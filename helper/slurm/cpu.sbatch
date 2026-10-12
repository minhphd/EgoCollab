#!/bin/bash
# Runs one Python command on CPUs with the project env, e.g. from the repo root:
#   sbatch -M kingspeak --array=0-11 helper/slurm/cpu.sbatch python -m helper.perception.frames --shard ARRAY/12
# ARRAY in the arguments becomes the array task id.
#SBATCH --job-name=cpu
#SBATCH --account=kmarino
#SBATCH --partition=kingspeak
#SBATCH --qos=kingspeak
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH=$HOME/.conda/envs/egocollab/bin:$PATH
"${@//ARRAY/${SLURM_ARRAY_TASK_ID:-0}}"
