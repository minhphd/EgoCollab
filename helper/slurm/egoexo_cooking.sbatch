#!/bin/bash
# Downloads the Ego-Exo4D cooking keystep takes with gaze (ego view, 448 px) on a transfer node.
#SBATCH --job-name=egoexo_dl
#SBATCH --account=dtn
#SBATCH --partition=notchpeak-dtn
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH=$HOME/.conda/envs/egocollab/bin:$PATH
uids=$(cat data/ego_exo4d/cooking_uids.txt)
dl() { egoexo -o data/ego_exo4d -y --s3_profile egoexo "$@" --uids $uids; }
dl --parts take_eye_gaze
dl --parts downscaled_takes/448 --views ego
