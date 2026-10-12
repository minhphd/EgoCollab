#!/bin/bash
# Downloads every HD-EPIC video and its per-frame gaze on a CHPC data transfer node, then
# builds the viewer episodes. Reruns skip finished files. Submit from the repo root:
#   sbatch helper/slurm/download_hd_epic.sbatch
#SBATCH --job-name=hd_epic_download
#SBATCH --account=dtn
#SBATCH --partition=notchpeak-dtn
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
PY=$HOME/miniforge3/envs/egocollab/bin/python
"$PY" -m helper.datasets.hd_epic download --all --workers 4
"$PY" -m helper.datasets.hd_epic check
"$PY" -m helper.datasets.hd_epic build
