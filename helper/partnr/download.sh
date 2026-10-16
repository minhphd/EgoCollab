#!/bin/bash
# Downloads the PARTNR data into data/partnr (seen by the code as third_party/partnr/partnr-planner/data):
# robot, humanoid and rearrangement assets, OVMM objects, the HSSD scenes (partnr branch), the PARTNR
# episodes with skill checkpoints, and the small CI scenes. Reruns resume. Run on a compute node.
set -euo pipefail
env=$HOME/.conda/envs/partnr
export PATH=$env/bin:$PATH  # git-lfs
cd third_party/partnr/partnr-planner

# clone without large files, then pull them; a rerun only pulls what is missing
lfs_clone() {  # url dir [branch]
  [ -d "$2/.git" ] || GIT_LFS_SKIP_SMUDGE=1 git clone -q ${3:+-b $3} "$1" "$2"
  git -C "$2" lfs install --local >/dev/null && git -C "$2" lfs pull
}

python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets hab_spot_arm hab3-episodes \
  habitat_humanoids --data-path data/ --no-replace --no-prune
lfs_clone https://huggingface.co/datasets/ai-habitat/OVMM_objects data/objects_ovmm
lfs_clone https://huggingface.co/datasets/hssd/hssd-hab data/versioned_data/hssd-hab partnr
lfs_clone https://huggingface.co/datasets/ai-habitat/partnr_episodes data/versioned_data/partnr_episodes
lfs_clone https://huggingface.co/datasets/ai-habitat/hssd-partnr-ci data/versioned_data/hssd-partnr-ci

mkdir -p data/datasets
ln -sfn versioned_data/hssd-hab data/hssd-hab
ln -sfn ../versioned_data/partnr_episodes data/datasets/partnr_episodes
ln -sfn versioned_data/partnr_episodes/checkpoints data/models
ln -sfn versioned_data/hssd-partnr-ci data/hssd-partnr-ci
ln -sfn versioned_data/partnr_episodes/test_rag data/test_rag
du -shL data/* 2>/dev/null
echo "PARTNR data in $(cd data && pwd -P)"
