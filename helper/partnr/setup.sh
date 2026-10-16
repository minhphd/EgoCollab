#!/bin/bash
# Installs PARTNR (Meta's habitat-llm) into third_party/partnr: the code with its submodules (habitat-lab,
# transformers-CFG) and a Python 3.9 environment in ~/.conda/envs/partnr with habitat-sim 0.3.3 (headless,
# bullet). Reruns skip what is already there. Run from the repo root; data comes from download.sh.
set -euo pipefail
root=$(pwd)/third_party/partnr
mkdir -p "$root" && cd "$root"
[ -d partnr-planner ] || git clone https://github.com/facebookresearch/partnr-planner.git
cd partnr-planner
git submodule sync -q && git submodule update -q --init --recursive
# the data lives with the other datasets in data/partnr
mkdir -p ../../../data/partnr && [ -e data ] || ln -s ../../../data/partnr data
env=$HOME/.conda/envs/partnr
if [ ! -x "$env/bin/python" ]; then
  MAMBA_ROOT_PREFIX=$HOME/.micromamba micromamba create -y -q -p "$env" -c conda-forge -c aihabitat \
    python=3.9.2 cmake=3.14.0 habitat-sim=0.3.3 withbullet headless git-lfs
fi
# torch 2.7 with CUDA 12.8 runs on Ampere, Hopper and Blackwell; the upstream pin (2.4.1, CUDA 12.4) lacks Blackwell
"$env/bin/python" -c "import torch" 2>/dev/null || \
  "$env/bin/pip" install -q torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
# ask pip, not import: the source folder here would satisfy an import
if ! "$env/bin/pip" show -q habitat-llm 2>/dev/null; then
  "$env/bin/pip" install -q -e third_party/habitat-lab/habitat-lab -e third_party/habitat-lab/habitat-baselines \
    -e third_party/transformers-CFG
  "$env/bin/pip" install -q -r requirements.txt
  "$env/bin/pip" install -q -e .
  # habitat-sim pins pillow 10.4; tensorboard 2.20+ needs protobuf 6, but requirements.txt pins 3.20.1
  "$env/bin/pip" install -q pillow==10.4.0 tensorboard==2.19.0
fi
echo "PARTNR installed in $root/partnr-planner"
