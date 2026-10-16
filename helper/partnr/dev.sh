#!/bin/bash
# Closed-loop robots on the 20-task dev set (helper/partnr/dev_tasks.txt), scored against the person alone
# (data/bip_partnr_dev/streams, recorded with name=rec_dev record_dir=data/bip_partnr_dev). Each argument is "name override ...". Run on the GPU node from the repo root:
#   helper/partnr/dev.sh "graph robot=graph completion=latent" "bip robot=bip action_tier=ego completion=latent"
set -euo pipefail
export PATH=$HOME/.conda/envs/egocollab/bin:$PATH OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
runs=$PWD/runs/partnr/${DEV:-dev}  # ALONE: the person-alone streams to score against (same person model)
only="($(grep -v '^#' helper/partnr/dev_tasks.txt | sed "s/.*/'&',/" | tr -d '\n'))"
dirs=()
for cond in "$@"; do
  set -- $cond
  name=$1; shift
  rm -rf "$runs/$name"
  $HOME/.conda/envs/partnr/bin/python -m helper.partnr.record name=$name runs=$runs workers=10 only="$only" "$@" | grep "episodes, " || true
  dirs+=("$runs/$name")
done
$HOME/.conda/envs/egocollab/bin/python -m helper.partnr.loop_score ${ALONE:-data/bip_partnr_dev/streams} "${dirs[@]}"
