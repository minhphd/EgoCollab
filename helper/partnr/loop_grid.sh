#!/bin/bash
# Closed-loop PARTNR runs: the same scripted person (same seeds) with each robot. Run on the GPU node from
# the repo root: helper/partnr/loop_grid.sh <runs_dir> [extra overrides, e.g. ordered_only=True]
set -euo pipefail
runs=$1; shift
py=$HOME/.conda/envs/partnr/bin/python
export PATH=$HOME/.conda/envs/egocollab/bin:$PATH OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
loop() {  # name, overrides...
  local name=$1; shift
  [ -f "$runs/$name/results_$name.json" ] || $py -m helper.partnr.record name=$name runs=$runs workers=8 "$@"
}
loop alone robot=idle "$@"
loop graph robot=graph "$@"
loop bip robot=bip action_tier=oracle completion=oracle "$@"
loop oracle robot=oracle "$@"
loop bip_ego robot=bip action_tier=ego completion=oracle "$@"
loop bip_ego_latent robot=bip action_tier=ego completion=latent "$@"
loop graph_latent robot=graph completion=latent "$@"
$HOME/.conda/envs/egocollab/bin/python -m helper.partnr.loop_score $runs/alone/streams $runs/graph $runs/bip $runs/oracle \
  $runs/bip_ego $runs/bip_ego_latent $runs/graph_latent
echo LOOP_DONE
