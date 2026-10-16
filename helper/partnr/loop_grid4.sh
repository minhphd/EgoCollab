#!/bin/bash
# Fourth closed-loop round on all tasks: a person who never notices the robot (notice_s = 1e9) keeps working
# on a subgoal the robot has taken. Run on the GPU node from the repo root: helper/partnr/loop_grid4.sh <runs_dir>
set -euo pipefail
runs=$1; shift
py=$HOME/.conda/envs/partnr/bin/python
export PATH=$HOME/.conda/envs/egocollab/bin:$PATH OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
loop() {
  local name=$1; shift
  [ -f "$runs/$name/results_$name.json" ] || $py -m helper.partnr.record name=$name runs=$runs workers=8 notice_s=1e9 "$@"
}
loop graph_rigid robot=graph
loop bip_rigid robot=bip look_s=3 prior_model=gemini-3.5-flash
loop oracle_rigid robot=oracle
loop bip_ego_latent_rigid robot=bip look_s=3 prior_model=gemini-3.5-flash action_tier=ego completion=latent
echo LOOP4_DONE
