#!/bin/bash
# Third closed-loop round on all tasks: the robot watches 3 s after the start and after each completion before
# choosing. Run on the GPU node from the repo root: helper/partnr/loop_grid3.sh <runs_dir>
set -euo pipefail
runs=$1; shift
py=$HOME/.conda/envs/partnr/bin/python
export PATH=$HOME/.conda/envs/egocollab/bin:$PATH OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
loop() {
  local name=$1; shift
  [ -f "$runs/$name/results_$name.json" ] || $py -m helper.partnr.record name=$name runs=$runs workers=8 "$@"
}
loop bip_look3 robot=bip look_s=3
loop bip_p35_look3 robot=bip look_s=3 prior_model=gemini-3.5-flash
loop graph_look3 robot=graph look_s=3
loop bip_p35_look3_ego_latent robot=bip look_s=3 prior_model=gemini-3.5-flash action_tier=ego completion=latent
echo LOOP3_DONE
