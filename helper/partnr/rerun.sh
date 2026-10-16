#!/bin/bash
# The full PARTNR experiment on the current testbed: record the person alone, export viewer episodes, the
# offline belief grid, then the main closed-loop robots. Run on the GPU node from the repo root:
#   helper/partnr/rerun.sh            (writes data/bip_partnr, data/episodes/partnr, runs/partnr/{belief,loop})
set -euo pipefail
export PATH=$HOME/.conda/envs/egocollab/bin:$PATH OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
py=$HOME/.conda/envs/partnr/bin/python
[ -f data/bip_partnr/results_rec_val_mini.json ] || $py -m helper.partnr.record name=rec_val_mini workers=8
$HOME/.conda/envs/egocollab/bin/python -m helper.partnr.export data/bip_partnr data/episodes
helper/partnr/belief_grid.sh data/bip_partnr/streams runs/partnr/belief &
runs=$PWD/runs/partnr/loop
loop() {
  local name=$1; shift
  [ -f "$runs/$name/results_$name.json" ] || $py -m helper.partnr.record name=$name runs=$runs workers=8 "$@"
}
loop alone robot=idle
loop graph robot=graph
loop bip robot=bip
loop bip_p35_look3 robot=bip look_s=3 prior_model=gemini-3.5-flash
loop oracle robot=oracle
loop bip_p35_look3_ego_latent robot=bip look_s=3 prior_model=gemini-3.5-flash action_tier=ego completion=latent
wait
$HOME/.conda/envs/egocollab/bin/python -m helper.partnr.loop_score $runs/alone/streams $runs/graph $runs/bip \
  $runs/bip_p35_look3 $runs/oracle $runs/bip_p35_look3_ego_latent
echo RERUN_DONE
