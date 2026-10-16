#!/bin/bash
# Fifth closed-loop round on all tasks: the cost of an ordering violation (in subgoal durations) traded
# against speed, for the best configuration. Run on the GPU node: helper/partnr/loop_grid5.sh <runs_dir>
set -euo pipefail
runs=$1; shift
py=$HOME/.conda/envs/partnr/bin/python
export PATH=$HOME/.conda/envs/egocollab/bin:$PATH OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
loop() {
  local name=$1; shift
  [ -f "$runs/$name/results_$name.json" ] || $py -m helper.partnr.record name=$name runs=$runs workers=8 "$@"
}
loop bip_p35_look3_prec1 robot=bip look_s=3 prior_model=gemini-3.5-flash beta_prec_rel=1.0
loop bip_p35_look3_prec05 robot=bip look_s=3 prior_model=gemini-3.5-flash beta_prec_rel=0.5
echo LOOP5_DONE
