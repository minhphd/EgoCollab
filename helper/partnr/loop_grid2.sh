#!/bin/bash
# Second closed-loop round on the tasks with required orderings: the prior's model, the true graph, and the
# conflict cost. Run on the GPU node from the repo root: helper/partnr/loop_grid2.sh <runs_dir>
set -euo pipefail
runs=$1; shift
py=$HOME/.conda/envs/partnr/bin/python
export PATH=$HOME/.conda/envs/egocollab/bin:$PATH OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
loop() {
  local name=$1; shift
  [ -f "$runs/$name/results_$name.json" ] || $py -m helper.partnr.record name=$name runs=$runs workers=6 ordered_only=True "$@"
}
loop bip_p35 robot=bip prior_model=gemini-3.5-flash
loop bip_truth robot=bip prior=truth s2=never
loop bip_b025 robot=bip beta_rel=0.25
loop bip_p35_b025 robot=bip prior_model=gemini-3.5-flash beta_rel=0.25
loop bip_p35_ego_latent robot=bip prior_model=gemini-3.5-flash action_tier=ego completion=latent
echo LOOP2_DONE
