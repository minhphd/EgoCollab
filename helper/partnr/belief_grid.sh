#!/bin/bash
# Offline BIP replays on the recorded PARTNR streams: perception tiers from oracle to realistic, then
# ablations. Gemini S2 only on the main rows (cost); ablations use the model-free repair. Run on a CPU
# node from the repo root: helper/partnr/belief_grid.sh <streams_dir> <runs_dir>
set -euo pipefail
streams=$1
runs=$2
py=$HOME/.conda/envs/egocollab/bin/python
run() {  # name, overrides...
  local name=$1; shift
  $py -m helper.bip.replay name=$name runs=$runs streams_dir=$streams "$@"
  $py -m helper.bip.score name=$name runs=$runs streams_dir=$streams
}
# perception tiers: actions (oracle skill and target, noisy target, held object only) x completions
run a_oracle_c_oracle action_tier=oracle completion=oracle
run a_noisy_c_oracle action_tier=noisy completion=oracle
run a_ego_c_oracle action_tier=ego completion=oracle
run a_oracle_c_latent action_tier=oracle completion=latent
run a_noisy_c_latent action_tier=noisy completion=latent
run a_ego_c_noisy action_tier=ego completion=noisy
run a_ego_c_latent action_tier=ego completion=latent
# ablations on the oracle and the realistic rows
for base in "a_oracle_c_oracle action_tier=oracle completion=oracle" "a_ego_c_latent action_tier=ego completion=latent"; do
  set -- $base
  name=$1; shift
  run ${name}_nogaze "$@" gaze=False s2=repair
  run ${name}_flatprior "$@" prior=flat s2=repair
  run ${name}_trueprior "$@" prior=truth s2=never
  run ${name}_nos2 "$@" s2=never
  run ${name}_repair "$@" s2=repair
done
# the prior's model: one call per task, so a stronger model costs little (S2 revisions stay on Flash Lite)
run m38_oracle action_tier=oracle completion=oracle model=gemini-3.8-flash s2=never
run m35_oracle action_tier=oracle completion=oracle model=gemini-3.5-flash s2=never
echo GRID_DONE
