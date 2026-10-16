#!/bin/bash
# Runs PARTNR's heuristic planner (no LLM, full observability) on the first N val_mini episodes and saves
# videos. Run from the repo root on a GPU node. Usage: helper/partnr/smoke_test.sh [N] [out_dir]
set -euo pipefail
n=${1:-2}
out=$(realpath -m "${2:-runs/partnr/smoke}")
mkdir -p "$out"
cd third_party/partnr/partnr-planner
idx=$(seq -s, 0 $((n - 1)))
HYDRA_FULL_ERROR=1 ~/.conda/envs/partnr/bin/python -m habitat_llm.examples.planner_demo \
  --config-name baselines/heuristic_full_obs.yaml \
  habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/val_mini.json.gz \
  "+episode_indices=[$idx]" evaluation.save_video=True \
  hydra.run.dir="$out" paths.results_dir="$out/results"
~/.conda/envs/partnr/bin/python scripts/read_results.py "$out/results" 2>/dev/null || ls -R "$out" | head -40
