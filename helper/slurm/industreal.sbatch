#!/bin/bash
# Downloads and unpacks the IndustReal recordings (train, val, test parts) on a transfer node.
#SBATCH --job-name=industreal_dl
#SBATCH --account=dtn
#SBATCH --partition=notchpeak-dtn
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR/data/industreal"
while read -r name url; do
  [ -f "$name.done" ] && continue
  until curl -fsSL -C - -o "$name" "$url"; do echo "retry $name in 30 min"; sleep 1800; done  # -f: an error reply is a failure, not the file
  unzip -q -o -DD "$name" && rm "$name" && touch "$name.done"  # -DD: files get the current date, not the archive's (scratch purges by date)
  echo "done $name"
done < files.txt
