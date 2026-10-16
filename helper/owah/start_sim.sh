#!/bin/bash
# Starts the VirtualHome simulator headless (Xvfb and Mesa inside the Apptainer image), on a compute node:
#   helper/owah/start_sim.sh [port] [display]        # then UnityCommunication(port=...) connects
set -euo pipefail
port=${1:-8080} display=${2:-99}
cd "$(dirname "$0")/../../third_party/owah/executable"
exe=$(ls linux_exec*.x86_64 | head -n 1)
module load apptainer 2>/dev/null || true
exec apptainer exec --bind "$PWD" virtualhome.sif bash -c \
  "Xvfb :$display -screen 0 640x480x24 >/dev/null 2>&1 & export DISPLAY=:$display; sleep 2; \
   ./$exe -batchmode -http-port=$port -screen-fullscreen 0 -screen-quality 4 -logfile -"
