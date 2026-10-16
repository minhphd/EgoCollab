#!/bin/bash
# Installs Online Watch-And-Help (NOPA) into third_party/owah: the code, VirtualHome's API (wah branch), the
# Watch-And-Help task files, the VirtualHome v2.3.0 Linux simulator with its Mesa container, and a Python 3.7
# environment in ~/.conda/envs/owah. Reruns skip what is already there. Run from the repo root (login node:
# downloads and installs only).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(pwd)/third_party/owah
mkdir -p "$root/executable" && cd "$root"
[ -d online_watch_and_help ] || git clone --depth 1 https://github.com/xavierpuigf/online_watch_and_help.git
[ -d virtualhome ] || git clone --depth 1 --branch wah https://github.com/xavierpuigf/virtualhome.git
# the wah API resets scenes with an action the v2.3 simulator dropped; apply the two-action reset once
git -C virtualhome apply --check "$here/virtualhome_v2.3.patch" 2>/dev/null && git -C virtualhome apply "$here/virtualhome_v2.3.patch"
wah=https://raw.githubusercontent.com/xavierpuigf/watch_and_help/main/dataset
for f in test_env_set_help.pik train_env_set_help.pik env_to_pred.json match_demo_test.json object_info.json watch_scenes_split.json; do
  [ -f "online_watch_and_help/dataset/$f" ] || curl -fsSL -o "online_watch_and_help/dataset/$f" "$wah/$f"
done
# virtual-home.org no longer serves the simulator; this fork mirrors v2.3.0 (patched) and a Mesa container.
sim=https://github.com/TieJean/virtualhome/releases/download/sim-v2.3.0-p1p2
cd executable
[ -f linux_exec_patched.zip ] || curl -fsSL -o linux_exec_patched.zip "$sim/linux_exec_patched.zip"
[ -f virtualhome.sif ] || curl -fsSL -o virtualhome.sif "$sim/virtualhome_ubuntu22.04_mesa251.sif"
echo "371fda39abb5dc8696677d569c52df1e5d23851f884b8cfc194a338c282cc9ba  linux_exec_patched.zip" | sha256sum -c -
ls linux_exec*.x86_64 >/dev/null 2>&1 || unzip -q -o -DD linux_exec_patched.zip  # -DD: current dates, so scratch keeps them
cd "$root"
env=$HOME/.conda/envs/owah
if [ ! -x "$env/bin/python" ]; then
  MAMBA_ROOT_PREFIX=$HOME/.micromamba micromamba create -y -q -p "$env" -c conda-forge python=3.7 pip
  "$env/bin/pip" install -q torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
  "$env/bin/pip" install -q -r "$here/requirements.txt"
fi
echo "Online Watch-And-Help installed in $root"
