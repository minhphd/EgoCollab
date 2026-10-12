#!/bin/bash
# Runs BIP replays in parallel on the current node and scores each: bip_sweep.sh "<name> <key=value ...>" ...
P=~/.conda/envs/egocollab/bin/python
for spec in "$@"; do
  set -- $spec; name=$1; shift
  ( $P -m helper.bip.run name=$name "$@" > logs/bip_$name.log 2>&1 && $P -m helper.bip.score name=$name "$@" > /dev/null 2>&1 ) &
done
wait
