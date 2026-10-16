# IndustReal

Tabletop assembly and maintenance of a construction-toy car (STEMFIE parts), filmed with a HoloLens 2:
84 recordings from 27 participants, with eye gaze, hand joints, head pose, depth and stereo, labelled
for procedure steps (including execution errors), hand actions and assembly states.

## Status

All 11 archives are downloaded and unpacked (55 GB), and all 84 recordings are viewer episodes in
`data/episodes/industreal/`. 4TU.ResearchData, the only host, was down for storage maintenance in early
October 2026; `helper/slurm/industreal.sbatch` retries every 30 min when it is.

## Source and licence

- Paper: Schoonbeek et al., *IndustReal: A Dataset for Procedure Step Recognition Handling Execution
  Errors in Egocentric Videos in an Industrial-Like Setting*, WACV 2024.
- Data: https://data.4tu.nl/datasets/b008dd74-020d-4ea4-a8ba-7bb60769d224 (DOI 10.4121/b008dd74-020d-4ea4-a8ba-7bb60769d224).
- Code and documentation: https://github.com/TimSchoonbeek/IndustReal (released under Apache 2.0; check
  the 4TU page for the data's own terms).

## The data

The recordings unpack into `data/industreal/` at the repo root (not in git).
[data/industreal/README.md](../../../data/industreal/README.md) describes the archives, the recording layout
and the label files.

## Code in this folder

`data.py`:

- `build(shard)`: one viewer episode per recording in `data/episodes/industreal/`. The RGB frames are
  encoded once into `data/industreal/videos/<recording>.mp4` (10 fps, times count from the first frame).
- Nodes are the procedure steps of `PSR_labels.csv`; a step's segment runs from the previous step's
  completion to its own. Actions are the `AR_labels.csv` segments. No edges: IndustReal gives no
  dependencies between steps.
- Gaze comes from `gaze.csv`; (0, 0) and points outside the frame count as missing.

## How to use

```bash
sbatch helper/slurm/industreal.sbatch                    # download and unpack, on a transfer node
sbatch -M kingspeak --array=0-7 helper/slurm/cpu.sbatch \
  python -m helper.datasets.industreal build --shard ARRAY/8
```

```python
from helper.datasets.industreal.data import DATA, recordings, rows
recs = recordings()                                   # recording -> split, for recordings on disk
steps = rows(DATA / "01_assy_1_1" / "PSR_labels.csv") # rows of the CSV
```

## Things to know

- The procedure steps are completion events of an assembly, so "subgoal complete" is directly
  labelled here, unlike in the kitchen datasets.
- Execution errors are labelled (`PSR_labels_with_errors.csv`), which makes the dataset a test of
  recovery from deviations.
- Participants read a printed instruction sheet; gaze on the sheet says little about the next part.
