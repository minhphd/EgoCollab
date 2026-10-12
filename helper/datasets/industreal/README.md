# IndustReal

Tabletop assembly and maintenance of a construction-toy car (STEMFIE parts), filmed with a HoloLens 2:
84 recordings from 27 participants, with eye gaze, hand joints, head pose, depth and stereo, labelled
for procedure steps (including execution errors), hand actions and assembly states.

## Status

The download has not completed: 4TU.ResearchData, the only host, has returned "503 Service Unavailable"
since early October 2026 (storage maintenance). `helper/slurm/industreal.sbatch` retries every 30 min
and unpacks each archive once it arrives. Only one test episode exists so far.

## Source and licence

- Paper: Schoonbeek et al., *IndustReal: A Dataset for Procedure Step Recognition Handling Execution
  Errors in Egocentric Videos in an Industrial-Like Setting*, WACV 2024.
- Data: https://data.4tu.nl/datasets/b008dd74-020d-4ea4-a8ba-7bb60769d224 (DOI 10.4121/b008dd74-020d-4ea4-a8ba-7bb60769d224).
- Code and documentation: https://github.com/TimSchoonbeek/IndustReal (released under Apache 2.0; check
  the 4TU page for the data's own terms).

## What the release contains

The 11 archives in `data/industreal/files.txt` (54 GB) are the ones this repo downloads:

| Archive | Size | Contents |
| --- | --- | --- |
| `train_p1.zip` to `train_p4.zip` | 20.4 GB | training recordings |
| `val_p1.zip`, `val_p2.zip` | 10.0 GB | validation recordings |
| `test_p1.zip` to `test_p3.zip` | 23.7 GB | test recordings |
| `action_recognition_labels.zip` | 0.07 MB | action recognition label files |
| `part_geometries.zip` | 17 MB | 3D models of the parts (fbx, 3mf) |

Not downloaded: `all_rgb_videos.zip` (5.0 GB), model weights for action recognition and assembly
state detection, assembly-state results, and the synthetic assembly-state data (16 GB).

Each recording folder (for example `01_assy_1_1`) holds:

| Path | Contents | Rate |
| --- | --- | --- |
| `rgb/*.jpg` | RGB camera, 1080 x 720 | 10 fps |
| `stereo_left/`, `stereo_right/` | grayscale, 640 x 480 | 10 fps |
| `depth/`, `ambient_light/` | long-throw depth and ambient light, 320 x 288 (frames duplicated to the RGB count) | 5 fps |
| `gaze.csv` | eye gaze in image coordinates | 10 fps |
| `hands.csv` | hand joint positions for both hands | 10 fps |
| `pose.csv` | head position, forward and up vectors | 10 fps |
| `PSR_labels.csv` | procedure steps: each completed step with its frame | |
| `PSR_labels_with_errors.csv` | the same, including incorrectly executed steps | |
| `PSR_labels_raw.csv` | per-component state (-1, 0 or 1) per frame | |
| `AR_labels.csv` | hand actions: action id, description, start and end frame | |
| `OD_labels.json` | assembly states, COCO format | |

`train.csv`, `val.csv` and `test.csv` list the recordings of each split. Recording names encode the
procedure: `assy` (assemble the car) or `main` (maintenance: replace the rear parts).

## Code in this folder

`data.py`:

- `build(shard)`: one viewer episode per recording in `data/episodes/industreal/`. The RGB frames are
  encoded once into `data/industreal/videos/<recording>.mp4` (10 fps, times count from the first frame).
- Nodes are the procedure steps of `PSR_labels.csv`; a step's segment runs from the previous step's
  completion to its own. Actions are the `AR_labels.csv` segments. Edges are the step precedence shared
  by the training recordings of the same procedure (at least two).
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
