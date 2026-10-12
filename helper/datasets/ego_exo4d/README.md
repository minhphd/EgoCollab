# Ego-Exo4D (cooking keystep takes with gaze)

Skilled people doing real activities, filmed at once by Aria glasses (ego) and fixed cameras (exo).
This repo uses the cooking takes that have keystep annotations and eye gaze: 341 takes over 11 recipes,
with the ego video downscaled to 448 px.

## Source and licence

- Paper: Grauman et al., *Ego-Exo4D: Understanding Skilled Human Activity from First- and
  Third-Person Perspectives*, CVPR 2024; docs: https://docs.ego-exo4d-data.org
- Licence: the Ego-Exo4D licence agreement, signed at https://ego4d.dev/request/ego-exo4d. Approval
  takes about two days; AWS keys arrive by email and expire after 14 days.
- Credentials: store the keys with `aws configure`. `data.py` uses the default AWS profile;
  `helper/slurm/egoexo_cooking.sbatch` uses a profile named `egoexo` (`aws configure --profile egoexo`).
  The repo-root file `EgoExoCreds` is git-ignored; never print or commit it.

## What is on disk

Everything lives in `data/ego_exo4d/` at the repo root (not in git), 81 GB.

| Path | Contents |
| --- | --- |
| `takes.json`, `captures.json`, `participants.json`, `physical_setting.json`, `visual_objects.json`, `metadata.json` | release metadata: one record per take (`take_uid`, `take_name`, `task_name`, `duration_sec`, `root_dir`, `frame_aligned_videos`, `has_trimmed_eye_gaze`, ...) |
| `annotations/keystep_train.json`, `keystep_val.json` | keystep annotations per take: `scenario` and `segments` (`step_id`, `step_name`, `start_time`, `end_time`, ...) |
| `annotations/splits.json` | the benchmark splits |
| `cooking_uids.txt` | the 341 take uids downloaded |
| `takes/<take_name>/frame_aligned_videos/downscaled/448/*.mp4` | every camera of the take at 448 px: the Aria ego stream (`aria*.mp4`) and the exo cameras (`cam0x.mp4`) |
| `takes/<take_name>/eye_gaze/` | `general_eye_gaze.csv`, `personalized_eye_gaze.csv` (3D), their `_2d.csv` versions (pixels on the 1408 px Aria frame, one row per three video frames, 10 Hz) and `summary.json` |

Takes per recipe: Cucumber and Tomato Salad 62, Omelet 59, Tomato and Eggs 46, Milk Tea 46, Noodles 35,
Sesame-Ginger Asian Salad 29, Scrambled Eggs 27, Coffee Latte 18, Chai Tea 14, Pasta 4, Sushi Rolls 1.

## Code in this folder

`data.py`:

- `download(num_takes, uids)`: metadata and keystep annotations, then the 448 px takes and their eye
  gaze. Without uids it picks the shortest keystep takes that have gaze and at least 6 segments.
- `build()`: one viewer episode per take in `data/episodes/ego_exo4d/`.
- `keystep_annotations()`, `ego_video(take)`, `load_gaze(take)` (normalized 2D gaze, personalized
  when present, else general).

## How to use

```bash
python -m helper.datasets.ego_exo4d download --num-takes 4      # the shortest keystep takes with gaze
python -m helper.datasets.ego_exo4d download --uids <take_uid> ...
sbatch helper/slurm/egoexo_cooking.sbatch                       # every take in cooking_uids.txt, on a transfer node
python -m helper.datasets.ego_exo4d build                       # viewer episodes
```

```python
import json
from helper.datasets.ego_exo4d.data import DATA, keystep_annotations, ego_video, load_gaze

takes = {t["take_uid"]: t for t in json.loads((DATA / "takes.json").read_text())}
ks = keystep_annotations()                       # take_uid -> {"scenario", "segments": [...]}
uid = next(u for u in ks if u in takes and ego_video(takes[u]).exists())
video, gaze = ego_video(takes[uid]), load_gaze(takes[uid])   # gaze[i] = [x, y] in [0, 1] or None, 10 Hz
```

## Viewer episodes (`data/episodes/ego_exo4d/`, 341)

- One per take. Nodes are the keysteps the take contains; segments are its keystep annotations
  (a keystep can occur several times).
- The graph is not ground truth: Ego-Exo4D has no dependency graph. An edge u to v is added when u
  starts before v in every annotated take of the scenario that contains both, with at least two such
  takes, then implied edges are removed (`episode.precedence_dag`). It is the order people followed,
  not the order the recipe requires.
- Gaze rows map to every third video frame.

## Things to know

- Keysteps are fine grained (for example "Add salt"), repeat within a take, and come from a taxonomy
  shared across scenarios; they are closer to actions than to the coarse subgoals of HD-EPIC.
- Downloads need the `egoexo` CLI (from the `ego4d` package) next to the Python interpreter.
