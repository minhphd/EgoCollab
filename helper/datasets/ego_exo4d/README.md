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

## The data

Everything lives in `data/ego_exo4d/` at the repo root (not in git), 81 GB.
[data/ego_exo4d/README.md](../../../data/ego_exo4d/README.md) describes the files and their formats.

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
  (a keystep can occur several times), each with its description, essential flag and taxonomy
  category. The four exo cameras play in sync (`view_cam0x.mp4`), and the info panel lists the take's
  facts and every keystep.
- No edges: Ego-Exo4D gives no dependencies between keysteps, and the viewer shows only what the
  dataset provides.
- Gaze rows map to every third video frame.

## Things to know

- Keysteps are fine grained (for example "Add salt"), repeat within a take, and come from a taxonomy
  shared across scenarios; they are closer to actions than to the coarse subgoals of HD-EPIC.
- Downloads need the `egoexo` CLI (from the `ego4d` package) next to the Python interpreter.
