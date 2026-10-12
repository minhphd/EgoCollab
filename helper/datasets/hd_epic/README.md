# HD-EPIC

Unscripted kitchen activity filmed with Project Aria glasses: 156 videos from 9 participants, with eye
gaze, head pose, 59,454 narrated actions, object movements and recipes. On top of HD-EPIC's own labels,
this folder holds a post-hoc task graph for every video: subgoals, the actions in each, and the
orderings between them.

## Source and licence

- Paper: Perrett et al., *HD-EPIC: A Highly-Detailed Egocentric Video Dataset*, CVPR 2025
  ([arXiv 2502.04144](https://arxiv.org/abs/2502.04144)); site: https://hd-epic.github.io
- Licence: CC BY-NC 4.0 (non-commercial).
- Annotations: https://github.com/hd-epic/hd-epic-annotations. Videos and the per-frame gaze and pose
  ("Intermediate Data") come from the OneDrive folders linked on the site; `data.py` holds the URLs.

## What is on disk

Everything lives in `data/hd_epic/` at the repo root (not in git).

| Path | Contents | Size |
| --- | --- | --- |
| `videos/<P0x>/<video>.mp4` | 156 RGB videos from the Aria camera, H.264, 1408 x 1408 fisheye, 30 fps | 116 GB |
| `framewise/<P0x>/<video>/framewise_info.jsonl` | one line per video frame: `T_world_device` (3 x 4 pose), `gaze_centre_in_pixels`, `gaze_direction_in_world`, timestamps | 2.2 GB |
| `framewise/<P0x>/<video>/device_calibration.json` | camera and sensor calibration of that recording (`camera-rgb`: fisheye model, `T_device_camera`) | |
| `annotations/` | the HD-EPIC annotations repo (below) | |
| `actions/<video>.txt` | numbered narrations per video, written by `export-actions` for graph annotation | |

154 of the 156 videos have gaze and pose; P02-20240210-113925 and P02-20240211-182630 have none released.

Derived data, also in `data/`: `frames448/` (every video at 2 fps, 448 px JPEG, 22 GB, from
`helper.perception.frames`) and `episodes/hd_epic/`, `episodes/hd_epic_task/` (viewer episodes, below).

## HD-EPIC's annotations (`data/hd_epic/annotations/`)

| Folder | File | What it holds |
| --- | --- | --- |
| `narrations-and-action-segments/` | `HD_EPIC_Narrations.pkl` | 59,454 actions: `narration` (free text), `start_timestamp`, `end_timestamp`, `verbs`, `nouns`, `pairs`, `main_actions`, their class ids, `hands`. `unique_narration_id` = `<video>-<index>`. |
| | `HD_EPIC_verb_classes.csv`, `HD_EPIC_noun_classes.csv` | verb and noun clusters with their instances |
| `high-level/` | `complete_recipes.json` | 69 recipes: `name`, `type`, `steps`, `captures` (which videos, step and prep times) |
| | `activities/P0x_recipe_timestamps.csv` | high-level activity segments: `video_id`, `recipe_id`, `high_level_activity_label`, `start_time`, `end_time` |
| `scene-and-object-movements/` | `assoc_info.json` | every moved object (about 54 per video): `name` and `tracks`, each a pick-up to put-down `time_segment` with mask ids |
| | `mask_info.json` | per mask: `frame_number`, `3d_location` (world), `bbox` (on the 1408 px frame), `fixture` |
| `eye-gaze-priming/` | `priming_info.json` | per object movement: when gaze first reached the pick-up or put-down location (`frame_primed`, `prime_gap` in s) |
| `audio-annotations/` | `HD_EPIC_Sounds.csv`, `.pkl` | sound events |
| `vqa-benchmark/` | 30 JSON files | the HD-EPIC VQA questions |
| `youtube-links/` | `HD_EPIC_YouTube_URLs.csv` | video links |

The annotations repo's own `README.md` documents every field.

## Post-hoc task graphs (`annotations/` in this folder, in git)

One JSON per task (155 files covering all 156 videos; a task can span videos), plus `GUIDELINES.md`
with the annotation rules:

```json
{
  "task": "Chopped Chickpea Salad",
  "videos": ["P07-20240529-131737", "P07-20240529-134410"],
  "nodes": [{"id": "lettuce_washed", "label": "Lettuce washed and dried", "group": "Salad",
             "actions": {"P07-20240529-131737": "40-52,57"}}],
  "edges": [{"from": "lettuce_washed", "to": "lettuce_in_bowl", "why": "Lettuce is washed before it goes in"}],
  "primitives": {"P07-20240529-131737": {"40": {"after": []}}}
}
```

- Two levels. Every narrated action (a primitive, numbered by its `unique_narration_id` index) has
  direct precondition edges (`after`). Subgoals group actions (`actions`: index ranges per video) and
  carry their own required orderings, each with a one-line reason (`why`). An edge (u, v) means v needs u.
- Edges are real preconditions only: an object in hand, a container open, a device on, an ingredient
  ready, an object in place. Observed order alone is not an edge.
- The graphs were drawn by LLM subagents and pass `check`, which verifies that every cross-subgoal
  action edge follows a subgoal path and every subgoal edge is backed by an action edge. About
  150,000 action edges were not read one by one; spot-check before trusting fine details.

## Code in this folder

| File | What it does |
| --- | --- |
| `data.py` | `download`, `build` (per-subgoal viewer episodes), `export-actions`, `check`; loaders `load_narrations()`, `action_times(narr, video)`, `load_framewise(video)` (normalized gaze per frame, head position) |
| `posthoc_graph.py` | the task-graph format: `find(video)`, `action_owner(graph)`, `parse_ranges`, `primitive_edges`, `validate`, `subgoal_episodes` |
| `tasks.py` | ground truth per video: `load(min_nodes)` gives a `Task` with subgoals, labels, required orderings, each subgoal's first and last action time, and `done(t)`, `current(t)`, `next(t)`; `export` writes one viewer episode per video |
| `gaze3d.py` | gaze fixations on objects in 3D: the Aria gaze ray against where each moved object rests (from the object-movement annotations), no detector needed |

## How to use

```bash
python -m helper.datasets.hd_epic download --all --workers 4   # videos, gaze, annotations (on a transfer node:
                                                                # helper/slurm/download_hd_epic.sbatch)
python -m helper.datasets.hd_epic check                        # validate the post-hoc graphs
python -m helper.datasets.hd_epic build                        # viewer episodes, one per subgoal
python -m helper.datasets.hd_epic.tasks export                 # viewer episodes, one per video
python -m helper.perception.frames --shard 0/1                 # 2 fps frame cache, data/frames448
```

```python
from helper.datasets.hd_epic import load_narrations, load_framewise
from helper.datasets.hd_epic.tasks import load
from helper.datasets.hd_epic.gaze3d import fixations

narr = load_narrations()                       # pandas DataFrame, one row per action
gaze, head_xy = load_framewise("P01-20240202-110250")   # gaze[k] = [x, y] in [0, 1] of frame k, or None
tasks = load(3)                                # 138 videos with >= 3 subgoals, in a fixed shuffled order
task = tasks["P04-20240414-175337"]
task.nodes, task.edges, task.label             # subgoal ids, required orderings (u, v), labels
task.current(120.0), task.next(120.0), task.done(120.0)
fixations("P04-20240414-175337")               # [{"t", "object" (instance id), "visible"}, ...]
```

`load(3)` keeps a fixed order (shuffled with seed 0): earlier work used its first 30 videos as a dev
split and the other 108 as the test split.

## Viewer episodes

- `data/episodes/hd_epic/` (1,225): one per subgoal per video. The clip runs from the subgoal's first
  action to its last with 1 s of context; nodes are that subgoal's actions (median 32) with their edges.
  Subgoals with fewer than 3 actions are skipped.
- `data/episodes/hd_epic_task/` (156): one per video; nodes are the subgoals with their required
  orderings, segments span each subgoal's first to last action, actions are the narrations.

## Things to know

- People work on subgoals in bouts: a subgoal is done in about 2.8 bouts on average, and returns to a
  started subgoal outnumber first starts two to one (2,279 against 1,058 over the 138 videos).
- 29% of subgoal starts happen while an annotated predecessor is still incomplete (12% while one has
  not started): the orderings hold between some actions of two subgoals, not always between whole
  subgoals.
- 10% of narrated actions belong to no subgoal; 95% of actions keep the subgoal of the previous one.
- Gaze leads the hands only briefly: fixated objects point to the next subgoal only in the last 2 s
  before a switch.
- P09 videos with nested narration ids are renumbered 1..n by start time.
