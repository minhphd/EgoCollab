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

## The data

Everything lives in `data/hd_epic/` at the repo root (not in git). [data/hd_epic/README.md](../../../data/hd_epic/README.md)
describes what is on disk and every annotation file HD-EPIC provides.

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
| `data.py` | `download`, `build` (one viewer episode per video with every HD-EPIC label), `export-actions`, `check`; per-layer readers `activities`, `movements` (with gaze priming and boxes), `sounds`, `ingredients`, `recipe_steps`; loaders `load_narrations()`, `action_times(narr, video)`, `load_framewise(video)` (normalized gaze per frame, head position) |
| `posthoc_graph.py` | the task-graph format: `find(video)`, `action_owner(graph)`, `parse_ranges`, `primitive_edges`, `validate` |
| `tasks.py` | ground truth per video: `load(min_nodes)` gives a `Task` with subgoals, labels, required orderings, each subgoal's first and last action time, and `done(t)`, `current(t)`, `next(t)` |
| `gaze3d.py` | gaze fixations on objects in 3D: the Aria gaze ray against where each moved object rests (from the object-movement annotations), no detector needed |

## How to use

```bash
python -m helper.datasets.hd_epic download --all --workers 4   # videos, gaze, annotations (on a transfer node:
                                                                # helper/slurm/download_hd_epic.sbatch)
python -m helper.datasets.hd_epic check                        # validate the post-hoc graphs
python -m helper.datasets.hd_epic build                        # viewer episodes, one per video
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

`data/episodes/hd_epic/` holds one episode per video with every HD-EPIC label: the narrations (with
verbs, nouns, main action and hands), the steps of every recipe captured in the video with their step and
prep times, and tracks for high-level activities, object movements, gaze priming, sounds and ingredient
weighing and adding, plus the boxes of each moved object at its pick-up and put-down and the recipe,
ingredient and nutrition facts. HD-EPIC gives no dependencies between steps, so the episodes have no edges.
Of the 40 videos without a recipe, 32 have activities, and every video has object movements and sounds. The post-hoc subgoal graphs are not shown; the
episodes once built from them are kept in `data/episodes_annotated/` (`hd_epic_subgoals/`, `hd_epic_task/`).

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
