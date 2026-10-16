# data/partnr/

The PARTNR data, 26 GB, made by `helper/partnr/download.sh`. The code sees this folder as
`third_party/partnr/partnr-planner/data`. Downloads land in `versioned_data/`; the other names are the
symlinks and folders the code expects.

| Path | Size | What |
| --- | --- | --- |
| `versioned_data/hssd-hab/` (`hssd-hab`) | 20 GB | HSSD scenes, `partnr` branch: furnished apartments, receptacles, semantic and navmesh files |
| `objects_ovmm/` | 1.8 GB | OVMM object models placed into the scenes |
| `versioned_data/partnr_episodes/` | 1.9 GB | the tasks, skill checkpoints, concept graphs and human traces (below) |
| `versioned_data/ycb/`, `replica_cad_dataset/` | 1.3 GB | YCB objects and ReplicaCAD, used by habitat-lab |
| `versioned_data/hssd-partnr-ci/` (`hssd-partnr-ci`) | 274 MB | two small scenes for the unit tests |
| `humanoids/`, `robots/`, `hab_spot_arm/`, `hab_fetch/` | 175 MB | humanoid avatars with motion data, Spot and Fetch robots |
| `hab3-episodes/`, `rearrange_*` | 317 MB | Habitat 3.0 rearrangement episodes, used by habitat-lab |
| `datasets/partnr_episodes` | | symlink to the task files |
| `models` | | symlink to the skill checkpoints |

## Tasks (`partnr_episodes/v0_0/`)

| File | Episodes | Scenes |
| --- | --- | --- |
| `train.json.gz` | 111,652, checked by people | 38 |
| `train_132k_unverified.json.gz` | 131,991, generated | 38 |
| `train_2k.json.gz`, `train_mini.json.gz` | 2,000 and 403, subsets of train | |
| `val.json.gz` | 1,000 | 12, disjoint from train |
| `val_mini.json.gz` | 369, a subset of val | 6 |
| `ci.json.gz` | 2 | the CI scenes |

Each episode holds the scene, the object placements (`rigid_objs`, `object_states`), the `instruction`,
and its success test: `evaluation_propositions` (for example `is_on_top` of named objects and
receptacles, with a `number` to satisfy), `evaluation_proposition_dependencies`, and
`evaluation_constraints` (`TemporalConstraint` orders propositions, `TerminalSatisfactionConstraint`
requires them to hold at the end, `SameArgConstraint` and `DifferentArgConstraint` tie arguments).

## Other contents of `partnr_episodes/`

- `checkpoints/`: neural skill policies (pick, place, point_nav, open, close).
- `conceptgraphs/`: precomputed concept graphs of HSSD scenes.
- `hitl_data/`: five archives (692 MB) of people solving tasks in the human-in-the-loop tool, alone
  (`p5_single_*`), in pairs (`p5_multi_val`) or with a learned robot (`p5_learn_*`). Each episode file
  has `frames`; a recorded frame holds the time `t`, per user the `camera_transform`, `held_object`,
  `hovered_object` and `events`, plus `object_states`, `agent_states` and `task_percent_complete`.
