# data/ego_exo4d

Ego-Exo4D as downloaded for this repo: the 341 cooking takes that have keystep annotations and eye gaze,
over 11 recipes, with every camera downscaled to 448 px. Code and how to use it:
[helper/datasets/ego_exo4d](../../helper/datasets/ego_exo4d/README.md).

- Paper: Grauman et al., *Ego-Exo4D*, CVPR 2024; docs: https://docs.ego-exo4d-data.org
- Licence: the Ego-Exo4D licence agreement (https://ego4d.dev/request/ego-exo4d); AWS keys required.
- Made by `python -m helper.datasets.ego_exo4d download` and `helper/slurm/egoexo_cooking.sbatch` (the uids
  in `cooking_uids.txt`).

## Layout (81 GB)

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

