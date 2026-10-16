# data/hd_epic

HD-EPIC as downloaded: 156 unscripted kitchen videos from 9 participants (P01 to P09), filmed with Project
Aria glasses, with per-frame gaze and pose and the HD-EPIC annotations. Code, the post-hoc subgoal
graphs and how to use them: [helper/datasets/hd_epic](../../helper/datasets/hd_epic/README.md).

- Paper: Perrett et al., *HD-EPIC: A Highly-Detailed Egocentric Video Dataset*, CVPR 2025; https://hd-epic.github.io
- Licence: CC BY-NC 4.0 (non-commercial).
- Made by `python -m helper.datasets.hd_epic download --all` (or `helper/slurm/download_hd_epic.sbatch` on a
  transfer node); the VRS recordings (1.9 TB) are not downloaded.

## Layout

| Path | Contents | Size |
| --- | --- | --- |
| `videos/<P0x>/<video>.mp4` | 156 RGB videos from the Aria camera, H.264, 1408 x 1408 fisheye, 30 fps | 116 GB |
| `framewise/<P0x>/<video>/framewise_info.jsonl` | one line per video frame: `T_world_device` (3 x 4 pose), `gaze_centre_in_pixels`, `gaze_direction_in_world`, timestamps | 2.2 GB |
| `framewise/<P0x>/<video>/device_calibration.json` | camera and sensor calibration of that recording (`camera-rgb`: fisheye model, `T_device_camera`) | |
| `annotations/` | the HD-EPIC annotations repo (below) | |
| `actions/<video>.txt` | numbered narrations per video, written by `export-actions` for graph annotation | |

154 of the 156 videos have gaze and pose; P02-20240210-113925 and P02-20240211-182630 have none released.

Derived data, also in `data/`: `frames448/` (every video at 2 fps, 448 px JPEG, 22 GB, from
`helper.perception.frames`) and `episodes/hd_epic/` (one viewer episode per video, described in the code README).

## Annotations (`annotations/`)

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

The annotations repo's own `annotations/README.md` documents every field.
