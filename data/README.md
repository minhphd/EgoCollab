# data/

Datasets, caches and derived files. Nothing here is in git except these README files; the code that
reads and writes each folder lives in `helper/`.

| Folder | Size | What | Made by |
| --- | --- | --- | --- |
| [hd_epic/](hd_epic/README.md) | 118 GB | HD-EPIC videos, per-frame gaze and pose, the HD-EPIC annotations | `python -m helper.datasets.hd_epic download` |
| [ego_exo4d/](ego_exo4d/README.md) | 81 GB | Ego-Exo4D cooking keystep takes with gaze, metadata, keystep annotations | `python -m helper.datasets.ego_exo4d download` |
| [industreal/](industreal/README.md) | 55 GB | IndustReal: 84 recordings with HoloLens 2 gaze and hands, labels, and MP4s of the RGB frames | `helper/slurm/industreal.sbatch` |
| [partnr/](partnr/README.md) | 26 GB | PARTNR scenes, objects, tasks, skill checkpoints and human traces | `helper/partnr/download.sh` |
| `episodes/` | | viewer episodes with every label each dataset provides: `hd_epic/` (156, one per video), `ego_exo4d/` (341, with exo views), `industreal/` (84), `partnr/` (221, recorded PARTNR runs) | each dataset's `build`; `helper.partnr.export` |
| `episodes_annotated/` | 690 MB | earlier episodes built from our post-hoc subgoal graphs: `hd_epic_subgoals/` (1,225, one per subgoal), `hd_epic_task/` (156, one per video, with BIP estimates); not shown in the viewer | the `backup` branch |
| `frames448/` | 22 GB | JPEG frame cache at 2 fps, 448 px: every HD-EPIC video and Ego-Exo4D take (`<video>.jpg.npy`, `<video>.end.npy`) | `python -m helper.perception.frames` |
| `bip/` | 1.3 GB | outputs of the BIP experiments (`backup` branch): observation streams, Gemini priors and object vocabularies, YOLO detections at 2 and 10 fps, hand contacts and fixations | `backup` branch, `helper/bip/` |
| `frames224/` | 42 GB | frame arrays at 224 px from the archived stage-1 human-model pipeline | archived code |
| `features/` | 36 GB | DINOv2 frame features and text label embeddings from the same archived pipeline | archived code |
| `motion/` | 144 KB | head-motion statistics from the archived prompted-VLM tracker | archived code |

The archived code is in `~/archive/EgoCollab-before-streamline-2026-10-11.tar.gz`. CHPC scratch deletes
files by date after about 60 days; unpack archives with `unzip -DD` so files get the current date.
