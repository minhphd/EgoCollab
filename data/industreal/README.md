# data/industreal

IndustReal: 84 recordings of 27 participants assembling or maintaining a construction-toy car, filmed
with a HoloLens 2. Code and how to use it: [helper/datasets/industreal](../../helper/datasets/industreal/README.md).

- Paper: Schoonbeek et al., WACV 2024; data: https://data.4tu.nl/datasets/b008dd74-020d-4ea4-a8ba-7bb60769d224
- Made by `helper/slurm/industreal.sbatch`, which downloads the archives in `files.txt` and unpacks each
  with `unzip -DD` (current file dates, so scratch does not purge them early).

## On disk

55 GB in all.

- `files.txt`: the 11 archives, one `name url` per line; `<archive>.done` marks each unpacked one (all 11).
- `<recording>/` (84 folders, such as `01_assy_0_1`): the unpacked recordings described below.
- `videos/`: the RGB frames of every recording encoded to MP4 by `python -m helper.datasets.industreal build`.

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

