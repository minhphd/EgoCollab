# EgoCollab: egocentric datasets and tools

Three egocentric datasets with eye gaze and task structure, their annotations, and reusable tools: a
Gemini client, sentence embeddings, YOLO object and hand detection, hand contacts and gaze fixations,
a frame cache, and an episode viewer. The `backup` branch keeps the fast-slow Bayesian inverse
planning experiments built on these tools.

## Layout

```
helper/
  datasets/
    episode.py          shared viewer episode format and graph utilities
    sharepoint.py       anonymous OneDrive client used by the HD-EPIC download
    hd_epic/            HD-EPIC: download, build, loaders, post-hoc subgoal graphs (annotations/), ground truth (tasks.py), 3D gaze fixations (gaze3d.py)
    ego_exo4d/          Ego-Exo4D cooking keystep takes: download, build, loaders
    industreal/         IndustReal assembly recordings: build
  llm/
    gemini.py           Gemini on Vertex AI (credentials from a .env)
    vocab.py            object names per subgoal from a task text, for detector prompts
    embed.py            all-MiniLM-L6-v2 sentence embeddings
  perception/
    frames.py           JPEG frame cache, video decoding, frames with the gaze as a red glow
    detect.py           YOLOE open-vocabulary objects and a YOLO hand detector
    contacts.py         hand-object contacts and gaze fixations from detections, causal
  viewer/               local web viewer for episodes (video, gaze, timelines, task graph)
  slurm/                cpu.sbatch and the dataset downloads
  environment.yml
data/                   datasets, frame cache, viewer episodes (not in git)
models/                 hf/ (Hugging Face cache), yolo/ (detector weights) (not in git)
runs/, logs/            experiment outputs (not in git)
```

Every dataset folder has a README: what the dataset is, what is on disk, the annotation formats, the
code, and how to use it.

| Dataset | What | On disk | README |
| --- | --- | --- | --- |
| HD-EPIC | 156 unscripted kitchen videos, Aria gaze and pose, 59,454 narrated actions, object movements, our post-hoc subgoal graphs | 118 GB | [helper/datasets/hd_epic](helper/datasets/hd_epic/README.md) |
| Ego-Exo4D | 341 cooking takes with keysteps and Aria gaze, ego at 448 px | 81 GB | [helper/datasets/ego_exo4d](helper/datasets/ego_exo4d/README.md) |
| IndustReal | 84 toy-car assembly recordings, HoloLens 2 gaze and hands, procedure-step labels | download pending | [helper/datasets/industreal](helper/datasets/industreal/README.md) |

## Setup

```bash
conda env create -f helper/environment.yml     # in home, never on scratch
conda activate egocollab
```

- **CHPC:** run everything on a compute node, never on a login node:
  `srun --jobid=<allocation> --overlap bash -c "cd $PWD; python -m ..."`. Scratch deletes files by date
  after about 60 days, so keep the conda environment in home. A file edited on the login node can read
  as its old version on a compute node for a few seconds; check before launching.
- **Hugging Face:** `export HF_HOME=$PWD/models/hf HF_HUB_OFFLINE=1` to load MiniLM from the local cache.
- **Gemini:** `helper/llm/gemini.py` reads a Vertex AI service account from `../MiniWeb-src/MiniWeb/.env`
  (`GOOGLE_CREDENTIALS_JSON`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`). Never print that file:
  the private key spans several lines. Models that answer: `gemini-3.5-flash` (default),
  `gemini-3.6-flash`, `gemini-3.1-pro-preview`.
- **YOLO weights** in `models/yolo/`: `yoloe-26l-seg.pt` (Ultralytics), `hand_yolov9c.pt`
  (Hugging Face `Bingsu/adetailer`) and the YOLOE text encoder `mobileclip2_b.ts`, fetched once.

## LLM tools (`helper/llm/`)

```python
from helper.llm.gemini import Gemini, generate
text, usage, latency = generate("Reply with {\"ok\": true}", schema={"type": "object"})
text, usage, latency = generate("What is in view?", images=[jpeg_bytes], cfg=Gemini(model="gemini-3.6-flash"))

from helper.llm.vocab import objects
objects("Dosa making", {"pan": "Pan hot and oiled", "pour": "Dosa batter poured"})
# {"pan": ["frying pan", "oil bottle", "spoon", "stove"], "pour": ["bowl", "frying pan", "ladle", "stove"]}

from helper.llm.embed import embed
E = embed(["pick up the spatula", "Pan scraped with a spatula"])   # unit vectors; E[0] @ E[1] = 0.65
```

`generate` takes an optional JSON schema (the reply then follows it), JPEG images, and retries
network errors with backoff.

## Perception tools (`helper/perception/`)

From a video to observations a live system would have:

```python
from helper.datasets.hd_epic import DATA, load_framewise
from helper.perception.frames import decode
from helper.perception.detect import Detector, run
from helper.perception.contacts import windows, fixations

vid = "P04-20240414-175337"
batches = decode(DATA / "videos" / vid[:3] / f"{vid}.mp4", start=0, end=60, fps=2)   # BGR, 448 px
det = run(Detector(), batches, ["frying pan", "spatula", "plate", "bowl"], fps=2, first=0)  # GPU
windows(det, t0=0, t1=60)              # every 3 s: objects a hand touched, objects seen
fixations(det, load_framewise(vid)[0], t0=0, t1=60)   # gaze held 0.3 s on one object box
```

- `frames.py`: `cache` stores a video as JPEGs at a fixed rate (`data/frames448` holds every HD-EPIC
  video at 2 fps); `Frames` reads them; `View` draws the gaze of the last second as a red glow and
  returns JPEGs for a VLM; `decode` streams frames from a video with ffmpeg.
- `detect.py`: YOLOE-26L finds objects named in the prompt list; a dedicated YOLOv9c detector finds
  hands, which YOLOE's "hand" prompt misses in egocentric views. Both together take 21.7 ms per frame on
  one RTX PRO 4000. CLI: `python -m helper.perception.detect --video <id> --names "a,b" --start 0 --end 60 --out det.json`.
- `contacts.py`: causal like a live system (a frame counts only `latency` s after it is taken). A
  hand touches an object when the hand box, grown by 15%, overlaps the object box; objects larger than
  a fifth of the frame never count. A fixation is gaze held 0.3 s inside one box (the smallest).

Measured on HD-EPIC, these cues are weak: the object in hand belongs to the current subgoal only 1.5
times as often as to another, and detecting at 10 fps instead of 2 changes little. They give no
signal for subgoal completion.

## Viewer

```bash
python -m helper.datasets.hd_epic build            # or ego_exo4d build, industreal build
python -m helper.viewer.server --port 8766         # http://127.0.0.1:8766
```

On a remote machine, forward the port: `ssh -L 8766:127.0.0.1:8766 <host>`. Pick a dataset, a video
or task, then an episode. Left: the ego video with the gaze point and a 0.6 s trail, subgoal and action
timelines. Right: the episode card and the graph coloured by status (done, active, ready, blocked). A
file `pred_<name>.json` next to an episode adds an inferred graph over time. Click a node or a timeline
to seek. Keys: space, arrows (5 s), `,` and `.` (one frame), `n` and `p` (next and previous episode).

## Episode format

An episode lives in `data/episodes/<dataset>/<id>/`:

- `video.mp4`: a symlink to the ego RGB video.
- `episode.json`: `nodes` (id, label, group), `edges` ([u, v]: v requires u), `segments` (node, start,
  end, kind), `actions`, `done_before`, `graph_source`, `edge_notes`, `clip`, `video_id`, `task`,
  `node_kind`, `stats`, `video` (size, fps, duration).
- `gaze.json`: `fps`, `t0` and `points`, one `[x, y]` in [0, 1] or `null` per sample.

A new dataset needs only a build script that writes this format with `episode.write_episode`.
