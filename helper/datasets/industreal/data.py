"""IndustReal (tabletop toy-car assembly and maintenance, HoloLens 2) as viewer episodes.

One episode per recording, from IndustReal's own labels: nodes are the procedure steps of PSR_labels.csv
(a step's segment runs from the previous step completion to its own) and actions are the AR_labels.csv
segments. IndustReal gives no dependencies between steps, so episodes have no edges. The video is encoded once
from the 10 fps RGB frames into <data>/videos/<recording>.mp4; times count from the first frame.

    python -m helper.datasets.industreal build [--shard 0/8]
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from ..episode import Action, Episode, Node, Segment, probe_video, write_episode

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data" / "industreal"
EPISODES = REPO / "data" / "episodes"
FPS = 10.0
TASK = {"assy": "Assemble the toy car", "main": "Maintain the toy car: replace the rear parts"}


def rows(path: Path) -> list[list[str]]:
    with open(path) as f:
        return [r for r in csv.reader(f) if r]


def recordings() -> dict[str, str]:
    """Recording -> its split (train, val or test), for recordings on disk."""
    out = {}
    for split in ("train", "val", "test"):
        out |= {r[0]: split for r in rows(DATA / f"{split}.csv") if (DATA / r[0] / "PSR_labels.csv").exists()}
    return out


def frame_numbers(rec: str) -> list[int]:
    return sorted(int(p.stem) for p in (DATA / rec / "rgb").glob("*.jpg"))


def encode(rec: str) -> Path:
    """The recording's frames as an H.264 video at FPS (frame order = file order)."""
    out = DATA / "videos" / f"{rec}.mp4"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.mp4")
        ffmpeg = Path(sys.executable).with_name("ffmpeg")
        subprocess.run([str(ffmpeg), "-v", "error", "-y", "-framerate", str(FPS), "-pattern_type", "glob",
                        "-i", str(DATA / rec / "rgb" / "*.jpg"), "-threads", "4", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "23", str(tmp)], check=True)
        tmp.rename(out)
    return out


def build_one(rec: str, kind_nodes: dict, labels: dict) -> Path:
    numbers = frame_numbers(rec)
    first, index = numbers[0], {n: i for i, n in enumerate(numbers)}
    t = lambda name: index.get(int(name.split(".")[0]), int(name.split(".")[0]) - first) / FPS
    kind = rec.split("_")[1]
    segments, prev = [], 0.0
    for r in rows(DATA / rec / "PSR_labels.csv"):
        segments.append(Segment(r[1], prev, t(r[0])))
        prev = max(prev, t(r[0]))
    actions = [Action(t(r[3]), t(r[4]), r[2].replace("_", " ")) for r in rows(DATA / rec / "AR_labels.csv")]
    gaze_at = {int(r[0].split(".")[0]): (float(r[1]), float(r[2])) for r in rows(DATA / rec / "gaze.csv")}
    video = encode(rec)
    info = probe_video(video)
    w, h = info["width"], info["height"]  # gaze.csv is in frame pixels, (0, 0) when missing
    gaze = []
    for n in numbers:
        g = gaze_at.get(n)
        inside = g and g != (0.0, 0.0) and 0 <= g[0] <= w and 0 <= g[1] <= h
        gaze.append([round(g[0] / w, 4), round(g[1] / h, 4)] if inside else None)
    ep = Episode("industreal", rec, TASK[kind], [Node(n, labels[n], TASK[kind]) for n in kind_nodes[kind]],
                 [], segments, actions, graph_source="IndustReal procedure steps (no dependencies are provided)",
                 video_id=rec, task=TASK[kind], node_kind="subgoal")
    return write_episode(EPISODES, ep, video, gaze)


def build(shard: str) -> None:
    recs = recordings()
    kind = lambda r: r.split("_")[1]
    steps = {r: [(x[1], x[2]) for x in rows(DATA / r / "PSR_labels.csv")] for r in recs}
    labels = {s: d for st in steps.values() for s, d in st}
    kind_nodes = {k: sorted({s for r in recs if kind(r) == k for s, _ in steps[r]}, key=int) for k in TASK}
    i, n = map(int, shard.split("/"))
    for rec in sorted(recs)[i::n]:
        print("built", build_one(rec, kind_nodes, labels), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["build"])
    ap.add_argument("--shard", default="0/1", help="i/n: every n-th recording starting at i")
    build(ap.parse_args().shard)


if __name__ == "__main__":
    main()
