"""Dataset-agnostic episode format read by the viewer.

An episode is a folder `<root>/<dataset>/<episode_id>/` holding:
  video.mp4     symlink to the egocentric RGB stream
  view_<name>.mp4  symlinks to other cameras of the same recording, played in sync (optional)
  episode.json  metadata, the dataset's steps (nodes, optional edges) and their segments, narrated actions,
                further label tracks, boxes on the video, and static info
  gaze.json     {"fps": float, "t0": float, "points": [[x, y] | null, ...]}, x and y normalised to [0, 1]

An episode either covers a whole video (`clip` is None) or one clip of it, such as one subgoal.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Node:
    id: str
    label: str
    group: str = ""  # e.g. recipe name; the viewer lays out each group separately


@dataclass
class Segment:
    node: str
    start: float
    end: float
    kind: str = "step"  # "step" means the node is being executed, "prep" means preparing for it
    detail: dict = field(default_factory=dict)  # extra labels of this occurrence, shown on hover


@dataclass
class Action:
    start: float
    end: float
    label: str
    detail: dict = field(default_factory=dict)  # e.g. verbs, nouns, hands


@dataclass
class Track:
    """One labelled timeline row. Items are {"start", "end", "label", "detail"}; end == start is an instant."""
    name: str
    items: list[dict]
    source: str = ""  # which annotation file the labels come from


@dataclass
class Box:
    t: float  # seconds
    box: list[float]  # [x1, y1, x2, y2] in [0, 1] of the frame
    label: str


@dataclass
class Episode:
    dataset: str
    id: str
    title: str
    nodes: list[Node]
    edges: list[tuple[str, str]]  # (u, v): v requires u
    segments: list[Segment]
    actions: list[Action] = field(default_factory=list)
    done_before: list[str] = field(default_factory=list)  # nodes completed before this video starts
    graph_source: str = ""  # one line on where the edges come from
    edge_notes: list[str] = field(default_factory=list)  # optional reason per edge, parallel to edges
    clip: tuple[float, float] | None = None  # playback window in video seconds
    video_id: str = ""  # source video, for grouping episodes
    task: str = ""  # task the episode belongs to, shown as context
    node_kind: str = "subgoal"  # what a node is: "subgoal" or "action"
    stats: dict = field(default_factory=dict)  # per-episode measurements, e.g. head displacement
    tracks: list[Track] = field(default_factory=list)
    boxes: list[Box] = field(default_factory=list)
    info: dict = field(default_factory=dict)  # static facts: {title: text or [lines]}
    views: list[str] = field(default_factory=list)  # names of the extra cameras (view_<name>.mp4)


def _ffprobe() -> str:
    local = Path(sys.executable).with_name("ffprobe")  # conda env binary, even when the env is not activated
    return str(local) if local.exists() else shutil.which("ffprobe") or "ffprobe"


def probe_video(path: Path) -> dict:
    out = subprocess.run(
        [_ffprobe(), "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate,codec_name:format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    info = json.loads(out.stdout)
    s = info["streams"][0]
    num, den = map(int, s["avg_frame_rate"].split("/"))
    return {"width": s["width"], "height": s["height"], "fps": num / den,
            "duration": float(info["format"]["duration"]), "codec": s["codec_name"]}


def write_episode(root: Path, ep: Episode, video: Path, gaze: list, gaze_stride: int = 1,
                  views: dict[str, Path] | None = None) -> Path:
    """Writes the episode folder; gaze row i belongs to video frame i * gaze_stride. `views` maps a camera
    name to another video of the same recording, time-aligned with `video`."""
    out = Path(root) / ep.dataset / ep.id
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("view_*.mp4"):
        old.unlink()
    ep.views = sorted(views or {})
    for name, path in [("video.mp4", video)] + [(f"view_{k}.mp4", v) for k, v in (views or {}).items()]:
        link = out / name
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(Path(path).resolve(), link)
    info = probe_video(video)
    (out / "episode.json").write_text(json.dumps(asdict(ep) | {"video": info}, indent=1))
    fps, t0 = info["fps"] / gaze_stride, 0.0
    if ep.clip:  # keep only the gaze inside the clip
        first = max(0, int(ep.clip[0] * fps))
        gaze, t0 = gaze[first: int(ep.clip[1] * fps) + 2], first / fps
    (out / "gaze.json").write_text(json.dumps({"fps": fps, "t0": t0, "points": gaze}, separators=(",", ":")))
    return out
