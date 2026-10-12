"""Dataset-agnostic episode format read by the viewer.

An episode is a folder `<root>/<dataset>/<episode_id>/` holding:
  video.mp4     symlink to the egocentric RGB stream
  episode.json  metadata, task graph, node segments and narrated actions
  gaze.json     {"fps": float, "t0": float, "points": [[x, y] | null, ...]}, x and y normalised to [0, 1]

An episode either covers a whole video (`clip` is None) or one clip of it, such as one subgoal.
"""

import itertools
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


@dataclass
class Action:
    start: float
    end: float
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


def transitive_reduction(nodes: list[str], edges: set[tuple[str, str]]) -> list[tuple[str, str]]:
    """Drops every edge (u, v) that is implied by a longer path u -> ... -> v."""
    succ = {n: {v for u, v in edges if u == n} for n in nodes}
    reach = {n: set() for n in nodes}
    for n in nodes:
        stack = list(succ[n])
        while stack:
            m = stack.pop()
            if m not in reach[n]:
                reach[n].add(m)
                stack.extend(succ[m])
    return sorted((u, v) for u, v in edges if not any(v in reach[w] for w in succ[u] if w != v))


def precedence_dag(orders: list[list[str]], nodes: list[str], min_support: int = 1) -> list[tuple[str, str]]:
    """Partial order from demonstrations: u -> v when u starts before v in every order containing both."""
    edges = set()
    for u, v in itertools.permutations(nodes, 2):
        both = [o for o in orders if u in o and v in o]
        if len(both) >= min_support and all(o.index(u) < o.index(v) for o in both):
            edges.add((u, v))
    return transitive_reduction(nodes, edges)


def write_episode(root: Path, ep: Episode, video: Path, gaze: list, gaze_stride: int = 1) -> Path:
    """Writes the episode folder; gaze row i belongs to video frame i * gaze_stride."""
    out = Path(root) / ep.dataset / ep.id
    out.mkdir(parents=True, exist_ok=True)
    link = out / "video.mp4"
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink(Path(video).resolve(), link)
    info = probe_video(video)
    (out / "episode.json").write_text(json.dumps(asdict(ep) | {"video": info}, indent=1))
    fps, t0 = info["fps"] / gaze_stride, 0.0
    if ep.clip:  # keep only the gaze inside the clip
        first = max(0, int(ep.clip[0] * fps))
        gaze, t0 = gaze[first: int(ep.clip[1] * fps) + 2], first / fps
    (out / "gaze.json").write_text(json.dumps({"fps": fps, "t0": t0, "points": gaze}, separators=(",", ":")))
    return out
