"""Ground truth per HD-EPIC video: the annotated subgoal graph and which subgoal is worked on when.

Built from the post-hoc task graphs in annotations/ next to this file. Every narrated action belongs to
at most one subgoal. A subgoal is done once its last action ends; the current subgoal at t is the one
whose action runs at t, with pauses inside one subgoal kept. These are our annotations, not HD-EPIC's,
so the viewer does not show them.

    from helper.datasets.hd_epic.tasks import load
    task = load(3)["P04-20240414-175337"]
"""

import json
import random
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from . import posthoc_graph
from .data import action_times, load_narrations


@dataclass
class Timeline:
    video: str
    task: str
    nodes: list[str]  # subgoal ids, in annotation order
    label: dict[str, str]
    edges: list[tuple[str, str]]  # (u, v): v needs u
    spans: list[tuple[float, float, str | None]]  # (start, end, owning subgoal) per action, by start

    def target(self, t: float, fill: bool = False) -> str | None:
        """Subgoal of the latest action running at t; with `fill`, a pause inside one subgoal keeps it."""
        running = [s for s in self.spans if s[0] <= t <= s[1]]
        if running or not fill:
            return running[-1][2] if running else None
        before = max((s for s in self.spans if s[1] < t), key=lambda s: s[1], default=None)
        after = min((s for s in self.spans if s[0] > t), key=lambda s: s[0], default=None)
        return before[2] if before and after and before[2] == after[2] else None

    def next_target(self, t: float, fill: bool = False) -> str | None:
        """The next different subgoal to start after t (None at the end of the task)."""
        now = self.target(t, fill)
        return next((s for a, _, s in self.spans if a > t and s and s != now), None)


@cache
def timelines() -> dict[str, Timeline]:
    narr, out = load_narrations(), {}
    for path in sorted(Path(posthoc_graph.ANNOTATIONS).glob("*.json")):
        g = json.loads(path.read_text())
        owner = posthoc_graph.action_owner(g)
        for vid in g["videos"]:
            times = action_times(narr, vid)
            if not times:
                continue
            spans = sorted((a, b, owner.get(f"{vid}-{i}")) for i, (a, b) in times.items())
            out[vid] = Timeline(vid, g["task"], [n["id"] for n in g["nodes"]], {n["id"]: n["label"] for n in g["nodes"]},
                                [(e["from"], e["to"]) for e in g["edges"]], spans)
    return out


@dataclass
class Task:
    video: str
    title: str
    nodes: list[str]  # subgoals with at least one action in the video
    label: dict[str, str]
    group: dict[str, str]
    edges: list[tuple[str, str]]  # required: (u, v) means u must be done before v starts
    first: dict[str, float]  # start of the first action of each subgoal
    last: dict[str, float]  # end of its last action
    timeline: Timeline

    @property
    def extent(self) -> tuple[float, float]:
        return min(self.first.values()), max(self.last.values())

    def done(self, t: float) -> set[str]:
        return {v for v in self.nodes if self.last[v] <= t}

    def current(self, t: float) -> str | None:
        return self.timeline.target(t, fill=True)

    def next(self, t: float) -> str | None:
        return self.timeline.next_target(t, fill=True)

    def order(self, t: float) -> list[str]:
        """Subgoals started by t, by first start."""
        return sorted((v for v in self.nodes if self.first[v] <= t), key=self.first.get)


@cache
def load(min_nodes: int = 1) -> dict[str, Task]:
    """Video -> task, in a fixed shuffled order (the first 30 with >= 3 subgoals are the dev split)."""
    out = {}
    for vid, tl in timelines().items():
        g = posthoc_graph.find(vid)
        first, last = {}, {}
        for a, b, v in tl.spans:
            if v:
                first[v], last[v] = min(first.get(v, a), a), max(last.get(v, b), b)
        nodes = [v for v in tl.nodes if v in first]
        if len(nodes) < min_nodes:
            continue
        group = {n["id"]: n.get("group", "") for n in g["nodes"]}
        edges = [(u, v) for u, v in tl.edges if u in first and v in first]
        out[vid] = Task(vid, tl.task, nodes, tl.label, group, edges, first, last, tl)
    keys = sorted(out)
    random.Random(0).shuffle(keys)
    return {k: out[k] for k in keys}


def clock(s: float) -> str:
    return f"{int(s) // 60}:{int(s) % 60:02d}"
