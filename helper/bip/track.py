"""Turns observations into S1 events through the scene graph (scene.py) and the bounded actions (actions.py).

Observations, in time order:
    view(t, triples)          detections from the person's ego view or the robot's view
    act(t, agent, action)     a recognized action of the person ("H") or the robot's own action ("R")
A person's action becomes an S1 action event with L[v] = P(a | g = v, s) on the scene before the action. A
subgoal whose predicate starts to hold becomes a completion event (followed by "the person moves on") when
the person did it, or a robot_done event when the robot's own Place did.
"""

from __future__ import annotations

import numpy as np

from .actions import OBSERVED, likelihood
from .config import Config
from .scene import Scene


class Tracker:
    def __init__(self, cfg: Config, st: dict):
        self.cfg, self.nodes = cfg, st["nodes"]
        self.specs = [st["spec"][v] for v in self.nodes]
        self.scene = Scene(st["furniture"])
        if cfg.premap:  # the map from a mapping run before the task
            self.scene.see(st.get("map", []), st.get("t0", 0.0))
        self.objects = {o for sp in self.specs for o in self.scene.candidates(sp)} | {sp["ref"] for sp in self.specs if sp.get("ref")}
        self.objects |= {a for a, rel, b in st.get("map", []) if rel in ("on", "in")} if cfg.premap else set()
        self.done: set[int] = {v for v, sp in enumerate(self.specs) if self.scene.holds(sp)}

    def _completions(self, t: float, robot: bool) -> list[tuple]:
        now = {v for v, sp in enumerate(self.specs) if self.scene.holds(sp)}
        out = []
        for v in sorted(now - self.done):
            out += [(t, 0, "robot_done", v)] if robot else [(t, 0, "completion", v), (t, 0.5, "action", np.ones(len(self.nodes) + 1))]
        self.done = now
        return out

    def view(self, t: float, triples) -> list[tuple]:
        self.scene.see(triples, t)
        self.objects |= {a for a, rel, b in triples if rel != "holding"} | {b for a, rel, b in triples if rel in ("holding", "next_to")}
        return self._completions(t, robot=False)

    def act(self, t: float, agent: str, action: tuple) -> list[tuple]:
        out = []
        if agent == "H" and action[0] in OBSERVED:
            out.append((t, 1, "action", likelihood(action, self.specs, self.scene, self.objects, self.cfg.eps_act)))
        self.scene.act(agent, action, t)
        return out + self._completions(t, robot=agent == "R")


def events(cfg: Config, st: dict) -> list[tuple]:
    """S1 events of a recorded stream's views and acts, in time order (offline replay)."""
    tr = Tracker(cfg, st)
    obs = [(v["t"], 0, "view", v) for v in st["views"]] + [(a["t"], 1, "act", a) for a in st["acts"]]
    out = []
    for t, _, kind, x in sorted(obs, key=lambda e: (e[0], e[1])):
        out += tr.view(t, x["triples"]) if kind == "view" else tr.act(t, x["actor"], tuple(x["action"]))
    return [e if len(e) == 5 else e + (None,) for e in out]
