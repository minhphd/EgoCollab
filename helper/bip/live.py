"""BIP as a robot runs it: one object fed the inputs of io.py in time order, returning commands.

    run = Live(cfg, task)                     # task: nodes, labels, objects, spec, furniture, map (the premapped home)
    for x in source:                          # SceneObs, EgoFrame, Act, Speech, in time order
        for cmd in run.on(x):                 # Command: execute a subgoal, ask, or tell where something is
            robot.send(cmd)

The fast path runs inline: the scene graph (track.py), S1 (belief.py) on every event, the partner's beliefs
(mind.py) on every frame, and the decision once per decide_every s. The slow path runs in the background, so its
1 to 2 s calls never stall the loop: S2's graph revisions (belief.py) and its judgement of what the person saw
(s2.saw), which joins the partner's beliefs when it returns.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from . import s2
from .belief import Belief
from .config import Config
from .decide import Costs, decide, tell_value
from .io import Act, Command, EgoFrame, SceneObs, Speech, fixated
from .mind import Mind, MindConfig
from .track import Tracker


class Live:
    def __init__(self, cfg: Config, task: dict, mind: MindConfig | None = None, costs: Costs | None = None,
                 slow_model: str = "gemini-3.5-flash"):
        self.cfg, self.task, self.costs = cfg, task, costs or Costs()
        self.st = {"id": task.get("id", "live"), "t0": 0.0, **task}
        self.tracker, self.mind = Tracker(cfg, self.st), Mind(mind)
        self.belief = Belief(cfg, self.st) if task["nodes"] else None  # no subgoals: beliefs and telling only
        if self.belief:
            self.belief.feed(0.0, "action", np.ones(len(task["nodes"]) + 1))  # the person starts some allowed subgoal
        self.furniture = set(task["furniture"])
        self.holder, self.place = {}, {}  # object -> who last picked it up; object -> its place in the scene graph
        self.frames, self.slow_model = [], slow_model  # recent frames for the slow path
        self.pool, self.pending = ThreadPoolExecutor(2), {}  # object -> Future of the slow path's judgement
        self.done_t, self.next_decide, self.told, self.expected, self.busy = [], 0.0, {}, {}, None

    # -- inputs ---------------------------------------------------------------------------------------
    def on(self, x) -> list[Command]:
        t = x.t
        if isinstance(x, SceneObs):
            self._events(self.tracker.view(t, x.triples))
            for a, rel, b in x.triples:
                if rel in ("on", "in"):
                    self._place(t, a, f"{rel} {b}")
                elif rel == "holding":
                    self.holder[b] = a
                    self._place(t, b, {"H": "person", "R": "robot"}.get(a, a))
        elif isinstance(x, Act):
            self._events(self.tracker.act(t, x.agent, (x.verb, *x.args)))
            if x.verb == "Pick":
                self.holder[x.args[0]] = x.agent
                self._place(t, x.args[0], "person" if x.agent == "H" else "robot")
            elif x.verb == "Place":
                self._place(t, x.args[0], f"{x.args[1]} {x.args[2]}")
        elif isinstance(x, EgoFrame):
            fix = x.fixated if x.fixated is not None else fixated(x.gaze, x.visible, self.furniture)
            if fix is not None and self.cfg.gaze and self.belief:
                self._events(self._fixation(t, fix, set(x.visible)))
            self.mind.view(t, set(x.visible), fix)
            if x.image is not None:
                self.frames = [f for f in self.frames if f[0] >= t - 60.0] + [(t, x.image)]
        elif isinstance(x, Speech):
            pass  # answers go to dialogue.py when the robot asks; the live loop keeps them for it
        self._join(t)
        return self._decide(t) if t >= self.next_decide else []

    def _place(self, t: float, o: str, place: str) -> None:
        if self.place.get(o) != place:
            self.place[o] = place
            self.mind.move(t, o, place, self.holder.get(o, "R"))

    def expect(self, o: str, p: float = 1.0) -> None:
        """A hint that the person will look for o soon (they asked for it, or the task says so), beside S1's."""
        self.expected[o] = p

    def _events(self, ev: list) -> None:
        for e in ev:
            if e[2] in ("completion", "robot_done"):
                self.done_t.append(e[0])
            if e[2] == "robot_done" and self.busy == e[3]:
                self.busy = None
            if self.belief:
                self.belief.feed(e[0], e[2], e[3])

    def _fixation(self, t: float, x: str, visible: set) -> list:
        O = [set(self.task["objects"][v]) for v in self.task["nodes"]]
        u = np.array([float(x in o) / max(1, len(o & (visible | {x}))) for o in O])
        return [(t, 2, "fixation", (u, 1 / max(1, len(visible))))]

    # -- the slow path for beliefs ------------------------------------------------------------------------
    def _join(self, t: float) -> None:
        for o, fut in list(self.pending.items()):
            if fut.done():
                p, usage, latency = fut.result()
                self.mind.evidence(o, p)
                del self.pending[o]

    def _ask_slow(self, t: float, o: str) -> None:
        moved = [e for e in self.mind.log if e[1] == "move" and e[2] == o and e[4] != "H"]
        if not moved or o in self.pending:
            return
        t0, t1 = moved[0][0] - 1.0, moved[-1][0] + 1.0
        imgs = [img for tt, img in self.frames if t0 <= tt <= t1][::2]
        log = [f"t = {e[0]:.1f} s: the robot moved the {o}; it is now {e[3]}" for e in moved]
        self.pending[o] = self.pool.submit(s2.saw, log, o, moved[-1][3], imgs, self.slow_model)

    # -- decision -------------------------------------------------------------------------------------------
    def seeking(self) -> dict:
        """P(the person looks for o soon): their current or planned next subgoal uses o."""
        out = {}
        if not self.belief:
            return dict(self.expected)
        F, nodes = self.belief.F, self.task["nodes"]
        p = F.marginal(F.g) + F.marginal(F.n)
        for v, name in enumerate(nodes):
            for o in self.task["objects"][name]:
                if o not in self.furniture:
                    out[o] = min(1.0, out.get(o, 0.0) + float(p[v]))
        return {o: max(out.get(o, 0.0), self.expected.get(o, 0.0)) for o in set(out) | set(self.expected)}

    def _decide(self, t: float) -> list[Command]:
        self.next_decide = t + self.cfg.decide_every
        cmds = []
        for o, p in self.seeking().items():
            if p >= 0.5 and self.mind.seek(t, o) and self.slow_model:
                self._ask_slow(t, o)
            elif p < 0.5:
                self.mind.seek(t, o, on=False)
            v = tell_value(self.mind.false_belief(o), p, self.costs)
            if v > 0 and self.told.get(o) != self.place.get(o):
                self.told[o] = self.place.get(o)
                cmds.append(Command(t, "tell", o, f"The {o} is {self.place.get(o)}.", {"value": round(v, 2)}))
                self.mind.inform(t, o)  # once told, they know
        if self.busy is not None or not self.belief:  # one subgoal at a time; done when the scene graph shows it
            return cmds
        gaps = np.diff([0.0] + sorted(self.done_t))
        d = np.full(len(self.task["nodes"]), float(np.mean(gaps)) if len(gaps) else 20.0)
        kind, x, info = decide(self.belief.F, d, d / 2, self.costs, "never")
        if kind == "execute":
            self.busy = x
            self.belief.feed(t, "claimed", self.belief.F.claimed | (1 << x))
            cmds.append(Command(t, "execute", self.task["nodes"][x], info={"value": round(info["value"], 2)}))
        return cmds
