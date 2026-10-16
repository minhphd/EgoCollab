"""The partner's beliefs about where things are: the robot's model of the person's own scene graph.

For each object o, b(o) is a distribution over the places the person may think o is (where they last registered it).
The robot's scene graph gives where o actually is. False belief: P(b(o) != actual(o)).

Generative model of the person:
    - their own actions they always register;
    - another agent's move of o they register only by attending o while it is in their view. Gaze is read as the entity
      it falls on, and its dwell accumulates as evidence: dwell on o registers o with 1 - exp(-dwell / tau_reg); o in
      view with the gaze elsewhere registers with hazard q_vis per second; never out of view. Attending o in the
      robot's hand within follow_s before it reaches its new place shows them where it went;
    - looking for o, their gaze goes to the believed place: the dwell on o's candidate places over the first look_s
      of looking at any of them is shared out, and each place's share is evidence with weight w_seek.
The first two update b forward in time (the fast path, every frame). The third is evidence about b at the start of
the search, applied once per search. When the fast path stays unsure while the person is about to look for o, the
slow path (a VLM on the person's view during the move) adds a likelihood for "they saw it".

    m = Mind(cfg)
    m.move(t, "mug_1", "in cabinet_3", actor="R")           # the robot's scene graph
    m.view(t, visible={"mug_1", "table_2"}, fixated="table_2")  # the person's view, every frame
    m.seek(t, "mug_1")                                     # S1: the person's next subgoal uses mug_1
    m.false_belief("mug_1"), m.believed("mug_1")
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MindConfig:
    q_fix: float = 0.9  # P(they follow o to its new place | they watched it go)
    tau_reg: float = 0.15  # s of dwell on o for 63% registration (0.3 s gives 86%)
    look_s: float = 0.5  # s of dwell on o's candidate places that make up the first look of a search
    q_vis: float = 0.02  # hazard per second of registering o in view while the gaze is elsewhere
    follow_s: float = 2.0  # attending o this soon before it reaches a new place shows where it went
    w_seek: float = 0.6  # the share of the first look that goes to the believed place, beyond chance
    gaze: bool = True  # without gaze: o in view during a move is registered with q_nogaze, once per move
    history_gaze: bool = True  # off: gaze is read only in the first look of a search
    q_nogaze: float = 0.5
    seek_evidence: bool = True  # use the first look of a search
    s2_band: tuple = (0.25, 0.75)  # call the slow path when P(false belief) falls in this band at a search
    min_fix: float = 0.0  # gaze on an entity counts once it stays there this long (a fixation filter for tracker noise)


def furniture_of(place: str | None) -> str | None:
    return place.split(" ", 1)[1] if place and " " in place else None


@dataclass
class Track:
    actual: str | None = None
    b: dict = field(default_factory=dict)  # believed place -> probability
    last_fix: float = -1e9  # last time the person's gaze fell on o
    seen_move: bool = False  # without gaze: o was in view since its last move
    seeking: bool = False
    looked: bool = False  # the first look of this search is used
    look: dict = field(default_factory=dict)  # place -> dwell during the first look


class Mind:
    def __init__(self, cfg: MindConfig | None = None):
        self.cfg, self.objs, self.t, self.log = cfg or MindConfig(), {}, None, []
        self.fix = (None, 0.0)  # the entity under the gaze and since when

    def _track(self, o: str) -> Track:
        return self.objs.setdefault(o, Track())

    def _register(self, tr: Track, q: float) -> None:
        """With probability q the person registers o where it is now."""
        if q <= 0 or tr.actual is None:
            return
        b = {p: (1 - q) * w for p, w in tr.b.items()}
        b[tr.actual] = b.get(tr.actual, 0.0) + q
        tr.b = b

    # -- inputs ---------------------------------------------------------------------------------------
    def move(self, t: float, o: str, place: str, actor: str) -> None:
        """o is now at place ("on f", "in f", or held: "robot", "person"), moved by actor ("H" the person, "R")."""
        tr, c = self._track(o), self.cfg
        first, prev = tr.actual is None, tr.actual
        tr.actual, tr.seen_move = place, False
        if actor == "H" or first or not tr.b:  # their own move, or o as the task starts: known
            tr.b = {place: 1.0}
        else:
            gaze = c.gaze and c.history_gaze
            watched = gaze and t - tr.last_fix <= c.follow_s
            held = tr.b.pop(prev, 0.0) if prev in ("robot", "person") else 0.0  # they knew it was in a hand
            if held:  # it left the hand: they follow it to its new place, or lose it
                f = c.q_fix if watched else (0.0 if gaze else c.q_nogaze)
                tr.b[place] = tr.b.get(place, 0.0) + f * held
                tr.b["unknown"] = tr.b.get("unknown", 0.0) + (1 - f) * held
            elif watched:  # watching o itself as it moves
                self._register(tr, c.q_fix)
        self.log.append((round(t, 2), "move", o, place, actor))

    def view(self, t: float, visible: set, fixated: str | None) -> None:
        """One frame of the person's view: the entities in view and the entity the gaze falls on."""
        dt = 0.0 if self.t is None else max(0.0, t - self.t)
        self.t = t
        c = self.cfg
        dwell = dt
        if c.min_fix > 0:
            if fixated != self.fix[0]:
                self.fix = (fixated, t)
            held = t - self.fix[1]
            dwell = held if held - dt < c.min_fix else dt  # once it becomes a fixation, all of its dwell counts
            fixated = fixated if held >= c.min_fix else None
        for o, tr in self.objs.items():
            g = c.gaze and (c.history_gaze or tr.seeking)
            if g and fixated == o:
                tr.last_fix = t
            if o not in visible:
                continue
            if g:
                self._register(tr, 1 - np.exp(-dwell / c.tau_reg) if fixated == o else 1 - np.exp(-c.q_vis * dt))
            elif not tr.seen_move:  # no gaze: in view during or after a move, registered once with q_nogaze
                tr.seen_move = True
                self._register(tr, c.q_nogaze)
        if c.gaze and c.seek_evidence and fixated:
            for o, tr in self.objs.items():
                if tr.seeking and not tr.looked:
                    self._first_look(tr, fixated, len(visible), dwell)

    def _first_look(self, tr: Track, fixated: str, n_visible: int, dt: float) -> None:
        """Accumulate dwell on o's candidate places while looking for o; after look_s of it, each place's share of the
        dwell is evidence about the believed place."""
        places = {furniture_of(p): p for p in set(tr.b) | {tr.actual} if furniture_of(p)}
        if fixated not in places:
            return
        tr.look[fixated] = tr.look.get(fixated, 0.0) + max(dt, 1e-3)
        total = sum(tr.look.values())
        if total < self.cfg.look_s:
            return
        tr.looked = True
        chance = 1.0 / max(1, n_visible)
        share = {f: w / total for f, w in tr.look.items()}
        L = {p: self.cfg.w_seek * share.get(furniture_of(p), 0.0) + (1 - self.cfg.w_seek) * chance for p in tr.b}
        z = sum(tr.b[p] * L[p] for p in tr.b)
        if z > 0:
            tr.b = {p: tr.b[p] * L[p] / z for p in tr.b}
        self.log.append((round(self.t or 0, 2), "first_look", dict(share)))

    def destination(self, o: str) -> dict:
        """Where the person will go for o: the believed places' furniture, by belief."""
        out = {}
        for p, w in self._track(o).b.items():
            f = furniture_of(p) or p
            out[f] = out.get(f, 0.0) + w
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def seek(self, t: float, o: str, on: bool = True) -> bool:
        """The person starts (or stops) looking for o. True when the slow path should judge o now."""
        tr = self._track(o)
        if on and not tr.seeking:
            tr.seeking, tr.looked, tr.look = True, False, {}
            lo, hi = self.cfg.s2_band
            return lo <= self.false_belief(o) <= hi
        tr.seeking = on
        return False

    def inform(self, t: float, o: str) -> None:
        """The robot told the person where o is: they now know."""
        tr = self._track(o)
        if tr.actual is not None:
            tr.b = {tr.actual: 1.0}
        self.log.append((round(t, 2), "told", o, tr.actual))

    def evidence(self, o: str, p_saw: float, weight: float = 1.0) -> None:
        """The slow path's judgement P(the person saw o reach its current place), as a likelihood."""
        tr = self._track(o)
        p = float(np.clip(p_saw, 0.02, 0.98)) ** weight
        q = (1 - float(np.clip(p_saw, 0.02, 0.98))) ** weight
        L = {x: (p if x == tr.actual else q) for x in tr.b}
        z = sum(tr.b[x] * L[x] for x in tr.b)
        tr.b = {x: tr.b[x] * L[x] / z for x in tr.b}

    # -- read-outs ------------------------------------------------------------------------------------
    def believed(self, o: str) -> dict:
        return dict(sorted(self._track(o).b.items(), key=lambda kv: -kv[1]))

    def false_belief(self, o: str) -> float:
        tr = self._track(o)
        return float(1.0 - tr.b.get(tr.actual, 0.0)) if tr.b else 0.0
