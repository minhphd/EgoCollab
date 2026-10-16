"""The scripted person: PARTNR's, calibrated on its people (5,041 human-in-the-loop episodes), in a BEHAVIOR home.

One step at a time, each as: go to the object, pick it, go to the target, [open it], place it, [close it].
    choice   P(v) prop. exp(-0.4 rank(v) - 4.4 dist(v) - 4.4 max(0, 1.5 - dist(v, where the robot was last seen))),
             over the steps not believed done whose required steps are believed done and not left to the robot
    pause    0.1 x the last step's work time (log-normal) before each step
    beliefs  where each object is (where last seen) and which steps are done (seen in place, done, or told so)
    robot    a step the robot is known to be on is left to it 2 s later, w.p. 0.9 (at once if it said so); an object
             gone from where it was believed means the robot has it: that step is left for 20 s; a step left to the
             robot for 90 s is taken back
    give up  after 2 failures on a step (forgiven every 30 s)
    way      around a robot in the path if that adds at most 2.5 m, else through it
Gaze: HD-EPIC statistics (Gaze).

    p = Person(task, world, seed)
    p.see(t, visible); p.act(t, claims, robot_xy); p.hear(t, kind, step)
    p.actions   what an observer could read every 3 s of a running skill: {t, verb, objects, visible, step}
"""

from __future__ import annotations

import numpy as np

from helper.behavior.sim import EYE

REACH_S, BACK_S, OPEN_S = 0.5, 0.4, 0.6  # s
DT = 1 / 15


class Gaze:
    """Fixations calibrated on HD-EPIC: gaze dwells on a target for a log-normal time (median 0.6 s), then picks the next:
    the object of an upcoming pick (65% of picks, 3.1 s ahead) or the target of a place (32%, 1.8 s ahead); else the
    current step's objects w.p. 0.3, the next step's 0.1, else anything in view. A fixation counts after 0.3 s."""

    def __init__(self, rng):
        self.rng, self.target, self.until, self.since, self.reported, self.primes = rng, None, -1.0, 0.0, False, []

    def plan(self, t_event: float, entity: str, pick: bool) -> None:
        p, lead = (0.65, 3.1) if pick else (0.32, 1.8)
        if self.rng.random() < p:
            self.primes.append((t_event - lead * np.exp(0.8 * self.rng.standard_normal()), t_event, entity))

    def step(self, t: float, masks: dict, cur: set, nxt: set) -> tuple[list[float], dict | None]:
        """The gaze point (image coordinates) and a fixation {t, object, visible} when one completes now."""
        primed = [e for a, b, e in self.primes if a <= t < b and e in masks]
        if t >= self.until or self.target not in masks or (primed and primed[0] != self.target):
            new = primed[0] if primed else None
            if new is None:
                r = self.rng.random()
                for pool, w in ((cur, 0.3), (nxt, 0.1)):
                    vis = sorted(pool & masks.keys())
                    if new is None and r < w and vis:
                        new = vis[self.rng.integers(len(vis))]
                    r -= w
                if new is None and masks:
                    new = sorted(masks)[self.rng.integers(len(masks))]
            if new != self.target:
                self.target, self.since, self.reported = new, t, False
            self.until = t + 0.6 * np.exp(0.5 * self.rng.standard_normal())
        cx, cy = (masks[self.target][:2] if self.target in masks else (0.5, 0.45))
        xy = [float(np.clip(cx + 0.02 * self.rng.standard_normal(), 0, 1)), float(np.clip(cy + 0.02 * self.rng.standard_normal(), 0, 1))]
        fix = None
        if self.target is not None and not self.reported and t - self.since >= 0.3:
            fix, self.reported = {"t": round(t, 2), "object": self.target, "visible": sorted(masks)}, True
        return xy, fix


class Person:
    def __init__(self, task: dict, world, seed: int):
        self.task, self.w, self.rng = task, world, np.random.default_rng([seed, 1])
        self.gaze = Gaze(np.random.default_rng([seed, 2]))
        self.steps = task["steps"]
        N = len(self.steps)
        self.req = {v: {u for u, x in task["orderings"] if x == v} for v in range(N)}
        self.where = {o: world.pos(o) for s in self.steps for o in (s["obj"],)}  # believed positions
        self.done, self.left, self.yields = set(), {}, {}  # believed done; step -> left to the robot until; decisions
        self.waiting = {}  # step -> when the person first left it to the robot
        self.fails, self.forgiven = [0] * N, 0.0
        self.cur, self.queue, self.run, self.wait_until, self.last_work, self.seg_start = None, [], None, 0.0, 15.0, 0.0
        self.robot_seen, self.robot_t = None, -9.0  # where and when the person last saw the robot
        self.picked_from = None  # the furniture the held object came off
        self.actions, self.log, self.segments = [], [], []
        self.completed = []  # pick, place, open and close done: what a clip recognizer would see end

    # -- what the person sees and hears ------------------------------------------------------------------------------
    def see(self, t: float, vis: dict) -> None:
        for s in self.steps:
            if s["obj"] in vis:
                self.where[s["obj"]] = self.w.pos(s["obj"])
        for v, s in enumerate(self.steps):
            if s["obj"] in vis and s["target"] in vis and s["obj"] not in self.w.held.values():
                (self.done.add if self.w.relation(s["obj"], s["rel"], s["target"]) else self.done.discard)(v)
        if "robot" in vis:
            self.robot_seen, self.robot_t = self.w.bot.xy.copy(), t

    def hear(self, t: float, kind: str, v: int) -> str | None:
        """The robot's words: announce/ask/have/claim mean it takes v; put means v is done. The answer to a question."""
        if kind == "put":
            self.done.add(v)
            self.where[self.steps[v]["obj"]] = self.w.pos(self.steps[v]["obj"])
            return None
        if kind == "ask" and self.cur == v:
            return "No, I'm doing that."
        self.left[v] = np.inf  # heard: left to the robot while it is on it
        return "Yes, please." if kind == "ask" else None

    def knows_robot_on(self, t: float, claims: dict) -> set:
        """Steps left to the robot now: one it said it takes (while it is on it), one whose object was gone (20 s), and
        one it is seen on (the robot in view within 3 s), 2 s after noticing, w.p. 0.9; none left to it for 90 s
        already (the robot is not getting it done: the person takes it back)."""
        out = {v for v, until in self.left.items() if (until == np.inf and v in claims) or (until != np.inf and until > t)}
        if t - self.robot_t < 3.0:
            for v, since in claims.items():
                if t - since >= 2.0:
                    if (v, since) not in self.yields:
                        self.yields[(v, since)] = bool(self.rng.random() < 0.9)
                    if self.yields[(v, since)]:
                        out.add(v)
        return {v for v in out if t - self.waiting.setdefault(v, t) < 90.0}

    # -- choosing ------------------------------------------------------------------------------------------------------
    def options(self, t: float, robot_on: set) -> list[int]:
        return [v for v in range(len(self.steps)) if v not in self.done and v not in robot_on and self.req[v] <= self.done
                and self.fails[v] < 2]

    def choose(self, options: list[int]) -> int:
        here = self.w.person_xy
        score = [-0.4 * v - 4.4 * float(np.linalg.norm(self.where[self.steps[v]["obj"]][:2] - here))
                 - (0.0 if self.robot_seen is None else
                    4.4 * max(0.0, 1.5 - float(np.linalg.norm(self.where[self.steps[v]["obj"]][:2] - self.robot_seen))))
                 for v in options]
        p = np.exp(np.array(score) - max(score))
        return options[self.rng.choice(len(options), p=p / p.sum())]

    # -- acting --------------------------------------------------------------------------------------------------------
    def act(self, t: float, claims: dict, robot_xy) -> None:
        """One frame: drop a step that is done or left to the robot (between skills), choose one, run its skills."""
        if t - self.forgiven >= 30.0:
            self.fails, self.forgiven = [min(f, 1) for f in self.fails], t
        robot_on = self.knows_robot_on(t, claims)
        between = self.run is None or self.run[0] == "go"
        if self.cur is not None and between and self.w.held["person"] is None and (self.cur in self.done or self.cur in robot_on):
            self._end(t, "done" if self.cur in self.done else "dropped")
        if self.cur is None:
            options = self.options(t, robot_on)
            if not options:  # nothing left that they know of: they look around (and see what is not in place)
                self.w.person_yaw += np.radians(40) * DT
                self.w.head = self._ahead()
                return
            self._start(t, self.choose(options))
        if t < self.wait_until:
            return
        if self.run is None:
            if not self.queue:
                self.done.add(self.cur)
                self._end(t, "finished")
                return
            self.run = (*self.queue.pop(0), {"t": 0.0, "read": -9.0})
        tool, arg, st = self.run
        st["t"] += DT
        if t - st["read"] >= 3.0:  # what an observer could read of this skill now
            objs = [arg] if tool != "place" else [arg, self.steps[self.cur]["target"]]
            self.actions.append({"t": round(t, 2), "verb": tool, "objects": objs, "step": self.cur})
            st["read"] = t
        status = getattr(self, "_" + tool)(t, arg, st, robot_xy)
        if status:
            self.log.append(f"{t:.1f} {tool}({arg}) {status}")
            if status == "ok" and tool in ("pick", "place", "open", "close"):
                self.completed.append({"t": round(t, 2), "verb": tool, "obj": arg,
                                       "target": self.steps[self.cur]["target"] if tool == "place" else None})
            self.run = None
            if status != "ok":
                if status != "in place":
                    self.fails[self.cur] += 1
                self._end(t, status)

    def _start(self, t: float, v: int) -> None:
        s = self.steps[v]
        self.cur, self.seg_start = v, t
        self.queue = [("go", s["obj"]), ("pick", s["obj"]), ("go", s["target"])] + \
                     ([("open", s["target"])] if s.get("closed") else []) + [("place", s["obj"])] + \
                     ([("close", s["target"])] if s.get("closed") else [])
        self.wait_until = t + 0.1 * self.last_work * float(np.exp(0.5 * self.rng.standard_normal()))
        self.log.append(f"{t:.1f} start {v}")

    def _end(self, t: float, why: str) -> None:
        o = self.w.held["person"]
        if o is not None:  # nothing stays in hand: back where it came from, else on the floor in front
            if self.picked_from is None or not self.w.put(o, "ontop", self.picked_from):
                self.w.held["person"] = None
                self.w.obj(o).set_position_orientation(position=[*(self.w.person_xy + 0.4 * np.array(
                    [np.cos(self.w.person_yaw), np.sin(self.w.person_yaw)])), 0.1])
            self.where[o] = self.w.pos(o)  # they know where they put it
        self.segments.append([round(self.seg_start, 2), round(t, 2), self.cur])
        self.last_work = max(3.0, t - max(self.seg_start, self.wait_until))
        self.log.append(f"{t:.1f} {why} {self.cur}")
        self.cur, self.queue, self.run, self.w.reach = None, [], None, None

    # -- skills: each returns "" while running, else "ok" or why it failed -------------------------------------------
    def _ahead(self) -> np.ndarray:
        xy, yaw = self.w.person_xy, self.w.person_yaw
        return np.array([xy[0] + 2.5 * np.cos(yaw), xy[1] + 2.5 * np.sin(yaw), max(1.0, self.w.eye[2] - 0.5)])

    def _go(self, t: float, x: str, st: dict, robot_xy) -> str:
        w, xy = self.w, self.w.person_xy
        if "path" not in st:
            st["at"] = self.where.get(x, w.top(x))  # where they believe it is
            under = w.support(st["at"], skip=self.where.keys()) if x in self.where else x
            point = st["at"] if x in self.where else None  # an object: stand beside it
            spot = w.stand_spot(under or x, xy, away=robot_xy, point=point)
            if spot is None:  # tighter
                spot = w.stand_spot(under or x, xy, gap=0.3, wall=0.25, point=point)
            if spot is None:  # anywhere by its furniture
                spot = w.stand_spot(under or x, xy, gap=0.3, wall=0.25)
            way = w.path(xy, spot) if spot is not None else None
            if way is None:
                return "cannot get near" if spot is None else "no way there"
            st["path"], st["around"] = [np.asarray(p, float) for p in way[1:]], None
        while st["path"] and np.linalg.norm(st["path"][0] - w.person_xy) < 0.05:
            st["path"].pop(0)
        if st["path"]:
            self._around(st, robot_xy)
            self._walk(st["path"][0])
            w.head = self._ahead()
            return ""
        to = np.asarray(st["at"], float)[:2] - w.person_xy
        want = float(np.arctan2(to[1], to[0]))
        d = (want - w.person_yaw + np.pi) % (2 * np.pi) - np.pi
        w.head = np.array([*np.asarray(st["at"], float)[:2], max(0.3, float(np.asarray(st["at"])[2]))])
        if abs(d) > np.radians(5):
            w.person_yaw += float(np.clip(d, -np.radians(200) * DT, np.radians(200) * DT))
            return ""
        return "ok"

    def _around(self, st: dict, robot_xy) -> None:
        """The robot within 0.5 m of the next 1.5 m: a way around it, if that adds at most 2.5 m."""
        if robot_xy is None or (st["around"] is not None and np.linalg.norm(robot_xy - st["around"]) < 0.3):
            return
        pts, ahead, d = [self.w.person_xy] + st["path"], [], 0.0
        for a, b in zip(pts, pts[1:]):
            for s in np.linspace(0, 1, 6):
                ahead.append(a + s * (b - a))
            d += float(np.linalg.norm(b - a))
            if d > 1.5:
                break
        if min(float(np.linalg.norm(p - robot_xy)) for p in ahead) < 0.5:
            st["around"] = np.asarray(robot_xy, float).copy()
            way = self.w.path(self.w.person_xy, st["path"][-1], avoid=[(robot_xy, 0.5)])
            length = lambda q: sum(float(np.linalg.norm(b - a)) for a, b in zip([self.w.person_xy] + list(q), list(q)))
            if way is not None and length(way[1:]) <= length(st["path"]) + 2.5:
                st["path"] = [np.asarray(p, float) for p in way[1:]]

    def _walk(self, nxt: np.ndarray) -> None:
        """One frame toward nxt: turning (200 deg/s) as they walk, slower off the heading."""
        w = self.w
        d = (float(np.arctan2(*(nxt - w.person_xy)[::-1])) - w.person_yaw + np.pi) % (2 * np.pi) - np.pi
        w.person_yaw += float(np.clip(d, -np.radians(200) * DT, np.radians(200) * DT))
        if abs(d) < np.radians(75):
            step = min(w.walk_speed * max(0.35, float(np.cos(d))) * DT, float(np.linalg.norm(nxt - w.person_xy)))
            w.person_xy = w.person_xy + step * (nxt - w.person_xy) / (np.linalg.norm(nxt - w.person_xy) + 1e-9)

    def _hand(self, st: dict, target, act) -> str:
        """Reach (0.5 s), act, come back (0.4 s)."""
        w = self.w
        w.head = np.asarray(target, float)
        if st["t"] < REACH_S:
            w.reach = (target, st["t"] / REACH_S)
            return ""
        if "err" not in st:
            st["err"] = act()
            if st["err"]:
                w.reach = None
                return st["err"]
        if st["t"] < REACH_S + BACK_S:
            w.reach = (target, 1.0 - (st["t"] - REACH_S) / BACK_S)
            return ""
        w.reach = None
        return "ok"

    def _pick(self, t: float, o: str, st: dict, robot_xy) -> str:
        w, v = self.w, self.cur
        if "checked" not in st:
            st["checked"] = True
            s = self.steps[v]
            if w.relation(o, s["rel"], s["target"]) and o not in w.held.values():  # already where it goes
                self.done.add(v)
                return "in place"
            if w.held["robot"] == o or float(np.linalg.norm(w.pos(o)[:2] - w.person_xy)) > 1.6:
                self.where[o] = w.pos(o)  # they look around and find it
                self.left[v] = t + 20.0  # gone: the robot has it
                return "not here"
            self.gaze.plan(t + REACH_S, o, True)
            self.picked_from = w.support(w.pos(o), skip=self.where.keys())
        return self._hand(st, w.pos(o), lambda: w.pick(o))

    def _place(self, t: float, o: str, st: dict, robot_xy) -> str:
        w, s = self.w, self.steps[self.cur]
        if "checked" not in st:
            st["checked"] = True
            self.gaze.plan(t + REACH_S, s["target"], False)
        return self._hand(st, w.top(s["target"]) + [0, 0, 0.12],
                          lambda: None if w.put(o, s["rel"], s["target"]) else "could not place")

    def _open(self, t: float, f: str, st: dict, robot_xy) -> str:
        return "" if st["t"] < OPEN_S else (self.w.open(f, True) or "ok")

    def _close(self, t: float, f: str, st: dict, robot_xy) -> str:
        return "" if st["t"] < OPEN_S else (self.w.open(f, False) or "ok")

    def place_at(self, xy, yaw: float) -> None:
        self.w.person_xy, self.w.person_yaw = np.asarray(xy, float).copy(), float(yaw)
        self.w.head = np.array([xy[0] + 2.5 * np.cos(yaw), xy[1] + 2.5 * np.sin(yaw), EYE - 0.5])
