"""The collaboration env, Gymnasium style: a person (scripted, animated, with gaze) does a task in a BEHAVIOR home; the
robot, a Stretch, is the agent.

    env = CollabEnv("lunch")                 low level, one frame (1/15 s) per step: base speeds, joint goals, gripper aperture
    env = SkillEnv(CollabEnv("lunch"))       high level: navigate / pick / place / open / close, run one frame per step
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)

obs, what the robot may use: its own state, the person's position, the objects' positions (the simulator stands in for
perception), and the events since the last step: the person's actions as a recognizer reads them (the object right
70% of the time, else another object in view), the person's gaze fixations, what the person said, and the robot's own
steps done. reward = -dt (time). info, privileged (the oracle and scoring only): steps done and who did them, the
person's step, the episode log. Either env's action may carry speech: {"say": (kind, step)}, kind one of announce
("I'll put ..."), ask ("Shall I put ...?"), have ("I have the ..."), put ("I put ...").
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from helper.stretch import skills as S

from .person import Person
from .sense import BANK, Recognizer
from .stretch_sim import LEVEL, TOP
from .tasks import TASKS

DT = 1 / 15
SAY = {"announce": "I'll {label}.", "ask": "Shall I {label}?", "have": "I have the {obj}.", "put": "I {label} .",
       "claim": "I'm getting the {obj}."}


class CollabEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, task: str, robot: bool = True, t_max: float = 300.0, acc: float = 0.7, render_mode=None,
                 clip: float = 3.0, bank=BANK):
        from .world import World
        self.task, self.robot_on, self.t_max, self.acc, self.render_mode = TASKS[task], robot, t_max, acc, render_mode
        self.clip, self.bank = clip, bank
        T = self.task
        self.w = World(T["scene"], T["rooms"], T["objects"], third=render_mode == "rgb_array")
        self.objects = [o["name"] for o in T["objects"]]
        self.names = self.objects + sorted({s["target"] for s in T["steps"]} - set(self.objects))
        self._setup()
        N = len(T["steps"])
        self.action_space = spaces.Dict({"base": spaces.Box(np.array([-S.V, -S.W]), np.array([S.V, S.W])),
                                         "joints": spaces.Box(np.array([0.0, 0.0, -1.57, -1.75]), np.array([1.1, 0.52, 0.56, 4.0])),
                                         "grip": spaces.Box(0.0, 1.0),
                                         "say": spaces.Tuple((spaces.Discrete(len(SAY) + 1), spaces.Discrete(N)))})
        self.observation_space = spaces.Dict({"robot": spaces.Box(-np.inf, np.inf, (6,)), "person": spaces.Box(-np.inf, np.inf, (2,)),
                                              "objects": spaces.Box(-np.inf, np.inf, (len(self.objects), 3)),
                                              "events": spaces.Sequence(spaces.Text(200))})

    def _setup(self) -> None:
        """The person's and the robot's starts; the objects at their starts (nearest the task's point for each, else on
        their furniture's side nearest the person's start, within reach from the floor); closed containers closed; the
        state saved."""
        w, T = self.w, self.task
        if T.get("starts"):
            self.person_start, self.robot_start = (np.asarray(q, float) for q in T["starts"])
        else:  # open floor by the first step's object's furniture
            f0 = next(o["start"][1] for o in T["objects"] if o["name"] == T["steps"][0]["obj"])
            spot = lambda here, away=None: next(q for gap in (1.2, 0.8, 0.5, 0.3)
                                                if (q := w.stand_spot(f0, here, gap=gap, wall=0.3, away=away)) is not None)
            self.person_start = spot(w.top(f0)[:2])
            self.robot_start = spot(self.person_start, away=self.person_start)
        for o in T["objects"]:
            rel, f = o["start"]
            if rel == "floor":  # a big thing stands where the task says: f = (x, y, yaw)
                from helper.behavior.sim import yaw_quat
                w.obj(o["name"]).set_position_orientation(position=[f[0], f[1], 1.0], orientation=yaw_quat(f[2]))
                lo, _ = w.box(o["name"])
                w.obj(o["name"]).set_position_orientation(position=w.pos(o["name"]) - [0, 0, lo[2] - 0.005])
                w.obj(o["name"]).keep_still()
                w.floor.block(*(b[:2] for b in w.box(o["name"])))  # the floor map does not hold it: walk around it
                continue
            w.held["person"] = o["name"]
            w.obj(o["name"]).set_position_orientation(position=w.top(f) + [0, 0, 0.3])
            if not w.put(o["name"], rel, f, near=o["near"] if o["near"] is not None else self.person_start):
                raise RuntimeError(f"cannot start {o['name']} {rel} {f}")
        for s in T["steps"]:
            if s.get("closed"):
                w.open(s["target"], False)
        ext = lambda o: w.box(o)[1] - w.box(o)[0]  # the ready gripper opens 17 cm: 12 cm across, to spare; open places only
        self.robot_can = [s.get("robot", True) and min(ext(s["obj"])[:2]) <= 0.12 and ext(s["obj"])[2] <= 0.30
                          and (s["rel"] == "ontop" or w.open_top(s["target"])) for s in T["steps"]]
        w.bot.place(self.robot_start, 0.0)
        for _ in range(10):
            w.step({})
        self.start = w.save()
        print(f"setup: person at {np.round(self.person_start, 2).tolist()}, robot at {np.round(self.robot_start, 2).tolist()}, "
              f"robot can do steps {[v for v, ok in enumerate(self.robot_can) if ok]}", flush=True)

    # -- gymnasium ---------------------------------------------------------------------------------------------------
    def reset(self, seed: int | None = None, options=None):
        super().reset(seed=seed)
        seed = 0 if seed is None else seed
        w = self.w
        w.restore(self.start)
        w.bot.stow()
        w.bot.place(self.robot_start if self.robot_on else np.array([100.0, 100.0]), 0.0)
        self.person = Person(self.task, w, seed)
        to = w.pos(self.task["steps"][0]["obj"])[:2] - self.person_start
        self.person.place_at(self.person_start, float(np.arctan2(to[1], to[0])))
        self.rng = np.random.default_rng([seed, 3])  # the recognizer
        self.rec, self.next_clip = Recognizer(self.bank, self.acc, np.random.default_rng([seed, 5])), self.clip
        self.t, self.k, self.vis = 0.0, 0, {}
        self.views = {}  # the render cameras' spots
        self.since, self.done_at, self.by, self.placed_by = {}, {}, {}, {}
        self.claims, self.hold_person = {}, 0.0
        self.events, self.log = [], {"done": [], "utterances": [], "frames": [], "fixations": []}
        self.n_actions = 0
        return self._obs(), self._info()

    def step(self, action: dict):
        w, p = self.w, self.person
        cmd = self._command(action) if self.robot_on else {}
        if "step" in action:  # the step the robot is on: what a person seeing it would take it to be doing
            v = action["step"]
            self.claims = {} if v is None else {v: self.claims.get(v, self.t)}
        if action.get("say"):
            self._say(*action["say"])
        if self.k > 0 and self.k % 2 == 0:  # what the person sees, at 7.5 Hz (raycasts are stale before the first step)
            self.vis = w.visible(sorted(set(self.names) | {"robot"}), strict=tuple(self.objects))
            p.see(self.t, self.vis)
        if self.t >= self.hold_person:
            p.act(self.t, self.claims, w.bot.xy if self.robot_on else None)
        w.step(cmd)
        cur = set(self.task["step_objects"][f"v{p.cur}"]) if p.cur is not None else set()
        nxt = next((v for v in range(len(self.task["steps"])) if v not in p.done and v != p.cur), None)
        _, fix = p.gaze.step(self.t, self.vis, cur, (set(self.task["step_objects"][f"v{nxt}"]) if nxt is not None else set()) | {"robot"})
        if fix:
            self.events.append({"t": fix["t"], "kind": "fixation", "object": fix["object"], "visible": fix["visible"]})
            self.log["fixations"].append(fix)
        for a in p.actions[self.n_actions:]:  # the recognizer: the object right w.p. acc, else another one in view
            objs = [o if self.rng.random() < self.acc or not self.vis else sorted(self.vis)[self.rng.integers(len(self.vis))]
                    for o in a["objects"]]
            self.events.append({"t": a["t"], "kind": "action", "verb": a["verb"], "objects": objs})
        self.n_actions = len(p.actions)
        if self.t + 1e-9 >= self.next_clip:  # a clip ended: what the verb recognizer and the detector report of it
            ev = self.rec.clip(self.next_clip, self.clip, p.completed, w.held["person"], set(self.vis) - {"robot"})
            self.log.setdefault("clips", []).append({"t": ev["t"], "truth": ev.pop("truth")})  # privileged: log only
            self.events.append(ev)
            self.next_clip += self.clip
        self._progress()
        self.t += DT
        self.k += 1
        if self.k % 15 == 0:
            self.log["frames"].append({"t": round(self.t, 2), "person": w.person_xy.round(3).tolist(), "robot": w.bot.xy.round(3).tolist(),
                                       "person_step": p.cur, "objects": {o: w.pos(o).round(3).tolist() for o in self.objects}})
        terminated = len(self.done_at) == len(self.task["steps"])
        return self._obs(), -DT, terminated, self.t >= self.t_max, self._info()

    def render(self):
        """Side by side: the person (with the robot when it is within 3 m) and the robot's arm. Each camera stands 2.4 m
        up on open floor with clear lines to what it films, keeps its spot while it sees it, re-chooses every 2 s and
        glides there (cuts when its line is lost); with no clear spot it looks down from over what it films."""
        w = self.w
        both = self.robot_on and float(np.linalg.norm(w.person_xy - w.bot.xy)) < 3.0
        pts = [np.array([*w.person_xy, 1.3])] + ([np.array([*w.bot.xy, 0.9])] if both else [])
        views = [self._film("third_cam", pts, (3.5, 4.25, 5.0, 2.5, 1.5))]
        if self.robot_on:
            views.append(self._film("robot_cam", [np.array([*w.bot.xy, 0.9]), w.bot.grasp_point()], (2.2, 2.8, 1.6, 1.1)))
        return np.hstack(views)

    def _film(self, cam: str, pts: list, radii: tuple) -> np.ndarray:
        from helper.behavior.sim import look_quat
        w, v = self.w, self.views.setdefault(cam, {"goal": None, "at": None, "look": None, "next": 0.0})
        center = np.mean(pts, axis=0)
        sees = lambda c: all(self._clear(c, p) for p in pts)
        if self.t >= v["next"] or v["goal"] is None:
            v["next"] = self.t + 2.0
            if v["goal"] is None or not sees(v["goal"]):
                ok = []
                for r in radii:  # far first; nearer in a corridor
                    spots = [np.array([*(center[:2] + r * np.array([np.cos(a), np.sin(a)])), 2.4])
                             for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)]
                    ok = [c for c in spots if w.floor.clearance(c[:2]) >= 0.3 and sees(c)]
                    if ok:
                        break
                last = v["goal"]
                v["goal"] = (min(ok, key=lambda c: 0.0 if last is None else float(np.linalg.norm(c - last))) if ok
                             else np.array([*(center[:2] + [0.4, 0.0]), 2.3]))  # none clear: from over it
        if v["at"] is None or not sees(v["at"]):
            v["at"], v["look"] = v["goal"], center
        v["at"] = v["at"] + 0.12 * (v["goal"] - v["at"])
        v["look"] = v["look"] + 0.15 * (center - v["look"])
        w.cams[cam].set_position_orientation(position=v["at"], orientation=look_quat(v["at"], v["look"]))
        return w.rgb(cam)

    def _clear(self, a, b) -> bool:
        """Nothing but the robot, or things within 25 cm of b (what it holds or reaches for), between a and b."""
        d = float(np.linalg.norm(np.asarray(b, float) - a))
        return all(self.w.name(h["rigidBody"]) == "robot" or float(h["distance"]) >= d - 0.25 for h in self.w._rays(a, b))

    # -- inside --------------------------------------------------------------------------------------------------------
    def _command(self, a: dict) -> dict:
        base, joints = a.get("base", (0.0, 0.0)), a.get("joints")
        cmd = {"v": float(base[0]), "w": float(base[1])}
        if joints is not None:
            cmd |= {"lift": float(joints[0]), "arm": float(joints[1]), "pitch": float(joints[2])}
            if len(joints) > 3:
                cmd["wrist_yaw"] = float(joints[3])
        if "grip" in a:
            cmd["grip"] = float(a["grip"])
        return cmd

    def _say(self, kind: str, v: int) -> None:
        s, label = self.task["steps"][v], self.task["labels"][v]
        text = SAY[kind].format(label=label, obj=s["obj"].rsplit("_", 1)[0].replace("_", " "))
        self.log["utterances"].append({"t": round(self.t, 2), "who": "robot", "text": text})
        reply = self.person.hear(self.t, kind, v)
        if kind in ("announce", "ask", "have", "claim"):
            self.claims.setdefault(v, self.t)
        if reply:
            self.log["utterances"].append({"t": round(self.t + 1.0, 2), "who": "person", "text": reply})
            self.events.append({"t": round(self.t + 1.0, 2), "kind": "speech", "text": reply, "step": v})
        self.hold_person = max(self.hold_person, self.t + 1.5)

    def _progress(self) -> None:
        """A step is done once its relation holds 0.5 s with its object out of every hand (undone once it moves 5 cm);
        credited to whoever set the object down last. An object dropped into an open container is pinned there then."""
        w = self.w
        for o in self.objects:  # who set it down: the robot's gripper let go of it, or the person's hand
            if w.held["robot"] == o and not w.bot.holding(w.obj(o)):
                w.held["robot"] = None
                self.placed_by[o] = "robot"
                for s in self.task["steps"]:
                    if s["obj"] == o:
                        w.settle(o, s["rel"], s["target"])
            elif w.held["robot"] is None and w.bot.holding(w.obj(o)):
                w.held["robot"] = o
        if w.held["person"] is not None:
            self.placed_by[w.held["person"]] = "person"
        for v, s in enumerate(self.task["steps"]):
            o = s["obj"]
            holds = o not in w.held.values() and w.relation(o, s["rel"], s["target"])
            if holds:
                self.since.setdefault(v, self.t)
            else:
                self.since.pop(v, None)
            if v not in self.done_at and holds and self.t - self.since[v] >= 0.5:
                w.settle(o, s["rel"], s["target"])  # landed in an open container: held there from now on
                self.done_at[v] = w.pos(o)
                self.by[v] = self.placed_by.get(o, "start")
                self.log["done"].append([round(self.t, 2), v, self.by[v]])
                if self.by[v] == "robot":
                    self.events.append({"t": round(self.t, 2), "kind": "robot_done", "step": v})
                    self.claims.pop(v, None)
            elif v in self.done_at and not holds and float(np.linalg.norm(w.pos(o) - self.done_at[v])) > 0.05:
                del self.done_at[v]
                self.log["done"].append([round(self.t, 2), v, "undone"])

    def _obs(self) -> dict:
        w, ev = self.w, self.events
        self.events = []
        return {"robot": np.array([*w.bot.xy, w.bot.yaw, *[w.bot.joints()[j] for j in ("lift", "arm")], float(w.held["robot"] is not None)]),
                "person": w.person_xy.copy(), "objects": np.stack([w.pos(o) for o in self.objects]), "events": ev}

    def _info(self) -> dict:
        p = self.person
        return {"t": round(self.t, 2), "done": set(self.done_at), "by": dict(self.by), "person_step": p.cur,
                "person_done": set(p.done), "log": self.log, "person_log": p.log, "person_segments": p.segments}


class SkillEnv(gym.Wrapper):
    """High-level actions on CollabEnv: {"skill": navigate / pick / place / open / close / stop / noop, "object",
    "target", "say"}. A skill runs its controller (helper/stretch/skills.py) one frame per step until it ends; a new skill
    replaces a running one; noop keeps it. info["skill"] = "running", "done", "failed: <why>" or "idle"."""

    def __init__(self, env: CollabEnv):
        super().__init__(env)
        self.run, self.status, self.geo = None, "idle", TOP  # geo: the wrist pose the held object was picked with
        self.over_what = "floor"  # the highest thing under the last arm's way looked at
        self.action_space = spaces.Dict({"skill": spaces.Discrete(7), "object": spaces.Text(64), "target": spaces.Text(64)})

    def reset(self, **kw):
        self.run, self.status, self.geo = None, "idle", TOP
        return self.env.reset(**kw)

    def step(self, action: dict):
        e = self.env
        if action.get("skill") not in (None, "noop"):
            self.run, self.status = self._start(action), "running"
        cmd = {"base": (0.0, 0.0)}
        part = self.run[0].parts[0] if self.run is not None else None  # what it holds passes through things until let go
        e.w.robot_ghost = not (isinstance(part, S.Place) and part.seq is not None and part.seq.k >= len(part.unfold()) + 3)
        closing = isinstance(part, S.Pick) and part.seq is not None and not part.missed and part.seq.k >= len(part.unfold()) + 3
        e.w.bot.picking = self.run[1] if closing else None  # the sticky grasp takes only this, only as the pick closes
        if self.run is not None:
            s = e.w.bot.state(person=e.w.person_xy, obj=e.w.obj(self.run[1]) if self.run[1] else None)
            c, self.status = self.run[0].step(s, DT)
            if self.status == "failed: nothing in the gripper":  # where the object went, against the fingertips
                off = e.w.pos(self.run[1]) - e.w.bot.grasp_point()
                self.status += f" ({self.run[1]} {100 * np.linalg.norm(off[:2]):.0f} cm off, {100 * off[2]:+.0f} cm up)"
            cmd = {"base": (c.get("v", 0.0), c.get("w", 0.0))}
            j = e.w.bot.goal | c
            cmd["joints"] = (j["lift"], j["arm"], j["pitch"], j["wrist_yaw"])
            cmd["grip"] = float(j["grip"])
            if self.status != "running":
                self.run = None
        if action.get("say"):
            cmd["say"] = action["say"]
        if "step" in action:
            cmd["step"] = action["step"]
        obs, r, term, trunc, info = e.step(cmd)
        info["skill"] = self.status
        return obs, r, term, trunc, info

    def _start(self, a: dict):
        """(controller, object it holds or grasps): navigate goes to stand by the object; pick drives to an approach pose
        then picks; place drives to the approach pose of a free spot on (or in) the target, then places."""
        e, w = self.env, self.env.w
        home = w.floor.region(w.bot.xy)  # base poses on floor the planner can reach; the base's radius is 0.17 m
        free = lambda q: w.floor.clearance(q) > 0.2 and w.floor.region(q) == home
        def tuck() -> list:  # the arm in and an empty wrist down before driving: a stopped or failed reach leaves them out
            j, held = w.bot.joints(), w.held["robot"] is not None
            down = abs(j["pitch"] - S.STOW) < 0.1
            if j["arm"] < 0.08 and j["lift"] > S.CARRY - 0.05 and (down or held):
                return []
            goals = [{"arm": 0.0} | ({} if held else {"grip": 1.0})]
            goals += [] if down or held else [{"lift": S.LIFT_MAX}, {"pitch": S.STOW, "wrist_yaw": 0.0}]
            return [S.Joints(goals + [{"lift": max(S.CARRY, j["lift"]) if held else S.CARRY}])]
        def go(xy, yaw=None):  # with something in hand, 30 cm clear of obstacles where the floor allows
            way = None if xy is None or not w.held["robot"] else w.path(w.bot.xy, xy, hard=0.3)
            way = way if way is not None or xy is None else w.path(w.bot.xy, xy)
            if way is None:
                return Fail(f"no way there (from {np.round(w.bot.xy, 2).tolist()} to {None if xy is None else np.round(xy, 2).tolist()})")
            return Chain(tuck() + [S.Navigate(way, yaw)])
        if a["skill"] == "stop":  # end what runs, the arm tucked
            return Chain(tuck() or [Do(lambda: None)]), None
        if a["skill"] == "navigate":
            return Chain([go(w.stand_spot(a["object"], w.bot.xy))]), None
        if a["skill"] in ("open", "close"):
            return Chain([go(w.stand_spot(a["object"], w.bot.xy)), Do(lambda: w.open(a["object"], a["skill"] == "open"))]), None
        o = a["object"]
        if a["skill"] == "pick":
            st, why = next(s for s in e.task["steps"] if s["obj"] == o), []
            for geo, name in ((TOP, "wrist down"), (LEVEL, "wrist level")):  # down unless the lift cannot carry o
                xyz, below = self._grasp(o, geo)
                no = self._fits(o, st, geo, below)
                if no:
                    why.append(f"{name}: cannot set it on {st['target']} ({no})")
                    continue
                at = S.approach(xyz, geo, free, w.bot.xy, axis=self._axis(o),
                                ok=lambda b, yaw: self._hemmed((b, yaw), xyz, o, geo) is None and self._swings(b, yaw, geo)
                                and self._through(b, xyz))
                if at is None:
                    near = S.approach(xyz, geo, free, w.bot.xy, axis=self._axis(o))
                    why.append(f"{name}: " + (f"{self._hemmed(near, xyz, o, geo)} under the way down to it" if near is not None
                                              else "no free floor in reach of it"))
                    continue
                up = max(float(w.box(o)[1][2]), self._over(at, xyz, o, geo) + below) + 0.03
                if not (xyz[2] - geo.z0 >= 0.0 and up - geo.z0 <= S.LIFT_MAX):
                    why.append(f"{name}: beyond the lift (grasp at {xyz[2]:.2f} m, over {self.over_what} at {up:.2f} m)")
                elif not self._room(xyz, float(w.box(o)[1][2]), geo, up):
                    why.append(f"{name}: no room over it for the arm")
                else:
                    self.geo = geo
                    return Chain([go(at[0], at[1]), S.Pick(xyz, geo, up)]), o
            return Chain([Fail("out of the arm's reach (" + "; ".join(why) + ")")]), o
        rel, f, geo = a.get("rel", "ontop"), a["target"], self.geo
        below = float(w.bot.grasp_point()[2] - w.box(o)[0][2])  # the held object's bottom under the fingertips
        rim = float(w.box(f)[1][2]) if w.open_top(f) else 0.0

        why = {"no free floor by it": 0, "beyond the lift": 0, "no room for the arm": 0}

        def plan(p):  # base pose and the fingertips' height on the way to spot p, if the lift can carry o over
            at = S.approach(p, geo, free, w.bot.xy, ok=lambda b, yaw: self._swings(b, yaw, geo) and self._through(b, p))
            if at is None:
                why["no free floor by it"] += 1
                return None
            up = max(max(rim, self._over(at, p, o, geo)) + below + 0.03, p[2] + 0.02 + below + 0.02)  # 2 cm over letting go
            if up - geo.z0 > S.LIFT_MAX:
                why["beyond the lift"] += 1
            elif not self._room(p, p[2] + below, geo, up):
                why["no room for the arm"] += 1
            else:
                return at, up
            return None
        spots = w.spots(o, rel, f)
        if rim:  # an open container: where o fits, middle first; let go with o's bottom 2 cm over the rim, so it drops in
            spots = [[x, y, rim] for x, y, _ in spots]
        else:  # a surface: the nearest free spot
            spots = sorted(spots, key=lambda p: float(np.linalg.norm(np.asarray(p[:2]) - w.bot.xy)))
        reach = next(((np.asarray(p, float), *r) for p in spots if (r := plan(p)) is not None), None)
        if reach is None:  # the nearest free spot the arm can reach from free floor
            return Chain([Fail(f"no reachable free spot {rel} {f} (of {len(spots)}: "
                               + ", ".join(f"{n} {k}" for k, n in why.items() if n) + ")")]), o
        spot, at, up = reach
        return Chain([go(at[0], at[1]), S.Place(spot, below, geo, up=up)]), o

    def _grasp(self, o: str, geo) -> tuple[np.ndarray, float]:
        """Where the fingertips close on o, and how far its bottom then hangs under them: wrist down, at its middle
        (on the upper 12 cm of a tall thing), so the first finger to touch does not tip it; wrist level, at most 5 cm
        over its bottom. Never under 2.5 cm: closing, the fingertips swing 2.1 cm below that point, and must not dig
        into the surface the object stands on."""
        lo, hi = self.env.w.box(o)
        h = float(hi[2] - lo[2])
        below = max(0.025, 0.5 * h, h - 0.12) if geo.pitch < -1.0 else max(0.025, min(0.5 * h, 0.05))
        return np.array([*(lo[:2] + hi[:2]) / 2, lo[2] + below]), below

    def _axis(self, o: str) -> float | None:
        """The world direction of o's long side when it is longer than the ready gripper's 17 cm opening could take
        across, else None."""
        lo, hi = self.env.w.box(o)
        ex, ey = hi[:2] - lo[:2]
        if max(ex, ey) <= 0.15:
            return None
        return 0.0 if ex > ey else np.pi / 2

    def _hemmed(self, at, xyz, skip: str, geo) -> str | None:
        """Another task object standing under the gripper's way down to xyz from base pose `at` (within 9 cm of the
        target across the arm's line, and along it from 3 cm beyond the target back 3 cm (wrist down) or 25 cm (wrist
        level: the fingers and wrist behind the tips); higher than 1 cm under the fingertips' goal), else None."""
        w = self.env.w
        u = S.unit(at[1] + geo.d)  # the arm's world direction
        n, back = np.array([-u[1], u[0]]), 0.25 if geo.pitch > -1.0 else 0.03
        top = float(w.box(skip)[1][2])
        for x in self.env.objects:
            if x == skip or x in w.held.values():
                continue
            lo, hi = w.box(x)
            if hi[2] < xyz[2] - 0.01 or lo[2] > top + 0.05:  # under the fingertips, or on a shelf above
                continue
            c = np.array([[a, b] for a in (lo[0], hi[0]) for b in (lo[1], hi[1])]) - np.asarray(xyz, float)[:2]
            if (c @ u).max() > -back and (c @ u).min() < 0.03 and (c @ n).max() > -0.09 and (c @ n).min() < 0.09:
                return x
        return None

    def _fits(self, o: str, st: dict, geo, below: float) -> str | None:
        """None if o, held `below` over its bottom, can be set on its step's target with this wrist pose (the lift
        reaches one of the target's ten nearest free spots, carries o over a container's rim, the arm has room), else
        why not."""
        w = self.env.w
        spots = sorted(w.spots(o, st["rel"], st["target"]), key=lambda p: float(np.linalg.norm(np.asarray(p[:2]) - w.bot.xy)))
        if not spots:
            return None  # nothing free now: whether it fits is for the place to find out
        rim = float(w.box(st["target"])[1][2]) if w.open_top(st["target"]) else 0.0
        no = "no spot"
        for p in spots[:10]:
            surface = rim or float(p[2])  # into an open container: let go over its rim
            z = surface + below + 0.02
            up = max(z + 0.02, rim + below + 0.03)
            if not (z - geo.z0 >= 0.0 and up - geo.z0 <= S.LIFT_MAX):
                no = f"lift for {z:.2f} m to {up:.2f} m"
            elif (above := self._above([*p[:2], surface], z, geo, up)) is not None:
                no = f"{above} over it"
            else:
                return None
        return no

    def _through(self, b, xyz) -> bool:
        """Whether the arm can reach over xyz from base point b: not through a wall (or a fridge), which stops both a
        ray just over xyz and one at 1.6 m, as furniture under the arm or a cabinet over it does not; task objects
        and the robot do not count. Rays down from above start inside a wall and miss it."""
        w, p = self.env.w, np.asarray(xyz, float)
        def stopped(z):
            return any(w.name(h["rigidBody"]) not in (*self.env.objects, "robot")
                       for h in w._rays([*np.asarray(b, float)[:2], z], [*p[:2], z]))
        return not (stopped(p[2] + 0.12) and stopped(1.6))

    def _swings(self, b, yaw: float, geo) -> bool:
        """Whether the wrist can turn level (and fold forward) at full lift from base pose (b, yaw): nothing over
        0.95 m where the gripper sweeps, 0.1 to 0.45 m out on the arm's side and up to 0.3 m ahead."""
        if abs(geo.pitch - S.STOW) <= 0.1:
            return True  # the wrist stays down
        w = self.env.w
        return all(w.height(*(np.asarray(b, float) + S.rot(yaw, (fx, -side))), top=1.3)[0] < 0.95
                   for fx in (-0.05, 0.1, 0.3) for side in (0.1, 0.25, 0.45))

    def _room(self, xyz, top: float, geo, up: float) -> bool:
        return self._above(xyz, top, geo, up) is None

    def _above(self, xyz, top: float, geo, up: float) -> str | None:
        """What the arm (its top 0.23 m over the lift) would meet over xyz (a shelf above) with the fingertips at height
        up: the first thing from `top` (the top of what is there) up to the arm's top, else None."""
        arm_top = up - geo.z0 + 0.23
        return self.env.w.hit(np.array([xyz[0], xyz[1], top + 0.02]), np.array([xyz[0], xyz[1], arm_top + 0.03]))

    def _over(self, at, xyz, skip: str, geo) -> float:
        """How high the highest thing (furniture, a rim, another object; not `skip`, what is in a hand or the robot)
        under the arm's way from base pose `at` to xyz is: rays down every 5 cm along it, on its line and 8 cm to
        either side, from under whatever is over xyz (a shelf above: the arm goes under it)."""
        w = self.env.w
        a, b = at[0] + S.rot(at[1], np.asarray(geo.p0, float)), np.asarray(xyz, float)[:2]
        n = np.array([-(b - a)[1], (b - a)[0]]) / max(float(np.linalg.norm(b - a)), 1e-9)
        skip = (skip, *[o for o in w.held.values() if o])
        above = [h for h in w._rays([*b, xyz[2] + 0.02], [*b, 1.5]) if w.name(h["rigidBody"]) not in (*skip, "robot")]
        top = min(1.25, xyz[2] + 0.02 + float(above[0]["distance"]) - 0.01) if above else 1.25
        z, self.over_what = max(w.height(*(a + t * (b - a) + side * n), skip=skip, top=top)
                                for t in np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / 0.05) + 1)) for side in (-0.08, 0.0, 0.08))
        return z


class Chain:
    """Controllers one after another; fails with the first that fails."""

    def __init__(self, parts: list):
        self.parts = parts

    def step(self, s: dict, dt: float):
        cmd, status = self.parts[0].step(s, dt)
        if status == "done" and len(self.parts) > 1:
            self.parts.pop(0)
            return cmd, "running"
        return cmd, status


class Do:
    def __init__(self, fn):
        self.fn = fn

    def step(self, s: dict, dt: float):
        self.fn()
        return {"v": 0.0, "w": 0.0}, "done"


class Fail:
    def __init__(self, why: str):
        self.why = why

    def step(self, s: dict, dt: float):
        return {"v": 0.0, "w": 0.0}, f"failed: {self.why}"
