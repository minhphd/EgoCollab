"""Collaborative rearrangement in BEHAVIOR-1K: a scripted person and a robot share one task's subgoals.

Tasks come from helper/behavior/collab_tasks.py (a challenge instance's BDDL goal as subgoals with orderings). The
person is PARTNR's scripted person (helper/partnr/person.py) on BEHAVIOR skills, with one change: what they count as
done, and where they think each object is, comes only from what they did or saw. A subgoal the robot finished out of
their sight sends them to where the object was, until they see it in place. The robot runs one of PARTNR's policies
(helper/partnr/robot.py) on the stream this loop builds: the person's skills, gaze fixations and completions. Gaze is
helper/partnr/gaze.py on the person's view.

Per run, against the person alone on the same seed:
    time_rel          task time with the robot / alone
    robot_redundant   robot trips for a subgoal the person had done or was doing (count, seconds)
    person_redundant  person trips for a subgoal the robot had done (count, seconds)
    robot_utterances, person_utterances

    python -m helper.behavior.collab <tasks.json> <out_dir> activity=setting_the_table policy=alone,bip seed=0
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from helper.bip.config import Config
from helper.bip.decide import Costs, tell_value
from helper.partnr.gaze import Gaze
from helper.partnr.person import Person

from .sim import B1K, DT, TURN, look_quat

ROOT = Path.home() / "BEHAVIOR-1K"
ROBOT_SPEED = 0.8  # m/s
DETOUR = 2.5  # m: the longest way around the robot worth taking; past it the person walks through
AVOID_ROBOT = 4.4  # the person's choice: per m an object is inside 1.5 m of where they last saw the robot
HAND_S = 0.8  # s a robot's pick or placement takes
REACH_S, BACK_S = 0.5, 0.4  # s the person's arm takes to reach, and to come back
SAY_S = 1.5  # s a spoken line takes
OPEN_S = 0.6
REACH = 1.6  # m within which an agent can pick or place
OK, FAIL = "Successful execution!", "Unexpected failure! "


def rooms_of(activity: str) -> list[str] | None:
    """The room instances the challenge loads for an activity (B100_task_misc.csv), or None for the whole home."""
    import csv
    f = ROOT / "datasets/2026-challenge-task-instances/metadata/B100_task_misc.csv"
    for row in csv.DictReader(open(f)):
        if row.get("Task") == activity:
            rooms = row.get("Rooms to inlcude") or row.get("Rooms to include") or ""  # the file's spelling
            return [r.strip() for r in rooms.replace(",", "\n").splitlines() if r.strip()] or None
    return None


def robot_spot(row: dict) -> list:
    """Where the robot starts: [x, y, yaw]."""
    return row.get("robot_spot") or [*(row.get("robot_start") or {}).get("position", row["person_start"][:2])[:2], 0.0]


def ahead(way: list, dist: float) -> list:
    """The first dist m of a path (points)."""
    out, left = [np.asarray(way[0], float)], dist
    for q in way[1:]:
        q = np.asarray(q, float)
        d = float(np.linalg.norm(q - out[-1]))
        if d >= left:
            out.append(out[-1] + (q - out[-1]) * left / max(d, 1e-9))
            break
        out.append(q)
        left -= d
    return out


def off(q, way: list) -> float:
    """Distance from point q to a path (points)."""
    q = np.asarray(q, float)[:2]
    if len(way) == 1:
        return float(np.linalg.norm(q - np.asarray(way[0], float)[:2]))
    best = np.inf
    for a, b in zip(way[:-1], way[1:]):
        a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
        k = float(np.clip((q - a) @ (b - a) / max(float((b - a) @ (b - a)), 1e-9), 0.0, 1.0))
        best = min(best, float(np.linalg.norm(q - (a + k * (b - a)))))
    return best


def length(xy, way) -> float:
    pts = np.vstack([np.asarray(xy, float)[:2]] + [np.asarray(q, float)[:2] for q in way])
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


class Skills:
    """PARTNR's skill interface on a BEHAVIOR scene: skill(agent, tool, arg) runs one frame of a skill and returns
    (None, response), the response empty until the skill ends. Agent 0 is the robot, 1 the person; the person walks
    to where they believe an object is (beliefs: object -> position)."""

    def __init__(self, sim: B1K, beliefs: dict):
        self.sim, self.beliefs = sim, beliefs
        self.run = {0: None, 1: None}  # (tool, arg, state) of the running skill
        self.placed_by, self.placed_at = {}, {}  # object -> who last set it down, and when
        self.in_place = lambda o: False  # set by the loop: whether o already satisfies its subgoal
        self.small = set()  # the task's movable objects (set by the loop)
        self.furniture = set()  # the task's furniture (set by the loop)
        self.failures = []  # (t, agent, tool, arg, message), in full
        self.agents = [type("A", (), {"reset": lambda s: None})(), None]

    def who(self, agent: int) -> str:
        return "robot" if agent == 0 else "person"

    def held(self, agent: int) -> str | None:
        return self.sim.held[self.who(agent)]

    def where(self, agent: int, x: str) -> np.ndarray:
        """Where agent goes for x: the person to where they believe it is, the robot to where it is."""
        if agent == 1 and x in self.beliefs:
            return np.asarray(self.beliefs[x], float)
        lo, hi = (np.asarray(v, float) for v in self.sim.obj(x).aabb)
        return (lo + hi) / 2

    def spot(self, agent: int, x: str, target: np.ndarray) -> np.ndarray:
        """Where to stand for x: off its furniture's edge and off walls, nearest the agent. For an object the person
        believes is elsewhere, the point they believe in."""
        here = self.sim.pose[self.who(agent)][0]
        believed = agent == 1 and x in self.beliefs
        base = None if believed else x
        under = self.sim.support(target, skip=self.small) if (believed or x in self.small) else None
        floorish = under is not None and any(k in under for k in ("floor", "carpet", "rug"))
        if under is not None and not floorish:  # a small thing on furniture: stand at the furniture's edge nearest it
            base = under
        elif floorish:  # on the floor: around the thing itself
            base = None
        other = self.sim.pose[self.who(1 - agent)][0]
        for gap, wall in ((0.45, 0.35), (0.3, 0.3), (0.2, 0.25)):  # nearer and tighter before giving up
            q = self.sim.stand_spot(base, here, point=target, gap=gap, wall=wall, near=target, away=other)
            if q is not None:
                return q
        return None

    def skill(self, agent: int, tool: str, arg: str) -> tuple[None, str]:
        run = self.run[agent]
        if run is None or run[:2] != (tool, arg):
            run = self.run[agent] = (tool, arg, {"t": 0.0})
        st = run[2]
        st["t"] += DT
        self.t = getattr(self, "t", 0.0)
        resp = getattr(self, "_" + tool.lower())(agent, arg, st)
        if resp:
            self.run[agent] = None
            if resp.startswith(FAIL):
                self.failures.append([round(getattr(self, "t", 0.0), 2), self.who(agent), tool, arg, resp[len(FAIL):]])
        return None, resp

    def _navigate(self, agent: int, x: str, st: dict) -> str:
        sim, who = self.sim, self.who(agent)
        xy, yaw = sim.pose[who]
        if "path" not in st:
            st["x"], st["target"] = x, self.where(agent, x)
            goal = self.spot(agent, x, st["target"])
            if goal is None:  # nowhere clear to stand near it
                return FAIL + f"cannot get near {x} at {np.round(st['target'], 2).tolist()} on {self.sim.support(st['target'])}"
            st["path"] = list(sim.path(xy, goal)[1:])
        while st["path"] and np.linalg.norm(np.asarray(st["path"][0]) - xy) < 0.05:
            st["path"].pop(0)
        if st["path"] and self._blocked(agent, xy, np.asarray(st["path"][0], float), st):
            return ""
        if st["path"]:
            self._walk(agent, xy, yaw, np.asarray(st["path"][0], float))
            return ""
        t = st["target"]  # arrived: face the target (the robot turns its arm side to it)
        want = float(np.arctan2(t[1] - xy[1], t[0] - xy[0]))
        geo = sim.arm_geometry() if who == "robot" else None
        if geo:
            want -= geo["angle"]
        d = (want - yaw + np.pi) % (2 * np.pi) - np.pi
        if abs(d) > np.radians(5):
            sim.place_agent(who, xy, yaw + float(np.clip(d, -TURN * DT, TURN * DT)))
            return ""
        return OK

    def _walk(self, agent: int, xy: np.ndarray, yaw: float, nxt: np.ndarray) -> None:
        """One frame toward nxt: turn toward it, stepping once roughly facing it."""
        who = self.who(agent)
        want = float(np.arctan2(nxt[1] - xy[1], nxt[0] - xy[0]))
        d = (want - yaw + np.pi) % (2 * np.pi) - np.pi
        turn = np.radians(200) if who == "person" else TURN  # people turn as they walk
        yaw += float(np.clip(d, -turn * DT, turn * DT))
        if abs(d) < np.radians(75):
            v = (self.sim.walk_speed if who == "person" else ROBOT_SPEED) * max(0.35, float(np.cos(d)))
            step = min(v * DT, float(np.linalg.norm(nxt - xy)))
            xy = xy + step * (nxt - xy) / (np.linalg.norm(nxt - xy) + 1e-9)
        self.sim.place_agent(who, xy, yaw)

    def _blocked(self, agent: int, xy: np.ndarray, nxt: np.ndarray, st: dict) -> bool:
        """The person has the right of way. The robot holds a step that would bring it within 0.5 m of the person,
        closer than now (4 s at most, then on it goes). The person never waits: with the robot within 0.5 m of their
        next 1.5 m, they take a way around it that adds at most DETOUR m, else walk on through it."""
        other = self.sim.pose[self.who(1 - agent)][0]
        if agent == 1:
            moved = float(np.linalg.norm(other - st.get("around", other + 9.0)))
            if st["path"] and moved > 0.3 and off(other, ahead([xy] + st["path"], 1.5)) < 0.5:
                st["around"] = other.copy()  # planned around the robot standing here
                way = self.sim.path(xy, st["path"][-1], avoid=[(other, 0.5)], strict=True)
                if way is not None and length(xy, way[1:]) <= length(xy, st["path"]) + DETOUR:
                    st["path"] = list(way[1:])
            return False
        step = 0.08 * (nxt - xy) / (np.linalg.norm(nxt - xy) + 1e-9)
        now, after = float(np.linalg.norm(other - xy)), float(np.linalg.norm(other - (xy + step)))
        if after >= 0.5 or after >= now:
            st.pop("waited", None)
            return False
        st["waited"] = st.get("waited", 0.0) + DT
        return st["waited"] < 4.0  # never a standstill

    def _near(self, agent: int, x: str) -> bool:
        lo, hi = (np.asarray(v, float) for v in self.sim.obj(x).aabb)
        return float(np.linalg.norm(((lo + hi) / 2)[:2] - self.sim.pose[self.who(agent)][0])) <= REACH

    def _arm(self, agent: int, st: dict, target, act) -> str:
        """The hand's part of a skill: the person reaches for target (REACH_S), act() happens at the hand, and the arm
        comes back (BACK_S); the robot waits HAND_S, then act(). act returns an error text or None."""
        if agent == 0:
            geo = self.sim.arm_geometry()
            if not geo:
                if st["t"] < HAND_S:
                    return ""
                err = act()
                return FAIL + err if err else OK
            return self._robot_arm(st, np.asarray(target, float), act, geo)
        if st["t"] < REACH_S:
            self.sim.reach = (target, st["t"] / REACH_S)
            return ""
        if "err" not in st:
            st["err"] = act()
            if st["err"]:
                self.sim.reach = None
                return FAIL + st["err"]
        if st["t"] < REACH_S + BACK_S:
            self.sim.reach = (target, 1.0 - (st["t"] - REACH_S) / BACK_S)
            return ""
        self.sim.reach = None
        return OK

    def _robot_arm(self, st: dict, target: np.ndarray, act, geo: dict) -> str:
        """The robot's hand: lift to the target's height and extend toward it (0.8 s), act, then retract and lower to
        carrying height (0.6 s)."""
        xy, yaw = self.sim.pose["robot"]
        a = yaw + geo["angle"]
        lift = target[2] - geo["z0"] + 0.02
        ext = float((target[:2] - xy) @ np.array([np.cos(a), np.sin(a)])) - geo["out0"]
        ease = lambda x: 0.5 - 0.5 * np.cos(np.pi * float(np.clip(x, 0, 1)))
        if st["t"] < 0.8:
            w = ease(st["t"] / 0.8)
            self.sim.set_arm(0.6 + w * (lift - 0.6), w * ext)
            return ""
        if "err" not in st:
            st["err"] = act()
            if st["err"]:
                self.sim.set_arm(0.6, 0.0)
                return FAIL + st["err"]
        if st["t"] < 1.4:
            w = ease((st["t"] - 0.8) / 0.6)
            self.sim.set_arm(lift + w * (0.6 - lift), (1 - w) * ext)
            return ""
        self.sim.set_arm(0.6, 0.0)
        return OK

    def _pick(self, agent: int, o: str, st: dict) -> str:
        if "checked" not in st:  # can it be picked at all: decided before the hand moves
            st["checked"] = True
            if agent == 1 and self.in_place(o):  # standing at it, the person sees it is already where it goes
                self.beliefs[o] = self.sim.pos(o).tolist()
                return FAIL + f"{o} is already in place"
            other = self.sim.held[self.who(1 - agent)]
            if other == o or not self._near(agent, o):
                if agent == 1 and o in self.beliefs:  # not where they thought: they look around and find it
                    self.beliefs[o] = self.sim.pos(o).tolist()
                return FAIL + f"{o} is not here"

        def act():
            have = self.sim.held[self.who(agent)]
            if have is not None and have != o:  # a hand is free first: what it holds goes on the nearest surface
                self._set_down(agent, have)
            self.sim.pick(self.who(agent), o)
        return self._arm(agent, st, self.sim.pos(o), act)

    def _set_down(self, agent: int, o: str) -> None:
        """Put o on the nearest piece of furniture that takes it, else on the floor in front."""
        who, sim = self.who(agent), self.sim
        xy, yaw = sim.pose[who]
        near = sorted((float(np.linalg.norm(sim.pos(f.name)[:2] - xy)) - (2.0 if f.name in self.furniture else 0.0), f.name)
                      for f in sim.scene.objects
                      if f.name in self.furniture or getattr(f, "category", "") in ("countertop", "breakfast_table", "coffee_table",
                                                                                   "shelf", "desk", "bottom_cabinet", "table",
                                                                                   "dining_table", "console_table"))
        for _, f in near[:4]:  # the task's own furniture first, then the nearest table or counter
            if sim.put(who, o, "ontop", f):
                self.placed_by[o] = who
                return
        sim.held[who] = None
        sim.obj(o).set_position_orientation(position=[xy[0] + 0.4 * np.cos(yaw), xy[1] + 0.4 * np.sin(yaw), 0.1])

    def _place(self, agent: int, arg: str, st: dict) -> str:
        o, rel, f = [x.strip() for x in arg.split(",")[:3]]
        who = self.who(agent)
        if self.sim.held[who] != o and "err" not in st:
            return FAIL + f"not holding {o}"

        def act():
            if not self.sim.put(who, o, rel, f):
                lo, hi = self.sim.box(o)
                flo, fhi = self.sim.box(f)
                return (f"could not place {o} {rel} {f} (o z {lo[2]:.3f}-{hi[2]:.3f}, f z {flo[2]:.3f}-{fhi[2]:.3f}, "
                        f"o centre {np.round((lo + hi)[:2] / 2, 2).tolist()} f box {np.round(flo[:2], 2).tolist()}-{np.round(fhi[:2], 2).tolist()})")
            self.placed_by[o], self.placed_at[o] = who, self.t
        return self._arm(agent, st, self.sim.top(f) + [0, 0, 0.12], act)  # above it: a carried thing never shoves it

    def _open(self, agent: int, f: str, st: dict) -> str:
        return self._arm(agent, st, self.sim.top(f) - [0, 0, 0.1], lambda: self.sim.open(f, True) and None)

    def _close(self, agent: int, f: str, st: dict) -> str:
        return self._arm(agent, st, self.sim.top(f) - [0, 0, 0.1], lambda: self.sim.open(f, False) and None)


class BPerson(Person):
    """PARTNR's person with BEHAVIOR positions; choose, nearest and misplace read the scene, not Habitat."""

    def __init__(self, cfg: Config, task: dict, furniture: list[str], seed: int, sim: B1K, skills: Skills):
        super().__init__(cfg, task, furniture, seed)
        self.bsim, self.skills = sim, skills
        self.robot_seen = None  # where the person last saw the robot (set by the loop)

    def _here(self) -> np.ndarray:
        return self.bsim.pose["person"][0]

    def choose(self, options: list[str], sim) -> str:
        if self.cfg.person != "hitl" or len(options) == 1:
            return options[self.rng.integers(len(options))]
        here, there = self._here(), self.robot_seen
        at = {v: self.skills.where(1, self.task["object"][v])[:2] for v in options}
        score = [-self.cfg.choice_rank * self.task["nodes"].index(v) - self.cfg.choice_dist * float(np.linalg.norm(at[v] - here))
                 - (0.0 if there is None else AVOID_ROBOT * max(0.0, 1.5 - float(np.linalg.norm(at[v] - there))))
                 for v in options]  # what the robot stands by is likely its next
        p = np.exp(np.array(score) - max(score))
        return options[self.rng.choice(len(options), p=p / p.sum())]

    def nearest(self, sim) -> str | None:
        here = self._here()
        near = [(float(np.linalg.norm(self.skills.where(0, f)[:2] - here)), f) for f in self.furniture[:200]]
        return min(near)[1] if near else None

    def misplace(self, sim) -> None:
        return  # placements in BEHAVIOR go where the script says


def official_done(sim: B1K, subs: dict) -> list | None:
    """The steps OmniGibson's own BDDL predicates (OnTop, Inside, NextTo, Under, Touching) count as done, checked once at
    the end beside ours; None when the check fails."""
    try:
        from omnigibson import object_states as S
        kinds = {"ontop": S.OnTop, "inside": S.Inside, "nextto": S.NextTo, "under": S.Under, "touching": S.Touching}
        return sorted(v for v, s in subs.items() if s["obj"] not in sim.held.values()
                      and sim.obj(s["obj"]).states[kinds[s["rel"]]].get_value(sim.obj(s["target"])))
    except Exception as e:
        print("official check failed:", repr(e)[:160], flush=True)
        return None


def holds(sim: B1K, s: dict) -> bool:
    """Whether subgoal s's relation holds (sim.relation's box geometry; OmniGibson's NextTo and OnTop on small
    targets such as plates reject placements people accept). An object in a hand is not placed."""
    return s["obj"] not in sim.held.values() and sim.relation(s["obj"], s["rel"], s["target"])


def head_target(sim: B1K, person, T: dict) -> np.ndarray:
    """Where the person's head points: ahead along the way while walking, else at what the running skill acts on."""
    xy, yaw = sim.pose["person"]
    eye_z = float(sim.eye[2]) if sim.eye[2] > 1.0 else 1.5
    if person.skill is None or person.skill[0] == "Navigate":
        return np.array([xy[0] + 2.5 * np.cos(yaw), xy[1] + 2.5 * np.sin(yaw), eye_z - 0.5])
    tool, arg = person.skill
    ent = arg.split(",")[2 if tool == "Place" else 0].strip()
    return sim.top(ent) if ent in [o.name for o in sim.scene.objects] else sim.head


def task_dict(row: dict, sim: B1K) -> dict:
    """PARTNR's task layout for the person and robot classes: nodes, orderings, a skill script per subgoal."""
    from omnigibson.object_states import Open
    nodes, scripts, obj, objects, labels, names = [], {}, {}, {}, {}, {}
    for i, s in enumerate(row["subgoals"]):
        v = f"v{i}"
        o, rel, f = s["obj"], s["rel"], s["target"]
        openable = rel in ("inside", "under") and Open in sim.obj(f).states  # under a sink: in its cabinet
        scripts[v] = ([("Navigate", o), ("Pick", o), ("Navigate", f)] + ([("Open", f)] if openable else [])
                      + [("Place", f"{o}, {rel}, {f}")] + ([("Close", f)] if openable else []))
        nodes.append(v); obj[v] = o; objects[v] = [o, f]
        word = lambda n: str(getattr(sim.obj(n), "category", "") or n.rsplit("_", 1)[0]).replace("_", " ")
        relw = {"ontop": "on", "inside": "in", "nextto": "next to", "under": "under", "touching": "against"}.get(rel, rel)
        labels[v], names[o], names[f] = f"{word(o)} {relw} the {word(f)}", word(o), word(f)
    labels = [labels[v] for v in nodes]
    edges = [(f"v{a}", f"v{b}") for a, b in row["orderings"]]
    return {"title": row["activity"].replace("_", " "), "nodes": nodes, "edges": edges, "scripts": scripts, "object": obj,
            "objects": objects, "labels": labels, "names": names, "spec": {}}


def overview(sim: B1K, row: dict) -> tuple[np.ndarray, np.ndarray]:
    """The room camera: 2.3 m up at a walkable point 2.5 to 4.5 m from the task's places that has clear lines to most
    of them (each object and where it goes), the farthest of the best; and the point it looks at."""
    names = sorted({x for s in row["subgoals"] for x in (s["obj"], s["target"])})
    pts = [(sim.top(n), n) for n in names]
    mid = np.mean([p for p, _ in pts], axis=0)
    best, fl = None, sim.floor()
    for r in (4.5, 3.5, 2.5):
        for a in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            q = mid[:2] + r * np.array([np.cos(a), np.sin(a)])
            if fl.clearance(q) < 0.3:
                continue
            cam = np.array([q[0], q[1], 2.3])
            score = (sum(sim.clear(cam, p, n) for p, n in pts), r)
            best = max(best, (score, cam.tolist())) if best else (score, cam.tolist())
    cam = np.asarray(best[1]) if best else np.array([*mid[:2], 3.0])
    return cam, mid


class Director:
    """The room camera for clips: 2.5 m up, 3 to 5 m from the person (or from the middle of both agents when they are
    within 5 m of each other), at a walkable point with clear lines to whom it frames. It keeps its spot while that
    still sees them, re-chooses every 2 s, and glides there."""

    def __init__(self, sim: B1K):
        self.sim, self.goal, self.cam, self.look, self.next = sim, None, None, None, 0.0

    def _sees(self, cam, pts) -> bool:
        return all(self.sim.clear(cam, p, n) for p, n in pts)

    def _framed(self, cam, pts) -> bool:
        """Whether, looking where it looks, the camera has everyone within 22 degrees of its axis."""
        if self.look is None:
            return True
        ax = (self.look - cam) / (np.linalg.norm(self.look - cam) + 1e-9)
        return all(float(ax @ ((p - cam) / (np.linalg.norm(p - cam) + 1e-9))) > np.cos(np.radians(22)) for p, _ in pts)

    def step(self, t: float) -> None:
        sim = self.sim
        pp, rp = sim.pose["person"][0], sim.pose["robot"][0]
        both = float(np.linalg.norm(pp - rp)) < 3.0
        center = np.array([*((pp + rp) / 2 if both else pp), 1.0])
        pts = [(np.array([*pp, 1.3]), None)] + ([(np.array([*rp, 0.9]), "robot")] if both else [])
        if t >= self.next or self.goal is None:
            self.next = t + 2.0
            if self.goal is None or not self._sees(self.goal, pts) or not self._framed(self.goal, pts):
                fl, best = sim.floor(), None
                for r in (3.5, 4.25, 5.0):
                    for a in np.linspace(0, 2 * np.pi, 24, endpoint=False):
                        q = center[:2] + r * np.array([np.cos(a), np.sin(a)])
                        if fl.clearance(q) < 0.3:
                            continue
                        cam = np.array([q[0], q[1], 2.4])
                        if not self._sees(cam, pts):
                            continue
                        move = float(np.linalg.norm(cam - self.goal)) if self.goal is not None else 0.0
                        best = min(best, (move, r, cam.tolist())) if best else (move, r, cam.tolist())
                if best:
                    self.goal = np.asarray(best[2])
                elif self.goal is None:
                    self.goal = center + np.array([2.0, 2.0, 1.5])
        self.cam = self.goal if self.cam is None else self.cam + 0.12 * (self.goal - self.cam)
        self.look = center if self.look is None else self.look + 0.15 * (center - self.look)
        sim.cams["third_cam"].set_position_orientation(position=self.cam, orientation=look_quat(self.cam, self.look))


def make_robot(policy: str, cfg: Config, seed: int):
    from helper.partnr import robot as R

    class Oracle(R.Robot):
        """Knows the task and what the person is on: a random open step (not done, its predecessors done) that the
        person is not on and nobody has claimed. Unlike PARTNR's, it ignores the person's failed attempts."""
        name = "oracle"
        declined = {}  # step -> until when the person said they are on it

        def choose(self, t: float, seen: dict) -> str | None:
            done, p = seen["done"], self.person
            ok = [v for v in self.task["nodes"] if v not in done and p.req[v] <= done and v != p.cur
                  and v not in self.claims and self.declined.get(v, -1) < t]
            if not ok:
                return None
            if p.cur is None or self.where is None:
                return ok[self.rng.integers(len(ok))]
            theirs = self.where(self.task["object"][p.cur])  # the step farthest from the person's
            return max(ok, key=lambda v: float(np.linalg.norm(self.where(self.task["object"][v])[:2] - theirs[:2])))

        def seek(self, v: str, done: set) -> float:
            """The chance the person goes for step v soon: their current step, else any ready one."""
            return 1.0 if self.person.cur == v else 0.3 if self.person.req[v] <= done else 0.05

    class Bip(R.Bip):
        """BIP; steps the person said they are on are blocked for a while."""
        declined = {}

        def observe(self, t: float, seen: dict) -> None:
            self.now = t
            super().observe(t, seen)

        def unknown(self):
            return np.array([self.declined.get(v, -1) >= getattr(self, "now", 0.0) for v in self.task["nodes"]])

        def seek(self, v: str, done: set) -> float:
            """The chance the person goes for step v soon: BIP's posterior that it is their current or next step."""
            F, i = self.B.F, self.task["nodes"].index(v)
            return float(min(1.0, F.marginal(F.g)[i] + F.marginal(F.n)[i]))

        def choose(self, t: float, seen: dict) -> str | None:
            v = super().choose(t, seen)
            if v is not None:
                return v
            done, p = seen["done"], self.person  # a step its belief counts done that the scene shows undone
            ever = {c["node"] for c in seen["completions"]}  # done once, knocked out of place since
            theirs = self.task["nodes"][self.read] if self.read is not None and self.read < len(self.task["nodes"]) else None
            redo = [u for u in self.task["nodes"] if u in ever and u not in done and p.req[u] <= done
                    and u != theirs and u not in self.claims and self.declined.get(u, -1) < t]
            return redo[0] if redo else None

    base = policy.split("+")[0]
    if base == "alone":
        return None
    r = {"idle": R.Robot, "graph": R.Graph, "oracle": Oracle, "bip": Bip}[base](cfg, seed)
    r.declined, r.where = {}, None  # where: object -> position, set by the loop
    return r


def run(row: dict, sim: B1K, policy: str, seed: int, out: Path, start_state, cfg: Config) -> dict:
    """One episode; returns its stream (also written to out/streams)."""
    sim.og.sim.load_state(start_state)
    sim.restored()
    T = task_dict(row, sim)
    subs = {v: row["subgoals"][i] for i, v in enumerate(T["nodes"])}
    beliefs = {o: sim.pos(o).tolist() for o in T["object"].values()}  # where the person thinks each object is
    skills = Skills(sim, beliefs)
    by_obj = {s["obj"]: s for s in row["subgoals"]}
    skills.furniture = {s["target"] for s in row["subgoals"]} - set(by_obj)
    skills.small = set(by_obj) | {s["target"] for s in row["subgoals"] if s["target"] in by_obj or s["rel"] == "nextto"}
    skills.in_place = lambda o: o in by_obj and holds(sim, by_obj[o])
    furniture = sorted({s["target"] for s in row["subgoals"]})
    person = BPerson(cfg, T, furniture, seed, sim, skills)
    robot = make_robot(policy, cfg, seed)
    if robot is not None:
        robot.start(T, f"{row['activity']}_{seed}", person)
        robot.where = sim.pos
    gaze = Gaze(cfg, seed)
    sim.place_agent("person", np.asarray(row["person_start"][:2], float), float(row["person_start"][2]))
    sim.head = head_target(sim, person, T)  # ahead along the way
    rs = robot_spot(row)
    sim.place_agent("robot", np.asarray(rs[:2], float) if robot is not None else np.array([100.0, 100.0]), float(rs[2]))
    seen = {"done": set(), "completions": [], "actions": person.actions, "fixations": [], "held": [], "frames": None}
    person_done, by, log, events = set(), {}, [], []
    since, done_at = {}, {}  # subgoal -> when its relation began to hold; -> where its object was when it counted
    objs_at = []
    utter = {"robot": [], "person": []}
    talk = policy.split("+")[1] if "+" in policy else "silent"
    heard_claims, asked, told, dwell, inview, moves = {}, set(), set(), {}, {}, {}  # moves: object -> (phase, since)
    assumed = {}  # step -> when the person found its object gone and left it to the robot
    log_n = 0  # person log lines read
    steps_of = {o: [v for v in T["nodes"] if T["object"][v] == o] for o in T["object"].values()}
    hold = {"robot": -1.0, "person": -1.0}  # until when each agent stands, speaking or listening
    costs = Costs()

    def say(who: str, text: str, t_: float) -> None:
        utter[who].append({"t": round(t_, 2), "text": text})
        hold[who] = max(hold[who], t_ + SAY_S)
    def tell_moves(t_: float) -> None:
        """Correction: each object the robot picks up, and again once it puts it in place, is worth telling about by
        the chance the person missed it (gaze dwell on it, or time in view without gaze) times the chance they go for
        it: while the robot carries it, certain if their action names it, else BIP's posterior; once in place, certain."""
        o_h = sim.held["robot"]
        if o_h is not None and moves.get(o_h, (None,))[0] != "held":
            moves[o_h] = ("held", t_)
            dwell[o_h] = inview[o_h] = 0.0
        for o, (phase, t0) in list(moves.items()):
            if phase == "held" and o != o_h:  # set down: in place once the scene counts it, else left for later
                moves[o] = phase, t0 = ("set", t_)
            if phase == "set":
                if any(v in done and by.get(v) == "robot" for v in steps_of.get(o, [])):
                    moves[o] = ("placed", t_)
                    dwell[o] = inview[o] = 0.0
                elif t_ - t0 > 1.0:
                    moves[o] = ("left", t_)
                continue
            if phase == "left" or (o, phase, t0) in told:
                continue
            look = gaze.target == o or (phase == "held" and gaze.target == "robot")
            dwell[o] += DT if look and (o in vis or "robot" in vis) else 0.0
            inview[o] += DT if o in vis or (phase == "held" and "robot" in vis) else 0.0
            p_seen = 1 - np.exp(-dwell[o] / 0.4) if talk == "tell" else 1 - np.exp(-inview[o] / 2.0)
            open_ = [v for v in steps_of.get(o, []) if v not in done] or steps_of.get(o, [])
            if not open_ or t_ - t0 < 1.5:  # a moment to see it first
                continue
            v = robot.cur if phase == "held" and robot.cur in open_ else open_[0]
            if phase == "placed":
                mine = [v_ for v_ in steps_of[o] if v_ in done and by.get(v_) == "robot"]
                if not mine:  # knocked out of place since: nothing to tell
                    continue
                v = mine[-1]
            if phase == "held":  # their action names it (they head for it), else BIP's posterior
                heading = any(o in a["entities"] for a in person.actions[-3:] if a["t"] >= t_ - 2.0)
                p_seek = 1.0 if heading else robot.seek(v, seen["done"])
            else:
                p_seek = 1.0  # an open step is done some time
            if talk != "always" and tell_value(1 - p_seen, p_seek, costs) <= 0:
                continue
            told.add((o, phase, t0))
            if phase == "held":
                say("robot", f"I have the {T['names'][o]}.", t_)
                heard_claims[v] = t_ - cfg.notice_s  # the person leaves it to the robot
            else:
                say("robot", f"I put the {T['labels'][T['nodes'].index(v)]}.", t_)
                person_done.add(v)
            beliefs[o] = sim.pos(o).tolist()
            hold["person"] = max(hold["person"], t_ + 0.8)
    robot_seen_at, near, vis = -9.0, [], {}
    task_names = {x for v in T["nodes"] for x in T["objects"][v]}
    all_furniture = [o.name for o in sim.scene.objects if o.name not in task_names and o.name != "robot"
                     and getattr(o, "category", "") not in ("walls", "floors", "ceilings", "background", "carpet")]
    t, t_max, wall = 0.0, float(row.get("t_max", 240.0)), time.time()
    eid = f"{row['activity']}_{policy}_{seed}"
    frames = out / "frames" / eid
    video = "third_cam" in sim.cams  # a clip: both views every frame
    if video:
        import cv2
        import shutil
        import subprocess
        tmp = out / "tmp" / eid
        for v in ("head", "third"):
            (tmp / v).mkdir(parents=True, exist_ok=True)
        director = Director(sim)
    k = 0
    while t < t_max:
        now = {v for v, s in subs.items() if holds(sim, s)}
        for v in now:  # held for 0.5 s counts; undone only once the object has moved 5 cm from where it was done
            since.setdefault(v, t)
        for v in list(since):
            if v not in now:
                since.pop(v)
        skills.t = t
        for v, t0 in since.items():
            just_placed = t - skills.placed_at.get(subs[v]["obj"], -9.0) < 1.0
            if v not in done_at and (t - t0 >= 0.5 or just_placed):
                done_at[v] = sim.pos(subs[v]["obj"])
        for v in list(done_at):
            moved = float(np.linalg.norm(sim.pos(subs[v]["obj"]) - done_at[v]))
            if v not in now and moved > 0.05:
                done_at.pop(v)
                held = [w for w, o in sim.held.items() if o == subs[v]["obj"]]
                events.append([round(t, 2), "undone", v, round(moved, 3), held[0] if held else None])
        done = set(done_at)
        for v in sorted(done - seen["done"]):  # a new completion, credited to whoever set the object down last
            by.setdefault(v, skills.placed_by.get(subs[v]["obj"], "start"))
            seen["completions"].append({"t": round(t, 2), "node": v, "by": by[v]})
            events.append([round(t, 2), "done", v, by[v]])
        seen["done"] = done
        if len(done) == len(subs):
            break
        if int(round(t / DT)) % 5 == 0:  # furniture near the person, for gaze to wander over
            here = sim.pose["person"][0]
            near = [f for f in all_furniture if float(np.linalg.norm(sim.pos(f)[:2] - here)) < 5.0]
        if k > 0 and k % 2 == 0:  # what the person sees, at 7.5 Hz; raycasts are stale before the first step
            vis = sim.visible(sorted(task_names | set(near) | {"robot"}), strict=tuple(T["object"].values()))
        for v, s in subs.items():  # what the person sees updates where they think things are and what is done
            if s["obj"] in vis:
                beliefs[s["obj"]] = sim.pos(s["obj"]).tolist()
                if v in done and s["target"] in vis:
                    person_done.add(v)
                elif v in person_done and v not in done and s["target"] in vis:  # knocked out of place, and seen
                    person_done.discard(v)
        if by and person.skills_done:
            person_done |= {v for v in done if by.get(v) == "person"}
        for i in range(log_n, len(person.log)):  # what the person found where they went
            line = person.log[i]
            v_at = next((v for v, s_ in subs.items() if f"[{s_['obj']}]" in line), None)
            if v_at is None:
                continue
            if line.endswith("is already in place"):  # seen in place: done, not a failure
                person_done.add(v_at)
                person.fails[v_at] = max(0, person.fails[v_at] - 1)
                person.log[i] += " (counted done)"
            elif line.endswith("is not here") and robot is not None:  # gone: the robot has it
                assumed[v_at] = t
                person.fails[v_at] = max(0, person.fails[v_at] - 1)
                person.log[i] += " (left to the robot)"
        log_n = len(person.log)
        if "robot" in vis:
            robot_seen_at = t
            person.robot_seen = sim.pose["robot"][0].copy()
        claims = {} if robot is None else {v: since for v, since in robot.claims.items() if t - robot_seen_at < 3.0}
        claims.update({v: t0 for v, t0 in heard_claims.items() if robot is not None and v in robot.claims})
        claims.update({v: t0 - cfg.notice_s for v, t0 in assumed.items() if t - t0 < 20.0 and v not in done})
        for agent, who in ((0, "robot"), (1, "person")):  # nothing stays in a hand without a step for it
            idle_hand = robot is None or robot.cur is None if who == "robot" else person.cur is None
            if sim.held[who] is not None and idle_hand and skills.run[agent] is None:
                skills._set_down(agent, sim.held[who])
        sim.reach = None  # set again by the person's running hand skill, if any
        if k % int(30 / DT) == 0:  # giving a step up lasts 30 s: things get moved, and someone has to finish it
            for agent_ in (person, robot):
                if agent_ is not None:
                    for v in [v for v, n in agent_.fails.items() if n >= cfg.max_fails and v not in done]:
                        agent_.fails[v] = cfg.max_fails - 1
        if t >= hold["person"]:
            person.act(t, skills, person_done, claims, sorted(vis))
        if robot is not None and t >= hold["robot"]:
            claimed_before = set(robot.claims)
            robot.act(t, skills, seen)
            for v in sorted(set(robot.claims) - claimed_before):  # a step just taken
                if talk == "announce":
                    say("robot", f"I'll put the {T['labels'][T['nodes'].index(v)]}.", t)
                    heard_claims[v] = t - cfg.notice_s  # heard: the person leaves it at once
                elif talk == "ask" and v not in asked:
                    asked.add(v)
                    say("robot", f"Shall I put the {T['labels'][T['nodes'].index(v)]}?", t)
                    if person.cur == v:  # they are on it: the robot leaves it
                        say("person", "No, I'm doing that.", t + 1.0)
                        robot.declined[v] = t + 15.0
                        robot._end(t, "declined")
                    else:
                        say("person", "Yes, please.", t + 1.0)
                        heard_claims[v] = t - cfg.notice_s
                    hold["person"] = max(hold["person"], t + SAY_S + 1.0)
        if talk in ("tell", "tellnogaze", "always") and robot is not None:
            tell_moves(t)
        cur = person.cur
        nxt = next((v for v in person.order if v not in person_done and v != cur), None)
        ents = lambda v: set(T["objects"][v]) if v else set()
        if person.skill and person.skill[0] in ("Pick", "Place") and getattr(person, "_planned", None) != person.skill:
            person._planned = person.skill
            ent = person.skill[1].split(",")[0 if person.skill[0] == "Pick" else 2].strip()
            gaze.plan(t + HAND_S, ent, "pick" if person.skill[0] == "Pick" else "place")
        sim.head = head_target(sim, person, T)
        if video:
            director.step(t)
        sim.step()
        partner = {"robot"} if robot is not None else set()  # glances at the robot, as at what comes next
        g, fix = gaze.step(t, {n: vis[n] for n in vis}, ents(cur), ents(nxt) | partner)
        if fix:
            seen["fixations"].append(fix)
        if int(round(t / DT)) % 15 == 0:  # where the task's objects are, once a second
            objs_at.append([round(t, 1), {o: [round(float(v), 3) for v in sim.pos(o)] for o in T["object"].values()}])
        log.append({"t": round(t, 2), "gaze": [round(x, 4) for x in g], "person": sim.pose["person"][0].tolist(),
                    "robot": sim.pose["robot"][0].tolist(), "cur": cur, "robot_cur": robot.cur if robot else None,
                    "held": dict(sim.held)})
        if frames:
            frames.mkdir(parents=True, exist_ok=True)
            if int(round(t * 15)) % 15 == 0:  # one head frame a second
                from PIL import Image
                Image.fromarray(sim.rgb("person_cam")).save(frames / f"{int(round(t * 10)):05d}.jpg")
        if video:
            cv2.imwrite(str(tmp / "head" / f"{k:06d}.jpg"), cv2.cvtColor(sim.rgb("person_cam"), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(tmp / "third" / f"{k:06d}.jpg"), cv2.cvtColor(sim.rgb("third_cam"), cv2.COLOR_RGB2BGR))
        k += 1
        t += DT
    person.close(t)
    if video:
        (out / "videos").mkdir(parents=True, exist_ok=True)
        ff = str(Path.home() / ".conda/envs/egocollab/bin/ffmpeg")
        for v in ("head", "third"):
            subprocess.run([ff, "-v", "error", "-y", "-framerate", f"{1 / DT:g}", "-i", str(tmp / v / "%06d.jpg"), "-c:v",
                            "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out / "videos" / f"{eid}_{v}.mp4")], check=True)
        shutil.rmtree(tmp, ignore_errors=True)
    st = {"activity": row["activity"], "scene": row["scene"], "policy": policy, "seed": seed, "subgoals": row["subgoals"],
          "names": T["names"],
          "orderings": row["orderings"], "seconds": round(t, 2), "finished": len(seen["done"]) == len(subs),
          "by": by, "person_segments": person.segments, "robot_segments": robot.segments if robot else [],
          "events": events, "final_done": sorted(seen["done"]), "official_done": official_done(sim, subs),
          "failures": skills.failures, "objects_at": objs_at, "utterances": utter, "fixations": seen["fixations"], "person_log": person.log,
          "robot_log": robot.log if robot else [], "log": log, "wall_s": round(time.time() - wall, 1)}
    (out / "streams").mkdir(parents=True, exist_ok=True)
    (out / "streams" / f"{row['activity']}_{policy}_{seed}.json").write_text(json.dumps(st))
    return st


def main() -> None:
    rows = json.load(open(sys.argv[1]))
    out = Path(sys.argv[2]).resolve()
    args = dict(a.split("=", 1) for a in sys.argv[3:])
    keep = set(args.get("activity", "").split(",")) - {""}
    rows = [{**r, "t_max": float(args.get("t_max", 240))} for r in rows if not keep or r["activity"] in keep]
    policies = args.get("policy", "alone,bip").split(",")
    seeds = [int(s) for s in args.get("seed", "0").split(",")]
    cfg = Config(person=args.get("person", "hitl"), idle_ratio=float(args.get("idle_ratio", 0.1)),
                 prior=args.get("prior", "truth"),  # Gemini's prior: computed on Redwood, its credentials stay there
                 surprise=float(args.get("surprise", "inf")))  # no Gemini revisions on the 5090  # PARTNR's 0.72 came from people driving a simulator
    for row in rows:
        sim = B1K(row["scene"], scene_file=str(ROOT / row["template"]), rooms=rooms_of(row["activity"]),
                  third=args.get("clip") == "1")
        try:
            from omnigibson.object_states import Open
            for s in row["subgoals"]:  # containers with doors or lids start closed; open ones take things from above
                if s["rel"] == "inside" and Open in sim.obj(s["target"]).states and s["target"] not in sim.inside:
                    sim.close_box(s["target"])
            here = np.asarray((row.get("robot_start") or {}).get("position", [0, 0])[:2], float)
            p0 = sim.stand_spot(None, here, point=here, gap=1.5)  # near the robot, off walls
            p0 = p0 if p0 is not None else here
            way = sim.path(p0, sim.pos(row["subgoals"][0]["obj"])[:2], strict=True)  # facing the first walk
            to = (way[1] if way is not None and len(way) > 1 else sim.pos(row["subgoals"][0]["obj"])[:2]) - p0
            h = to / (np.linalg.norm(to) + 1e-9)
            row = {**row, "person_start": [*p0, float(np.arctan2(h[1], h[0]))]}
            sim.place_agent("person", row["person_start"][:2], row["person_start"][2])
            r0 = sim.stand_spot(None, p0, point=p0 - 1.3 * h, gap=0.0, away=p0)  # the robot behind them
            if r0 is not None:
                row = {**row, "robot_spot": [*np.asarray(r0, float).tolist(), row["person_start"][2]]}
            sim.place_agent("robot", np.asarray(robot_spot(row)[:2], float), float(robot_spot(row)[2]))
            sim.arm_geometry()  # measured before the start state: every episode begins with the arm at carrying height
            for _ in range(5):
                sim.step()
            start = sim.og.sim.dump_state(serialized=False)
            for seed in seeds:
                for policy in policies:
                    if (out / "streams" / f"{row['activity']}_{policy}_{seed}.json").exists():
                        continue
                    st = run(row, sim, policy, seed, out, start, cfg)
                    print(json.dumps({k: st[k] for k in ("activity", "policy", "seed", "seconds", "finished", "by", "official_done", "wall_s")}),
                          flush=True)
        finally:
            sim.og.clear()


if __name__ == "__main__":
    main()
