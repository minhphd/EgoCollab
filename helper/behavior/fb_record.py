"""False-belief tasks in BEHAVIOR-1K (docs/false_belief_tasks.md), recorded with BIP's live loop running online.

The person puts X on A; the robot later moves X into the closed container B; the person comes back for X. The six
conditions differ only in what the person sees and attends during the move. Gaze is read as the entity it falls on
(a small object within aoi of the gaze point first, else whatever a ray through the gaze point hits), never as pixels. The live loop (helper/bip/
live.py) reads the person's view, the gaze labels and both agents' actions and decides when to tell the person where
X is. Every condition of a design runs from the same start, so watched and missed share their video.

    python -m helper.behavior.fb_record <designs.json> <out_dir> [only=b000,b003] [validate=1] [act_on_tells=0]   # RTX GPU
act_on_tells=0 records the silent robot: BIP's tells are logged, the person never hears them.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .sim import B1K, DT, EYE, TURN, WALK, look_quat

CONDS = ("left", "turned", "watched", "missed", "back_watched", "back_missed")
BELIEF = {"left": "A", "turned": "A", "watched": "B", "missed": "A", "back_watched": "B", "back_missed": "A"}


@dataclass
class FB:
    max_s: float = 150.0
    frame_every: float = 0.5  # s between the frames kept as images
    dwell_s: float = 0.6  # free gaze: median dwell
    dwell_sigma: float = 0.5
    jitter: float = 0.02  # gaze noise, image widths
    aoi: float = 0.04  # an object within this of the gaze point is the one looked at
    keep_away: float = 0.15  # gaze that avoids X stays this far from it
    p_antic: float = 0.8  # the first look goes to the believed place
    look1: float = 1.2
    look2: float = 1.0
    min_area: float = 2e-5
    search_s: float = 3.0  # s the person looks around after finding the believed place empty
    tell_s: float = 2.5  # s of the person's time to hear where X is
    ffmpeg: str = str(Path.home() / ".conda/envs/egocollab/bin/ffmpeg")


# -- the person's attention and the gaze label ---------------------------------------------------------------
class Gaze:
    def __init__(self, fb: FB, rng, small: set):
        self.fb, self.rng, self.small = fb, rng, small
        self.target, self.until = None, -1.0

    def step(self, t: float, mode: tuple, masks: dict, sim: B1K) -> tuple:
        fb, kind = self.fb, mode[0]
        if kind == "seq":
            (first, _), (second, at2), t0 = mode[1][0], mode[1][1], mode[2]
            self.target = first if t - t0 < at2 else second
        elif kind == "focus" and mode[1] in masks:
            self.target = mode[1]
        else:
            exclude, strict = (tuple(mode[1]), mode[2]) if kind == "free" else ((), False)
            near = lambda n: strict and any(np.hypot(masks[n][0] - masks[e][0], masks[n][1] - masks[e][1]) <= fb.keep_away
                                             for e in exclude if e in masks and e in self.small)
            if t >= self.until or self.target not in masks or self.target in exclude or near(self.target):
                names_ = [n for n in sorted(masks) if n not in exclude and not near(n)]
                self.target = names_[self.rng.integers(len(names_))] if names_ else None
                self.until = t + fb.dwell_s * float(np.exp(fb.dwell_sigma * self.rng.standard_normal()))
        if self.target in masks:
            cx, cy, _ = masks[self.target]
            xy = [cx + fb.jitter * self.rng.standard_normal(), cy + fb.jitter * self.rng.standard_normal()]
        else:
            xy = [0.5 + 0.1 * self.rng.standard_normal(), 0.45 + 0.1 * self.rng.standard_normal()]
        xy = [float(np.clip(v, 0, 0.999)) for v in xy]
        objs = [(np.hypot(c[0] - xy[0], c[1] - xy[1]), n) for n, c in masks.items() if n in self.small]
        near_obj = min(objs) if objs else (1.0, None)
        read = near_obj[1] if near_obj[0] <= fb.aoi else sim.gaze_hit(xy)
        return xy, read, self.target


# -- scripted agents -------------------------------------------------------------------------------------------
class Agent:
    """Steps: ("walk", xy), ("face", point), ("wait", s), ("pick", o), ("place", o, rel, f), ("open", f, bool),
    ("signal", flag), ("until", flag), ("head", mode, ...), ("attend", mode, ...). Bookkeeping steps run at once."""

    def __init__(self, who: str, sim: B1K, steps: list, flags: dict, log: list, events: list):
        self.who, self.sim, self.steps, self.flags, self.log, self.events = who, sim, list(steps), flags, log, events
        self.path, self.until_t, self.head, self.attend = None, None, ("auto",), ("free", (), False)

    @property
    def finished(self) -> bool:
        return not self.steps and self.path is None

    def act(self, t: float) -> None:
        sim = self.sim
        while self.steps:
            kind, *a = self.steps[0]
            if kind == "goforx":
                believed = "B" if "told" in self.flags or BELIEF[self.flags["cond"]] == "B" else "A"
                self.flags["went_for"] = believed
                self.steps[:1] = fetch_steps(self.flags["fb"], self.flags["d"], believed)
                continue
            if kind == "walk" and self.who == "person" and "told" in self.flags and self.flags.get("went_for") == "A" \
                    and "at_A" not in self.flags and not self.flags.get("redirected"):
                self.flags["redirected"] = round(t, 2)  # heard where X is on the way: turn to the container
                self.path = None
                self.steps = fetch_steps(self.flags["fb"], self.flags["d"], "B")
                continue
            if kind == "walk":
                if self.path is None:
                    xy, yaw = sim.pose[self.who]
                    self.path = list(sim.path(xy, a[0])[1:]) or [np.asarray(a[0], float)]
                if self._walk():
                    self.path = None
                    self.steps.pop(0)
                    continue
                return
            if kind == "face":
                xy, yaw = sim.pose[self.who]
                want = float(np.arctan2(a[0][1] - xy[1], a[0][0] - xy[0]))
                d = (want - yaw + np.pi) % (2 * np.pi) - np.pi
                if abs(d) > np.radians(3):
                    sim.place_agent(self.who, xy, yaw + np.clip(d, -TURN * DT, TURN * DT))
                    return
            elif kind == "wait":
                self.until_t = t + a[0] if self.until_t is None else self.until_t
                if t < self.until_t:
                    return
                self.until_t = None
            elif kind == "until" and a[0] not in self.flags:
                return
            elif kind == "signal":
                self.flags[a[0]] = round(t, 2)
            elif kind == "head":
                self.head = tuple(a)
            elif kind == "attend":
                self.attend = tuple(a) + ((round(t, 2),) if a[0] == "seq" else ())
            elif kind == "pick":
                sim.pick(self.who, a[0])
                self.events.append((round(t, 2), self.who, "Pick", a[0]))
            elif kind == "place":
                ok = sim.place(self.who, a[0], a[1], a[2])
                self.events.append((round(t, 2), self.who, "Place", a[0], a[1], a[2], ok))
            elif kind == "open":
                sim.open(a[0], a[1])
                self.events.append((round(t, 2), self.who, "Open" if a[1] else "Close", a[0]))
            self.steps.pop(0)
            self.log.append([round(t, 2), self.who, kind, *[str(x) for x in a]])

    def _walk(self) -> bool:
        """One frame along the path: turn toward the next waypoint, then step WALK * DT. True on arrival."""
        sim, (xy, yaw) = self.sim, self.sim.pose[self.who]
        while self.path and np.linalg.norm(self.path[0] - xy) < 0.05:
            self.path.pop(0)
        if not self.path:
            return True
        nxt = np.asarray(self.path[0], float)
        want = float(np.arctan2(nxt[1] - xy[1], nxt[0] - xy[0]))
        d = (want - yaw + np.pi) % (2 * np.pi) - np.pi
        yaw = yaw + float(np.clip(d, -TURN * DT, TURN * DT))
        if abs(d) < np.radians(45):
            step = min((sim.walk_speed if self.who == "person" else WALK) * DT, float(np.linalg.norm(nxt - xy)))
            xy = xy + step * (nxt - xy) / (np.linalg.norm(nxt - xy) + 1e-9)
        sim.place_agent(self.who, xy, yaw)
        return False


def person_script(fb: FB, d: dict, cond: str, first: str) -> list:
    X, A, B, P = d["X"], d["A"], d["B"], d["spot"]["xy"]
    avoid = ("free", (X, A, B), True)
    mid = d["mid"]
    pre = [("attend", "free", (), False), ("walk", d["X_from_xy"]), ("face", d["X_from_top"]), ("head", "at", X),
           ("attend", "focus", X), ("wait", 0.6), ("pick", X), ("head", "auto"), ("walk", d["A_xy"]), ("face", d["A_top"]),
           ("head", "at", A), ("place", X, "on", A), ("signal", "placed"), ("head", "at", X), ("wait", 1.5),
           ("head", "auto"), ("attend", *avoid)]
    face = [("face", mid), ("head", "mid")]
    watch = ("attend", "focus", X) if cond.endswith("watched") else ("attend", *avoid)
    mid_steps = {
        "left": [("walk", d["Q_xy"]), ("signal", "away"), ("until", "moved"), ("wait", 1.0), ("walk", P), *face, ("wait", 1.0)],
        "turned": [("walk", P), ("face", 2 * np.asarray(P) - np.asarray(mid[:2])), ("head", "ahead"), ("signal", "ready"),
                   ("until", "moved"), ("wait", 1.0), *face, ("wait", 1.0)],
        "watched": [("walk", P), *face, ("signal", "ready"), ("until", "approach"), watch, ("until", "moved"),
                    ("wait", 1.0), ("attend", *avoid), ("wait", 0.5)],
        "back_watched": [("walk", d["Q_xy"]), ("signal", "away"), ("until", "picked"), ("walk", P), *face, ("signal", "back"),
                         watch, ("until", "moved"), ("wait", 1.0), ("attend", *avoid), ("wait", 0.5)]}
    mid_steps["missed"], mid_steps["back_missed"] = mid_steps["watched"], mid_steps["back_watched"]
    other = B if first == A else A
    look = [("attend", "seq", ((first, 0.0), (other, fb.look1))), ("wait", fb.look1), ("signal", "query"), ("wait", fb.look2)]
    return pre + mid_steps[cond] + look + [("signal", "go"), ("goforx",)]


def fetch_steps(fb: FB, d: dict, believed: str) -> list:
    """Going for X: to the believed place; if X is not there, look around, then to the container, open it, take X."""
    X, A, B = d["X"], d["A"], d["B"]
    at_b = [("walk", d["B_xy"]), ("face", d["B_top"]), ("head", "at", B), ("attend", "focus", B), ("signal", "at_B"),
            ("open", B, True), ("wait", 0.8), ("attend", "focus", X), ("pick", X), ("signal", "got"), ("wait", 0.5)]
    if believed == "B":
        return at_b
    return [("walk", d["A_xy"]), ("face", d["A_top"]), ("head", "at", A), ("attend", "focus", A), ("signal", "at_A"),
            ("wait", 1.5), ("head", "ahead"), ("attend", "free", (), False), ("wait", fb.search_s), ("signal", "searched")] + at_b


def robot_script(d: dict, cond: str) -> list:
    start = [("until", "away"), ("wait", 1.0)] if cond in ("left", "back_watched", "back_missed") else [("until", "ready"), ("wait", 1.5)]
    hold = [("until", "back"), ("wait", 0.5)] if cond.startswith("back") else []
    return start + [("walk", d["rA_xy"]), ("face", d["A_top"]), ("signal", "approach"), ("wait", 0.8), ("pick", d["X"]),
                    ("signal", "picked"), ("walk", d["rB_xy"]), ("face", d["B_top"]), *hold, ("open", d["B"], True),
                    ("wait", 0.8), ("place", d["X"], "in", d["B"]), ("signal", "placed_in"), ("wait", 0.8),
                    ("open", d["B"], False), ("signal", "moved"), ("walk", d["park_xy"])]


# -- one task ---------------------------------------------------------------------------------------------------
def place_of(events: list, x: str, start: str) -> str:
    """Where x is, from the agents' own picks and places (object-state checks miss some containers)."""
    loc = f"on {start}"
    for e in events:
        if e[2] == "Pick" and e[3] == x:
            loc = e[1]
        elif e[2] == "Place" and e[3] == x:
            loc = f"{e[4]} {e[5]}"
    return loc


def location(sim: B1K, d: dict) -> str:
    for who in ("person", "robot"):
        if sim.held[who] == d["X"]:
            return who
    from omnigibson.object_states import Inside, OnTop
    x = sim.obj(d["X"])
    if x.states[Inside].get_value(sim.obj(d["B"])):
        return f"in {d['B']}"
    if x.states[OnTop].get_value(sim.obj(d["A"])):
        return f"on {d['A']}"
    if d.get("X_from") and x.states[OnTop].get_value(sim.obj(d["X_from"])):
        return f"on {d['X_from']}"
    return "unknown"


def run(fb: FB, sim: B1K, d: dict, cond: str, out: Path, start_state, act_on_tells: bool = True) -> dict:
    import cv2
    from helper.bip.config import Config
    from helper.bip.io import Act, EgoFrame, SceneObs
    from helper.bip.live import Live
    sim.og.sim.load_state(start_state, serialized=False)
    sim.restored()
    sim.place_agent("person", d["person0"][:2], d["person0"][2])
    sim.place_agent("robot", d["robot0"][:2], d["robot0"][2])
    for _ in range(2):
        sim.step()
    seed = int(d["design"][1:])
    rng = np.random.default_rng([seed, 1])
    first = d["A"] if (BELIEF[cond] == "A") == (rng.random() < fb.p_antic) else d["B"]
    flags, steps_log, events = {"cond": cond, "fb": fb, "d": d}, [], []
    person = Agent("person", sim, person_script(fb, d, cond, first), flags, steps_log, events)
    robot = Agent("robot", sim, robot_script(d, cond), flags, steps_log, events)
    gaze = Gaze(fb, np.random.default_rng([seed, 2]), d["small"])
    furniture = set(d["furniture"])
    live = Live(Config(premap=False), {"id": d["design"], "nodes": [], "labels": [], "objects": {}, "spec": {},
                                       "furniture": sorted(furniture), "map": []}, slow_model="")
    X = d["X"]
    live.on(SceneObs(0.0, [(X, "on", d["X_from"])]))
    tid = f"{d['design']}_{cond}"
    tmp, fdir = out / "tmp" / tid, out / "frames" / tid
    for p in (tmp / "head", tmp / "third", fdir):
        shutil.rmtree(p, ignore_errors=True)
        p.mkdir(parents=True)
    log, track, tells, n_ev, k, masks_all = [], [], [], 0, 0, {}
    t0 = time.time()
    while sim.t < fb.max_s + 1e9:
        t = round(k * DT, 3)
        person.act(t)
        robot.act(t)
        sim.head = head_point(sim, d, person, t)
        sim.step()
        rgb = sim.rgb()
        full = k % round(fb.frame_every / DT) == 0
        if full:  # every entity in view twice a second; each frame only those that matter now
            masks_all = {n: v for n, v in sim.visible(strict=(X,)).items() if v[2] >= fb.min_area}
        now = sim.visible([n for n in (X, d["A"], d["B"], gaze.target) if n], strict=(X,))
        masks = {**{n: v for n, v in masks_all.items() if n not in (X, d["A"], d["B"], gaze.target)},
                 **{n: v for n, v in now.items() if v[2] >= fb.min_area}}
        xy, read, target = gaze.step(t, person.attend, masks, sim)
        loc = place_of(events, X, d["X_from"])
        if not track or track[-1][1] != loc:
            track.append([t, loc])
        for e in events[n_ev:]:  # both agents' actions reach the live loop: the robot's own, the person's recognized
            if e[2] == "Pick":
                live.on(Act(e[0], "H" if e[1] == "person" else "R", "Pick", (e[3],)))
            elif e[2] == "Place":
                live.on(Act(e[0], "H" if e[1] == "person" else "R", "Place", (e[3], e[4], e[5], None)))
            else:
                live.on(Act(e[0], "H" if e[1] == "person" else "R", e[2], (e[3],)))
        n_ev = len(events)
        if "moved" in flags and X not in live.expected:
            live.expect(X)  # the person will want X; live, S1's next subgoal says so
        for c in live.on(EgoFrame(t, {n: v for n, v in masks.items()}, tuple(xy), fixated=read or "")):
            if c.kind == "tell":
                tells.append({"t": t, "p_fb": round(live.mind.false_belief(X), 3)})
                if act_on_tells and "told" not in flags:
                    flags["told"] = round(t, 2)  # the person hears where X is
        log.append({"t": t, "gaze": [round(v, 4) for v in xy], "read": read, "attend": target,
                    "vis": {c: [round(v, 4) for v in masks[n]] for c, n in (("X", X), ("A", d["A"]), ("B", d["B"])) if n in masks},
                    "seen": sorted(masks) if k % round(fb.frame_every / DT) == 0 else None,
                    "person": [round(float(v), 3) for v in sim.pose["person"][0]] + [round(sim.pose["person"][1], 3)],
                    "robot": [round(float(v), 3) for v in sim.pose["robot"][0]], "p_fb": round(live.mind.false_belief(X), 4),
                    "dest": {k_: round(v, 3) for k_, v in list(live.mind.destination(X).items())[:3]}})
        cv2.imwrite(str(tmp / "head" / f"{k:06d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(tmp / "third" / f"{k:06d}.jpg"), cv2.cvtColor(sim.rgb("third_cam"), cv2.COLOR_RGB2BGR))
        if k % round(fb.frame_every / DT) == 0:
            cv2.imwrite(str(fdir / f"{int(round(t * 10)):05d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        k += 1
        if (person.finished and robot.finished) or t > fb.max_s:
            break
    for name in ("head", "third"):
        subprocess.run([fb.ffmpeg, "-v", "error", "-y", "-framerate", f"{1 / DT:g}", "-i", str(tmp / name / "%06d.jpg"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out / "videos" / f"{tid}_{name}.mp4")], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    live.pool.shutdown(wait=False)
    st = {**{k_: v for k_, v in d.items() if k_ not in ("small", "furniture")}, "task": tid, "cond": cond, "fb": asdict(fb),
          "first_look": "A" if first == d["A"] else "B", "belief": BELIEF[cond], "false_belief": BELIEF[cond] == "A",
          "flags": {k_: v for k_, v in flags.items() if k_ not in ("fb", "d")}, "steps": steps_log, "events": events, "x_track": track, "log": log, "tells": tells,
          "seconds": t, "wall_s": round(time.time() - t0, 1)}
    st["reject"] = checks(st)
    (out / "streams").mkdir(parents=True, exist_ok=True)
    (out / "streams" / f"{tid}.json").write_text(json.dumps(st))
    return st


def head_point(sim: B1K, d: dict, person: Agent, t: float) -> np.ndarray:
    """Where the person's head points: the middle of A and B, an entity, straight ahead, or away."""
    xy, yaw = sim.pose["person"]
    mode = person.head
    ahead = np.array([xy[0] + 3 * np.cos(yaw), xy[1] + 3 * np.sin(yaw), EYE - 0.4])
    if mode[0] == "mid":
        return np.asarray(d["mid"], float) + [0, 0, 0.3]
    if mode[0] == "at":
        return sim.pos(mode[1]) if mode[1] == d["X"] else np.asarray(sim.top(mode[1]), float)
    if mode[0] == "auto" and person.path:  # 2.5 m ahead along the path, a little below eye height
        pts, here, left = [np.asarray(q, float) for q in person.path], np.asarray(xy, float), 2.5
        for q in pts:
            seg = float(np.linalg.norm(q - here))
            if seg >= left:
                return np.array([*(here + left * (q - here) / seg), EYE - 0.3])
            left, here = left - seg, q
        return np.array([*here, EYE - 0.3])
    return ahead


def checks(st: dict) -> list[str]:
    f, cond, log = st["flags"], st["cond"], st["log"]
    span = lambda a, b, pred: sum(DT for r in log if a <= r["t"] <= b and pred(r))
    if any(k not in f for k in ("placed", "moved", "query")):
        return [f"never reached {k}" for k in ("placed", "moved", "query") if k not in f]
    start = f["back"] if cond.startswith("back") else f["approach"]
    out = []
    if cond not in ("left", "turned") and span(start, f["placed_in"], lambda r: "X" in r["vis"]) < 1.0:
        out.append("X in view under 1 s of the move")
    X = st["X"]
    if cond.endswith("watched") and span(f["placed_in"] - 2.0, f["placed_in"], lambda r: r["attend"] == X) < 0.5:
        out.append("gaze not on X at the placement")
    if cond.endswith("missed") and span(start, f["moved"], lambda r: r["attend"] == X or r["read"] == X) > 0:
        out.append("gaze touched X during the move")
    if cond in ("left", "turned") and span(f["approach"], f["moved"], lambda r: "X" in r["vis"]) > 0.2:
        out.append("X visible during the move")
    if span(f["moved"] + 0.5, f["query"], lambda r: "X" in r["vis"]) > 0.4:
        out.append("X visible inside the closed container")
    near_q = [r for r in log if abs(r["t"] - f["query"]) <= 0.5]
    if not (any("A" in r["vis"] for r in near_q) and any("B" in r["vis"] for r in near_q)):
        out.append("A and B not both in view at the query")
    return out


# -- a design: its places, the watching spot, and a dry run -------------------------------------------------------------
def prepare(fb: FB, sim: B1K, d: dict) -> dict | None:
    """Fill a design's places, validate it (X sampled on X_from, a spot that sees A, B and X, X hidden in the closed B),
    and save the start state. None when it fails."""
    A, B = d["A"], d["B"]
    sim.close_box(B)
    a, b = sim.top(A), sim.top(B)
    mid = (a + b) / 2
    d = {**d, "mid": mid.tolist(), "A_top": a.tolist(), "B_top": b.tolist()}
    for k, name in (("A_xy", A), ("B_xy", B), ("Q_xy", d["Q"]), ("X_from_xy", d["X_from"])):
        # a body with eyes ahead of it needs its spot off the furniture's edge, not its centre
        p = sim.stand_spot(name, mid) if sim.human is not None else sim.free_near(sim.top(name), 0.55, 1.1)
        if p is None:
            return {**d, "fail": f"no free spot near {name}"}
        d[k] = p.tolist()
    d["X_from_top"] = sim.top(d["X_from"]).tolist()
    if not any(sim.place("person", d["X"], "on", d["X_from"]) for _ in range(4)):  # sampling is random: retry
        return {**d, "fail": "X does not sample on X_from"}
    sim.step()
    start = sim.og.sim.dump_state(serialized=False)
    sim.place("person", d["X"], "on", A)
    best = []
    for r in (1.6, 2.2, 2.8, 3.4):
        for th in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            q = mid[:2] + r * np.array([np.cos(th), np.sin(th)])
            if not sim._trav(q) or (sim.human is not None and not sim.clear_of_walls(q)):
                continue
            yaw = float(np.arctan2(mid[1] - q[1], mid[0] - q[0]))
            sim.place_agent("person", q, yaw)
            sim.head = mid + [0, 0, 0.3]
            sim.step()
            m = sim.visible([A, B, d["X"]], strict=(d["X"],))
            sa, sb, sx = (m.get(n, (0, 0, 0))[2] for n in (A, B, d["X"]))
            if sx > 0 and min(sa, sb) >= 0.003:
                best.append((min(sa, sb), q.tolist(), yaw, sx))
    if not best:
        return {**d, "fail": "no spot sees A, B and X"}
    sim.place("robot", d["X"], "in", B)
    sim.open(B, False)
    sim.step()
    for score, q, yaw, sx in sorted(best, reverse=True)[:6]:
        sim.place_agent("person", q, yaw)
        sim.step()
        m = sim.visible([A, B, d["X"]], strict=(d["X"],))
        if d["X"] not in m and A in m and B in m:
            d["spot"] = {"xy": q, "yaw": yaw, "areas": {"A": m[A][2], "B": m[B][2], "X_on_A": sx}}
            break
    else:
        return {**d, "fail": "X stays visible in the closed container"}
    sim.og.sim.load_state(start, serialized=False)
    sim.restored()
    if sim.human is not None:  # off walls, facing the first place they walk to
        q0 = sim.stand_spot(None, d["X_from_xy"], point=sim.top(d["Q"]), gap=1.2)
        q0 = q0 if q0 is not None else sim.free_near(sim.top(d["Q"]), 1.2, 2.0)
        way = sim.path(q0, d["X_from_xy"])
        to = np.asarray(way[min(3, len(way) - 1)]) - q0
        d["person0"] = [*q0.tolist(), float(np.arctan2(to[1], to[0]))]
    else:
        d["person0"] = [*sim.free_near(sim.top(d["Q"]), 1.2, 2.0).tolist(), 0.0]
    plen = lambda a_, b_: float(np.sum(np.linalg.norm(np.diff(sim.path(a_, b_), axis=0), axis=1)))
    sp = d["spot"]["xy"]
    d["dist"] = {"spot_A": plen(sp, d["A_xy"]), "spot_B": plen(sp, d["B_xy"]), "A_B": plen(d["A_xy"], d["B_xy"]),
                 "Q_spot": plen(d["Q_xy"], sp)}
    eye = np.array([*d["spot"]["xy"], 1.6])
    for k_, name in (("rA_xy", A), ("rB_xy", B)):  # the robot works where the person sees what it holds
        q_ = sim.work_spot(sim.top(name), eye)
        d[k_] = (q_ if q_ is not None else np.asarray(d[k_[1:]])).tolist()
    d["third"] = third_spot(sim, d)
    r0 = sim.stand_spot(None, d["Q_xy"], point=sim.top(d["Q"]), gap=1.6, wall=0.3) if sim.human is not None else None
    r0 = r0 if r0 is not None else sim.free_near(sim.top(d["Q"]), 1.6, 2.6)
    r0 = r0 if r0 is not None else np.asarray(d["Q_xy"])
    d["robot0"], d["park_xy"] = [*r0.tolist(), 0.0], r0.tolist()  # it comes from, and goes back to, another room
    d["fail"] = None
    return d, start


def third_spot(sim: B1K, d: dict) -> list:
    """The room camera, 2.2 m up at a traversable point 2 to 4.5 m from the middle of A and B with clear lines to A, B
    and the person's head at the watching spot; the best of these is the farthest."""
    mid = np.asarray(d["mid"], float)
    head = np.array([*d["spot"]["xy"], 1.6])
    best = None
    for r in (4.5, 3.8, 3.1, 2.4):
        for th in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            q = mid[:2] + r * np.array([np.cos(th), np.sin(th)])
            if not sim._trav(q):
                continue
            p = np.array([q[0], q[1], 2.2])
            n = sum(sim.clear(p, x, t) for x, t in ((np.asarray(d["A_top"]), d["A"]), (np.asarray(d["B_top"]), d["B"]), (head, None)))
            if best is None or n > best[0]:
                best = (n, p.tolist())
            if n == 3:
                return p.tolist()
    return best[1] if best else [*d["spot"]["xy"], 2.2]


def third_view(sim: B1K, d: dict) -> None:
    p, mid = np.asarray(d["third"], float), np.asarray(d["mid"], float)
    sim.cams["third_cam"].set_position_orientation(position=p, orientation=look_quat(p, mid))


def main() -> None:
    designs = json.load(open(sys.argv[1]))
    out = Path(sys.argv[2]).resolve()
    args = dict(a.split("=", 1) for a in sys.argv[3:])
    if args.get("only"):
        keep = set(args["only"].split(","))
        designs = [d for d in designs if d["design"] in keep]
    conds = args.get("conds", ",".join(CONDS)).split(",")
    (out / "videos").mkdir(parents=True, exist_ok=True)
    fb, rows = FB(), []
    for scene in sorted({d["scene"] for d in designs}):
        here = [d for d in designs if d["scene"] == scene]
        kinds = sorted({d["X_kind"] for d in here})
        sim = B1K(scene, extra=[(f"X_{k}", k) for k in kinds])  # one object per kind, parked out of sight
        try:
            for d in here:
                done = all((out / "streams" / f"{d['design']}_{c}.json").exists() for c in conds)
                if done or (out / "failed" / f"{d['design']}.json").exists():
                    continue
                d = {**d, "X": f"X_{d['X_kind']}"}
                d = {**d, "X_from": d.get("X_from") or nearest_surface(sim, d)}
                try:
                    res = prepare(fb, sim, d)
                except Exception as e:
                    import traceback
                    res = {"fail": f"{type(e).__name__}: {str(e)[:200]}", "tb": traceback.format_exc()[-1200:]}
                if isinstance(res, dict):  # failed
                    rows.append({"design": d["design"], "fail": res.get("fail"), **({"tb": res["tb"]} if "tb" in res else {})})
                    print(json.dumps(rows[-1]), flush=True)
                    (out / "failed").mkdir(parents=True, exist_ok=True)
                    (out / "failed" / f"{d['design']}.json").write_text(json.dumps(rows[-1]))
                    continue
                d, start = res
                d["small"] = {f"X_{k}" for k in kinds}
                d["furniture"] = [o.name for o in sim.scene.objects if not o.name.startswith("X_") and o.name not in ("person_body", "robot")]
                third_view(sim, d)
                print(json.dumps({"design": d["design"], "spot": d["spot"]}), flush=True)
                if args.get("validate") == "1":
                    rows.append({"design": d["design"], "fail": None, "spot": d["spot"]})
                    continue
                for c in conds:
                    if (out / "streams" / f"{d['design']}_{c}.json").exists():  # resumable: crashes restart the scene
                        continue
                    try:
                        st = run(fb, sim, d, c, out, start, act_on_tells=args.get("act_on_tells", "1") == "1")
                        rows.append({"task": st["task"], "reject": st["reject"], "tells": st["tells"], "seconds": st["seconds"],
                                     "wall_s": st["wall_s"]})
                    except Exception as e:
                        import traceback
                        rows.append({"task": f"{d['design']}_{c}", "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-800:]})
                    print(json.dumps(rows[-1])[:600], flush=True)
        finally:
            if sim is not None:
                sim.og.clear()
    (out / f"results_{args.get('tag', 'run')}.json").write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} rows", flush=True)


def nearest_surface(sim: B1K, d: dict) -> str:
    """A surface other than A for X to start on: the nearest table or counter to A, 1 to 6 m away."""
    from .fb_design import SURFACES
    a = sim.top(d["A"])
    cands = [(float(np.linalg.norm(sim.top(o.name)[:2] - a[:2])), o.name) for o in sim.scene.objects
             if getattr(o, "category", "") in SURFACES and o.name != d["A"]]
    cands = [c for c in cands if 1.0 <= c[0] <= 6.0]
    return min(cands)[1] if cands else d["A"]


if __name__ == "__main__":
    main()
