"""False-belief episodes in PARTNR: a Sally-Anne task in a home.

The person puts an object X on a surface A. Later the robot moves X into a closed container B in the same room
(Open, Place within, Close). The person then comes back for X and first looks where they think it is. The
conditions differ only in what the person sees and attends while X moves:

    left           out of the room for the whole move                                   false belief
    turned         in the room, facing away                                             false belief
    watched        in the room, the move in view, gaze follows X                        true belief
    missed         in the room, the move in view, gaze elsewhere                        false belief
    back_watched   away for the pick, back in view for the placement, gaze follows X    true belief
    back_missed    as back_watched, gaze elsewhere                                      false belief

watched and missed (and the two back_ conditions) run the same script, so the person's view is the same and
only the gaze differs. The person learns where X is only by attending X while it is visible. Coming back, the
first look goes to the believed place with p_antic (anticipatory looking), else to the other place first; the
query is at the end of that first look. Each check below must pass, or the task is rejected.

    python -m helper.partnr.fb_record <designs.json> <out_dir> mode=validate [workers=4]    # GPU node; FB_GPUS=0,1
    python -m helper.partnr.fb_record <tasks.json> <out_dir> [workers=4] [only=t000,t001]
validate: a dry run of the scripted moves and the spot survey per design. tasks.json: designs (fb_design.py) with
"task", "cond", "seed" and the validated "spot" added.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from helper.bip.config import Config, parse

from .perceive import relation, sim_position

START: dict = {}  # design -> both agents' start poses, from its first condition in this worker
BELIEF = {"left": "A", "turned": "A", "watched": "B", "missed": "A", "back_watched": "B", "back_missed": "A"}


@dataclass
class FB:
    fps: float = 30.0
    video_every: int = 2  # 15 fps videos and logs
    frame_every: float = 0.5  # s between the frames with gaze kept for the VLM
    frame_px: int = 512
    max_s: float = 170.0
    min_area: float = 2e-5  # a mask this large counts as visible
    dwell_s: float = 0.6  # free gaze: median dwell (HD-EPIC, as helper/partnr/gaze.py)
    dwell_sigma: float = 0.5
    fix_min: float = 0.3  # s on one entity before it counts as a fixation
    jitter: float = 0.02  # gaze point noise, in image widths (about 1.5 degrees)
    aoi: float = 0.04  # an object within this of the gaze point is the one looked at (area of interest margin)
    keep_away: float = 0.15  # gaze that avoids X stays this far from it in the image
    p_antic: float = 0.8  # first look to the believed place
    look1: float = 1.2  # s of the first look; the query is at its end
    look2: float = 1.0
    turn_dps: float = 150.0  # body turn speed
    ffmpeg: str = str(Path.home() / ".conda/envs/egocollab/bin/ffmpeg")


# -- the two agents' scripts ------------------------------------------------------------------------------
def person_script(fb: FB, d: dict, cond: str, first: str) -> list:
    X, A, B, Q = d["X"], d["A"], d["B"], d["Q"]
    P = ",".join(str(v) for v in d["spot"]["pos"] + [d["spot"]["yaw"]])
    avoid = ("free", (X, A, B), True)
    pre = [("attend", "free", (), False), ("skill", "Navigate", X), ("attend", "focus", X), ("skill", "Pick", X),
           ("skill", "Navigate", A), ("skill", "Place", f"{X}, on, {A}, none, none"), ("signal", "placed"),
           ("head", "at", X), ("wait", 1.5), ("head", "auto"), ("attend", *avoid)]
    face = [("turn", "mid"), ("head", "mid")]
    watch = ("attend", "focus", X) if cond.endswith("watched") else ("attend", *avoid)
    mid = {"left": [("skill", "Navigate", Q), ("signal", "away"), ("until", "moved"), ("wait", 1.0),
                    ("skill", "Walk", P), *face, ("wait", 1.0)],
           "turned": [("skill", "Walk", P), ("turn", "away"), ("head", "ahead"), ("signal", "ready"),
                      ("until", "moved"), ("wait", 1.0), *face, ("wait", 1.0)],
           "watched": [("skill", "Walk", P), *face, ("signal", "ready"), ("until", "approach"), watch,
                       ("until", "moved"), ("wait", 1.0), ("attend", *avoid), ("wait", 0.5)],
           "back_watched": [("skill", "Navigate", Q), ("signal", "away"), ("until", "picked"), ("skill", "Walk", P),
                            *face, ("signal", "back"), watch, ("until", "moved"), ("wait", 1.0), ("attend", *avoid),
                            ("wait", 0.5)]}
    mid["missed"], mid["back_missed"] = mid["watched"], mid["back_watched"]
    other = B if first == A else A
    look = [("attend", "seq", ((first, 0.0), (other, fb.look1))), ("wait", fb.look1), ("signal", "query"), ("wait", fb.look2)]
    believed = BELIEF[cond]
    after = ([("head", "auto"), ("attend", "focus", A), ("skill", "Navigate", A), ("head", "at", A), ("wait", 2.0),
              ("attend", "free", (), False), ("head", "ahead"), ("wait", 1.5)] if believed == "A" else
             [("head", "auto"), ("attend", "focus", B), ("skill", "Navigate", B), ("skill", "Open", B),
              ("attend", "focus", X), ("skill", "Pick", X), ("wait", 1.0)])
    return pre + mid[cond] + look + after


def robot_script(d: dict, cond: str) -> list:
    X, B = d["X"], d["B"]
    start = [("until", "away"), ("wait", 1.0)] if cond in ("left", "back_watched", "back_missed") else [("until", "ready"), ("wait", 1.5)]
    hold = [("until", "back"), ("wait", 0.5)] if cond.startswith("back") else []
    return start + [("skill", "Navigate", X), ("signal", "approach"), ("skill", "Pick", X), ("signal", "picked"),
                    ("skill", "Navigate", B), *hold, ("skill", "Open", B), ("wait", 0.8), ("skill", "Place", f"{X}, within, {B}, none, none"),
                    ("signal", "placed_in"), ("wait", 0.8), ("skill", "Close", B), ("signal", "moved"), ("skill", "Navigate", d["park"])]


class Runner:
    """One agent's steps. ("skill", tool, arg), ("wait", s), ("until", flag), ("signal", flag), ("head", mode, ...),
    ("attend", mode, ...), ("turn", mode). Bookkeeping steps run at once; a failed skill raises."""

    def __init__(self, uid: int, steps: list, flags: dict, log: list, anchor: str = ""):
        self.uid, self.steps, self.flags, self.log, self.anchor = uid, list(steps), flags, log, anchor
        self.skill, self.until_t, self.turning = None, None, None
        self.head, self.attend, self.skills = ("auto",), ("free", (), False), []

    @property
    def finished(self) -> bool:
        return self.skill is None and self.turning is None and not self.steps

    def act(self, t: float, sim, turn):
        while self.skill is None and self.turning is None and self.steps:
            kind, *a = self.steps[0]
            if kind == "wait":
                self.until_t = t + a[0] if self.until_t is None else self.until_t
                if t < self.until_t:
                    return None
                self.until_t = None
            elif kind == "until" and a[0] not in self.flags:
                return None
            elif kind == "signal":
                self.flags[a[0]] = round(t, 2)
            elif kind == "head":
                self.head = tuple(a)
            elif kind == "attend":
                self.attend = tuple(a) + ((round(t, 2),) if a[0] == "seq" else ())
            elif kind == "turn":
                self.turning = a[0]
            elif kind == "skill":
                self.skill = (a[0], a[1])
            self.steps.pop(0)
            self.log.append([round(t, 2), self.uid, kind, *[str(x) for x in a]])
        if self.turning is not None:
            if turn(self.turning):
                self.turning = None
            return None
        if self.skill is None:
            return None
        walk = self.skill[0] == "Walk"  # to a point: PARTNR's navigation with its goal set directly
        if walk:
            nav = sim.agents[self.uid].tools["Navigate"].skill
            if not nav.target_is_set:
                import magnum as mn
                x, y, z, yaw = (float(v) for v in self.skill[1].split(","))
                nav.target_pos, nav.target_base_pos = np.array([x, y, z]), mn.Vector3(x, y, z)
                nav.target_base_rot, nav.target_is_set, nav.face_object = yaw, True, False
        vec, resp = sim.skill(self.uid, "Navigate", self.anchor) if walk else sim.skill(self.uid, *self.skill)
        if resp and walk:
            sim.agents[self.uid].tools["Navigate"].skill.face_object = True
        if resp:
            if not resp.startswith("Successful"):
                raise RuntimeError(f"agent {self.uid} {self.skill[0]}[{self.skill[1]}]: {resp[:100]}")
            self.skills.append([round(t, 2), *self.skill])
            self.skill = None
        return vec


# -- the person's body, head and gaze -----------------------------------------------------------------------
def sync(sim) -> None:
    """Hand the person's pose to PARTNR's walk controller, which re-applies its own stored pose on every idle step."""
    ctrl = getattr(sim.env.env.env.env._env.task, "humanoid_controller", None)
    if ctrl is not None:
        ctrl.obj_transform_base = sim.sim.agents_mgr[1].articulated_agent.base_transformation
        ctrl.prev_orientation = None


def body_yaw(sim, point: np.ndarray) -> float:
    """The point's bearing in the person's body frame (0 straight ahead), as Sim.head_look measures it."""
    import magnum as mn
    agent = sim.sim.agents_mgr[1].articulated_agent
    d = np.array(agent.sim_obj.transformation.inverted().transform_point(mn.Vector3(*point)))
    return float(np.arctan2(d[0], d[2]))


class Body:
    def __init__(self, fb: FB, sim, d: dict, furniture: set):
        self.fb, self.sim, self.d, self.furniture, self.sign, self.look = fb, sim, d, furniture, None, None
        a, b = sim_position(sim, d["A"]), sim_position(sim, d["B"])
        self.mid = (a + b) / 2 + [0.0, 0.6, 0.0]

    def here(self) -> np.ndarray:
        return np.array(self.sim.sim.agents_mgr[1].articulated_agent.base_pos)

    def point(self, mode: tuple, skill) -> np.ndarray | None:
        kind = mode[0]
        if kind == "mid":
            return self.mid
        if kind == "away":
            h = self.here()
            return h + (h - self.mid) * [1, 0, 1] + [0, 1.0, 0]
        if kind == "at":
            p = sim_position(self.sim, mode[1])
            return None if p is None else p + [0.0, 0.6 if mode[1] in self.furniture else 0.0, 0.0]
        if kind == "auto" and skill is not None:  # the skill's target while handling it or within 3 m of it
            tool, arg = skill
            name = arg.split(",")[2].strip() if tool == "Place" else arg.split(",")[0].strip()
            p = self.point(("at", name), None)
            near = p is not None and np.linalg.norm((self.here() - p)[[0, 2]]) < 3.0
            return p if p is not None and (tool != "Navigate" or near) else None
        return None

    def turn(self, mode: str, instant: bool = False) -> bool:
        """Rotate the body toward the mode's point at turn_dps (at once when instant); True once facing it. The
        rotation values for now and for the goal are found by search, since base_rot's getter and setter disagree."""
        agent = self.sim.sim.agents_mgr[1].articulated_agent
        if self.sign is None:
            target, grid = self.point((mode,), None), np.linspace(-np.pi, np.pi, 181)
            import magnum as mn
            f0 = np.array(agent.base_transformation.transform_vector(mn.Vector3(1, 0, 0)))[[0, 2]]
            fits = []
            for th in grid:
                agent.base_rot = float(th)
                f = np.array(agent.base_transformation.transform_vector(mn.Vector3(1, 0, 0)))[[0, 2]]
                fits.append((float(f @ f0 / (np.linalg.norm(f) * np.linalg.norm(f0) + 1e-9)), abs(body_yaw(self.sim, target))))
            start, goal = grid[int(np.argmax([a for a, _ in fits]))], grid[int(np.argmin([b for _, b in fits]))]
            gap = (goal - start + np.pi) % (2 * np.pi) - np.pi
            n = 1 if instant else max(1, int(abs(np.degrees(gap)) / self.fb.turn_dps * self.fb.fps))
            self.sign = [float(start + gap * k / n) for k in range(1, n + 1)]  # the path of rotation values
            agent.base_rot = float(start)
        agent.base_rot = self.sign.pop(0)
        sync(self.sim)
        if not self.sign:
            self.sign = None
            return True
        return False

    def aim(self, mode: tuple, skill) -> None:
        goal = self.point(mode, skill)
        if goal is None:
            self.look = None
            self.sim.head_look(None)
            return
        self.look = goal if self.look is None else self.look + 0.2 * (goal - self.look)
        self.sim.head_look(self.look)


class Gaze:
    """The gaze point from the attended entity, and what an observer reads off it: the object within aoi of the
    point (nearest), else the entity under it. Fixations: the same read held for fix_min."""

    def __init__(self, fb: FB, rng):
        self.fb, self.rng = fb, rng
        self.target, self.until, self.read, self.read_since, self.fixes = None, -1.0, None, 0.0, []

    def choose(self, masks: dict, exclude: tuple, strict: bool, furniture: set) -> str | None:
        names = [n for n in sorted(masks) if n not in exclude]
        if strict:  # stay clear of the excluded entities in the image
            far = lambda n: all(np.hypot(masks[n][0] - masks[e][0], masks[n][1] - masks[e][1]) > self.fb.keep_away
                                for e in exclude if e in masks and e not in furniture)
            names = [n for n in names if far(n)]
        return names[self.rng.integers(len(names))] if names else None

    def step(self, t: float, mode: tuple, masks: dict, pan: np.ndarray, sim, furniture: set) -> tuple:
        fb = self.fb
        kind = mode[0]
        if kind == "seq":
            (first, _), (second, at2), t0 = mode[1][0], mode[1][1], mode[2]
            self.target = first if t - t0 < at2 else second
        elif kind == "focus" and mode[1] in masks:
            self.target = mode[1]
        else:  # free gaze, or a focus whose entity is out of view
            exclude, strict = (tuple(mode[1]), mode[2]) if kind == "free" else ((), False)
            near = strict and self.target in masks and any(  # what the gaze holds must stay clear of the excluded
                np.hypot(masks[self.target][0] - masks[e][0], masks[self.target][1] - masks[e][1]) <= fb.keep_away
                for e in exclude if e in masks and e not in furniture)
            if t >= self.until or self.target not in masks or self.target in exclude or near:
                self.target = self.choose(masks, exclude, strict, furniture)
                self.until = t + fb.dwell_s * float(np.exp(fb.dwell_sigma * self.rng.standard_normal()))
        if self.target in masks:
            cx, cy, _ = masks[self.target]
            xy = [cx + fb.jitter * self.rng.standard_normal(), cy + fb.jitter * self.rng.standard_normal()]
        else:
            xy = [0.5 + 0.1 * self.rng.standard_normal(), 0.45 + 0.1 * self.rng.standard_normal()]
        xy = [float(np.clip(v, 0, 0.999)) for v in xy]
        objs = [(np.hypot(c[0] - xy[0], c[1] - xy[1]), n) for n, c in masks.items() if n not in furniture]
        near = min(objs) if objs else (1.0, None)
        read = near[1] if near[0] <= fb.aoi else sim.name_of(int(pan[int(xy[1] * pan.shape[0]), int(xy[0] * pan.shape[1])]))
        if read != self.read:
            if self.read is not None and t - self.read_since >= fb.fix_min:
                self.fixes.append([round(self.read_since, 2), round(t, 2), self.read])
            self.read, self.read_since = read, t
        return xy, read, self.target

    def close(self, t: float) -> None:
        if self.read is not None and t - self.read_since >= self.fb.fix_min:
            self.fixes.append([round(self.read_since, 2), round(t, 2), self.read])


# -- one task ---------------------------------------------------------------------------------------------
def location(sim, x: str) -> str:
    if sim.held(0) == x:
        return "robot"
    if sim.held(1) == x:
        return "person"
    r = relation(sim, x)
    return f"{'in' if r[0] == 'in' else 'on'} {r[1]}" if r else "unknown"


def dot(img: np.ndarray, xy: list) -> np.ndarray:
    import cv2
    h, w = img.shape[:2]
    out = img.copy()
    cv2.circle(out, (int(xy[0] * w), int(xy[1] * h)), max(5, w // 48), (255, 40, 40), 2)
    return out


def checks(st: dict) -> list[str]:
    """Reasons to reject the task; empty when it shows what its condition claims."""
    fb, f, cond, log = st["fb"], st["flags"], st["cond"], st["log"]
    dt = fb["video_every"] / fb["fps"]
    span = lambda a, b, key, pred: sum(dt for r in log if a <= r["t"] <= b and pred(r[key]))
    vis = lambda name: (lambda v: v is not None and name in v)
    bad = [k for k in ("placed", "moved", "query") if k not in f]
    if bad:
        return [f"never reached {k}" for k in bad]
    start = f["back"] if cond.startswith("back") else f["approach"]
    seen_x = span(start, f["placed_in"], "vis", vis("X"))
    attended = span(start, f["placed_in"], "attend", lambda a: a == st["X"])
    out = []
    if cond in ("watched", "missed", "back_watched", "back_missed") and seen_x < 1.0:
        out.append(f"X in view only {seen_x:.1f} s of the move")
    if cond.endswith("watched") and span(f["placed_in"] - 2.0, f["placed_in"], "attend", lambda a: a == st["X"]) < 0.5:
        out.append("gaze not on X at the placement")
    if cond.endswith("missed") and (attended > 0 or span(start, f["moved"], "read", lambda r: r == st["X"]) > 0):
        out.append("gaze touched X during the move")
    if cond in ("left", "turned") and span(f["approach"], f["moved"], "vis", vis("X")) > 0.2:
        out.append("X visible during the move")
    if span(f["moved"] + 0.5, f["query"], "vis", vis("X")) > 0.2:
        out.append("X visible inside the closed container")
    q = min(log, key=lambda r: abs(r["t"] - f["query"]))
    if not (vis("A")(q["vis"]) and vis("B")(q["vis"])):
        out.append("A and B not both in view at the query")
    return out


def run(fb: FB, sim, d: dict, out: Path) -> dict:
    import random
    import cv2
    import magnum as mn
    fresh(sim, d)
    poses = START.setdefault(d["design"], {u: (list(sim.sim.agents_mgr[u].articulated_agent.base_pos),
                                               sim.sim.agents_mgr[u].articulated_agent.base_rot) for u in (0, 1)})
    for u, (pos, rot) in poses.items():  # resets place the agents at random: every condition starts alike
        sim.sim.agents_mgr[u].articulated_agent.base_pos = mn.Vector3(*pos)
        sim.sim.agents_mgr[u].articulated_agent.base_rot = rot
    sync(sim)  # the walk keeps its last pose and heading across resets
    ctrl = getattr(sim.env.env.env.env._env.task, "humanoid_controller", None)
    if ctrl is not None:
        ctrl.walk_mocap_frame = 0
    sim.head_look(None)  # the head camera also keeps the last task's aim
    sim.step({1: sim.zero})
    random.seed(d["seed"])  # PARTNR's navigation samples its goal poses: the conditions of a design walk alike
    np.random.seed(d["seed"])
    sim.sim.seed(d["seed"])
    rng = np.random.default_rng([d["seed"], 1])
    first = d["A"] if (BELIEF[d["cond"]] == "A") == (rng.random() < fb.p_antic) else d["B"]
    furniture = {f.name for f in sim.env.full_world_graph.get_all_furnitures()}
    flags, steps_log = {}, []
    body = Body(fb, sim, d, furniture)
    person = Runner(1, person_script(fb, d, d["cond"], first), flags, steps_log, anchor=d["A"])
    robot = Runner(0, robot_script(d, d["cond"]), flags, steps_log)
    gaze = Gaze(fb, np.random.default_rng([d["seed"], 2]))  # the same draws in both members of a pair until they part
    tid = d["task"]
    tmp = out / "tmp" / tid
    for k in ("head", "third"):
        shutil.rmtree(tmp / k, ignore_errors=True)
        (tmp / k).mkdir(parents=True)
    fdir = out / "frames" / tid
    shutil.rmtree(fdir, ignore_errors=True)
    fdir.mkdir(parents=True)
    names = {d["X"]: "X", d["A"]: "A", d["B"]: "B"}
    objects = sorted(o.name for o in sim.env.full_world_graph.get_all_objects())
    world = {o: location(sim, o) for o in objects}  # the ground truth every condition may read: all objects, all times
    world0, world_track, views = dict(world), [], []
    log, x_track, step, t0 = [], [], 0, time.time()
    while step < fb.max_s * fb.fps:
        t = step / fb.fps
        masks = sim.masks(min_area=fb.min_area)
        pan = sim.batch["agent_1_head_panoptic"][0, ..., 0].cpu().numpy()
        vec1 = person.act(t, sim, body.turn)
        vec1 = sim.zero if vec1 is None else vec1  # a no-op keeps the turned body; without it the last pose returns
        vec0 = robot.act(t, sim, lambda m: True)
        body.aim(person.head, person.skill)
        xy, read, target = gaze.step(t, person.attend, masks, pan, sim, furniture)
        loc = location(sim, d["X"])
        if not x_track or x_track[-1][1] != loc:
            x_track.append([round(t, 2), loc])
        if step % fb.video_every == 0:
            k = step // fb.video_every
            log.append({"t": round(t, 2), "gaze": [round(v, 4) for v in xy], "read": read, "attend": target,
                        "vis": {names[n]: [round(v, 4) for v in masks[n]] for n in names if n in masks},
                        "person": [round(float(v), 3) for v in body.here()],
                        "robot": [round(float(v), 3) for v in sim.sim.agents_mgr[0].articulated_agent.base_pos]})
            cv2.imwrite(str(tmp / "head" / f"{k:06d}.jpg"), cv2.cvtColor(sim.frame("agent_1_head_rgb"), cv2.COLOR_RGB2BGR))
            third = np.concatenate([cv2.resize(sim.frame("agent_1_third_rgb"), (384, 384)),
                                    cv2.resize(sim.frame("agent_0_third_rgb"), (384, 384))], 1)
            cv2.imwrite(str(tmp / "third" / f"{k:06d}.jpg"), cv2.cvtColor(third, cv2.COLOR_RGB2BGR))
        if step % round(fb.frame_every * fb.fps) == 0:
            for o in objects:
                now = location(sim, o)
                if now != world[o]:
                    world[o] = now
                    world_track.append([round(t, 2), o, now])
            agent = sim.sim.agents_mgr[1].articulated_agent
            views.append({"t": round(t, 2), "seen": sorted(masks), "gaze": read, "heading": round(float(agent.base_rot), 3)})
            img = cv2.resize(sim.frame("agent_1_head_rgb"), (fb.frame_px, fb.frame_px))
            cv2.imwrite(str(fdir / f"{int(round(t * 10)):05d}.jpg"), cv2.cvtColor(dot(img, xy), cv2.COLOR_RGB2BGR))
        if person.finished and robot.finished:
            break
        sim.step({0: vec0, 1: vec1})
        step += 1
    t_end = step / fb.fps
    gaze.close(t_end)
    for k in ("head", "third"):
        subprocess.run([fb.ffmpeg, "-v", "error", "-y", "-framerate", f"{fb.fps / fb.video_every:g}", "-i",
                        str(tmp / k / "%06d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                        str(out / "videos" / f"{tid}_{k}.mp4")], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    st = {**d, "fb": asdict(fb), "first_look": "A" if first == d["A"] else "B", "belief": BELIEF[d["cond"]],
          "false_belief": BELIEF[d["cond"]] == "A", "flags": flags, "steps": steps_log, "x_track": x_track,
          "world0": world0, "world_track": world_track, "views": views,
          "fixations": gaze.fixes, "log": log, "person_skills": person.skills, "robot_skills": robot.skills,
          "frames": str(fdir), "seconds": round(t_end, 2), "wall_s": round(time.time() - t0, 1)}
    st["reject"] = checks(st)
    (out / "streams").mkdir(parents=True, exist_ok=True)
    (out / "streams" / f"{tid}.json").write_text(json.dumps(st))
    return st


def fresh(sim, d: dict) -> None:
    """Reset to the design's episode through another one: a reset of the same episode keeps moved objects."""
    sim.reset(d["bounce"])
    sim.reset(d["episode"])
    aom = sim.sim.get_articulated_object_manager()  # articulated furniture renders as background until tagged
    for ao in aom.get_objects_by_handle_substring().values():
        for node in ao.visual_scene_nodes:
            node.semantic_id = ao.object_id + 100
    sim.names = {}
    if location(sim, d["X"]) != f"on {d['X_from']}":
        raise RuntimeError(f"after reset X is {location(sim, d['X'])}")


def drive(sim, uid: int, tool: str, arg: str, max_s: float = 30.0) -> bool:
    """Run one skill to its end, the other agent standing still."""
    for _ in range(int(max_s * 30)):
        vec, resp = sim.skill(uid, tool, arg)
        if resp:
            return resp.startswith("Successful")
        sim.step({uid: vec})
    return False


def view_from(fb: FB, sim, body, d: dict, pos) -> dict:
    """Visible areas of A, B and X from a standing point, facing the middle of A and B (the person is moved there;
    validation only)."""
    import magnum as mn
    sim.sim.agents_mgr[1].articulated_agent.base_pos = mn.Vector3(*pos)
    sync(sim)
    body.turn("mid", instant=True)
    sim.head_look(body.mid)
    for _ in range(2):
        sim.step({1: sim.zero})
    m = sim.masks(min_area=fb.min_area)
    return {k: round(m.get(d[k], (0, 0, 0))[2], 5) for k in ("A", "B", "X")}


def candidates(sim, body) -> list:
    """Navigable standing points 1.8 to 4.2 m from the middle of A and B, on the person's island."""
    import magnum as mn
    pf, here = sim.sim.pathfinder, np.array(sim.sim.agents_mgr[1].articulated_agent.base_pos)
    island, out = pf.get_island(mn.Vector3(*here)), []
    for r in (1.8, 2.4, 3.0, 3.6, 4.2):
        for th in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            p = np.array([body.mid[0] + r * np.cos(th), here[1], body.mid[2] + r * np.sin(th)])
            q = np.array(pf.snap_point(mn.Vector3(*p), island))
            if np.all(np.isfinite(q)) and np.linalg.norm((q - p)[[0, 2]]) < 0.4:
                out.append([round(float(v), 3) for v in q])
    return out


def validate_design(fb: FB, sim, d: dict) -> tuple[str | None, dict | None]:
    """The scripted moves once, unrecorded, with the survey of standing points between them: (failure, spot). The spot
    sees A and B and the object on A, and no longer sees the object once it is in the closed container."""
    fresh(sim, d)
    X, A, B = d["X"], d["A"], d["B"]
    for tool, arg in [("Navigate", X), ("Pick", X), ("Navigate", A), ("Place", f"{X}, on, {A}, none, none")]:
        if not drive(sim, 1, tool, arg):
            return f"person {tool}[{arg}]", None
    furniture = {f.name for f in sim.env.full_world_graph.get_all_furnitures()}
    body = Body(fb, sim, {**d, "P_used": None}, furniture)
    home = list(np.array(sim.sim.agents_mgr[1].articulated_agent.base_pos))
    seen = [(view_from(fb, sim, body, d, p), p) for p in candidates(sim, body)]
    good = sorted(((min(v["A"], v["B"]), p, v["X"]) for v, p in seen if v["X"] > 0 and min(v["A"], v["B"]) >= 0.002), reverse=True)
    if not good:
        return "no point sees A, B and the object", None
    view_from(fb, sim, body, d, home)  # out of the robot's way
    for tool, arg in [("Navigate", X), ("Pick", X), ("Navigate", B), ("Open", B), ("Place", f"{X}, within, {B}, none, none"), ("Close", B)]:
        if not drive(sim, 0, tool, arg):
            return f"robot {tool}[{arg}]", None
    for score, p, x_on_a in good[:6]:
        v = view_from(fb, sim, body, d, p)
        if v["X"] == 0 and min(v["A"], v["B"]) >= 0.002:
            dx, dz = body.mid[0] - p[0], body.mid[2] - p[2]
            return None, {"pos": p, "yaw": round(float(np.arctan2(-dz, dx)), 3), "areas": dict(v, X_on_A=x_on_a)}
    return "the object stays visible in the closed container", None


def worker(args) -> list[dict]:
    fb, tasks, gpu, out = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from .env import Sim
    cfg = parse(["split=val", "head_px=512"], Config)
    sim, rows = Sim(cfg, sorted({d["episode"] for d in tasks} | {d["bounce"] for d in tasks}), out / "partnr_logs"), []
    for d in tasks:
        try:
            st = run(fb, sim, d, out)
            rows.append({"task": d["task"], "cond": d["cond"], "seconds": st["seconds"], "reject": st["reject"]})
        except Exception as e:  # one bad task must not stop the worker
            rows.append({"task": d["task"], "cond": d["cond"], "error": f"{type(e).__name__}: {str(e)[:200]}"})
        print(json.dumps(rows[-1]), flush=True)
    return rows


def validate(args) -> list[dict]:
    fb, designs, gpu, out = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from .env import Sim
    cfg = parse(["split=val", "head_px=512"], Config)
    sim, rows = Sim(cfg, sorted({d["episode"] for d in designs} | {d["bounce"] for d in designs}), out / "partnr_logs"), []
    for d in designs:
        try:
            fail, spot = validate_design(fb, sim, d)
            rows.append({"design": d["design"], "fail": fail, "spot": spot})
        except Exception as e:
            rows.append({"design": d["design"], "fail": f"{type(e).__name__}: {str(e)[:200]}", "spot": None})
        print(json.dumps(rows[-1]), flush=True)
    return rows


def main() -> None:
    tasks = json.load(open(sys.argv[1]))
    out = Path(sys.argv[2]).resolve()
    args = dict(a.split("=", 1) for a in sys.argv[3:])
    gpus = os.environ.get("FB_GPUS", "0,1,2,3").split(",")  # the devices workers take in turn
    if args.get("mode") == "validate":  # tasks.json holds designs here
        n = int(args.get("workers", 4))
        if args.get("only"):
            tasks = [d for d in tasks if d["design"] in set(args["only"].split(","))]
        out.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(n, mp_context=mp.get_context("spawn")) as pool:
            rows = [r for rs in pool.map(validate, [(FB(), tasks[i::n], gpus[i % len(gpus)], out) for i in range(n)]) for r in rs]
        (out / f"validate{args.get('tag', '')}.json").write_text(json.dumps(rows, indent=1))
        print(f"{len(rows)} designs, {sum(1 for r in rows if not r['fail'])} pass", flush=True)
        return
    if args.get("only"):
        keep = set(args["only"].split(","))
        tasks = [d for d in tasks if d["task"] in keep]
    (out / "videos").mkdir(parents=True, exist_ok=True)
    fb, n = FB(), int(args.get("workers", 4))
    by_design = {}
    for d in tasks:  # a design's conditions stay with one worker, which shares their survey
        by_design.setdefault(d["design"], []).append(d)
    chunks = [[] for _ in range(n)]
    for ds in sorted(by_design.values(), key=len, reverse=True):
        min(chunks, key=len).extend(ds)
    with ProcessPoolExecutor(n, mp_context=mp.get_context("spawn")) as pool:
        rows = [r for rs in pool.map(worker, [(fb, c, gpus[i % len(gpus)], out) for i, c in enumerate(chunks) if c]) for r in rs]
    prev = out / "results.json"
    old = {r["task"]: r for r in json.load(open(prev))} if prev.exists() else {}
    old.update({r["task"]: r for r in rows})
    prev.write_text(json.dumps(sorted(old.values(), key=lambda r: r["task"]), indent=1))
    ok = [r for r in rows if not r.get("error") and not r.get("reject")]
    print(f"{len(rows)} tasks, {len(ok)} kept", flush=True)


if __name__ == "__main__":
    main()
