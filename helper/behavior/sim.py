"""One BEHAVIOR-1K home in OmniGibson for collaboration tasks, driven by script.

The person is an Isaac Sim character skinned every frame (helper/behavior/human.py) with the head camera (RGB) at its
eyes, walking the scene's traversable map at its walk clip's pace (person="mannequin" keeps the earlier rigid body
with the camera above it); the robot is a Stretch moved the same way. What the person sees and where their gaze lands come from physics raycasts from the eye
(Isaac Sim's segmentation annotators crash on some views): an object is in view when a ray reaches a point on it
first, and the gaze falls on whatever the ray through the gaze point hits. Objects are carried, and held inside
closed containers, by setting their pose every frame; they are placed on surfaces with OmniGibson's OnTop and moved in
and out of containers opened with Open. The robot is visual-only, so it never pushes furniture. A third camera films
the room for clips.

    sim = B1K("Beechwood_0_int", extra=[("X", "mug")])
    sim.walk("person", goal_xy); sim.look("person", point); frame = sim.person_frame()
Run in the `behavior` env on an RTX GPU (OMNI_KIT_ACCEPT_EULA=YES).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

EYE = 1.6  # m, the person's eye height
WALK = 1.2  # m/s
TURN = np.radians(120)  # rad/s
DT = 1 / 15  # s of task time per frame
DATA = Path(os.environ.get("B1K_DATASETS", Path.home() / "BEHAVIOR-1K/datasets"))


def yaw90(q) -> np.ndarray:
    """Quaternion (x, y, z, w) q turned 90 degrees about the world vertical."""
    h = np.sqrt(0.5)
    ax, ay, az, aw = 0.0, 0.0, h, h
    bx, by, bz, bw = (float(v) for v in q)
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def look_quat(pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """(x, y, z, w) of a camera at pos looking at target; z up, the camera looks along its -z."""
    from scipy.spatial.transform import Rotation
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f) + 1e-9
    r = np.cross(f, [0, 0, 1.0])
    r = r / (np.linalg.norm(r) + 1e-9) if np.linalg.norm(r) > 1e-6 else np.array([1.0, 0, 0])
    u = np.cross(r, f)
    return Rotation.from_matrix(np.stack([r, u, -f], 1)).as_quat()


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])


def body_quat(yaw: float) -> np.ndarray:
    """(x, y, z, w) of the mannequin standing and facing yaw (the model lies along its x axis, face up)."""
    from scipy.spatial.transform import Rotation
    return (Rotation.from_euler("z", yaw) * Rotation.from_euler("y", np.pi / 2)).as_quat()


def model_of(kind: str) -> str:
    return sorted(p.name for p in (DATA / "behavior-1k-assets/objects" / kind).iterdir())[0]


class B1K:
    def __init__(self, scene: str, extra: list[tuple[str, str]] = (), px: int = 512, third: bool = True,
                 scene_file: str | None = None, rooms: list[str] | None = None, person: str = "human",
                 character: str = "M_Medical_01"):
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        if person == "human":
            from .human import patch_app
            patch_app()
        import omnigibson as og
        from omnigibson.macros import gm
        gm.HEADLESS, gm.ENABLE_OBJECT_STATES = True, True
        gm.ENABLE_TRANSITION_RULES = False  # recipes (cooking, garbage) only change objects; they crashed a placement
        self.og = og
        cam = lambda name, px_: {"sensor_type": "VisionSensor", "name": name, "relative_prim_path": f"/{name}",
                                 "modalities": ["rgb"],
                                 "sensor_kwargs": {"image_height": px_, "image_width": px_},
                                 "position": [0.0, 0.0, 10.0], "orientation": [0.0, 0.0, 0.0, 1.0]}
        objs = [{"type": "DatasetObject", "name": n, "category": k, "model": model_of(k), "kinematic_only": True,
                 "position": [2.0 * i, 0.0, -20.0]} for i, (n, k) in enumerate(extra)]  # parked below the floor
        if person == "mannequin":
            objs.append({"type": "DatasetObject", "name": "person_body", "category": "mannequin", "model": model_of("mannequin"),
                         "kinematic_only": True, "visual_only": True, "position": [0.0, 5.0, -20.0]})
        scene_cfg = {"type": "InteractiveTraversableScene", "scene_model": scene}
        if scene_file:  # a task instance: the home with the task's objects in their sampled places
            scene_cfg["scene_file"] = str(scene_file)
        if rooms:
            scene_cfg["load_room_instances"] = list(rooms)
        self.scene_model = scene
        cfg = {"scene": scene_cfg,
               "robots": [{"type": "Stretch", "name": "robot", "obs_modalities": [], "position": [0.0, 10.0, -20.0]}],
               "objects": objs,
               "env": {"external_sensors": [cam("person_cam", px)] + ([cam("third_cam", 512)] if third else [])}}
        self.env = og.Environment(configs=cfg)
        self.env.reset()
        self.scene = self.env.scene
        rb = self.env.robots[0]
        self.robot = rb[0] if isinstance(rb, (list, tuple)) else rb  # robots also come one list per scene
        self.body = self.scene.object_registry("name", "person_body") if person == "mannequin" else None
        es = self.env.external_sensors
        es = es[0] if isinstance(es, (list, tuple)) else es  # one dict per scene
        self.cams = dict(es) if isinstance(es, dict) else {s.name: s for s in es}
        self.pose = {"person": None, "robot": None}  # (xy, yaw)
        self.held = {"person": None, "robot": None}
        self.ghosts = set()  # objects in a hand: no collisions (see _ghosts)
        self.inside = {}  # container -> the middle of its closed box (fill with close_box)
        self.pinned = {}  # object -> where it rests inside a container; set every frame (objects here are dynamic)
        self.pinned_orn = {}  # object -> how it rests there, when set (a toothbrush stands in a cup)
        self.head = np.array([1.0, 0.0, EYE])  # where the person's head points
        self.eye = np.array([0.0, 0.0, EYE])  # where the person's eyes are
        self.walk_speed, self.human, self._last_xy = WALK, None, None
        self.reach = None  # (point, share of the way) the person's right hand reaches for
        self.t = 0.0
        self.robot.visual_only = True  # moved by teleport: no collisions or gravity, so it never pushes furniture
        if self.body is not None:
            self.body.set_position_orientation(position=[0.0, 5.0, -20.0], orientation=body_quat(0.0))
        for _ in range(3):
            og.sim.step()
        if self.body is not None:
            self.body_z = -20.0 - float(self.body.aabb[0][2])  # lifts the standing mannequin's feet to the floor
        else:
            from .human import Human
            self.human = Human(og.sim, character)
            self.walk_speed = self.human.speed
            self.open_doors()

    # -- the scene --------------------------------------------------------------------------------------
    def obj(self, name: str):
        return self.scene.object_registry("name", name)

    def pos(self, name: str) -> np.ndarray:
        return np.asarray(self.obj(name).get_position_orientation()[0], float)

    def top(self, name: str) -> np.ndarray:
        """The middle of an object's top surface (its bounding box)."""
        lo, hi = (np.asarray(v, float) for v in self.obj(name).aabb)
        return np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2]])

    def path(self, a, b, avoid=(), strict: bool = False) -> np.ndarray | None:
        """Waypoints (N, 2) from a to b: with the animated person, from Floor (full-resolution clearance, bodies kept off
        furniture); with the mannequin, OmniGibson's shortest path. When there is none: None if strict, else the
        straight line."""
        a2, b2 = np.asarray(a[:2], float), np.asarray(b[:2], float)
        if self.human is not None:
            p = self.floor().plan(a2, b2, avoid)
            return p if p is not None or strict else np.stack([a2, b2])
        import torch as th
        try:
            p, _ = self.scene.trav_map.get_shortest_path(0, th.tensor(a2), th.tensor(b2), entire_path=True)
            p = np.asarray(p.cpu() if hasattr(p, "cpu") else p, float)
        except Exception:
            p = None
        return p if p is not None and p.ndim == 2 and len(p) >= 1 else np.stack([a2, b2])

    def open_doors(self) -> int:
        """Open every door in the home, so people walk through doorways rather than doors."""
        from omnigibson.object_states import Open
        n = 0
        for o in self.scene.objects:
            if "door" in getattr(o, "category", "") and Open in o.states:
                n += bool(o.states[Open].set_value(True))
        for _ in range(3):
            self.og.sim.step()
        return n

    def floor(self):
        """The scene's walking room at 1 cm (helper/behavior/floor.py)."""
        if not hasattr(self, "_floor"):
            from .floor import Floor
            lay = DATA / "behavior-1k-assets/scenes" / self.scene_model / "layout"
            png = lay / "floor_trav_no_door_0.png"  # doors are opened at load (open_doors)
            self._floor = Floor(png if png.exists() else lay / "floor_trav_0.png")
        return self._floor

    def clear_of_walls(self, xy, wall: float = 0.4) -> bool:
        """Whether xy has wall m of free floor around it (at 1 cm)."""
        return self.floor().clearance(xy) >= wall

    def support(self, point, skip=(), start=None) -> str | None:
        """The furniture right below a point: the first object a ray down meets, skipping `skip` (from `start` m up,
        default 0.3 m above the point, so it begins above the thing that rests there)."""
        import torch as th
        from omnigibson.utils.sampling_utils import raytest
        p = np.asarray(point, float)
        z0 = p[2] + 0.3 if start is None else start
        hits = raytest(start_point=th.tensor([p[0], p[1], z0], dtype=th.float32),
                       end_point=th.tensor([p[0], p[1], p[2] - 1.5], dtype=th.float32), only_closest=False)
        names = {o.name for o in self.scene.objects}
        for h in sorted(hits or [], key=lambda h: float(h["distance"])):
            n = self._name(h["rigidBody"])
            if n in names and n not in skip:
                return n
        return None

    def stand_spot(self, name: str | None, here, point=None, gap: float = 0.45, wall: float = 0.35,
                   near=None, away=None) -> np.ndarray | None:
        """Where to stand to work at an object (or a point): a traversable point gap to gap + 0.6 m from its box's
        edge with wall m of free floor around it (so the eyes stay out of furniture and walls), nearest to here."""
        import torch as th
        if name is not None:
            lo, hi = (np.asarray(v, float) for v in self.obj(name).aabb)
        else:
            lo = hi = np.asarray(point, float)
        fl, best = self.floor(), None
        home = fl.region(here)  # spots must be reachable by a body from here
        for r in np.linspace(gap, gap + 0.6, 4):
            for a in np.linspace(0, 2 * np.pi, 32, endpoint=False):
                d = np.array([np.cos(a), np.sin(a)])
                edge = (lo[:2] + hi[:2]) / 2 + d * float(np.abs(d) @ ((hi - lo)[:2] / 2))
                q = edge + r * d
                if not self._trav(q):
                    continue
                if fl.clearance(q) < wall or fl.region(q) != home:
                    continue
                if away is not None and float(np.linalg.norm(q - np.asarray(away, float)[:2])) < 0.8:
                    continue  # not where someone else stands
                score = float(np.linalg.norm(q - np.asarray(here, float)[:2])) + 0.5 * r
                if near is not None:  # nearest the thing to reach for first, then the walker
                    score = float(np.linalg.norm(q - np.asarray(near, float)[:2])) + 0.3 * score
                best = min(best, (score, q.tolist())) if best else (score, q.tolist())
        return np.asarray(best[1]) if best else None

    def free_near(self, target, r_min: float = 0.6, r_max: float = 1.2) -> np.ndarray | None:
        """The traversable point nearest target between r_min and r_max of it, or None."""
        for r in np.linspace(r_min, r_max, 5):
            for th in np.linspace(0, 2 * np.pi, 24, endpoint=False):
                q = np.asarray(target[:2], float) + r * np.array([np.cos(th), np.sin(th)])
                if self._trav(q):
                    return q
        return None

    def _trav(self, xy) -> bool:
        import torch as th
        tm = self.scene.trav_map
        m = tm.floor_map[0]
        rc = tm.world_to_map(th.tensor(np.asarray(xy, float)[:2], dtype=th.float32))
        r, c = int(rc[0]), int(rc[1])
        return 0 <= r < m.shape[0] and 0 <= c < m.shape[1] and bool(m[r, c] > 0)

    # -- agents ----------------------------------------------------------------------------------------
    def place_agent(self, who: str, xy, yaw: float) -> None:
        self.pose[who] = (np.asarray(xy, float), float(yaw))
        if who == "robot":
            self.robot.set_position_orientation(position=[xy[0], xy[1], 0.0], orientation=yaw_quat(yaw))
            self.robot.keep_still()
        else:
            self._body()
        self._carry(who)

    def _body(self) -> None:
        """The mannequin faces where the head points, just behind the eye camera, so its arms stay out of the view."""
        if self.body is None:  # the animated person is posed in step()
            return
        xy, yaw = self.pose["person"]
        d = self.head[:2] - xy
        yaw = float(np.arctan2(d[1], d[0])) if np.linalg.norm(d) > 0.05 else yaw
        back = 0.2 * np.array([np.cos(yaw), np.sin(yaw)])
        self.body.set_position_orientation(position=[xy[0] - back[0], xy[1] - back[1], self.body_z], orientation=body_quat(yaw))

    def _pin(self) -> None:
        for o, p in self.pinned.items():
            self.obj(o).set_position_orientation(position=p, orientation=self.pinned_orn.get(o))
            self.obj(o).keep_still()

    def _ghosts(self) -> None:
        """A thing in a hand passes through others (a held bottle swept past a mug knocked it over), and is solid
        again once set down."""
        now = {o for o in self.held.values() if o is not None}
        for o in sorted(now ^ self.ghosts):
            self.obj(o).visual_only = o in now
        self.ghosts = now

    # -- the robot's arm (Stretch: a lift and a telescoping arm out of its side) ---------------------------------
    ARM = ("joint_arm_l0", "joint_arm_l1", "joint_arm_l2", "joint_arm_l3")

    def arm_geometry(self) -> dict | None:
        """Measured once: the arm's direction in the base frame (rad), the grasp point's height at lift 0, and its reach
        from the base centre at no extension. None when the robot has no such arm (then objects are carried as before)."""
        if hasattr(self, "_arm_geo"):
            return self._arm_geo
        self._arm_geo = None
        try:
            names = list(self.robot.joints.keys())
            self._rj = {n: names.index(n) for n in ("joint_lift", *self.ARM) if n in names}
            if len(self._rj) < 5 or "link_grasp_center" not in self.robot.links:
                return None
            xy, yaw = self.pose["robot"] or (np.zeros(2), 0.0)
            g = lambda: np.asarray(self.robot.links["link_grasp_center"].get_position_orientation()[0], float)
            self.set_arm(0.0, 0.0)
            self.og.sim.step()
            p0 = g()
            self.set_arm(0.0, 0.4)
            self.og.sim.step()
            p1 = g()
            self.set_arm(0.6, 0.0)  # carrying height
            self.og.sim.step()
            ax = np.arctan2(p1[1] - p0[1], p1[0] - p0[0]) - yaw
            out = (p0[:2] - np.asarray(xy, float)) @ np.array([np.cos(yaw + ax), np.sin(yaw + ax)])
            self._arm_geo = {"angle": float(ax), "z0": float(p0[2]), "out0": float(out)}
        except Exception as e:
            print("robot arm unavailable:", repr(e)[:120], flush=True)
        return self._arm_geo

    def set_arm(self, lift: float, ext: float) -> None:
        """Lift (m above its lowest) and total extension (m) of the robot's arm."""
        q = self.robot.get_joint_positions().clone()
        q[self._rj["joint_lift"]] = float(np.clip(lift, 0.0, 1.1))
        for n in self.ARM:
            q[self._rj[n]] = float(np.clip(ext, 0.0, 0.52)) / 4
        self.robot.set_joint_positions(q)

    def grasp_point(self) -> np.ndarray:
        return np.asarray(self.robot.links["link_grasp_center"].get_position_orientation()[0], float)

    def _carry(self, who: str) -> None:
        o = self.held[who]
        if o is None:
            return
        xy, yaw = self.pose[who]
        if who == "person" and self.human is not None and self.human.sk is not None:  # in the right hand, a little ahead
            p = self.human.hand() + 0.08 * np.array([np.cos(yaw), np.sin(yaw), 0.0])
        elif who == "robot" and getattr(self, "_arm_geo", None):  # in the gripper
            p = self.grasp_point() - np.array([0.0, 0.0, 0.03])
        else:
            reach = 0.45
            p = [xy[0] + reach * np.cos(yaw), xy[1] + reach * np.sin(yaw), 1.0 if who == "person" else 1.1]  # held up, in sight
        self.obj(o).set_position_orientation(position=p)

    def restored(self) -> None:
        """After og.sim.load_state: nothing held, pinned or moved, and the person's view and stride as at the start, so
        an episode does not begin where the last one ended."""
        self.held, self.pinned, self.pinned_orn = {"person": None, "robot": None}, {}, {}
        self._ghosts()  # load_state keeps collision flags: whatever was in a hand is solid again
        self._shapes, self._rest_orn = {}, {}
        self.head, self.eye = np.array([1.0, 0.0, EYE]), np.array([0.0, 0.0, EYE])
        self._last_xy, self.reach, self.t = None, None, 0.0
        if self.human is not None:
            self.human.reset()

    def box(self, o: str) -> tuple[np.ndarray, np.ndarray]:
        """o's world box at its current pose, from its collision bounding box in its own frame (the simulator's own
        box goes stale while a thing is carried)."""
        from scipy.spatial.transform import Rotation
        try:
            c, q, ext, _ = self.obj(o).get_base_aligned_bbox(visual=False)
            c, q, ext = (np.asarray(x.cpu() if hasattr(x, "cpu") else x, float) for x in (c, q, ext))
            half = np.abs(Rotation.from_quat(q).as_matrix()) @ (ext / 2)
            return c - half, c + half
        except Exception:
            lo, hi = (np.asarray(v, float) for v in self.obj(o).aabb)
            return lo, hi

    def shape(self, o: str) -> tuple[np.ndarray, np.ndarray]:
        """(box centre minus origin, box size) of o, as measured while it last rested: a carried object's box in the
        simulator is stale."""
        self._shapes = getattr(self, "_shapes", {})
        if o not in self.held.values() or o not in self._shapes:
            lo, hi = (np.asarray(v, float) for v in self.obj(o).aabb)
            self._shapes[o] = ((lo + hi) / 2 - self.pos(o), hi - lo)
        return self._shapes[o]

    def pick(self, who: str, o: str) -> None:
        self.shape(o)  # measured at rest, before it moves
        self._rest_orn = getattr(self, "_rest_orn", {})
        q = self.obj(o).get_position_orientation()[1]
        self._rest_orn[o] = np.asarray(q.cpu() if hasattr(q, "cpu") else q, float)  # how it rested
        self.held[who] = o
        self.pinned.pop(o, None)
        self.pinned_orn.pop(o, None)
        self._carry(who)

    def place(self, who: str, o: str, rel: str, f: str) -> bool:
        from omnigibson.object_states import Inside, OnTop
        self.held[who] = None
        self._ghosts()
        if rel == "in" and f in self.inside:  # the middle of the closed container: Inside samples an open one's box
            self.pinned[o] = self.inside[f]
            self._pin()
            return True
        ok = self.obj(o).states[Inside if rel == "in" else OnTop].set_value(self.obj(f), True)
        if not ok:  # put it at the middle of the surface, or inside the box, by hand
            top = self.top(f)
            lo, hi = (np.asarray(v, float) for v in self.obj(f).aabb)
            z = top[2] + 0.05 if rel == "on" else (lo[2] + hi[2]) / 2
            self.obj(o).set_position_orientation(position=[top[0], top[1], z])
        return bool(ok)

    def relation(self, o: str, rel: str, f: str) -> bool:
        """Geometric relations between boxes: ontop (o rests within 4 cm above f's top, its centre over f), inside (o's
        centre in f's box), nextto (on the same level, boxes under 12 cm apart), under (o's centre over f's footprint, its
        top 10 cm or more below f's top: inside a cabinet's base or on the floor beneath a table)."""
        lo, hi = self.box(o)
        flo, fhi = self.box(f)
        c = (lo + hi) / 2
        over = flo[0] <= c[0] <= fhi[0] and flo[1] <= c[1] <= fhi[1]
        if rel == "ontop":  # resting between f's base and its rim (a plate is a few cm deep), or just on top
            return over and min(fhi[2] - 0.04, flo[2] - 0.01) <= lo[2] <= fhi[2] + 0.04  # a plate is thin
        if rel == "inside":
            return bool(np.all(c >= flo) and np.all(c <= fhi))
        if rel in ("nextto", "touching"):
            gap = np.maximum(0.0, np.maximum(flo[:2] - hi[:2], lo[:2] - fhi[:2]))
            return float(np.linalg.norm(gap)) < 0.12 and abs(lo[2] - flo[2]) < 0.06 and not over
        if rel == "under":  # beneath its top surface, within its footprint (a sink cabinet stands on the floor)
            return over and hi[2] <= fhi[2] - 0.1 and lo[2] >= flo[2] - 0.05
        return False

    def _surfaces(self, f: str, x: float, y: float, top: float, bottom: float, floor: bool = False,
                  also=()) -> list[float]:
        """Heights of f's surfaces (the floor's, if asked; those of objects in also) that a ray down at (x, y) meets,
        lowest first."""
        import torch as th
        from omnigibson.utils.sampling_utils import raytest
        hits = raytest(start_point=th.tensor([x, y, top + 0.05], dtype=th.float32),
                       end_point=th.tensor([x, y, bottom - 0.05], dtype=th.float32), only_closest=False) or []
        keep = lambda n: n == f or n in also or (floor and "floor" in n)
        return sorted(top + 0.05 - float(h["distance"]) for h in hits if keep(self._name(h["rigidBody"])))

    def _footing(self, f: str, gx: float, gy: float, z: float, half, height: float, top: float, bottom: float,
                 floor: bool = False, also=(), flat: bool = True) -> bool:
        """Whether o's footprint at (gx, gy) is clear at z: rays at its corners meet nothing (a wall, a rim, a shelf
        above) up to the height of o, and meet a surface within 2 cm of z (flat) or anywhere below (in a container)."""
        for dx in (-half[0], half[0]):
            for dy in (-half[1], half[1]):
                zs = self._surfaces(f, gx + dx, gy + dy, top, bottom, floor, also)
                under = any(abs(z_ - z) < 0.02 for z_ in zs) if flat else any(z_ <= z + 0.02 for z_ in zs)
                if not under or any(z + 0.02 < z_ < z + height for z_ in zs):
                    return False
        return True

    def _others(self, o: str, f: str, lo, hi, above: float) -> list:
        """Boxes of other objects within f's footprint, from its bottom to `above` m over its top."""
        out = []
        for x in self.scene.objects:
            if x.name in (o, f):
                continue
            try:  # an object moved before has its box from its pose (the simulator's own can be stale after a move)
                xl, xh = self.box(x.name) if x.name in getattr(self, "_shapes", {}) else (np.asarray(v, float) for v in x.aabb)
            except Exception:
                continue
            if xh[0] > lo[0] and xl[0] < hi[0] and xh[1] > lo[1] and xl[1] < hi[1] and xl[2] < hi[2] + above and xh[2] > lo[2]:
                out.append((xl, xh))
        return out

    @staticmethod
    def _free(gx, gy, z, half, height, others) -> bool:
        return all(gx + half[0] < xl[0] or gx - half[0] > xh[0] or gy + half[1] < xl[1] or gy - half[1] > xh[1]
                   or xh[2] < z + 0.005 or xl[2] > z + height for xl, xh in others)

    def free_tops(self, o: str, f: str, n: int = 5, level: float | None = None) -> list:
        """Points (x, y, surface z) on f where o fits clear of other objects: on its topmost surface under each point,
        or, given a level, on the surface within 4 cm of it; nearest f's middle first."""
        lo, hi = self.box(f)
        olo, ohi = self.box(o)
        ext = ohi - olo
        half = ext[:2] / 2 + 0.01
        others = self._others(o, f, lo, hi, 0.3)
        c, pts = (lo + hi) / 2, []
        for gx in np.linspace(lo[0] + half[0], hi[0] - half[0], n):
            for gy in np.linspace(lo[1] + half[1], hi[1] - half[1], n):
                zs = self._surfaces(f, gx, gy, hi[2], lo[2])
                if level is not None:
                    zs = [z for z in zs if abs(z - level) < 0.04]
                if not zs:
                    continue
                z = zs[-1] if level is None else min(zs, key=lambda z_: abs(z_ - level))
                if self._free(gx, gy, z, half, ext[2], others) and self._footing(f, gx, gy, z, half, ext[2], hi[2], lo[2]):
                    pts.append((float(np.hypot(gx - c[0], gy - c[1])), [gx, gy, z]))
        return [q for _, q in sorted(pts)]

    def inner_spots(self, o: str, f: str, n: int = 10, floor: bool = False) -> list:
        """Points (x, y, surface z) inside or beneath f (shelves of a bookcase, the bottom of a bin, the floor under a
        table when floor is asked): surfaces below f's top with room above for o and no other object there, lowest
        first."""
        lo, hi = self.box(f)
        olo, ohi = self.box(o)
        ext = ohi - olo
        half = ext[:2] / 2 + 0.01
        others = self._others(o, f, lo, hi, 0.0)
        within = {x.name for x in self.scene.objects if x.name in getattr(self, "_shapes", {}) and x.name not in (o, f)
                  and np.all(np.abs(self.pos(x.name)[:2] - (lo + hi)[:2] / 2) < (hi - lo)[:2] / 2)
                  and lo[2] <= self.pos(x.name)[2] <= hi[2]}  # put in before: once the bottom is full, on top of them
        c, pts = (lo + hi) / 2, []
        for gx in np.linspace(lo[0] + half[0], hi[0] - half[0], n):
            for gy in np.linspace(lo[1] + half[1], hi[1] - half[1], n):
                zs = self._surfaces(f, gx, gy, hi[2], lo[2], floor, within)
                for k, z in enumerate(zs):
                    ceiling = zs[k + 1] if k + 1 < len(zs) else hi[2] + 1.0
                    if z > hi[2] - 0.03 or ceiling - z < ext[2] + 0.02:  # the top is "on"; a shelf needs headroom
                        continue
                    if self._free(gx, gy, z, half, ext[2], others) and self._footing(f, gx, gy, z, half, ext[2], hi[2],
                                                                                    lo[2], floor, within, flat=False):
                        pts.append((z + 0.3 * float(np.hypot(gx - c[0], gy - c[1])), [gx, gy, z]))  # lowest first
        return [q for _, q in sorted(pts)]

    def _open_container(self, f: str) -> bool:
        """A box or basket open at the top: rays down its middle meet its floor first, under its rim."""
        lo, hi = self.box(f)
        c = (lo + hi) / 2
        zs = self._surfaces(f, c[0], c[1], hi[2], lo[2])
        return len(zs) == 1 and zs[0] < lo[2] + 0.5 * (hi[2] - lo[2])

    def _turned(self, o: str, at, orn, find) -> tuple[list, np.ndarray]:
        """(spots, orientation): find() with o as oriented, else with o turned 90 degrees about the vertical (a long toy
        train along a shallow shelf). Leaves o in that orientation."""
        for q in (orn, yaw90(orn)):
            self.obj(o).set_position_orientation(position=at + [0, 0, 0.3], orientation=q)
            spots = find()
            if spots:
                return spots, q
        self.obj(o).set_position_orientation(position=at + [0, 0, 0.3], orientation=orn)
        return [], orn

    def _fit(self, o: str, f: str, at, orn) -> tuple | None:
        """(spot, orientation) for o in open container f: lying as it rests, else upright (a toothbrush in a cup),
        narrowest footprint first. Leaves o in that orientation."""
        h = np.sqrt(0.5)
        ups = []
        for q in (np.array([h, 0.0, 0.0, h]), np.array([0.0, h, 0.0, h])):  # its model frame turned 90 deg about x, y
            self.obj(o).set_position_orientation(position=at + [0, 0, 0.3], orientation=q)
            lo, hi = self.box(o)
            ups.append((float(np.prod((hi - lo)[:2])), q))
        for k, q in enumerate([orn, yaw90(orn)] + [q for _, q in sorted(ups, key=lambda x: x[0])]):
            self.obj(o).set_position_orientation(position=at + [0, 0, 0.3], orientation=q)
            spot = self.stack_spot(o, f, tall=k >= 2)
            if spot is not None:
                return spot, q
        self.obj(o).set_position_orientation(position=at + [0, 0, 0.3], orientation=orn)
        return None

    def stack_spot(self, o: str, f: str, n: int = 12, tall: bool = False) -> list | None:
        """Where o goes in an open container f (a toy box, a basket): over f's floor, clear of its walls, resting on
        whatever is already there; lowest, then nearest the middle. None when o's top would stand above f (its middle,
        when tall: a toothbrush stands out of a cup)."""
        lo, hi = self.box(f)
        olo, ohi = self.box(o)
        ext = ohi - olo
        half = ext[:2] / 2 + 0.01
        others = self._others(o, f, lo, hi, 0.0)
        c, best = (lo + hi) / 2, None
        for gx in np.linspace(lo[0] + half[0], hi[0] - half[0], n):
            for gy in np.linspace(lo[1] + half[1], hi[1] - half[1], n):
                zs = self._surfaces(f, gx, gy, hi[2], lo[2])
                if not zs or zs[0] > hi[2] - (ext[2] / 2 if tall else ext[2]) or not self._footing(
                        f, gx, gy, zs[0], half, ext[2], hi[2], lo[2], flat=False):
                    continue
                under = [xh[2] for xl, xh in others if not (gx + half[0] < xl[0] or gx - half[0] > xh[0]
                                                            or gy + half[1] < xl[1] or gy - half[1] > xh[1])]
                z = max([zs[0]] + under)
                if z + ext[2] / 2 > hi[2] - 0.01:  # it has to be in, not on top
                    continue
                score = z + 0.3 * float(np.hypot(gx - c[0], gy - c[1]))
                if best is None or score < best[0]:
                    best = (score, [gx, gy, z])
        return None if best is None else best[1]

    def put(self, who: str, o: str, rel: str, f: str) -> bool:
        """Set o down in relation rel to f (ontop or on, inside, nextto, under); kept in hand when it will not go."""
        rel = "ontop" if rel == "on" else rel
        if rel == "inside" and f in self.inside:  # the middle of the closed container (see close_box)
            self.held[who] = None
            self._ghosts()
            self.pinned[o] = self.inside[f]
            self._pin()
            return True
        self.held[who] = None
        self._ghosts()  # solid before any physics step below
        orn, best, held_at = None, None, self.pos(o)  # the flatter of how it last rested and its model's own
        for q in (getattr(self, "_rest_orn", {}).get(o), np.array([0.0, 0.0, 0.0, 1.0])):  # orientation: a cupcake
            if q is None:  # stays upright, a knife lies down
                continue
            self.obj(o).set_position_orientation(position=held_at + [0, 0, 0.3], orientation=q)
            blo, bhi = self.box(o)
            if best is None or bhi[2] - blo[2] < best:
                orn, best = q, bhi[2] - blo[2]
        self.obj(o).set_position_orientation(position=held_at + [0, 0, 0.3], orientation=orn)  # spots fit this footprint
        lo, hi = self.box(f)
        olo, ohi = self.box(o)
        ext = ohi - olo
        c, xy = (lo + hi) / 2, self.pose[who][0]
        if rel in ("nextto", "touching"):  # free points on what f rests on, 1 to 10 cm from f, nearest f first
            below = self.support(c, skip={o, f}, start=lo[2] + 0.005)
            spots = []
            if below is not None:
                for q in self.free_tops(o, below, n=12, level=lo[2]):  # on the level f stands on
                    gap = np.maximum(0.0, np.maximum(lo[:2] - (np.array(q[:2]) + ext[:2] / 2),
                                                     (np.array(q[:2]) - ext[:2] / 2) - hi[:2]))
                    if 0.01 <= float(np.linalg.norm(gap)) <= 0.10:
                        spots.append((float(np.linalg.norm(np.array(q[:2]) - c[:2])), q))
                spots = [q for _, q in sorted(spots)][:6]
            if not spots:
                toward = (xy - c[:2]) / (np.linalg.norm(xy - c[:2]) + 1e-9)
                side = np.array([-toward[1], toward[0]])
                spots = [[*(c[:2] + (float(np.abs(d) @ ((hi - lo)[:2] / 2 + ext[:2] / 2)) + 0.03) * d), lo[2]]
                         for d in (side, -side, toward)]
        elif rel == "under":  # inside the cabinet below its top, or the floor beneath it
            spots, orn = self._turned(o, held_at, orn, lambda: self.inner_spots(o, f, floor=True)[:12])  # none: kept
        elif rel == "ontop":  # a free point on the top (BEHAVIOR's sampler can leave o overlapping f, which flings f)
            spots, orn = self._turned(o, held_at, orn, lambda: self.free_tops(o, f)[:6])
            spots = spots or [[c[0], c[1], hi[2]]]
        elif (q := self._fit(o, f, held_at, orn) if self._open_container(f) else None) is not None:  # a box, a basket,
            q, orn = q  # a cup: on its floor or on what is in it, held there (pushed about, a box of small things spills)
            blo, bhi = self.box(o)
            shift = np.array([q[0] - (blo[0] + bhi[0]) / 2, q[1] - (blo[1] + bhi[1]) / 2, q[2] + 0.005 - blo[2]])
            self.obj(o).set_position_orientation(position=self.pos(o) + shift, orientation=orn)
            self.pinned[o], self.pinned_orn[o] = self.pos(o), orn
            self._pin()
            if self.relation(o, rel, f):
                return True
            self.pinned.pop(o, None)
            spots, orn = self._turned(o, held_at, orn, lambda: self.inner_spots(o, f)[:12])
        else:  # inside a bookcase, a cabinet or a cup: a free spot on one of its shelves, or its bottom
            spots, orn = self._turned(o, held_at, orn, lambda: self.inner_spots(o, f)[:12])
        for p in spots:  # p: (x, y, surface z); o goes down in that orientation, its box 1 cm above the surface
            p = np.asarray(p, float)
            self.obj(o).set_position_orientation(position=p + [0, 0, 0.3], orientation=orn)
            blo, bhi = self.box(o)
            shift = np.array([p[0] - (blo[0] + bhi[0]) / 2, p[1] - (blo[1] + bhi[1]) / 2, p[2] + 0.01 - blo[2]])
            self.obj(o).set_position_orientation(position=self.pos(o) + shift, orientation=orn)
            self.obj(o).keep_still()
            for _ in range(4):
                self.og.sim.step()
            if not self.relation(o, rel, f):
                continue
            p0 = self.pos(o)
            for _ in range(10):  # it has to stay: a thing balanced on a rim falls off
                self.og.sim.step()
            if self.relation(o, rel, f) and float(np.linalg.norm(self.pos(o) - p0)) < 0.03:
                return True
        self.held[who] = o  # still in hand
        self._ghosts()
        return False

    def place_beside(self, who: str, o: str, f: str) -> bool:
        """Put o next to f on whatever f rests on (beside it, toward the agent first), let it settle, and check NextTo."""
        from omnigibson.object_states import NextTo
        self.held[who] = None
        self._ghosts()
        lo, hi =(np.asarray(v, float) for v in self.obj(f).aabb)
        olo, ohi = (np.asarray(v, float) for v in self.obj(o).aabb)
        c, xy = (lo + hi) / 2, self.pose[who][0]
        toward = (xy - c[:2]) / (np.linalg.norm(xy - c[:2]) + 1e-9)
        side = np.array([-toward[1], toward[0]])
        for d in (side, -side, toward):
            gap = float(np.abs(d) @ ((hi - lo)[:2] / 2 + (ohi - olo)[:2] / 2)) + 0.04
            p = [c[0] + gap * d[0], c[1] + gap * d[1], lo[2] + (ohi - olo)[2] / 2 + 0.01]
            self.obj(o).set_position_orientation(position=p)
            for _ in range(3):
                self.og.sim.step()
            if self.obj(o).states[NextTo].get_value(self.obj(f)):
                return True
        return False

    def close_box(self, f: str) -> None:
        """Close container f and keep the middle of its closed box as the place for things put in it."""
        self.open(f, False)
        self.og.sim.step()
        lo, hi = (np.asarray(v, float) for v in self.obj(f).aabb)
        self.inside[f] = (lo + hi) / 2

    def open(self, f: str, on: bool) -> bool:
        from omnigibson.object_states import Open
        if Open not in self.obj(f).states:  # an open basket or bin: nothing to open
            return False
        return bool(self.obj(f).states[Open].set_value(on))

    # -- frames ---------------------------------------------------------------------------------------------------
    def step(self) -> None:
        if self.pose["person"] is not None:
            xy, yaw = self.pose["person"]
            if self.human is not None:  # the eyes: ahead of and above the head joint of the posed body
                moving = self._last_xy is not None and float(np.linalg.norm(xy - self._last_xy)) > 1e-3
                self.human.update(xy, yaw, moving, DT, reach=self.reach, look=self.head)
                self.eye = self.human.eye()
                self._last_xy = np.array(xy, float)
            else:
                self.eye = np.array([xy[0], xy[1], EYE])
            self.cams["person_cam"].set_position_orientation(position=self.eye, orientation=look_quat(self.eye, self.head))
            self._body()
        for who in ("person", "robot"):
            if self.pose[who] is not None:
                self._carry(who)
        self._pin()
        self._ghosts()
        self.og.sim.step()
        self.t += DT

    def rgb(self, cam: str = "person_cam") -> np.ndarray:
        obs, _ = self.cams[cam].get_obs()
        return obs["rgb"][..., :3].cpu().numpy().astype(np.uint8)

    # the person's view, by raycasts ---------------------------------------------------------------------------------
    HFOV = 2 * np.arctan(20.995 / (2 * 17.0))  # OmniGibson's default camera: aperture 20.995, focal length 17

    def basis(self) -> tuple:
        xy, yaw = self.pose["person"]
        if self.human is not None:  # the eyes at the current pose, at the last measured eye height
            z = float(self.eye[2]) if self.eye[2] > 1.0 else EYE
            eye = np.array([xy[0] + 0.14 * np.cos(yaw), xy[1] + 0.14 * np.sin(yaw), z])
        else:
            eye = np.array([xy[0], xy[1], EYE])
        f = np.asarray(self.head, float) - eye
        f /= np.linalg.norm(f) + 1e-9
        r = np.cross(f, [0, 0, 1.0])
        r /= np.linalg.norm(r) + 1e-9
        return eye, f, r, np.cross(r, f)

    def project(self, p, basis=None) -> tuple | None:
        """Image coordinates (u, v) in [0, 1] of a world point, or None behind the eye."""
        eye, f, r, u = basis or self.basis()
        d = np.asarray(p, float) - eye
        z = d @ f
        if z < 0.05:
            return None
        k = 2 * np.tan(self.HFOV / 2)
        return 0.5 + (d @ r) / z / k, 0.5 - (d @ u) / z / k

    def hit(self, a, b) -> str | None:
        """The object a ray from a to b hits first, or None."""
        import torch as th
        from omnigibson.utils.sampling_utils import raytest
        res = raytest(start_point=th.tensor(a, dtype=th.float32), end_point=th.tensor(b, dtype=th.float32), only_closest=True)
        return self._name(res["rigidBody"]) if res.get("hit") else None

    def gaze_hit(self, uv) -> str | None:
        """Whatever the gaze through image point uv falls on."""
        eye, f, r, u = self.basis()
        k = np.tan(self.HFOV / 2)
        d = f + (2 * uv[0] - 1) * k * r - (2 * uv[1] - 1) * k * u
        return self.hit(eye, eye + 12.0 * d / np.linalg.norm(d))

    def clear(self, a, b, target: str | None = None) -> bool:
        """Nothing blocks the line from a to b (hitting target itself counts as clear)."""
        import torch as th
        from omnigibson.utils.sampling_utils import raytest
        a, b = np.asarray(a, float), np.asarray(b, float)
        res = raytest(start_point=th.tensor(a, dtype=th.float32), end_point=th.tensor(b, dtype=th.float32), only_closest=True)
        if not res.get("hit"):
            return True
        return (target is not None and self._name(res["rigidBody"]) == target) or float(res["distance"]) >= np.linalg.norm(b - a) - 0.05

    def work_spot(self, target, eye, r_min: float = 0.55, r_max: float = 1.2, hold: float = 1.1) -> np.ndarray | None:
        """Where the robot stands to work at target: a traversable point from which the object it holds (0.45 m toward
        target, hold m up) is in clear view of eye, and which keeps off the line from eye to target."""
        t2, e2 = np.asarray(target[:2], float), np.asarray(eye[:2], float)
        look = (e2 - t2) / (np.linalg.norm(e2 - t2) + 1e-9)
        best = None
        home = self.floor().region(eye[:2]) if self.human is not None else None
        for r in np.linspace(r_min, r_max, 5):
            for th in np.linspace(0, 2 * np.pi, 24, endpoint=False):
                q = t2 + r * np.array([np.cos(th), np.sin(th)])
                if not self._trav(q):
                    continue
                if home is not None and (self.floor().clearance(q) < 0.3 or self.floor().region(q) != home):
                    continue  # the robot's base needs room too
                h = np.array([*(q + 0.45 * (t2 - q) / r), hold])
                seen = self.clear(eye, h)
                score = (0 if seen else 1, float(look @ (q - t2) / r))  # in view first, then away from the line
                best = min(best, (score, q.tolist())) if best else (score, q.tolist())
        return np.asarray(best[1]) if best else None

    def side_spot(self, target, away_from, r_min: float = 0.55, r_max: float = 1.2) -> np.ndarray | None:
        """A traversable point near target on the side away from away_from (out of its line of sight to target)."""
        t2, a2 = np.asarray(target[:2], float), np.asarray(away_from[:2], float)
        look = (a2 - t2) / (np.linalg.norm(a2 - t2) + 1e-9)
        best = None
        for r in np.linspace(r_min, r_max, 5):
            for th in np.linspace(0, 2 * np.pi, 24, endpoint=False):
                q = t2 + r * np.array([np.cos(th), np.sin(th)])
                if self._trav(q):
                    score = float(look @ (q - t2) / r)  # -1 behind target, +1 in the line of sight
                    best = min(best, (score, r, q.tolist())) if best else (score, r, q.tolist())
        return np.asarray(best[2]) if best else None

    def visible(self, names: list[str] | None = None, max_dist: float = 8.0, strict: tuple = ()) -> dict:
        """{object: (cx, cy, area)} in the person's view: projected box area times the share of sample points a ray
        from the eye reaches first; (cx, cy) the mean of the reached points."""
        basis = self.basis()
        eye = basis[0]
        out = {}
        for o in names if names is not None else self.candidates():
            try:
                lo, hi = self.bounds(o)
            except Exception:
                continue
            c = (lo + hi) / 2
            if np.linalg.norm(c - eye) > max_dist:
                continue
            corners = [self.project([x, y, z], basis) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
            corners = [q for q in corners if q is not None]
            if not corners:
                continue
            us, vs = [q[0] for q in corners], [q[1] for q in corners]
            box = (max(0.0, min(1.0, max(us))) - max(0.0, min(1.0, min(us)))) * (max(0.0, min(1.0, max(vs))) - max(0.0, min(1.0, min(vs))))
            if box <= 0:
                continue
            half, toward = (hi - lo) / 2, np.sign(eye - c)  # the faces that point toward the eye
            pts = [c, [c[0], c[1], hi[2] - 0.01]]
            for z in (0.3, 0.7):
                zz = lo[2] + z * (hi[2] - lo[2])
                pts += [[c[0] + toward[0] * half[0] * 0.95, c[1], zz], [c[0], c[1] + toward[1] * half[1] * 0.95, zz]]
            seen, ghost = [], o == "robot" or o in self.ghosts  # no collision shape: seen when nothing is in between
            for p in pts:
                q = self.project(p, basis)
                if q is None or not (0 <= q[0] <= 1 and 0 <= q[1] <= 1):
                    continue
                if (self.hit(eye, eye + (np.asarray(p) - eye) * 0.97) is None) if ghost else \
                        self.hit(eye, eye + (np.asarray(p) - eye) * 1.02) == o:
                    seen.append(q)
            if len(seen) >= (2 if o in strict else 1):  # a strict object needs two rays, not one through a gap
                out[o] = (float(np.mean([q[0] for q in seen])), float(np.mean([q[1] for q in seen])), float(box * len(seen) / len(pts)))
        return out

    def bounds(self, o: str) -> tuple[np.ndarray, np.ndarray]:
        """An object's box. A carried object's box follows its pose (the simulator's box updates only at rest): its size
        and offset from the pose are taken while it rests."""
        lo, hi = (np.asarray(v, float) for v in self.obj(o).aabb)
        if o not in self.held.values():
            if o.startswith("X_"):
                self._shape = getattr(self, "_shape", {})
                self._shape[o] = ((lo + hi) / 2 - self.pos(o), hi - lo)
            return lo, hi
        off, ext = getattr(self, "_shape", {}).get(o, (np.zeros(3), hi - lo))
        c = self.pos(o) + off
        return c - ext / 2, c + ext / 2

    def candidates(self) -> list[str]:
        """Objects the person may look at: everything but the house's shell, the person's own body and parked objects."""
        if not hasattr(self, "_cands"):
            shell = ("walls", "floors", "ceilings", "background", "carpet", "downlight", "electric_switch")
            self._cands = [o.name for o in self.scene.objects if getattr(o, "category", "") not in shell
                           and o.name != "person_body"]
        return [n for n in self._cands if not n.startswith("X_") or self.pos(n)[2] > -5]

    @staticmethod
    def _name(v) -> str:
        """An object name from a USD path."""
        parts = [p for p in str(v).split("/") if p and p != "World" and not p.startswith("scene_")]
        return parts[0].removeprefix("articulated__") if parts else str(v)
