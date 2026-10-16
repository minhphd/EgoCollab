"""One BEHAVIOR home in OmniGibson for the collaboration env: the animated person, a Stretch, the task's objects.

The person is an Isaac Sim character skinned every frame (helper/behavior/human.py), its eye camera in front of its
eyes; it is moved along planned paths by its owner (helper/collab/person.py). The Stretch is driven through
helper/collab/stretch_sim.py. What the person sees comes from physics rays from the eye. Objects in the person's hand
follow the hand and pass through things (visual_only) until set down; objects set in an open container are held there.
One frame is 1/15 s: two 1/30 s physics steps.

    w = World(scene, rooms, objects)        objects: [{"name", "category"}]
    w.step(robot_cmd)                        one frame
    w.put(o, rel, f) / w.pick(o)             the person's hand; w.settle(o, rel, f) after the robot lets go
    w.relation(o, rel, f), w.visible(names), w.path(a, b), w.stand_spot(name, here)
    s = w.save(); w.restore(s)
"""

from __future__ import annotations

import os

import numpy as np

from helper.behavior.sim import DATA, EYE, look_quat, model_of, yaw90, yaw_quat

from .stretch_sim import CONFIG, StretchSim

SHELL = ("walls", "floors", "ceilings", "background", "carpet", "downlight", "electric_switch")


class World:
    def __init__(self, scene: str, rooms: list[str], objects: list[dict], third: bool = False, px: int = 512,
                 character: str = "M_Medical_01"):
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        from helper.behavior.human import Human, patch_app
        patch_app()  # before omnigibson starts Isaac Sim
        import omnigibson as og
        from omnigibson.macros import gm
        from helper.behavior.floor import Floor
        gm.HEADLESS, gm.ENABLE_OBJECT_STATES, gm.ENABLE_TRANSITION_RULES = True, True, False
        self.og = og
        cam = lambda name, w, h: {"sensor_type": "VisionSensor", "name": name, "relative_prim_path": f"/{name}",
                                  "modalities": ["rgb"], "sensor_kwargs": {"image_height": h, "image_width": w},
                                  "position": [0.0, 0.0, 10.0], "orientation": [0.0, 0.0, 0.0, 1.0]}
        objs = [{"type": "DatasetObject", "name": o["name"], "category": o["category"],
                 "model": o.get("model") or model_of(o["category"]), "position": [2.0 * i, 0.0, -20.0]}
                for i, o in enumerate(objects)]
        cfg = {"scene": {"type": "InteractiveTraversableScene", "scene_model": scene, "load_room_instances": list(rooms)},
               "robots": [CONFIG], "objects": objs,
               "env": {"external_sensors": [cam("person_cam", px, px)] + ([cam("third_cam", 640, 480), cam("robot_cam", 640, 480)] if third else []),
                       "action_frequency": 30, "physics_frequency": 120}}
        self.env = og.Environment(configs=cfg)
        self.env.reset()
        self.scene = self.env.scene
        rb = self.env.robots[0]
        self.bot = StretchSim(rb[0] if isinstance(rb, (list, tuple)) else rb)
        es = self.env.external_sensors
        es = es[0] if isinstance(es, (list, tuple)) else es
        self.cams = dict(es) if isinstance(es, dict) else {s.name: s for s in es}
        self.task_objects = [o["name"] for o in objects]
        for i, o in enumerate(self.task_objects):  # back to their parking spots, still: the env places them
            self.obj(o).set_position_orientation(position=[2.0 * i, 0.0, -20.0])
            self.obj(o).keep_still()
        self.human = Human(og.sim, character)
        self.walk_speed = self.human.speed
        self.open_doors()
        lay = DATA / "behavior-1k-assets/scenes" / scene / "layout"
        self.floor = Floor(lay / "floor_trav_no_door_0.png")
        self.inside = {}  # closed container -> the middle of its closed box
        self.reset_state()

    def reset_state(self) -> None:
        """Nothing held, pinned or moved; the person standing, looking ahead."""
        self.held = {"person": None, "robot": None}
        self.ghosts, self.pinned, self.shapes, self.rest_orn = set(), {}, {}, {}
        self.robot_ghost = False
        self.person_xy, self.person_yaw = np.zeros(2), 0.0
        self.head, self.eye, self.reach, self.last_xy = np.array([1.0, 0.0, EYE]), np.array([0.0, 0.0, EYE]), None, None
        self.human.reset()

    # -- objects ----------------------------------------------------------------------------------------------------
    def obj(self, name: str):
        return self.scene.object_registry("name", name)

    def pos(self, name: str) -> np.ndarray:
        return np.asarray(self.obj(name).get_position_orientation()[0], float)

    def box(self, o: str) -> tuple[np.ndarray, np.ndarray]:
        """o's world box at its current pose (from its base-aligned collision box: the simulator's own box is stale
        while a thing moves)."""
        from scipy.spatial.transform import Rotation
        try:
            c, q, ext, _ = self.obj(o).get_base_aligned_bbox(visual=False)
            c, q, ext = (np.asarray(x.cpu() if hasattr(x, "cpu") else x, float) for x in (c, q, ext))
            half = np.abs(Rotation.from_quat(q).as_matrix()) @ (ext / 2)
            return c - half, c + half
        except Exception:
            lo, hi = (np.asarray(v, float) for v in self.obj(o).aabb)
            return lo, hi

    def top(self, name: str) -> np.ndarray:
        lo, hi = self.box(name)
        return np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2]])

    def relation(self, o: str, rel: str, f: str) -> bool:
        """ontop: o rests within 4 cm of f's top, its centre over f; inside: o's centre in f's box."""
        lo, hi = self.box(o)
        flo, fhi = self.box(f)
        c = (lo + hi) / 2
        if rel == "inside":
            return bool(np.all(c >= flo) and np.all(c <= fhi))
        over = flo[0] <= c[0] <= fhi[0] and flo[1] <= c[1] <= fhi[1]
        return over and min(fhi[2] - 0.04, flo[2] - 0.01) <= lo[2] <= fhi[2] + 0.04

    def open(self, f: str, on: bool) -> None:
        from omnigibson.object_states import Open
        if Open in self.obj(f).states:  # an open basket or bin has nothing to open
            self.obj(f).states[Open].set_value(on)

    def open_doors(self) -> None:
        """Every door open, so people walk through doorways."""
        for o in self.scene.objects:
            if "door" in getattr(o, "category", ""):
                self.open(o.name, True)
        for _ in range(3):
            self.og.sim.step()

    # -- placement (the person's hand; the robot's release) ----------------------------------------------------------
    def _rays(self, a, b) -> list:
        """What a ray from a to b meets, nearest first, but meta links (a container's fillable volume reaches over its
        rim: rays meet it, things do not)."""
        import torch as th
        from omnigibson.utils.sampling_utils import raytest
        hits = raytest(start_point=th.tensor(np.asarray(a, float), dtype=th.float32),
                       end_point=th.tensor(np.asarray(b, float), dtype=th.float32), only_closest=False) or []
        return sorted((h for h in hits if "meta__" not in str(h["rigidBody"])), key=lambda h: float(h["distance"]))

    def _surfaces(self, f: str, x: float, y: float, top: float, bottom: float, also=()) -> list[float]:
        """Heights of f's surfaces (and those of objects in `also`) that a ray down at (x, y) meets, lowest first."""
        hits = self._rays([x, y, top + 0.05], [x, y, bottom - 0.05])
        return sorted(top + 0.05 - float(h["distance"]) for h in hits if self.name(h["rigidBody"]) in (f, *also))

    def _footing(self, f: str, gx: float, gy: float, z: float, half, height: float, top: float, bottom: float,
                 also=(), flat: bool = True) -> bool:
        """o's footprint at (gx, gy) is clear at z: rays at its corners meet nothing up to o's height (a wall, a rim, a
        shelf above) and meet a surface within 2 cm of z (flat) or anywhere below (in a container)."""
        for dx in (-half[0], half[0]):
            for dy in (-half[1], half[1]):
                zs = self._surfaces(f, gx + dx, gy + dy, top, bottom, also)
                under = any(abs(z_ - z) < 0.02 for z_ in zs) if flat else any(z_ <= z + 0.02 for z_ in zs)
                if not under or any(z + 0.02 < z_ < z + height for z_ in zs):
                    return False
        return True

    def _others(self, o: str, f: str, lo, hi, above: float) -> list:
        """Boxes of the task's other objects within f's footprint, from its bottom to `above` m over its top."""
        out = []
        for x in self.task_objects:
            if x in (o, f):
                continue
            xl, xh = self.box(x)
            if xh[0] > lo[0] and xl[0] < hi[0] and xh[1] > lo[1] and xl[1] < hi[1] and xl[2] < hi[2] + above and xh[2] > lo[2]:
                out.append((xl, xh))
        return out

    @staticmethod
    def _free(gx, gy, z, half, height, others) -> bool:
        return all(gx + half[0] < xl[0] or gx - half[0] > xh[0] or gy + half[1] < xl[1] or gy - half[1] > xh[1]
                   or xh[2] < z + 0.005 or xl[2] > z + height for xl, xh in others)

    def spots(self, o: str, rel: str, f: str, n: int = 10) -> list:
        """Points (x, y, surface z) where o goes: on f's top clear of other objects, nearest its middle (ontop); on f's
        shelves or bottom with headroom, lowest first (inside); in an open container, on what is already there."""
        lo, hi = self.box(f)
        olo, ohi = self.box(o)
        ext, c, pts = ohi - olo, (lo + hi) / 2, []
        half = ext[:2] / 2 + 0.01
        if rel == "ontop":
            others = self._others(o, f, lo, hi, 0.3)
            for gx in np.linspace(lo[0] + half[0], hi[0] - half[0], n):
                for gy in np.linspace(lo[1] + half[1], hi[1] - half[1], n):
                    zs = self._surfaces(f, gx, gy, hi[2], lo[2])
                    if zs and self._free(gx, gy, zs[-1], half, ext[2], others) and \
                            self._footing(f, gx, gy, zs[-1], half, ext[2], hi[2], lo[2]):
                        pts.append((float(np.hypot(gx - c[0], gy - c[1])), [gx, gy, zs[-1]]))
            return [q for _, q in sorted(pts)]
        others = self._others(o, f, lo, hi, 0.0)
        within = [x for x in self.task_objects if x not in (o, f) and self.relation(x, "inside", f)]  # put in before
        if self.open_top(f):  # a bowl, box or basket: beside what is in it, else on it, never into it; low, middle first (held there)
            boxes = [self.box(x) for x in within]
            for gx in np.linspace(lo[0], hi[0], n + 2)[1:-1]:
                for gy in np.linspace(lo[1], hi[1], n + 2)[1:-1]:
                    zs = self._surfaces(f, gx, gy, hi[2], lo[2], within)
                    if zs and zs[0] < hi[2] - 0.01 and self._free(gx, gy, max(zs), half, ext[2], boxes):
                        z = max(zs)
                        fits = self._footing(f, gx, gy, z, half, ext[2], hi[2], lo[2], within, flat=False)
                        pts.append(((not fits, round(z, 2), float(np.hypot(gx - c[0], gy - c[1]))), [gx, gy, z]))
            return [q for _, q in sorted(pts)]
        for gx in np.linspace(lo[0] + half[0], hi[0] - half[0], n):
            for gy in np.linspace(lo[1] + half[1], hi[1] - half[1], n):
                zs = self._surfaces(f, gx, gy, hi[2], lo[2], within)
                for k, z in enumerate(zs):
                    ceiling = zs[k + 1] if k + 1 < len(zs) else hi[2] + 1.0
                    if z > hi[2] - 0.03 or ceiling - z < ext[2] + 0.02:
                        continue  # the top itself is "on"; a shelf needs headroom
                    if self._free(gx, gy, z, half, ext[2], others) and \
                            self._footing(f, gx, gy, z, half, ext[2], hi[2], lo[2], within, flat=False):
                        pts.append((z + 0.3 * float(np.hypot(gx - c[0], gy - c[1])), [gx, gy, z]))
        return [q for _, q in sorted(pts)]

    def open_top(self, f: str) -> bool:
        """A bowl, box or basket: a ray down its middle meets its floor first, under its rim."""
        lo, hi = self.box(f)
        c = (lo + hi) / 2
        zs = self._surfaces(f, c[0], c[1], hi[2], lo[2])
        return len(zs) == 1 and zs[0] < lo[2] + 0.5 * (hi[2] - lo[2])

    def pick(self, o: str) -> None:
        """Into the person's hand (it follows the hand and passes through things)."""
        q = self.obj(o).get_position_orientation()[1]
        self.rest_orn[o] = np.asarray(q.cpu() if hasattr(q, "cpu") else q, float)
        self.held["person"] = o
        self.pinned.pop(o, None)

    def put(self, o: str, rel: str, f: str, near=None) -> bool:
        """From the person's hand into the first spot where o stays put (its flatter resting orientation, else turned
        90 degrees), nearest `near` if given; kept in hand when none will do."""
        self.held["person"] = None
        self._ghosts()
        at = self.pos(o)
        best = None
        for q in (self.rest_orn.get(o), np.array([0.0, 0.0, 0.0, 1.0])):  # flatter of how it rested and its model's own
            if q is not None:
                self.obj(o).set_position_orientation(position=at + [0, 0, 0.3], orientation=q)
                lo, hi = self.box(o)
                best = (hi[2] - lo[2], q) if best is None or hi[2] - lo[2] < best[0] else best
        for orn in (best[1], yaw90(best[1])):
            self.obj(o).set_position_orientation(position=at + [0, 0, 0.3], orientation=orn)
            spots = self.spots(o, rel, f)
            if near is not None:
                spots = sorted(spots, key=lambda p: float(np.linalg.norm(np.asarray(p[:2]) - np.asarray(near, float)[:2])))
            for p in spots[:8]:
                if self._set(o, rel, f, np.asarray(p, float), orn):
                    return True
        self.held["person"] = o
        return False

    def _set(self, o: str, rel: str, f: str, p: np.ndarray, orn) -> bool:
        """o's box bottom 1 cm over p; held in place in an open container, else it must stay put (3 cm, 10 steps)."""
        self.obj(o).set_position_orientation(position=p + [0, 0, 0.3], orientation=orn)
        lo, hi = self.box(o)
        shift = np.array([p[0] - (lo[0] + hi[0]) / 2, p[1] - (lo[1] + hi[1]) / 2, p[2] + 0.01 - lo[2]])
        self.obj(o).set_position_orientation(position=self.pos(o) + shift, orientation=orn)
        self.obj(o).keep_still()
        if rel == "inside" and self.open_top(f):
            if not self.relation(o, rel, f):
                return False
            self.pinned[o] = (self.pos(o), orn)
            return True
        for _ in range(4):
            self.og.sim.step()
        p0 = self.pos(o)
        for _ in range(10):
            self.og.sim.step()
        return self.relation(o, rel, f) and float(np.linalg.norm(self.pos(o) - p0)) < 0.03

    def settle(self, o: str, rel: str, f: str) -> bool:
        """After the robot lets o go: in an open container it is held where it landed; whether the relation holds."""
        if rel == "inside" and self.open_top(f) and self.relation(o, rel, f):
            q = self.obj(o).get_position_orientation()[1]
            self.pinned[o] = (self.pos(o), np.asarray(q.cpu() if hasattr(q, "cpu") else q, float))
        return self.relation(o, rel, f)

    def _pin(self) -> None:
        for o, (p, q) in self.pinned.items():
            if o not in self.held.values():
                self.obj(o).set_position_orientation(position=p, orientation=q)
                self.obj(o).keep_still()

    def _ghosts(self) -> None:
        """What the person holds passes through things until set down; what the robot holds, while `robot_ghost`."""
        now = ({self.held["person"]} | ({self.held["robot"]} if self.robot_ghost else set())) - {None}
        for o in sorted(now ^ self.ghosts):
            self.obj(o).visual_only = o in now
        self.ghosts = now

    def height(self, x: float, y: float, skip=(), top: float = 1.25) -> tuple[float, str]:
        """How high the highest thing under `top` at (x, y) is, and what it is ((0, "floor") for bare floor), but `skip`
        and the robot."""
        hits = [h for h in self._rays([x, y, top], [x, y, -0.05]) if self.name(h["rigidBody"]) not in (*skip, "robot")]
        return (top - float(hits[0]["distance"]), self.name(hits[0]["rigidBody"])) if hits else (0.0, "floor")

    def support(self, point, skip=()) -> str | None:
        """The furniture right under a point: the first object a ray down from 0.3 m above it meets, skipping `skip`."""
        p = np.asarray(point, float)
        for h in self._rays([p[0], p[1], p[2] + 0.3], [p[0], p[1], p[2] - 1.5]):
            n = self.name(h["rigidBody"])
            if n not in skip and n != "robot" and self.obj(n) is not None:
                return n
        return None

    # -- the floor ----------------------------------------------------------------------------------------------------
    def path(self, a, b, avoid=(), hard: float | None = None) -> np.ndarray | None:
        return self.floor.plan(np.asarray(a, float)[:2], np.asarray(b, float)[:2], avoid, hard)

    def stand_spot(self, name: str, here, gap: float = 0.45, wall: float = 0.35, away=None, point=None) -> np.ndarray | None:
        """Where to stand to work at something: floor a path from `here` reaches, with wall m of room around it, 0.8 m
        from `away`. Beside `point` (a thing on the furniture): the nearest such spot to it, within 1.2 m. Else gap to
        gap + 0.6 m off the edge of name's box, nearest here."""
        home = self.floor.region(here)
        ok = lambda q: (self.floor.clearance(q) >= wall and self.floor.region(q) == home
                        and (away is None or float(np.linalg.norm(q - np.asarray(away, float)[:2])) >= 0.8))
        rays = [np.array([np.cos(a), np.sin(a)]) for a in np.linspace(0, 2 * np.pi, 48, endpoint=False)]
        here = np.asarray(here, float)[:2]
        reached = lambda qs: next((q for q in qs[:6] if self.path(here, q) is not None), None)  # a region can hide a gap
        if point is not None:
            p = np.asarray(point, float)[:2]
            for r in np.arange(0.35, 1.25, 0.1):
                q = reached(sorted((p + r * d for d in rays if ok(p + r * d)), key=lambda q: float(np.linalg.norm(q - here))))
                if q is not None:
                    return q
            return None
        lo, hi = self.box(name)
        c, cands = (lo[:2] + hi[:2]) / 2, []
        for r in np.linspace(gap, gap + 0.6, 4):
            for d in rays:
                q = c + d * float(np.abs(d) @ ((hi - lo)[:2] / 2)) + r * d
                if ok(q):
                    cands.append((float(np.linalg.norm(q - here)) + 0.5 * r, q))
        return reached([q for _, q in sorted(cands, key=lambda x: x[0])])

    # -- sight ---------------------------------------------------------------------------------------------------------
    @staticmethod
    def name(v) -> str:
        """An object name from a USD path ("robot" for any Stretch link)."""
        parts = [p for p in str(v).split("/") if p and p != "World" and not p.startswith("scene_")]
        n = parts[0].removeprefix("articulated__") if parts else str(v)
        return "robot" if n.startswith("controllable__") else n

    def hit(self, a, b) -> str | None:
        """The first thing a ray from a to b meets, else None."""
        hits = self._rays(a, b)
        return self.name(hits[0]["rigidBody"]) if hits else None

    def basis(self) -> tuple:
        z = float(self.eye[2]) if self.eye[2] > 1.0 else EYE
        eye = np.array([*(self.person_xy + 0.14 * np.array([np.cos(self.person_yaw), np.sin(self.person_yaw)])), z])
        f = np.asarray(self.head, float) - eye
        f /= np.linalg.norm(f) + 1e-9
        r = np.cross(f, [0, 0, 1.0])
        r /= np.linalg.norm(r) + 1e-9
        return eye, f, r, np.cross(r, f)

    def visible(self, names: list[str], strict: tuple = ()) -> dict:
        """{name: (cx, cy, area)} for what the person sees: rays from the eye to sample points of each object's box (its
        middle, the middle of its top, and points on the faces toward the eye) that reach it first; (cx, cy) in [0, 1]
        image coordinates, area the box's projected area times the share reached. A strict object needs two rays."""
        eye, f, r, u = basis = self.basis()
        k = 2 * np.tan(np.arctan(20.995 / (2 * 17.0)))  # OmniGibson's default camera: aperture 20.995, focal 17
        proj = lambda p: (None if (d := np.asarray(p, float) - eye) @ f < 0.05
                          else (0.5 + (d @ r) / (d @ f) / k, 0.5 - (d @ u) / (d @ f) / k))
        out = {}
        for o in names:
            lo, hi = self.box(o) if o != "robot" else self.robot_box()
            c = (lo + hi) / 2
            if np.linalg.norm(c - eye) > 8.0:
                continue
            corners = [q for q in (proj([x, y, z]) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])) if q]
            if not corners:
                continue
            us, vs = np.clip([q[0] for q in corners], 0, 1), np.clip([q[1] for q in corners], 0, 1)
            box = (us.max() - us.min()) * (vs.max() - vs.min())
            if box <= 0:
                continue
            half, toward = (hi - lo) / 2, np.sign(eye - c)
            pts = [c, [c[0], c[1], hi[2] - 0.01]] + [p for z in (0.3, 0.7) for p in (
                [c[0] + toward[0] * half[0] * 0.95, c[1], lo[2] + z * (hi[2] - lo[2])],
                [c[0], c[1] + toward[1] * half[1] * 0.95, lo[2] + z * (hi[2] - lo[2])])]
            ghost = o in self.ghosts  # no collision shape: seen when nothing is in between
            seen = [q for p in pts if (q := proj(p)) and 0 <= q[0] <= 1 and 0 <= q[1] <= 1
                    and ((self.hit(eye, eye + (np.asarray(p) - eye) * 0.97) is None) if ghost
                         else self.hit(eye, eye + (np.asarray(p) - eye) * 1.02) == o)]
            if len(seen) >= (2 if o in strict else 1):
                out[o] = (float(np.mean([q[0] for q in seen])), float(np.mean([q[1] for q in seen])), float(box * len(seen) / len(pts)))
        return out

    def robot_box(self) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = (np.asarray(v, float) for v in self.bot.rb.aabb)
        return lo, hi

    # -- frames --------------------------------------------------------------------------------------------------------
    def step(self, robot_cmd: dict | None = None) -> None:
        """One frame: pose the person (walking or standing, reaching, looking at `head`), their eye camera and what they
        hold; objects pinned in containers; then the robot's command over two physics steps."""
        xy, yaw = self.person_xy, self.person_yaw
        moving = self.last_xy is not None and float(np.linalg.norm(xy - self.last_xy)) > 1e-3
        self.human.update(xy, yaw, moving, 1 / 15, reach=self.reach, look=self.head)
        self.eye, self.last_xy = self.human.eye(), xy.copy()
        self.cams["person_cam"].set_position_orientation(position=self.eye, orientation=look_quat(self.eye, self.head))
        self._ghosts()
        self.bot.frame(robot_cmd or {}, each=self._hold)

    def _hold(self) -> None:
        """Before every physics step: the person's object in their hand, pinned objects in place."""
        o = self.held["person"]
        if o is not None and self.human.sk is not None:
            p = self.human.hand() + 0.08 * np.array([np.cos(self.person_yaw), np.sin(self.person_yaw), 0.0])
            self.obj(o).set_position_orientation(position=p)
        self._pin()

    def rgb(self, cam: str = "person_cam") -> np.ndarray:
        return self.cams[cam].get_obs()[0]["rgb"][..., :3].cpu().numpy().astype(np.uint8)

    def save(self):
        return self.og.sim.dump_state(serialized=False)

    def restore(self, state) -> None:
        """Back to a saved state with nothing carried over: no hold, pin or ghost; the person's stride and view reset.
        No physics step here: one from a restored state depends on what ran before (PhysX caches)."""
        self.og.sim.load_state(state, serialized=False)
        for o in self.ghosts:
            self.obj(o).visual_only = False
        self.reset_state()
