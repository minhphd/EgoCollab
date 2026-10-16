"""Navigate, pick and place for a Stretch, as pure controllers: `skill.step(state, dt) -> (command, status)`. They
read only the robot's state, so the same code drives the simulated robot (helper/collab/env.py) and the real one.

    state   {"xy": (x, y), "yaw", "lift", "arm", "pitch", "wrist_yaw", "holding": bool, "person": (x, y) or None}
            map frame, m, rad
    command {"v", "w", "lift", "arm", "pitch", "wrist_yaw", "grip"}   base speeds (m/s, rad/s); joint goals (m, rad);
            grip = the gripper's aperture, 1 open, 0.5 ready (shaped over an object), 0 closed
    status  "running", "done" or "failed: <why>"

The driver below the skills moves each joint to its goal at the joint's speed limit, as Stretch's own driver does.
The robot drives with the wrist down (the gripper over its base); a reach with the wrist level unfolds it at full lift
and folds it back there, so the swinging gripper clears the furniture it works at. What it holds level it carries level
with the gripper turned forward (wrist yaw), so the object stays upright and clear of the robot.
The arm reaches out of the base's side along a line offset from the base centre; `Geometry` says where (measure it
once on each robot): gripper(xy, yaw, ext) = xy + R(yaw) (p0 + ext u(d)), height z0 + lift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

V, W = 0.3, 0.8  # m/s, rad/s: Stretch's base limits for navigation
LOOK = 0.3  # m, pure-pursuit look-ahead
CARRY = 0.95  # m of lift while driving: the arm (0.16 to 0.23 m over the lift) clears desks, tables and counters
ARM_MAX, LIFT_MAX = 0.52, 1.1
STOW = -np.pi / 2  # rad, wrist pitch while driving: down
FOLD = np.pi / 2  # rad of wrist yaw that turns a level gripper forward, for carrying what it holds level


@dataclass
class Geometry:
    p0: tuple  # (x, y) of the gripper centre in the base frame, arm in
    d: float  # rad, the arm's direction in the base frame
    z0: float  # m, gripper centre height at lift 0
    pitch: float = -np.pi / 2  # wrist pitch these were measured at: the gripper points down, for objects on surfaces


def wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def unit(a: float) -> np.ndarray:
    return np.array([np.cos(a), np.sin(a)])


def rot(yaw: float, v) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def aim(xy, target, geo: Geometry) -> tuple[float, float] | None:
    """Heading and arm extension that put the gripper over target's (x, y) from base position xy, or None if the
    target is inside the arm's reach circle: the target lies on the line p0 + t u(d) in the base frame."""
    rel = np.asarray(target, float)[:2] - np.asarray(xy, float)[:2]
    p0, u = np.asarray(geo.p0, float), unit(geo.d)
    h2 = float(p0 @ p0 - (p0 @ u) ** 2)  # squared distance from the base centre to the arm's line
    r2 = float(rel @ rel)
    if r2 < h2:
        return None
    t = -float(p0 @ u) + np.sqrt(r2 - h2)  # how far along the line the target is
    q = p0 + t * u
    return wrap(float(np.arctan2(rel[1], rel[0]) - np.arctan2(q[1], q[0]))), t


def approach(target, geo: Geometry, free, near, reaches=(0.25, 0.35, 0.45), axis: float | None = None,
             ok=None) -> tuple[np.ndarray, float] | None:
    """Base (x, y) and heading that put the gripper, arm `reach` m out, over target's (x, y): one candidate per 10
    degrees of the arm's world direction that `free(xy)` and `ok(xy, heading)` accept (within 20 degrees of `axis`,
    either way, if given: the fingers close across the arm's line), the shortest reach first, then nearest to `near`.
    Reaches stop 7 cm short of the arm's 0.52 m, for where the base actually stops."""
    t = np.asarray(target, float)[:2]
    for reach in reaches:
        out = []
        for a in np.radians(np.arange(0, 360, 10)):
            if axis is not None and abs(np.sin(a - axis)) > np.sin(np.radians(20)):
                continue
            yaw = wrap(a - geo.d)
            b = t - rot(yaw, np.asarray(geo.p0, float) + reach * unit(geo.d))
            if free(b) and (ok is None or ok(b, yaw)):
                out.append((float(np.linalg.norm(b - np.asarray(near, float)[:2])), b, yaw))
        if out:
            return min(out, key=lambda o: o[0])[1:]
    return None


def turn(yaw: float, goal: float) -> dict:
    return {"v": 0.0, "w": float(np.clip(2.0 * wrap(goal - yaw), -W, W))}


class Navigate:
    """Follow `path` (map-frame points) and turn to `yaw`: steer toward the point LOOK m further along the path than
    the robot's projection on it (pure pursuit), turning in place when that point is more than 0.5 rad off and slowing
    for the last 0.3 m. Holds still while the person is within 0.5 m ahead (4 s at most); fails after 8 s in which the
    path left does not shrink."""

    def __init__(self, path, yaw: float | None = None):
        self.path, self.yaw = [np.asarray(p, float)[:2] for p in path], yaw
        self.mark, self.still, self.waited = np.inf, 0.0, 0.0  # path left at the last progress

    def _project(self, xy: np.ndarray) -> None:
        """Drop the path behind xy's projection on it."""
        P, best = self.path, (np.inf, 0, self.path[0])
        for i in range(len(P) - 1):
            a, b = P[i], P[i + 1]
            q = a + float(np.clip((xy - a) @ (b - a) / max(float((b - a) @ (b - a)), 1e-9), 0, 1)) * (b - a)
            if np.linalg.norm(xy - q) < best[0] - 1e-6:
                best = (float(np.linalg.norm(xy - q)), i, q)
        self.path = [best[2]] + P[best[1] + 1:]

    def _along(self, d: float) -> np.ndarray:
        for a, b in zip(self.path, self.path[1:]):
            seg = float(np.linalg.norm(b - a))
            if seg >= d:
                return a + (b - a) * d / max(seg, 1e-9)
            d -= seg
        return self.path[-1]

    def step(self, s: dict, dt: float) -> tuple[dict, str]:
        xy, yaw = np.asarray(s["xy"], float), s["yaw"]
        if np.linalg.norm(self.path[-1] - xy) > 0.05:
            self._project(xy)
            left = float(np.linalg.norm(self.path[0] - xy)) + sum(
                float(np.linalg.norm(b - a)) for a, b in zip(self.path, self.path[1:]))
            if left < self.mark - 0.05:
                self.mark, self.still = left, 0.0
            else:
                self.still += dt
            if self.still > 8.0:
                return {"v": 0.0, "w": 0.0}, "failed: no progress"
            to = self._along(LOOK) - xy
            e = wrap(float(np.arctan2(to[1], to[0])) - yaw)
            if abs(e) > 0.5:
                return turn(yaw, yaw + e), "running"
            p = s.get("person")
            if p is not None and np.linalg.norm(np.asarray(p) - xy) < 0.5 and (np.asarray(p) - xy) @ unit(yaw) > 0:
                self.waited += dt
                if self.waited < 4.0:  # the person has the right of way
                    return {"v": 0.0, "w": 0.0}, "running"
            v = min(V * float(np.cos(e)), 0.8 * left + 0.05)
            return {"v": v, "w": float(np.clip(2.0 * v * np.sin(e) / LOOK, -W, W))}, "running"
        if self.yaw is not None and abs(wrap(self.yaw - yaw)) > 0.03:
            return turn(yaw, self.yaw), "running"
        return {"v": 0.0, "w": 0.0}, "done"


class Joints:
    """Joint goals in order, each held until reached (lift 1 cm, arm 2 cm or 8 cm drawn in, wrist 0.05 rad, gripper
    aperture 0.02; closing, until it holds something or is shut), 0.2 s at least for a gripper goal."""

    def __init__(self, goals: list[dict]):
        self.goals, self.k, self.t, self.cmd = goals, 0, 0.0, {"v": 0.0, "w": 0.0}

    def step(self, s: dict, dt: float) -> tuple[dict, str]:
        if self.k == len(self.goals):
            return dict(self.cmd), "done"
        g = self.goals[self.k]
        self.cmd.update(g)
        self.t += dt
        shut = g.get("grip", 1.0) < 0.25
        if all(abs(s[j] - g[j]) < (0.08 if j == "arm" and g[j] == 0.0 else tol)  # drawn in: 8 cm out will do
               for j, tol in (("lift", 0.01), ("arm", 0.02), ("pitch", 0.05), ("wrist_yaw", 0.05)) if j in g) and \
                ("grip" not in g or self.t > 0.2 and ((s["holding"] or s["grip"] < 0.02) if shut else abs(s["grip"] - g["grip"]) < 0.02)):
            self.k, self.t = self.k + 1, 0.0
        elif self.t > 15.0:
            at = {j: round(float(s[j]), 3) for j in g if j in ("lift", "arm", "pitch", "wrist_yaw")}
            return dict(self.cmd), f"failed: {g} not reached (at {at})"
        return dict(self.cmd), "running"


class _Reach:
    """Turn in place until the arm's line passes through the target, then run the joint goals built from the extension
    and the lifts that put the fingertips at the target and at `up`, the height they travel at (over what is in the
    way, at least 2 cm over the target)."""

    def __init__(self, xyz, geo: Geometry, up: float):
        self.xyz, self.geo, self.up, self.seq = np.asarray(xyz, float), geo, float(up), None

    def goals(self, ext: float, lift: float, up: float) -> list[dict]:
        raise NotImplementedError

    def step(self, s: dict, dt: float) -> tuple[dict, str]:
        if self.seq is None:
            a = aim(s["xy"], self.xyz, self.geo)
            if a is None:
                return {"v": 0.0, "w": 0.0}, "failed: too close to aim"
            yaw, ext = a
            if abs(wrap(yaw - s["yaw"])) > 0.01:
                return turn(s["yaw"], yaw), "running"
            lift, up = float(self.xyz[2] - self.geo.z0), float(max(self.up, self.xyz[2] + 0.02) - self.geo.z0)
            if not (0.0 <= ext <= ARM_MAX and 0.0 <= lift and up <= LIFT_MAX):
                return {"v": 0.0, "w": 0.0}, f"failed: out of reach (arm {ext:.2f} m, lift {lift:.2f} m, over at {up:.2f} m)"
            self.lifts = (lift, up)
            self.seq = Joints(self.unfold() + self.goals(ext, lift, up) + self.fold())
        return self.seq.step(s, dt)

    def unfold(self) -> list[dict]:
        """From how the robot drives to this reach's wrist pose: down stays down; level unfolds at full lift (a held
        object turned back from forward)."""
        if abs(self.geo.pitch - STOW) <= 0.1:
            return [{"pitch": STOW, "wrist_yaw": 0.0}]
        return [{"lift": LIFT_MAX, "arm": 0.0}, {"pitch": self.geo.pitch, "wrist_yaw": 0.0}]

    def fold(self) -> list[dict]:
        """Back to how the robot drives: holding something level, the gripper turned forward at full lift (it stays
        level, so the object stays upright and as high as it came off); empty, stowed."""
        if isinstance(self, Pick):
            return [{"lift": LIFT_MAX}, {"wrist_yaw": FOLD}] if abs(self.geo.pitch - STOW) > 0.1 else []
        return self.stow()

    def stow(self) -> list[dict]:
        """Empty-handed to how the robot drives: the wrist down at full lift (if level), then carrying height."""
        return ([{"lift": LIFT_MAX}, {"pitch": STOW}] if abs(self.geo.pitch - STOW) > 0.1 else []) + [{"lift": CARRY}]


class Pick(_Reach):
    """Open, up to `up` (default 10 cm over the target), arm out with the gripper shaped (ready), down until the
    fingertips are at xyz, close, back up, arm in. Fails when nothing is held after closing, once open, back up, in
    and stowed."""

    def __init__(self, xyz, geo: Geometry, up: float | None = None):
        super().__init__(xyz, geo, float(xyz[2]) + 0.10 if up is None else up)
        self.missed = False

    def goals(self, ext: float, lift: float, up: float) -> list[dict]:
        return [{"lift": up, "grip": 1.0}, {"arm": ext, "grip": 0.5}, {"lift": lift}, {"grip": 0.0}, {"lift": up}, {"arm": 0.0}]

    def step(self, s: dict, dt: float) -> tuple[dict, str]:
        cmd, status = super().step(s, dt)
        if self.seq is not None and not self.missed and self.seq.k >= len(self.unfold()) + 4 and not s["holding"]:
            self.missed = True  # nothing held: open, back up, arm in and stowed, then say so (never drive reached out)
            self.seq = Joints([{"lift": self.lifts[1], "grip": 1.0}, {"arm": 0.0}] + self.stow())
            return dict(cmd, grip=1.0), "running"
        return cmd, "failed: nothing in the gripper" if self.missed and status == "done" else status


class Place(_Reach):
    """Up to `up` (the fingertips' height that carries the held object over what is in the way; default 10 cm over
    the target), arm out, down until the object's bottom is `drop` m over the surface, open, back up, arm in, carrying
    height. spot = (x, y, surface z); `below` = how far the object's bottom hangs under the fingertips."""

    def __init__(self, spot, below: float, geo: Geometry, drop: float = 0.02, up: float | None = None):
        x, y, z = (float(c) for c in spot)
        super().__init__((x, y, z + below + drop), geo, z + below + drop + 0.10 if up is None else up)

    def goals(self, ext: float, lift: float, up: float) -> list[dict]:
        return [{"lift": up}, {"arm": ext}, {"lift": lift}, {"grip": 1.0}, {"lift": up}, {"arm": 0.0}]
