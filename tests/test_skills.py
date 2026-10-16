"""helper/stretch/skills.py against a kinematic Stretch: a unicycle base and joints that move to their goals at the
real robot's speeds; closing the gripper within 3 cm of the object holds it.

    python -m pytest tests/test_skills.py -q
"""

import numpy as np

from helper.stretch.skills import FOLD, Geometry, Navigate, Pick, Place, approach, rot, unit, wrap

GEO = Geometry(p0=(-0.1, -0.25), d=-np.pi / 2, z0=0.2)  # out of the right side, 10 cm behind the centre
LEVEL = Geometry(p0=(-0.1, -0.5), d=-np.pi / 2, z0=0.4, pitch=0.0)  # the same arm with the wrist level
DT = 1 / 15


class Toy:
    def __init__(self, xy, yaw, obj=None):
        self.s = {"xy": np.array(xy, float), "yaw": float(yaw), "lift": 0.6, "arm": 0.0, "pitch": -np.pi / 2,
                  "wrist_yaw": 0.0, "grip": 1.0, "holding": False, "person": None}
        self.obj, self.grip = None if obj is None else np.array(obj, float), False

    def gripper(self) -> np.ndarray:
        s = self.s
        geo = LEVEL if abs(s["pitch"]) < 0.05 else GEO  # the wrist level or down
        return np.array([*(s["xy"] + rot(s["yaw"], np.asarray(geo.p0) + s["arm"] * unit(geo.d))), geo.z0 + s["lift"]])

    def apply(self, cmd: dict) -> None:
        s = self.s
        s["yaw"] = wrap(s["yaw"] + cmd.get("w", 0.0) * DT)
        s["xy"] = s["xy"] + cmd.get("v", 0.0) * DT * unit(s["yaw"])
        for j, speed in (("lift", 0.15), ("arm", 0.15), ("pitch", 1.0), ("wrist_yaw", 1.0)):
            if j in cmd:
                s[j] += float(np.clip(cmd[j] - s[j], -speed * DT, speed * DT))
        s["grip"] = float(cmd.get("grip", s["grip"]))  # the aperture: 0 closed, 1 open
        closed = s["grip"] < 0.25
        if closed and not self.grip and self.obj is not None and np.linalg.norm(self.gripper() - self.obj) < 0.03:
            s["holding"] = True
        if not closed:
            s["holding"] = False
        self.grip = closed
        if s["holding"]:
            self.obj = self.gripper()

    def run(self, skill, t_max: float = 120.0) -> str:
        status, t = "running", 0.0
        while status == "running" and t < t_max:
            cmd, status = skill.step(self.s, DT)
            self.apply(cmd)
            t += DT
        return status


def test_navigate_reaches_the_goal_and_heading():
    bot = Toy((0, 0), 0.0)
    assert bot.run(Navigate([(0, 0), (2, 0), (2, 2)], yaw=np.pi)) == "done"
    assert np.linalg.norm(bot.s["xy"] - (2, 2)) < 0.06 and abs(wrap(bot.s["yaw"] - np.pi)) < 0.06


def test_navigate_turns_in_place_before_driving():
    bot = Toy((0, 0), np.pi)  # facing away from the goal
    cmd, _ = Navigate([(0, 0), (1, 0)]).step(bot.s, DT)
    assert cmd["v"] == 0.0 and cmd["w"] != 0.0


def test_navigate_holds_for_the_person_then_goes_on():
    bot, nav = Toy((0, 0), 0.0), Navigate([(0, 0), (3, 0)])
    bot.s["person"] = (0.4, 0.0)  # 0.4 m ahead, standing still
    held = 0.0
    for _ in range(int(3.0 / DT)):
        cmd, _ = nav.step(bot.s, DT)
        held += DT * (cmd["v"] == 0.0)
        bot.apply(cmd)
    assert held > 2.9  # waits while the person is there (up to 4 s)
    bot.s["person"] = None
    assert bot.run(nav) == "done"


def test_pick_and_place():
    obj = (1.0, 0.5, 0.75)
    base, yaw = approach(obj, GEO, free=lambda xy: True, near=(0, 0), reaches=(0.25,))
    bot = Toy(base, yaw, obj=obj)
    assert bot.run(Pick(obj, GEO)) == "done" and bot.s["holding"]
    spot = (1.0, 0.6, 0.70)  # 10 cm to the side, on a surface at 0.70 m: the base turns to aim
    assert bot.run(Place(spot, below=0.0, geo=GEO, drop=0.0)) == "done" and not bot.s["holding"]
    assert np.linalg.norm(bot.obj[:2] - spot[:2]) < 0.03 and abs(bot.obj[2] - 0.70) < 0.03


def test_pick_out_of_reach_fails():
    bot = Toy((0, 0), 0.0, obj=(0, -2.0, 0.7))  # 2 m out of the arm side
    assert bot.run(Pick((0, -2.0, 0.7), GEO)).startswith("failed: out of reach")


def test_pick_with_nothing_there_fails():
    obj = (1.0, 0.5, 0.75)
    base, yaw = approach(obj, GEO, free=lambda xy: True, near=(0, 0), reaches=(0.25,))
    bot = Toy(base, yaw, obj=(5.0, 5.0, 0.75))  # the object is elsewhere
    assert bot.run(Pick(obj, GEO)) == "failed: nothing in the gripper"
    assert bot.s["arm"] < 0.08 and abs(bot.s["lift"] - 0.95) < 0.02  # drawn in and at carrying height, to drive


def test_place_carries_the_object_at_up_before_reaching_out():
    obj = (1.0, 0.5, 0.75)
    base, yaw = approach(obj, GEO, free=lambda xy: True, near=(0, 0), reaches=(0.25,))
    bot = Toy(base, yaw, obj=obj)
    assert bot.run(Pick(obj, GEO)) == "done"
    up = 0.40 + 0.05 + 0.03  # a box's rim at 0.40 m, the object's bottom 5 cm under the fingertips, 3 cm to spare
    place, lift_at_reach = Place((1.0, 0.5, 0.30), below=0.05, geo=GEO, up=up), None
    status = "running"
    while status == "running":
        cmd, status = place.step(bot.s, DT)
        bot.apply(cmd)
        if lift_at_reach is None and bot.s["arm"] > 0.01:
            lift_at_reach = bot.s["lift"]
    assert status == "done" and lift_at_reach >= up - GEO.z0 - 0.02


def test_pick_fails_when_the_way_over_is_too_high():
    obj = (1.0, 0.5, 0.75)
    base, yaw = approach(obj, GEO, free=lambda xy: True, near=(0, 0), reaches=(0.25,))
    bot = Toy(base, yaw, obj=obj)
    assert bot.run(Pick(obj, GEO, up=1.5)).startswith("failed: out of reach")  # 1.5 m: a lift of 1.3 m is needed


def test_a_level_reach_unfolds_and_folds_the_wrist_at_full_lift():
    obj = (1.0, 0.5, 0.95)
    base, yaw = approach(obj, LEVEL, free=lambda xy: True, near=(0, 0), reaches=(0.25,))
    bot, lifts = Toy(base, yaw, obj=obj), set()
    pick, status = Pick(obj, LEVEL), "running"
    while status == "running":
        cmd, status = pick.step(bot.s, DT)
        if abs(bot.s["pitch"] - cmd.get("pitch", bot.s["pitch"])) > 0.01:  # the wrist is swinging
            lifts.add(round(bot.s["lift"], 2))
        bot.apply(cmd)
    assert status == "done" and bot.s["holding"] and lifts == {1.1}
    assert abs(bot.s["pitch"]) < 0.06 and abs(bot.s["wrist_yaw"] - FOLD) < 0.06  # carried level, turned forward

