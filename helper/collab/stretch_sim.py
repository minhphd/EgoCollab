"""The simulated Stretch under the skills (helper/stretch/skills.py): what Stretch's own driver does on the real robot.

The base follows the commanded (v, w) exactly (a unicycle, pinned to its commanded pose so contact cannot shove it:
OmniGibson's Stretch drags its fixed caster). The lift, telescoping arm, wrist and fingers are physical joints whose
targets move to the commanded goals at the real robot's speeds; both fingers are driven to the same angle (0.6 rad
open, 0 closed, the tips meeting at eef_link: OmniGibson's binary gripper swings one finger across), and what a
finger touches as they close sticks at once (sticky grasp).

    CONFIG                                  the robot's entry in an og.Environment config
    bot = StretchSim(robot); bot.place(xy, yaw)
    bot.frame(cmd)                          one 1/15 s frame: two 1/30 s physics steps
    bot.state(person=xy), bot.holding(obj), bot.geometry(pitch)
"""

from __future__ import annotations

import numpy as np

from helper.stretch.skills import CARRY, STOW, Geometry, rot, unit, wrap

RAW = {"command_input_limits": None, "command_output_limits": None}  # commands in metres and radians
CONFIG = {"type": "Stretch", "name": "robot", "obs_modalities": [], "position": [0.0, 0.0, 0.0],
          "grasping_mode": "sticky", "action_normalize": False,
          "controller_config": {"base": {"name": "DifferentialDriveController", **RAW},
                                "camera": {"name": "JointController", "use_delta_commands": False, **RAW},
                                "arm_0": {"name": "JointController", "use_delta_commands": False, "motor_type": "position", **RAW},
                                "gripper_0": {"name": "JointController", "use_delta_commands": False, "motor_type": "position", **RAW}}}
SPEED = {"lift": 0.15, "arm": 0.4, "pitch": 3.0, "wrist_yaw": 3.0, "grip": 1.0}  # m/s, m/s, rad/s, rad/s, finger rad/s:
# Stretch 3's fastest (Hello Robot's stretch_body 0.7.31, robot_params_SE3.py "max"); the base's is 0.3 m/s
CLOSE = 0.4  # rad/s, the fingers closing: gently, so the first to touch does not flick a light round thing away
FINGER_OPEN = 0.6  # rad, each finger fully open (the sticky grasp lets go only there); grip = aperture * this
GRASP_WINDOW = 1 / 60  # s of touch before the sticky grasp holds; OmniGibson's 0.3 s lets a light thing bounce off
                       # the closing finger, each bounce starting the count again, and be pushed away
TOP = Geometry(p0=(-0.020, -0.121), d=-np.pi / 2, z0=-0.124)  # wrist down; StretchSim.geometry() on open floor
LEVEL = Geometry(p0=(-0.019, -0.402), d=-np.pi / 2, z0=0.109, pitch=0.0)  # wrist level: fingers out along the arm
ARM = ["joint_arm_l3", "joint_arm_l2", "joint_arm_l1", "joint_arm_l0"]
STEPS = 2  # physics steps per 1/15 s frame (OmniGibson steps 1/30 s)


class StretchSim:
    def __init__(self, robot):
        import omnigibson as og
        from omnigibson.robots import robot as og_robot
        og_robot.m.GRASP_WINDOW = GRASP_WINDOW
        self.og, self.rb = og, robot
        names = list(robot.joints.keys())
        self.j = {n: names.index(n) for n in names}
        self.xy, self.yaw, self.z = np.zeros(2), 0.0, 0.0
        self.picking = None  # the object a pick is for: the sticky grasp takes nothing else (a neighbour, one pinned)
        touching = robot._calculate_in_hand_object
        robot._calculate_in_hand_object = lambda arm="default": (
            lambda c: c if c is not None and c[0].name == self.picking else None)(touching(arm=arm))
        self.stow()

    def stow(self) -> None:
        """Joint goals and targets back to how the robot drives: lift at carrying height, arm in, wrist down, open. A
        restored state brings the joints back, not these."""
        self.goal = {"v": 0.0, "w": 0.0, "lift": CARRY, "arm": 0.0, "pitch": STOW, "wrist_yaw": 0.0, "grip": 1.0}
        self.target = {"lift": CARRY, "arm": 0.0, "pitch": STOW, "wrist_yaw": 0.0, "grip": FINGER_OPEN}  # toward the goals
        self.held = None  # finger angles held since the grasp took

    def place(self, xy, yaw: float) -> None:
        from helper.behavior.sim import yaw_quat
        self.xy, self.yaw, self.z = np.asarray(xy, float)[:2].copy(), float(yaw), 0.0  # the base on the floor
        self.rb.set_position_orientation(position=[*self.xy, self.z], orientation=yaw_quat(self.yaw))
        self.rb.keep_still()

    def joints(self) -> dict:
        q = np.asarray(self.rb.get_joint_positions(), float)
        return {"lift": float(q[self.j["joint_lift"]]), "arm": float(sum(q[self.j[n]] for n in ARM)),
                "pitch": float(q[self.j["joint_wrist_pitch"]]), "wrist_yaw": float(q[self.j["joint_wrist_yaw"]])}

    def holding(self, obj) -> bool:
        from omnigibson.controllers.controller_base import IsGraspingState
        return obj is not None and self.rb.is_grasping(candidate_obj=obj) == IsGraspingState.TRUE

    def state(self, person=None, obj=None) -> dict:
        j = self.joints()
        return {"xy": self.xy.copy(), "yaw": self.yaw, "lift": j["lift"], "arm": j["arm"], "pitch": j["pitch"],
                "wrist_yaw": j["wrist_yaw"], "grip": self.target["grip"] / FINGER_OPEN, "holding": self.holding(obj),
                "person": person}

    def grasp_point(self) -> np.ndarray:
        return np.asarray(self.rb.links["eef_link"].get_position_orientation()[0], float)

    def frame(self, cmd: dict, each=None) -> None:
        """Apply a skill command for one 1/15 s frame; `each()` runs before every physics step."""
        import torch as th
        from helper.behavior.sim import yaw_quat
        self.goal.update(cmd)
        dt = 1 / 15 / STEPS
        for _ in range(STEPS):
            for k in self.target:
                goal = FINGER_OPEN * float(self.goal[k]) if k == "grip" else self.goal[k]
                down = CLOSE if k == "grip" else SPEED[k]
                self.target[k] += float(np.clip(goal - self.target[k], -down * dt, SPEED[k] * dt))
            self.yaw = wrap(self.yaw + self.goal["w"] * dt)
            self.xy = self.xy + self.goal["v"] * dt * unit(self.yaw)
            self.rb.set_position_orientation(position=[*self.xy, self.z], orientation=yaw_quat(self.yaw))
            t = self.target
            fingers = self._held_fingers()  # checked every frame, so a let-go clears it
            fingers = fingers if self.goal["grip"] < 1.0 else None
            if fingers is not None:  # holding: each finger stays where the grasp took
                t["grip"] = max(fingers)
            a = [0.0, 0.0, 0.0, -0.3, t["lift"], *[t["arm"] / 4] * 4, t["wrist_yaw"], t["pitch"], 0.0, *(fingers or [t["grip"]] * 2)]
            self.rb.apply_action(th.tensor(a, dtype=th.float32))
            if each is not None:
                each()
            self.og.sim.step()

    def _held_fingers(self) -> list[float] | None:
        """[right, left] finger targets (the action's order) while the sticky grasp holds something: each finger where it
        was when the grasp took; None otherwise. Squeezing on fights the grasp joint and jams the arm and wrist."""
        from omnigibson.controllers.controller_base import IsGraspingState
        if self.rb.is_grasping() != IsGraspingState.TRUE:
            self.held = None
        elif self.held is None:
            q = np.asarray(self.rb.get_joint_positions(), float)
            self.held = [float(q[self.j[f"joint_gripper_finger_{side}"]]) for side in ("right", "left")]
        return self.held

    def geometry(self, pitch: float = -np.pi / 2) -> Geometry:
        """Measure where the gripper centre is: arm in, then 0.4 m out, at lift 0.4 m with the wrist at `pitch`."""
        for ext in (0.0, 0.4):
            self.goal.update({"lift": 0.4, "arm": ext, "pitch": pitch})
            for _ in range(int(15 * 6)):
                self.frame({})
            if ext == 0.0:
                e0, lift0 = self.grasp_point(), self.joints()["lift"]
        e1 = self.grasp_point()
        p0 = rot(-self.yaw, e0[:2] - self.xy)
        d = wrap(float(np.arctan2(*rot(-self.yaw, e1[:2] - e0[:2])[::-1])))
        self.goal.update({"arm": 0.0, "lift": CARRY})
        return Geometry(p0=(float(p0[0]), float(p0[1])), d=d, z0=float(e0[2] - lift0), pitch=pitch)
