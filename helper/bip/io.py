"""What BIP reads from the robot and the partner's glasses, and what it sends back. Every source, live or replayed,
speaks these types; a robot only has to fill them.

Inputs, each stamped with t in seconds on one clock:
    SceneObs   the robot's scene graph detections: (object, "on" | "in", furniture), (object, "next_to", object),
               (agent, "holding", object), with agents "H" (the person) and "R" (the robot)
    EgoFrame   one frame of the person's glasses: entities in view with image boxes, the gaze point, and the image
    Act        a recognized action of the person ("H") or the robot's own ("R"): Pick, Place, Open, Close
    Speech     what the person said
Output:
    Command    execute a subgoal, ask a question, tell the person where something is, or nothing new
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ACTIONS = ("Pick", "Place", "Open", "Close")


@dataclass
class SceneObs:
    t: float
    triples: list


@dataclass
class EgoFrame:
    t: float
    visible: dict  # entity -> (cx, cy, area), image coordinates in [0, 1]
    gaze: tuple | None = None  # (x, y) in the same image coordinates, None when the eyes are not tracked
    image: bytes | None = None  # JPEG with the gaze drawn, kept for the slow path
    fixated: str | None = None  # the entity the gaze falls on, if a source already knows it


@dataclass
class Act:
    t: float
    agent: str  # "H" or "R"
    verb: str  # one of ACTIONS
    args: tuple = ()  # Pick (o,), Place (o, "on" | "in", furniture, next_to or None), Open (f,), Close (f,)


@dataclass
class Speech:
    t: float
    text: str


@dataclass
class Command:
    t: float
    kind: str  # "execute", "ask", "tell"
    target: str | None = None  # the subgoal to execute, or the object told about
    text: str = ""
    info: dict = field(default_factory=dict)


def fixated(gaze: tuple | None, visible: dict, furniture: set, aoi: float = 0.04) -> str | None:
    """The entity the gaze falls on: the nearest object within aoi of the gaze point, else the largest entity whose
    box centre is within 0.15 (furniture), else None. The same rule labels gaze in simulation."""
    if gaze is None or not visible:
        return None
    d = {n: float(np.hypot(c[0] - gaze[0], c[1] - gaze[1])) for n, c in visible.items()}
    objs = [n for n in visible if n not in furniture and d[n] <= aoi]
    if objs:
        return min(objs, key=d.get)
    near = [n for n in visible if n in furniture and d[n] <= 0.15]
    return max(near, key=lambda n: visible[n][2]) if near else None
