"""Simulated perception for the robot's scene graph (helper/bip/scene.py), with the noise of helper.bip.config.

view: what one camera (the person's head camera or the robot's head camera) detects: each visible object with
the furniture it rests on or in, pairs of visible objects on the same furniture within next_to_m, and the
object held in front of the camera. Relations come from the simulator's placement of the objects in view, as a
3D scene-graph detector would report them; an object out of view is never reported. Misses (view_miss) and
wrong furniture (view_rel_err) are added.
recognize: the person's completed Pick, Place, Open or Close as an action recognizer reports it: missed with
act_miss, its object swapped for another visible one with act_arg_err.
"""

from __future__ import annotations

import numpy as np

from helper.bip.actions import OBSERVED
from helper.bip.config import Config

CONTAINERS = ("chest_of_drawers", "cabinet", "fridge", "drawer", "wardrobe", "dresser", "oven", "dishwasher", "washer")


def parse(tool: str, arg: str) -> tuple:
    """A PARTNR skill call as a bounded action."""
    if tool == "Place":
        obj, prep, rec, *rest = [x.strip() for x in arg.split(",")]
        ref = rest[1] if len(rest) > 1 and rest[0] == "next_to" and rest[1] != "none" else None
        return ("Place", obj, "in" if prep == "within" else "on", rec, ref)
    return (tool, arg.strip())


def relation(sim, o: str) -> tuple[str, str] | None:
    wg = sim.env.full_world_graph
    try:
        f = wg.find_furniture_for_object(wg.get_node_from_name(o))
    except Exception:
        return None
    if f is None:
        return None
    return ("in" if any(c in f.name for c in CONTAINERS) else "on"), f.name


def view(cfg: Config, sim, key: str, uid: int, agent: str, furniture: set, rng) -> list[tuple]:
    masks = sim.masks(key)
    held, out, placed = sim.held(uid), [], {}
    for o in sorted(n for n in masks if n not in furniture):
        if rng.random() < cfg.view_miss:
            continue
        if o == held:
            out.append((agent, "holding", o))
            continue
        r = relation(sim, o)
        if r is None:
            continue
        rel, f = r
        if rng.random() < cfg.view_rel_err:
            others = sorted(x for x in masks if x in furniture and x != f) or sorted(furniture - {f})
            f = others[rng.integers(len(others))]
        out.append((o, rel, f))
        placed[o] = f
    pos = {o: sim_position(sim, o) for o in placed}
    names = sorted(placed)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if placed[a] == placed[b] and pos[a] is not None and pos[b] is not None and np.linalg.norm(pos[a] - pos[b]) < cfg.next_to_m:
                out.append((a, "next_to", b))
    return out


def survey(cfg: Config, sim) -> list[tuple]:
    """The robot's prior map of the home when the task starts: every object with its furniture and next-to
    relations (a mapping run done before the task; the map goes stale as objects move)."""
    objs = [o.name for o in sim.env.full_world_graph.get_all_objects()]
    placed = {o: r for o in objs if (r := relation(sim, o)) is not None}
    out = [(o, rel, f) for o, (rel, f) in placed.items()]
    pos = {o: sim_position(sim, o) for o in placed}
    names = sorted(placed)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if placed[a][1] == placed[b][1] and pos[a] is not None and pos[b] is not None and np.linalg.norm(pos[a] - pos[b]) < cfg.next_to_m:
                out.append((a, "next_to", b))
    return out


def sim_position(sim, name: str) -> np.ndarray | None:
    try:
        return np.array(sim.env.full_world_graph.get_node_from_name(name).properties["translation"])
    except Exception:
        return None


def recognize(cfg: Config, tool: str, arg: str, visible: list[str], furniture: set, rng) -> list | None:
    if tool not in OBSERVED or rng.random() < cfg.act_miss:
        return None
    a = list(parse(tool, arg))
    kind = (a[1] in furniture)  # a wrong read stays within its kind: furniture for Open and Close, objects otherwise
    others = [x for x in visible if x != a[1] and (x in furniture) == kind]
    if rng.random() < cfg.act_arg_err and others:
        a[1] = others[rng.integers(len(others))]
    return a
