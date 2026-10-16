"""The bounded action set over a scene graph, the person's plan for a subgoal, and P(a | g, s).

Actions: ("Navigate", x), ("Pick", o), ("Place", o, rel, f, ref), ("Open", f), ("Close", f), ("Wait",).
Only manipulations are observed (Pick, Place, Open, Close); walking leaves no evidence without the person's
position. For subgoal g = (o rel f, next to ref) in state s, the next manipulation is Place(o, rel, f, ref)
when the person holds o, else Pick(o), preceded by Open of o's furniture when o is in it and it is not open.
    P(a | g, s) = (1 - eps) [a in next(g, s)] / |next(g, s)| + eps / |A_obs(s)|
with A_obs(s) the observable actions in s: Pick of every known unheld object, Place of each held object on or
in any furniture, Open and Close of any furniture.
"""

from __future__ import annotations

import numpy as np

from .scene import Scene

OBSERVED = ("Pick", "Place", "Open", "Close")


def n_observable(s: Scene, objects: set) -> int:
    """|A_obs(s)| without listing it."""
    free = sum(1 for o in objects if s.holder(o) is None)
    return max(1, free + 2 * len(s.held) * len(s.furniture) + 2 * len(s.furniture))


def plan(spec: dict, s: Scene, agent: str = "H") -> list[tuple]:
    """The next manipulations toward the subgoal from state s ([] when it already holds). A category subgoal
    can be served by any candidate that is not in a hand and not already in place."""
    if s.holds(spec):
        return []
    tos, cands = spec.get("tos") or [spec["to"]], s.candidates(spec)
    if s.held.get(agent) in cands:
        ref = spec.get("ref") or (s.placed_ref(spec) if spec.get("refs") else None)
        return [("Place", s.held[agent], spec["rel"], f, ref) for f in tos]
    out = []
    for o in cands:
        rel, f = s.parent.get(o, (None, None))
        if "objects" in spec and (s.holder(o) is not None or f in tos):
            continue
        out += [a for a in ([("Open", f)] if rel == "in" and f not in s.opened else []) + [("Pick", o)] if a not in out]
    return out


def same(a: tuple, b: tuple) -> bool:
    """Whether two actions match: same verb and object, and for Place the same furniture."""
    if a[0] != b[0] or len(a) < 2 or len(b) < 2 or a[1] != b[1]:
        return False
    return a[0] != "Place" or a[3] == b[3]


def likelihood(a: tuple, specs: list[dict], s: Scene, objects: set, eps: float) -> np.ndarray:
    """P(a | g = v, s) for every subgoal v, then P(a | no subgoal left) = 1 / |A_obs(s)|, the null model."""
    floor = eps / n_observable(s, objects)
    out = np.full(len(specs), floor)
    for v, spec in enumerate(specs):
        nxt = plan(spec, s)
        if any(same(a, b) for b in nxt):
            out[v] += (1 - eps) / len(nxt)
    return np.append(out, floor / eps)
