"""Observations to S1 events, per perception tier (cfg.action_tier, cfg.completion).

An event is (t, order, kind, x, info): kind action with x = L[v] = P(a_t | g_t = v, s_t); fixation with
x = (u, 1 / |X|), u[v] = 1[x in O(v)] / |O(v) & X| for the fixated object x among the visible X; completion v;
detected v (a late report). Events at the same t run in `order`.
    scene    the scene graph (track.py): actions scored by the plan likelihood (actions.py), completions read
             off the graph's predicates
    oracle   the person's skill and target every few seconds; L[v] = exp(beta * (m_v - max m)), m_v the share
             of the action's entities in O(v)
    noisy    the same with the target swapped for another visible entity with 1 - action_acc
    ego      only the object in the person's hand, read once per second by a detector that misses it with
             hand_miss and names a random visible object instead with 1 - hand_acc
Completions outside the scene tier: oracle (exact), noisy (det_rate found, det_delay late) or none (latent).
Every completion is followed by an uninformative action: the person moves on.
"""

from __future__ import annotations

import numpy as np

from .config import Config


def match(ents: list[str], objects: list[set]) -> np.ndarray:
    E = set(ents)
    return np.array([len(E & O) / max(1, len(E)) for O in objects])


def likelihood(cfg: Config, ents: list[str], objects: list[set]) -> np.ndarray:
    """Relative likelihood per subgoal, then the null model's: an action that matches no subgoal."""
    m = match(ents, objects)
    return np.append(np.exp(cfg.beta_action * (m - m.max())), np.exp(-cfg.beta_action * m.max()))


def read(true: str, visible, acc: float, miss: float, rng) -> str | None:
    """A detector's read of an object: missed with P(miss), else right with P(acc), else a random visible one."""
    if rng.random() < miss:
        return None
    others = sorted(set(visible) - {true})
    return true if rng.random() < acc or not others else others[rng.integers(len(others))]


def actions(cfg: Config, st: dict, rng) -> list[tuple]:
    """(t, order, "action", L, text) events for the chosen tier."""
    O, out = [set(st["objects"][v]) for v in st["nodes"]], []
    if cfg.action_tier == "ego":
        every = sorted(set().union(*O))
        for h in st["held"]:
            x = read(h["object"], h.get("visible", every), cfg.hand_acc, cfg.hand_miss, rng) if h["object"] else None
            if x:
                out.append((h["t"], 1, "action", likelihood(cfg, [x], O), f"holding {x}"))
        return out
    for a in st["actions"]:
        ents = list(a["entities"])
        if cfg.action_tier == "noisy" and ents and rng.random() > cfg.action_acc:
            others = [x for x in a.get("visible", []) if x not in ents] or [x for O_ in O for x in O_ if x not in ents]
            if others:
                ents[0] = others[rng.integers(len(others))]
        out.append((a["t"], 1, "action", likelihood(cfg, ents, O), f"{a['verb']} {' '.join(ents)}"))
    return out


def fixation(cfg: Config, f: dict, O: list[set], rng) -> tuple:
    """S1's fixation event from a recorded fixation, its object label read with accuracy fix_acc."""
    X = set(f["visible"]) | {f["object"]}
    x = read(f["object"], X, cfg.fix_acc, 0.0, rng)
    u = np.array([float(x in o) / max(1, len(o & X)) for o in O])
    return (f["t"], 2, "fixation", (u, 1 / len(X)), x)


def fixations(cfg: Config, st: dict, rng) -> list[tuple]:
    O = [set(st["objects"][v]) for v in st["nodes"]]
    return [fixation(cfg, f, O, rng) for f in st["fixations"]]


def completions(cfg: Config, st: dict, rng) -> list[tuple]:
    """Completion events, each followed by "the person moves on": an uninformative action at the same time."""
    index, ones = {v: i for i, v in enumerate(st["nodes"])}, np.ones(len(st["nodes"]) + 1)
    if cfg.completion == "oracle":
        seen = [(c["t"], "completion", c["node"]) for c in st["completions"]]
    elif cfg.completion == "noisy":
        seen = [(c["t"] + rng.exponential(cfg.det_delay), "detected", c["node"]) for c in st["completions"] if rng.random() < cfg.det_rate]
    else:
        seen = []
    return [e for t, kind, v in seen for e in ((t, 0, kind, index[v], v), (t, 0.5, "action", ones, "moves on"))]


def events(cfg: Config, st: dict) -> list[tuple]:
    rng = np.random.default_rng([cfg.seed, sum(map(ord, st["id"]))])
    start = [(st["t0"], 1, "action", np.ones(len(st["nodes"]) + 1), "start")]  # the person starts some allowed subgoal
    if cfg.action_tier == "scene":  # actions and completions both come from the scene graph
        from .track import events as scene_events
        out = start + scene_events(cfg, st) + (fixations(cfg, st, rng) if cfg.gaze else [])
        return sorted(out, key=lambda e: (e[0], e[1]))
    out = start + actions(cfg, st, rng) + completions(cfg, st, rng) + (fixations(cfg, st, rng) if cfg.gaze else [])
    return sorted(out, key=lambda e: (e[0], e[1]))
