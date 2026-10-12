"""Fits perception weights by EM on labeled dev videos and reports how informative each cue is.

With the true current and next subgoals known, the mixture components (U_O(g), U_O(n), the noise floor
U_X) are fixed numbers, so EM only updates the weights. `tier=real` fits the detector cues instead:
handled objects (w_hand) and YOLO gaze fixations; it also measures p_done from the timelines.

    python -m helper.bip.fit split=dev
    python -m helper.bip.fit split=dev tier=real
"""

import sys

import numpy as np

from . import llm, stream
from .config import parse
from .tasks import load


def components(cfg) -> np.ndarray:
    """Rows of [U_O(g)(x), U_O(n)(x), U_X(x), mean U_O(v)(x) over other subgoals v] per fixation."""
    tasks, rows = load(cfg.min_nodes), []
    for vid in stream.split(cfg):
        st, task = stream.read(cfg, vid), tasks[vid]
        p = llm.prior(cfg, st, llm.ids_for(cfg, st["nodes"], vid))
        sets = {v: {o for o, nm in st["objects"].items() if nm in set(p["objects"].get(v, []))} for v in st["nodes"]}
        for f in st["fixations"]:
            g, n = task.current(f["t"]), task.next(f["t"])
            if g is None or n is None:
                continue
            X = set(f["visible"])
            U = lambda v: float(f["object"] in sets[v]) / max(1, len(sets[v] & X))
            others = [U(v) for v in st["nodes"] if v not in (g, n)]
            rows.append([U(g), U(n), 1 / len(X), np.mean(others) if others else 0.0])
    return np.array(rows)


def em(C: np.ndarray, iters: int = 200) -> np.ndarray:
    w = np.full(3, 1 / 3)
    for _ in range(iters):
        r = C[:, :3] * w
        r /= r.sum(1, keepdims=True)
        w = r.mean(0)
    return w


def main() -> None:
    cfg = parse(sys.argv[1:])
    C = components(cfg)
    hit = C[:, :2] > 0
    print(f"fixations {len(C)}; object in O(current) {hit[:, 0].mean():.3f}, in O(next) {hit[:, 1].mean():.3f}, "
          f"in O(other subgoal) on average {(C[:, 3] > 0).mean():.3f}")
    print(f"mean U: current {C[:, 0].mean():.3f}, next {C[:, 1].mean():.3f}, other {C[:, 3].mean():.3f}, noise {C[:, 2].mean():.3f}")
    w = em(C)
    print(f"EM weights: w_cur {w[0]:.3f}, w_next {w[1]:.3f}, w_noise {w[2]:.3f}")


def lookahead(cfg) -> None:
    """Mean U_O(next)(x) against U_O(other)(x) by time left until the next subgoal starts."""
    tasks, bins = load(cfg.min_nodes), {(0, 2): [], (2, 5): [], (5, 10): [], (10, 30): [], (30, 1e9): []}
    for vid in stream.split(cfg):
        st, task = stream.read(cfg, vid), tasks[vid]
        p = llm.prior(cfg, st, llm.ids_for(cfg, st["nodes"], vid))
        sets = {v: {o for o, nm in st["objects"].items() if nm in set(p["objects"].get(v, []))} for v in st["nodes"]}
        starts = [(a, s) for a, _, s in task.timeline.spans if s]
        for f in st["fixations"]:
            g, n = task.current(f["t"]), task.next(f["t"])
            if g is None or n is None:
                continue
            t_switch = next((a for a, s in starts if a > f["t"] and s == n), None)
            X = set(f["visible"])
            U = lambda v: float(f["object"] in sets[v]) / max(1, len(sets[v] & X))
            others = [U(v) for v in st["nodes"] if v not in (g, n)]
            for (a, b), rows in bins.items():
                if t_switch is not None and a <= t_switch - f["t"] < b:
                    rows.append([U(n), np.mean(others) if others else 0.0, U(g)])
    for (a, b), rows in bins.items():
        r = np.array(rows)
        print(f"{a:>3}-{b:<4} s before the switch: {len(r):5d} fixations, U next {r[:, 0].mean():.3f}, "
              f"other {r[:, 1].mean():.3f}, current {r[:, 2].mean():.3f}")


def real(cfg) -> None:
    """Handled-object and gaze mixtures on detections, and how often leaving a subgoal means it is complete."""
    from . import perceive
    tasks, hands, fixes, left = load(cfg.min_nodes), [], [], []
    for vid in stream.split(cfg):
        st, task = stream.read(cfg, vid), tasks[vid]
        vocab = llm.vocab(cfg, st, llm.ids_for(cfg, st["nodes"], vid))
        sets = {v: set(vocab.get(v, [])) for v in st["nodes"]}
        obs = perceive.read(cfg, vid)
        for w in obs["windows"]:
            g = task.current(w["t"] - cfg.window / 2)
            X = set(w["seen"]) | set(w["handled"])
            if g is None:
                continue
            for x in w["handled"]:
                others = [float(x in sets[v]) / max(1, len(sets[v] & X)) for v in st["nodes"] if v != g]
                hands.append([float(x in sets[g]) / max(1, len(sets[g] & X)), 1 / len(X), np.mean(others) if others else 0.0])
        for f in obs["fixations"]:
            g, n = task.current(f["t"]), task.next(f["t"])
            if g is None or n is None:
                continue
            X = set(f["visible"])
            U = lambda v: float(f["object"] in sets[v]) / max(1, len(sets[v] & X))
            others = [U(v) for v in st["nodes"] if v not in (g, n)]
            fixes.append([U(g), U(n), 1 / len(X), np.mean(others) if others else 0.0])
        owned = [(t, s) for t, _, s in task.timeline.spans if s]
        for (_, a), (t, b) in zip(owned, owned[1:]):
            if a != b:  # the person moves from a to b at t
                left.append(task.last[a] <= t)
    H, Fx = np.array(hands), np.array(fixes)
    w = np.full(2, 0.5)
    for _ in range(200):
        r = H[:, :2] * w
        r /= r.sum(1, keepdims=True)
        w = r.mean(0)
    print(f"handled objects {len(H)}: in O(current) {(H[:, 0] > 0).mean():.3f}; mean U current {H[:, 0].mean():.3f}, "
          f"other subgoal {H[:, 2].mean():.3f}; EM w_hand {w[0]:.3f}")
    print(f"YOLO fixations {len(Fx)}: mean U current {Fx[:, 0].mean():.3f}, next {Fx[:, 1].mean():.3f}, other {Fx[:, 3].mean():.3f}; "
          f"EM weights cur/next/noise {np.round(em(Fx), 3).tolist()}")
    print(f"switches {len(left)}: the subgoal left was complete {np.mean(left):.3f} (p_done)")


if __name__ == "__main__":
    cfg = parse(sys.argv[1:])
    if cfg.tier == "real":
        real(cfg)
    else:
        main()
        lookahead(cfg)
