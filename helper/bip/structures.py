"""BIP when the subgoals themselves are uncertain: a few candidate structures (for example a task family's
recipes), each with its own scene tracker and S1 filter. A structure's weight is its prior times its evidence;
a subgoal's P_exist is the weight of the structures that contain it (subgoals match by name, e.g. "plate1").

    B = StructureBelief(cfg, structures, prior)   structures: {name: st} with nodes, spec, objects, labels, edges
    B.observe(t, views, acts, fixations)          the same observations into every structure
    B.claim(keys); B.weights(); B.values(...)     V(execute key) mixed over structures (README section 6)
"""

from __future__ import annotations

import numpy as np

from . import observe
from .config import Config
from .decide import act_values, exec_prob
from .s1 import Filter
from .track import Tracker


class StructureBelief:
    def __init__(self, cfg: Config, structures: dict, prior: dict, seed: int = 0):
        self.cfg, self.st, self.names = cfg, structures, list(structures)
        self.idx = {n: {v: i for i, v in enumerate(st["nodes"])} for n, st in structures.items()}
        self.O = {n: [set(st["objects"][v]) for v in st["nodes"]] for n, st in structures.items()}
        self.T = {n: Tracker(cfg, st) for n, st in structures.items()}
        self.F = {n: Filter(cfg, len(self.idx[n]), [frozenset((self.idx[n][u], self.idx[n][v]) for u, v in st["edges"])],
                            [0.0], seed + k) for k, (n, st) in enumerate(structures.items())}
        for F in self.F.values():  # the person starts some allowed subgoal
            F.action(np.ones(F.N + 1))
        self.log_prior = {n: float(np.log(max(1e-6, prior[n]))) for n in self.names}
        self.keys = sorted({v for st in structures.values() for v in st["nodes"]})
        self.done_t = {n: [] for n in self.names}  # completion times seen in each structure (the person's in person_t)
        self.person_t = {n: [] for n in self.names}
        self.rng = np.random.default_rng(seed)

    def observe(self, views: list, acts: list, fixations: list) -> None:
        """New observations, in time order, into every structure's tracker and filter."""
        obs = [(v["t"], 0, v) for v in views] + [(a["t"], 1, a) for a in acts]
        for n in self.names:
            T, F, ev = self.T[n], self.F[n], []
            for t, _, x in sorted(obs, key=lambda e: (e[0], e[1])):
                out = T.view(t, x["triples"]) if "triples" in x else T.act(t, x["actor"], tuple(x["action"]))
                self.done_t[n] += [e[0] for e in out if e[2] in ("completion", "robot_done")]
                self.person_t[n] += [e[0] for e in out if e[2] == "completion"]
                ev += [e[:4] for e in out]
            ev += [observe.fixation(self.cfg, f, self.O[n], self.rng)[:4] for f in fixations]
            step = {"action": F.action, "fixation": lambda a: F.fixation(*a), "completion": F.completion,
                    "detected": F.detected, "robot_done": F.robot_done}
            for t, _, kind, x in sorted(ev, key=lambda e: (e[0], e[1])):
                step[kind](x)

    def claim(self, keys: set) -> None:
        for n, F in self.F.items():
            F.claimed = sum(1 << i for v, i in self.idx[n].items() if v in keys)

    def weights(self) -> dict:
        lw = np.array([self.log_prior[n] + float(self.F[n].logZ[0]) for n in self.names])
        w = np.exp(lw - lw.max())
        return dict(zip(self.names, w / w.sum()))

    def lead(self) -> str:
        w = self.weights()
        return max(w, key=w.get)

    def p_exist(self, key: str) -> float:
        w = self.weights()
        return sum(w[n] for n in self.names if key in self.idx[n])

    def values(self, d: float, rem: float, costs, beta_wrong: float, blocked=None, w: dict | None = None) -> dict:
        """V(execute key) = sum over the structures that send key where the most of them do (the robot goes there) of
        w * V_s(key), minus beta_wrong (1 - P_right), with P_right their weight: a placement is wrong, and saves
        nothing, if the subgoal does not exist or goes elsewhere.
        A key is never executed with P_exist below 0.5, or when a structure of weight above 0.2 rules it out
        (done, started, claimed); `blocked(key)` adds the robot's own reasons (an object it never located).
        `w` replaces the structure weights (normalized here), to value a question's possible answers."""
        w = self.weights() if w is None else {n: x / sum(w.values()) for n, x in w.items()}
        out = {}
        per = {}
        for n in self.names:
            F = self.F[n]
            per[n] = act_values(F, F.joint(), exec_prob(F), np.full(F.N, d), np.full(F.N, rem), costs)
        for key in self.keys:
            have = [n for n in self.names if key in self.idx[n]]
            pe = sum(w[n] for n in have)
            vals = [(w[n], per[n][self.idx[n][key]]) for n in have]
            if pe < 0.5 or any(wn > 0.2 and not np.isfinite(v) for wn, v in vals) or (blocked and blocked(key)):
                continue
            dest = {n: self.st[n]["spec"][key].get("to") if self.st[n].get("spec", {}).get(key) else None for n in have}
            to = {}
            for n in have:
                to[dest[n]] = to.get(dest[n], 0.0) + w[n]
            f = max(to, key=to.get)
            out[key] = sum(wn * v for n, (wn, v) in zip(have, vals) if dest[n] == f and np.isfinite(v)) - beta_wrong * (1 - to[f])
        return out

    def classes(self) -> dict:
        """Per subgoal key: (P_exist, P(person on it), P(done)), mixed over structures; for read-outs and clips."""
        w, out = self.weights(), {}
        for key in self.keys:
            pe = cur = done = 0.0
            for n in self.names:
                if key in self.idx[n]:
                    F, i = self.F[n], self.idx[n][key]
                    pe += w[n]
                    cur += w[n] * float(F.marginal(F.g)[i])
                    done += w[n] * float(F.done_probs()[i])
            out[key] = (round(pe, 3), round(cur, 3), round(done, 3))
        return out
