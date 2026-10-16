"""The fast-slow belief: S1 (s1.py) updates on every event; S2 (s2.py) proposes graphs when S1 is surprised.

    B = Belief(cfg, st)        S2's prior over orderings -> a pool of graphs -> one S1 particle filter per graph
    B.feed(t, kind, x)         one event: S1 update; on surprise, an S2 call whose graphs join after its latency
    B.estimate()               done, current, ranked next subgoals, P(u before v)

Events (kind, x): action L[v] = P(a | g = v, s); fixation (u, 1 / |X|); completion v; detected v; robot_done v;
claimed (bitmask of the robot's subgoals). Surprise: log P(event | no orderings) - log P(event | pool), from a
reference filter R that has the empty graph; above cfg.surprise (and past the cooldown and call budget), S2
revises the three leading graphs. New graphs are scored on the whole history before they join, so every graph's
weight stays prior times marginal likelihood.
"""

from __future__ import annotations

import numpy as np

from . import s2
from .config import Config
from .graph import ancestors, log_prior, repair, sample_pool
from .s1 import Filter


class Belief:
    def __init__(self, cfg: Config, st: dict):
        self.cfg, self.st, self.N = cfg, st, len(st["nodes"])
        self.rng = np.random.default_rng(cfg.seed)
        self.edges, self.prior_call = s2.prior_edges(cfg, st)  # P(G | l): p_uv per ordering
        pool = sample_pool(cfg.pool, self.N, self.edges, self.rng)
        self.F = Filter(cfg, self.N, pool, [log_prior(self.edges, g) for g in pool], cfg.seed)
        self.R = Filter(cfg, self.N, [frozenset()], [0.0], cfg.seed + 1)  # reference: no orderings
        self.history, self.log, self.starts, self.seen, self.read = [], [], [], set(), None
        self.calls, self.pending, self.last_call = [], None, -np.inf

    # -- S1 -------------------------------------------------------------------------------------------
    def apply(self, F: Filter, kind: str, x):
        """One event into a filter; returns the predictive probability per graph (None for bookkeeping)."""
        if kind == "claimed":
            F.claimed = x
            return None
        return {"action": F.action, "fixation": lambda a: F.fixation(*a), "detected": F.detected,
                "completion": F.completion, "robot_done": F.robot_done}[kind](x)

    def feed(self, t: float, kind: str, x) -> None:
        """The main step: join S2's pending graphs if their latency has passed, update S1, maybe call S2."""
        cfg, st = self.cfg, self.st
        if self.pending and self.pending[0] <= t:
            self.pending[2]["added"] = self.add(self.pending[1])
            self.pending = None
        w = self.F.graph_weights()
        pred = self.apply(self.F, kind, x)
        ref = self.apply(self.R, kind, x)
        self.history.append((kind, x))
        if kind in ("completion", "detected", "robot_done"):
            self.log.append(f"{t:.0f} s {s2.ids_for(cfg, st)[st['nodes'][x]]} complete")
        if kind == "action":
            self.note(t)
            surprise = float(np.log(max(1e-300, ref[0])) - np.log(max(1e-300, w @ pred)))
            if (cfg.s2 != "never" and surprise > cfg.surprise and not self.pending
                    and t - self.last_call >= cfg.cooldown and len(self.calls) < cfg.max_calls):
                self.fire(t, surprise)

    def note(self, t: float) -> None:
        """Log S1's read of the current subgoal when it changes, for S2's prompt and for repairs."""
        F, nodes = self.F, self.st["nodes"]
        cur = int(np.argmax(F.marginal(F.g)))
        if cur == self.read:
            return
        self.read = cur
        self.log.append(f"{t:.0f} s {'back to' if cur in self.seen else 'started'} {s2.ids_for(self.cfg, self.st)[nodes[cur]]}")
        if cur not in self.seen:
            self.starts.append((cur, F.done_mask()))
            self.seen.add(cur)

    # -- S2 -------------------------------------------------------------------------------------------
    def fire(self, t: float, surprise: float) -> None:
        """Ask S2 (Gemini, or the model-free repair) for corrected graphs; they arrive after its latency."""
        cfg, F, nodes = self.cfg, self.F, self.st["nodes"]
        self.last_call = t
        w = F.graph_weights()
        top = [(F.graphs[k], w[k]) for k in np.argsort(-w)[:3]]
        call = {"t": round(t, 2), "surprise": round(surprise, 2)}
        if cfg.s2 == "repair":
            graphs, latency = [repair(self.N, g, self.starts) for g, _ in top], 0.0
        else:
            cur = int(np.argmax(F.marginal(F.g)))
            ids = s2.ids_for(cfg, self.st)
            unmet = [ids[nodes[u]] for u in range(self.N) if (F.anc[np.argmax(w)][cur] >> u & 1) and not (F.done_mask() >> u & 1)]
            event = (f"the person appears to work on {ids[nodes[cur]]}"
                     + (f" while {', '.join(unmet)} {'is' if len(unmet) == 1 else 'are'} not complete" if unmet else "")
                     + "; the hypotheses explain this poorly.")
            imgs = s2.frames(self.st, t, cfg.s2_frames, cfg.s2_frame_gap)
            graphs, usage, latency = s2.revise(cfg, self.st, top, self.log, event, imgs)
            call |= {k: usage.get(k, 0) for k in ("prompt_tokens", "completion_tokens", "thinking_tokens")}
        call |= {"latency": round(latency, 2), "graphs": len(graphs)}
        self.calls.append(call)
        self.pending = (t + latency, graphs, call)

    def add(self, graphs: list[frozenset]) -> int:
        """Join new acyclic graphs to the pool, each scored on the whole event history first."""
        have, new = set(self.F.graphs), []
        for g in graphs:
            try:
                ancestors(self.N, g)
            except ValueError:
                continue
            if g not in have:
                new.append(g)
                have.add(g)
        if not new:
            return 0
        G = Filter(self.cfg, self.N, new, [log_prior(self.edges, g) for g in new], int(self.rng.integers(1 << 30)))
        for kind, x in self.history:
            self.apply(G, kind, x)
        F = self.F
        F.graphs += new
        F.log_prior = np.concatenate([F.log_prior, G.log_prior])
        F.anc = np.concatenate([F.anc, G.anc])
        for a in ("g", "n", "S", "D", "logW", "logZ"):
            setattr(F, a, np.concatenate([getattr(F, a), getattr(G, a)]))
        return len(new)

    # -- read-outs ------------------------------------------------------------------------------------
    def estimate(self) -> dict:
        F, nodes = self.F, self.st["nodes"]
        cur, nxt, E = F.marginal(F.g), F.marginal(F.n), F.edge_probs()
        return {"done": sorted(nodes[v] for v in range(self.N) if F.done_mask() >> v & 1),
                "current": nodes[int(np.argmax(cur))] if cur.sum() > 0 else None,
                "next": self.ranked_next(cur, nxt),
                "edges": [[nodes[u], nodes[v], round(float(E[u, v]), 3)]
                          for u in range(self.N) for v in range(self.N) if E[u, v] >= 0.05]}

    def ranked_next(self, cur: np.ndarray, nxt: np.ndarray) -> list[str]:
        """Open subgoals other than the current one, by P(planned next), then by P(predecessors done)."""
        F = self.F
        ready = np.einsum("km,kmn->n", F.joint(), (F.anc[:, None, :] & ~(F.D | F.C)[..., None]) == 0)
        c, done = int(np.argmax(cur)), F.done_mask()
        open_ = [v for v in range(self.N) if v != c and not done >> v & 1]
        return [self.st["nodes"][v] for v in sorted(open_, key=lambda v: (-nxt[v], -ready[v]))][:3]

    def snapshot(self, t: float, kind: str, info) -> dict:
        F = self.F
        return {"t": round(t, 2), "kind": kind, "info": info, "cur": F.marginal(F.g).round(3).tolist(),
                "next": F.marginal(F.n).round(3).tolist(), "done": F.done_probs().round(3).tolist(),
                "edges": F.edge_probs().round(3).tolist()}
