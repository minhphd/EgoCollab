"""S1: the fast filter over task graphs, the current subgoal g and the planned next subgoal n.

Each graph in the pool runs its own particle filter over (g, n, started set, done set D); the observed
completed set C is shared. Without observed completions (cfg.completion latent or noisy), leaving an
unfinished subgoal marks it done in a particle with probability p_done. Graph weights are prior times
marginal likelihood, so a graph that cannot explain a start loses weight by the deviation rate delta.
Events:
    action(L)   L[v] = likelihood of the action under subgoal v
    fixation(u, noise)   u[v] = U_O(v)(x), the gaze term of subgoal v for the fixated object x
    completion(v)        subgoal v is observed complete; only the subgoal being worked on can finish
    detected(v)          a completion detector reports v, possibly late: done, no evidence about g
"""

import numpy as np

from .config import Config


def ancestors(n: int, edges) -> np.ndarray:
    """Bitmask of the transitive predecessors of every node; raises on a cycle."""
    preds = [0] * n
    for u, v in edges:
        preds[v] |= 1 << u
    anc = preds[:]
    for _ in range(n):
        new = [a | sum_or(anc[u] for u in range(n) if a >> u & 1) for a in anc]
        if new == anc:
            break
        anc = new
    if any(anc[v] >> v & 1 for v in range(n)):
        raise ValueError("cycle")
    return np.array(anc, dtype=np.int64)


def sum_or(xs) -> int:
    out = 0
    for x in xs:
        out |= x
    return out


class Filter:
    def __init__(self, cfg: Config, n: int, graphs: list[frozenset], log_prior: list[float], seed: int = 0):
        self.cfg, self.N, self.rng = cfg, n, np.random.default_rng(seed)
        self.graphs, self.log_prior = list(graphs), np.array(log_prior, dtype=float)
        self.anc = np.stack([ancestors(n, g) for g in graphs])
        K, M = len(graphs), cfg.particles
        self.g = np.full((K, M), -1)
        self.n = np.full((K, M), -1)
        self.S = np.zeros((K, M), dtype=np.int64)
        self.D = np.zeros((K, M), dtype=np.int64)  # done in this particle's history (latent completion)
        self.logW = np.full((K, M), -np.log(M))
        self.logZ = np.zeros(K)
        self.C = 0
        self.claimed = 0  # bitmask of subgoals the robot is doing: the person picks around them
        self.v = np.arange(n)

    # -- belief readout ------------------------------------------------------------------------
    def graph_weights(self) -> np.ndarray:
        lw = self.log_prior + self.logZ
        w = np.exp(lw - lw.max())
        return w / w.sum()

    def joint(self) -> np.ndarray:
        W = np.exp(self.logW - self.logW.max(1, keepdims=True))
        return self.graph_weights()[:, None] * W / W.sum(1, keepdims=True)

    def done_probs(self) -> np.ndarray:
        """P(v is complete) for every subgoal."""
        J, Dp = self.joint(), self.D | self.C
        return np.array([(J * ((Dp >> v) & 1)).sum() for v in range(self.N)])

    def done_mask(self) -> int:
        """Bitmask of the subgoals more likely complete than not."""
        return sum(1 << v for v, p in enumerate(self.done_probs()) if p >= 0.5)

    def marginal(self, x: np.ndarray) -> np.ndarray:
        """Distribution of a particle variable (g or n) over subgoals."""
        J, ok = self.joint(), x >= 0
        return np.bincount(x[ok], J[ok], minlength=self.N)

    def edge_probs(self) -> np.ndarray:
        """P((u, v) in the closure of G), as an N x N matrix."""
        clos = (self.anc[:, None, :] >> self.v[:, None]) & 1  # [K, u, v] = u is an ancestor of v
        return np.einsum("k,kuv->uv", self.graph_weights(), clos)

    # -- model pieces ----------------------------------------------------------------------------
    def done(self, x: np.ndarray) -> np.ndarray:
        return (x >= 0) & (((self.D | self.C) >> np.maximum(x, 0)) & 1).astype(bool)

    def options(self, S: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
        """Unnormalized choice weights [K, M, N] over subgoals to work on, given the started sets S."""
        cfg, v = self.cfg, self.v
        Dp = self.D | self.C
        done = ((Dp[..., None] >> v) & 1).astype(bool)
        sat = (S | Dp) if cfg.rule == "started" else Dp
        if extra is not None:  # the subgoal about to be worked on counts as done for planning
            sat = sat | (np.int64(1) << np.maximum(extra, 0)) * (extra >= 0)
        started = ((S[..., None] >> v) & 1).astype(bool)
        ready = (self.anc[:, None, :] & ~sat[..., None]) == 0
        done = done | ((self.claimed >> v) & 1).astype(bool)
        return np.where(done, 0.0, np.where(started, cfg.resume, np.where(ready, 1.0, cfg.delta)))

    def _sample(self, P: np.ndarray) -> np.ndarray:
        c = P.cumsum(-1)
        u = self.rng.random(P.shape[:-1])[..., None] * c[..., -1:]
        return np.minimum((c < u).sum(-1), self.N - 1)

    def _weigh(self, inc: np.ndarray) -> np.ndarray:
        """Multiply particle weights by inc [K, M]; returns each graph's predictive probability."""
        W = np.exp(self.logW - self.logW.max(1, keepdims=True))
        W /= W.sum(1, keepdims=True)
        pred = (W * inc).sum(1)
        self.logZ += np.log(np.maximum(pred, 1e-300))
        self.logW += np.log(np.maximum(inc, 1e-300))
        self._resample()
        return pred

    def _resample(self) -> None:
        W = np.exp(self.logW - self.logW.max(1, keepdims=True))
        W /= W.sum(1, keepdims=True)
        ess = 1 / (W ** 2).sum(1)
        M = W.shape[1]
        for k in np.flatnonzero(ess < M / 2):
            idx = np.minimum(np.searchsorted(W[k].cumsum(), (self.rng.random() + np.arange(M)) / M), M - 1)
            self.g[k], self.n[k], self.S[k], self.D[k] = self.g[k, idx], self.n[k, idx], self.S[k, idx], self.D[k, idx]
            self.logW[k] = -np.log(M)

    # -- events ------------------------------------------------------------------------------------
    def action(self, L: np.ndarray) -> np.ndarray:
        cfg, g, n = self.cfg, self.g, self.n
        w = self.options(self.S)
        if not w.sum(-1).any():  # everything complete in every particle: nothing left to explain
            return np.ones(len(self.graphs))
        excl = np.where(self.v == g[..., None], 0.0, w)
        pi = excl / np.maximum(excl.sum(-1, keepdims=True), 1e-300)
        plan_ok = (n >= 0) & ~self.done(n) & (n != g)
        plan = (self.v == n[..., None]).astype(float)
        switch = np.where(plan_ok[..., None], cfg.plan * plan + (1 - cfg.plan) * pi, pi)
        keep = (g >= 0) & ~self.done(g)
        T = np.where(keep[..., None], cfg.stay * (self.v == g[..., None]) + (1 - cfg.stay) * switch, switch)
        q = T * L
        pred = self._weigh(q.sum(-1))
        g2 = self._sample(q)
        moved = g2 != self.g
        if cfg.completion != "oracle":  # leaving an unfinished subgoal may mean it is complete
            left = moved & (self.g >= 0) & ~self.done(self.g) & (self.rng.random(g2.shape) < cfg.p_done)
            self.D |= np.where(left, np.int64(1) << np.maximum(self.g, 0), 0)
        self.S |= np.int64(1) << g2
        redraw = moved | (self.n == g2) | self.done(self.n) | (self.n < 0)
        w2 = self.options(self.S, extra=g2)
        w2 = np.where(self.v == g2[..., None], 0.0, w2)
        n2 = np.where(w2.sum(-1) > 0, self._sample(w2), -1)
        self.g, self.n = g2, np.where(redraw, n2, self.n)
        return pred

    def fixation(self, u: np.ndarray, noise: float) -> np.ndarray:
        cfg = self.cfg
        U = np.append(u, 0.0)  # index -1 (no subgoal) has no gaze term
        L = cfg.w_cur * U[self.g] + cfg.w_next * U[self.n] + cfg.w_noise * noise
        return self._weigh(L ** cfg.gaze_temper)

    def answer(self, a: int | None, choices: list[int], eps: float) -> np.ndarray:
        """The person's answer to "which subgoal next?": a choice, or None for "something else"."""
        hit = (self.n == a) if a is not None else ~np.isin(self.n, choices)
        return self._weigh(np.where(hit, 1 - eps, eps / len(choices)))

    def robot_done(self, v: int) -> None:
        """The robot finished v: it is complete, and says nothing about the person."""
        self.C |= 1 << v
        self.claimed &= ~(1 << v)

    def detected(self, v: int) -> None:
        self.C |= 1 << v

    def completion(self, v: int) -> np.ndarray:
        pred = self._weigh(np.where(self.g == v, 1 - self.cfg.eps_done, self.cfg.eps_done))
        self.C |= 1 << v
        return pred
