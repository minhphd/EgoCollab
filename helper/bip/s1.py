"""S1, the fast filter: P(g_t, n_t, D_t | G, o_1:t) for every graph G of the pool, and P(G | o_1:t).

State per particle: the person's current subgoal g, their planned next subgoal n, the started set S and the
done set D (bitmasks). C is the done set observed by everyone (scene graph completions, the robot's own work).
Each graph k holds M particles; its weight is prior times marginal likelihood, log P(G_k | l) + log Z_k, where
Z_k multiplies the predictive probability of every event (_weigh). Read-outs average over graphs and particles.

The person model, the terms of the joint (README.md):
    policy     pi(v | D, S, G, Q) prop. to 0 if v is done or the robot has claimed it (Q), `resume` if v is
               started, 1 if every predecessor of v in G is done, `delta` otherwise           (options)
    transition P(g_t | g_t-1, n_t-1): stay on an unfinished g with `stay`; otherwise switch, to the planned n
               with `plan`, else by pi; a new n is drawn from pi when g changes or n is used up   (action)
    action     P(a_t | g_t, s_t) = L[g_t], from observe.py / actions.py                          (action)
    gaze       P(x_t | g_t, n_t, X_t) = w_cur U_O(g)(x) + w_next U_O(n)(x) + w_noise / |X|      (fixation)
    completion P(v completes | g_t) = 1 - eps_done if g_t = v, else eps_done                     (completion)
Without observed completions, leaving an unfinished-looking subgoal marks it done with p_done (action).
"""

from __future__ import annotations

import numpy as np

from .config import Config
from .graph import ancestors


class Filter:
    def __init__(self, cfg: Config, n: int, graphs: list[frozenset], log_prior: list[float], seed: int = 0):
        self.cfg, self.N, self.rng = cfg, n, np.random.default_rng(seed)
        self.graphs, self.log_prior = list(graphs), np.array(log_prior, dtype=float)
        self.anc = np.stack([ancestors(n, g) for g in graphs])
        K, M = len(graphs), cfg.particles
        self.g = np.full((K, M), -1)  # current subgoal, -1 before any evidence
        self.n = np.full((K, M), -1)  # planned next subgoal
        self.S = np.zeros((K, M), dtype=np.int64)  # started subgoals
        self.D = np.zeros((K, M), dtype=np.int64)  # done in this particle's history (inferred)
        self.logW = np.full((K, M), -np.log(M))
        self.logZ = np.zeros(K)  # log marginal likelihood of each graph
        self.C = 0  # done for everyone (observed)
        self.claimed = 0  # the robot's current subgoals: the person picks around them
        self.v = np.arange(n)

    # -- read-outs ---------------------------------------------------------------------------------
    def graph_weights(self) -> np.ndarray:
        """P(G_k | o_1:t) over the pool."""
        lw = self.log_prior + self.logZ
        w = np.exp(lw - lw.max())
        return w / w.sum()

    def joint(self) -> np.ndarray:
        """Weight of every (graph, particle) pair."""
        W = np.exp(self.logW - self.logW.max(1, keepdims=True))
        return self.graph_weights()[:, None] * W / W.sum(1, keepdims=True)

    def done_probs(self) -> np.ndarray:
        """P(v is done) for every subgoal."""
        J, Dp = self.joint(), self.D | self.C
        return np.array([(J * ((Dp >> v) & 1)).sum() for v in range(self.N)])

    def done_mask(self) -> int:
        """Bitmask of the subgoals more likely done than not."""
        return sum(1 << v for v, p in enumerate(self.done_probs()) if p >= 0.5)

    def marginal(self, x: np.ndarray) -> np.ndarray:
        """Distribution of a particle variable (g or n) over subgoals."""
        J, ok = self.joint(), x >= 0
        return np.bincount(x[ok], J[ok], minlength=self.N)

    def edge_probs(self) -> np.ndarray:
        """P(u must precede v) as an N x N matrix, over the transitive closure of each graph."""
        clos = (self.anc[:, None, :] >> self.v[:, None]) & 1  # [K, u, v] = u is an ancestor of v
        return np.einsum("k,kuv->uv", self.graph_weights(), clos)

    # -- the person model ----------------------------------------------------------------------------
    def done(self, x: np.ndarray) -> np.ndarray:
        return (x >= 0) & (((self.D | self.C) >> np.maximum(x, 0)) & 1).astype(bool)

    def options(self, S: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
        """The policy pi, unnormalized, [K, M, N]: which subgoal the person would pick next."""
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

    # -- events: each returns the predictive probability per graph ----------------------------------------
    def action(self, L: np.ndarray) -> np.ndarray:
        """A person action with likelihood L[v] = P(a_t | g_t = v, s_t) for v < N and L[N] = P(a_t | no subgoal
        left), the null model: propagate g by the transition and weight by L (the proposal is the posterior over
        g_t), then redraw the planned next subgoal n. A particle with nothing possible keeps g and costs L[N]."""
        cfg, g, n = self.cfg, self.g, self.n
        L, null = L[:-1], L[-1]
        w = self.options(self.S)
        excl = np.where(self.v == g[..., None], 0.0, w)
        pi = excl / np.maximum(excl.sum(-1, keepdims=True), 1e-300)
        mine = lambda x: (x >= 0) & ((self.claimed >> np.maximum(x, 0)) & 1).astype(bool)
        plan_ok = (n >= 0) & ~self.done(n) & (n != g) & ~mine(n)
        plan = (self.v == n[..., None]).astype(float)
        switch = np.where(plan_ok[..., None], cfg.plan * plan + (1 - cfg.plan) * pi, pi)
        keep = (g >= 0) & ~self.done(g)
        T = np.where(keep[..., None], cfg.stay * (self.v == g[..., None]) + (1 - cfg.stay) * switch, switch)
        none = T.sum(-1) < 1e-12  # every subgoal done or claimed: the person's actions are off-task
        q = T * L
        pred = self._weigh(np.where(none, null, q.sum(-1)))
        g2 = np.where(none, g, self._sample(np.where(none[..., None], 1.0, q)))
        moved = g2 != self.g
        if cfg.completion != "oracle":  # leaving an unfinished-looking subgoal may mean it is done
            left = moved & (self.g >= 0) & ~self.done(self.g) & (self.rng.random(g2.shape) < cfg.p_done)
            self.D |= np.where(left, np.int64(1) << np.maximum(self.g, 0), 0)
        self.S |= np.where(g2 >= 0, np.int64(1) << np.maximum(g2, 0), 0)
        redraw = moved | (self.n == g2) | self.done(self.n) | (self.n < 0) | mine(self.n)
        w2 = self.options(self.S, extra=g2)
        w2 = np.where(self.v == g2[..., None], 0.0, w2)
        n2 = np.where(w2.sum(-1) > 0, self._sample(w2), -1)
        self.g, self.n = g2, np.where(redraw, n2, self.n)
        return pred

    def fixation(self, u: np.ndarray, noise: float) -> np.ndarray:
        """A gaze fixation on object x among the visible X: u[v] = U_O(v)(x), noise = 1 / |X|."""
        cfg = self.cfg
        U = np.append(u, 0.0)  # index -1 (no subgoal yet) has no gaze term
        L = cfg.w_cur * U[self.g] + cfg.w_next * U[self.n] + cfg.w_noise * noise
        return self._weigh(L ** cfg.gaze_temper)

    def completion(self, v: int) -> np.ndarray:
        """v was seen to become done by the person. With cfg.completion_lik, also evidence that they were on v;
        off by default, since a structure without v sees no such event and would escape its cost."""
        pred = (self._weigh(np.where(self.g == v, 1 - self.cfg.eps_done, self.cfg.eps_done)) if self.cfg.completion_lik
                else np.ones(len(self.graphs)))
        self.C |= 1 << v
        return pred

    def detected(self, v: int) -> None:
        """A late completion report: v is done, with no evidence about what the person is on now."""
        self.C |= 1 << v

    def robot_done(self, v: int) -> None:
        """The robot finished v: done, and says nothing about the person."""
        self.C |= 1 << v
        self.claimed &= ~(1 << v)

    def answer(self, a: int | None, choices: list[int], eps: float) -> np.ndarray:
        """The person's answer to "which subgoal next?": a choice, or None for "something else"."""
        hit = (self.n == a) if a is not None else ~np.isin(self.n, choices)
        return self._weigh(np.where(hit, 1 - eps, eps / len(choices)))

    # -- particle mechanics -------------------------------------------------------------------------
    def _sample(self, P: np.ndarray) -> np.ndarray:
        c = P.cumsum(-1)
        u = self.rng.random(P.shape[:-1])[..., None] * c[..., -1:]
        return np.minimum((c < u).sum(-1), self.N - 1)

    def _weigh(self, inc: np.ndarray) -> np.ndarray:
        """Multiply particle weights by inc [K, M]; adds each graph's predictive probability to log Z."""
        W = np.exp(self.logW - self.logW.max(1, keepdims=True))
        W /= W.sum(1, keepdims=True)
        pred = (W * inc).sum(1)
        self.logZ += np.log(np.maximum(pred, 1e-300))
        self.logW += np.log(np.maximum(inc, 1e-300))
        self._resample()
        return pred

    def _resample(self) -> None:
        """Systematic resampling of a graph's particles when their effective size drops below M / 2."""
        W = np.exp(self.logW - self.logW.max(1, keepdims=True))
        W /= W.sum(1, keepdims=True)
        ess = 1 / (W ** 2).sum(1)
        M = W.shape[1]
        for k in np.flatnonzero(ess < M / 2):
            idx = np.minimum(np.searchsorted(W[k].cumsum(), (self.rng.random() + np.arange(M)) / M), M - 1)
            self.g[k], self.n[k], self.S[k], self.D[k] = self.g[k, idx], self.n[k, idx], self.S[k, idx], self.D[k, idx]
            self.logW[k] = -np.log(M)
