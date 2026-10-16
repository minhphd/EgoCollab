"""BIP, Bayesian inverse planning of a person's task, as the robot runs it. All of it is in this file; each piece sits
under the equation it computes.

The task is a set of subgoals V ("put the cup in the cabinet"). Some must come before others (u -> v: v cannot start
until u is done); the robot does not know which: the orderings form an unknown graph G. Gemini (S2) writes, once per
task, the probability of each ordering, P(G), and each subgoal as a short list of steps a recognizer can see, R_v
(pick(cup) -> place(cup, cabinet)); a subgoal is done once its last required step is. From what it sees of the person,
the robot infers G and where the person is in the task:
    o_t   a clip (the recognizer's verb probabilities, and the object and receptacle a detector read in it), a gaze
          fixation (an object among those in view), or one of the robot's own subgoals done

    bel = Belief(task, Params())          task: {"title", "nodes", "labels", "objects": {node: [object, target]}}
    bel.clip(t, verbs, obj, receptacle, visible); bel.fixation(t, obj, visible); bel.robot_done(t, v); bel.claim(t, vs)
    bel.current(), bel.next(), bel.done()  P(g = v), P(n = v), P(v done) for every subgoal v
    v = decide(bel, t)                    the subgoal the robot should take now (an index), or None to wait
    tell_value(p_missed, p_seek, P)       > 0: telling the person about a move is worth their time

Per particle: g the subgoal the person is on, b the probabilities of the one they plan next (kept exactly: every
fixation updates them), k the steps done of every subgoal (a progress vector; S and D, the subgoals started and done,
follow from it). Shared by all particles: C the subgoals the robot has
done, Q those it has claimed.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from functools import reduce
from operator import or_
from pathlib import Path

import numpy as np

VERBS = ("pick", "place", "open", "close", "other")  # the recognizer's classes; "other": no step completed in the clip


@dataclass
class Params:
    particles: int = 64  # M particles per graph
    pool: int = 64  # K graphs sampled from the prior
    seed: int = 0
    # the person model P(z_t | z_t-1, G): counts on HD-EPIC's annotated videos of the training and validation
    # participants, 3 s clips (helper/collab/fit.py, data/collab/fit/person_clip3.json); lambda, upsilon and kappa are
    # not annotated there, so they stay hand-set
    s_done: float = 0.29  # P(a clip leaves the current subgoal) once its required steps are done
    s_mid: float = 0.044  # ... before that (an interruption)
    adv: float = 0.42  # a: P(a clip completes a step of the subgoal the person is on, given one remains)
    adv2: float = 0.36  # P(it completes the next step too), given one
    plan: float = 0.5  # lambda: P(a switch goes to the planned next subgoal)
    resume: float = 3.0  # rho: pi weight of a started subgoal (a ready one: 1)
    delta: float = 0.40  # pi weight of a subgoal whose required predecessors are not done
    claimed: float = 0.3  # kappa: pi times this for a subgoal the robot claimed (the person may not know it is)
    undo: float = 0.01  # upsilon: P(a clip undoes a subgoal the person did)
    # the sensors
    clip: float = 3.0  # s per clip event
    verbs: tuple = VERBS  # the recognizer's classes, "other" (no action) last; pick and place keep their names
    verb_prior: tuple = (0.2, 0.2, 0.2, 0.2, 0.2)  # the class prior the recognizer trained under (class-balanced)
    background: tuple | None = None  # P(verb read | a clip that completes no listed step); None: all on "other"
    obj_acc: float = 0.7  # P(the detector reads the right object)
    # gaze, uniform saliency: maximum likelihood on 36,562 fixations of HD-EPIC's training participants (fit.py gaze=1)
    gamma_cur: float = 1.3  # log boost of the current subgoal's objects
    gamma_next: float = 0.14  # ... of the planned next subgoal's
    eps: float = 0.18  # share of fixations that follow neither, spread over what is in view
    # S2: Gemini writes the prior, the step lists, and revisions
    surprise: float = 1.4  # nats: revise when a clip is this much less likely under the pool than without orderings
    cooldown: float = 20.0  # s between revisions
    max_calls: int = 4  # revisions per episode
    revisions: int = 3  # graphs per revision
    model: str = "gemini-3.5-flash-lite"
    cache: str = str(Path(__file__).resolve().parents[2] / "data/collab/prior")
    # the decision
    beta_rel: float = 1.0  # c: a conflict costs this many subgoal durations
    beta_prec_rel: float = 2.0  # c_prec: an ordering violation costs this many subgoal durations
    slowdown: float = 3.0  # robot time per unit of person time on a subgoal (Stretch drives at 0.3 m/s, people 1.2)
    notice: float = 2.0  # s until the person notices the robot on a subgoal and leaves it
    rem_curve: tuple = (1.0, 0.798, 0.654, 0.546, 0.483, 0.492, 0.467, 0.446, 0.493, 0.54, 0.608, 0.877, 0.797)
    rem_step: float = 0.25  # E[time left | time spent x] on a subgoal, in durations, at x = 0, rem_step, ... (PARTNR)
    # telling
    search: float = 20.0  # s a person loses looking for an object where it no longer is
    talk: float = 3.0  # s of the person's time to hear a sentence


def bit(x) -> np.ndarray:
    """1 << x where x >= 0, else 0 (x = -1 means no subgoal)."""
    x = np.asarray(x)
    return np.where(x >= 0, np.int64(1) << np.maximum(x, 0), np.int64(0))


def has(mask, v) -> np.ndarray:
    """Whether subgoal v (an index, or an array of them; -1 = none) is in mask."""
    v = np.asarray(v)
    return (v >= 0) & ((np.asarray(mask) >> np.maximum(v, 0)) & 1).astype(bool)


# -- the task graph G --------------------------------------------------------------------------------------------------
# P(G) = prod_listed p_uv^[uv in G] (1 - p_uv)^[uv not in G] prod_(uv in G, not listed) 0.02, p_uv from Gemini (S2).

def ancestors(n: int, edges) -> np.ndarray:
    """anc[v] = bit mask of every subgoal that must be done before v (transitive); raises on a cycle."""
    anc = [0] * n
    for u, v in edges:
        anc[v] |= 1 << u
    for _ in range(n):
        new = [a | reduce(or_, (anc[u] for u in range(n) if a >> u & 1), 0) for a in anc]
        if new == anc:
            break
        anc = new
    if any(anc[v] >> v & 1 for v in range(n)):
        raise ValueError("cycle")
    return np.array(anc, dtype=np.int64)


def log_prior(p: dict, g: frozenset) -> float:
    """log P(G); orderings Gemini did not list get p = 0.02."""
    return sum(np.log(q if e in g else 1 - q) for e, q in p.items()) + np.log(0.02) * len(g - p.keys())


def sample_pool(size: int, n: int, p: dict, rng) -> list[frozenset]:
    """The most probable acyclic graph and up to size - 1 distinct samples from P(G)."""
    def build(order, keep) -> frozenset:
        g = set()
        for e in order:
            if keep(e):
                try:
                    ancestors(n, g | {e})
                    g.add(e)
                except ValueError:  # would close a cycle
                    pass
        return frozenset(g)
    pool = {build(sorted(p, key=p.get, reverse=True), lambda e: p[e] >= 0.5)}
    for _ in range(size * 20):
        if len(pool) >= size:
            break
        order = [list(p)[i] for i in rng.permutation(len(p))]
        pool.add(build(order, lambda e: rng.random() < p[e]))
    return sorted(pool, key=sorted)


# -- the step lists R_v ------------------------------------------------------------------------------------------------
# A step: {"verb", "obj", "target" (a place's receptacle, else None), "optional"}; a subgoal: {"steps", "undo"}, the
# undo being the step that would reverse it. The subgoal is done once its last step that is not optional is.

def required(steps: list[dict]) -> int:
    """How many of the steps must be done for the subgoal to be done."""
    return max((j + 1 for j, s in enumerate(steps) if not s["optional"]), default=len(steps))


def fallback(obj: str, target: str, closed: bool = False) -> dict:
    """pick(o) -> place(o, f), with an optional open and close of f when it has a door."""
    pick, place = {"verb": "pick", "obj": obj, "target": None, "optional": False}, \
        {"verb": "place", "obj": obj, "target": target, "optional": False}
    door = lambda v: {"verb": v, "obj": target, "target": None, "optional": True}
    return {"steps": [pick, door("open"), place, door("close")] if closed else [pick, place],
            "undo": {"verb": "pick", "obj": obj, "target": None, "optional": False}}


def checked(r: dict | None, vocab: set, verbs=VERBS[:4]) -> dict | None:
    """A step list from Gemini in the internal form, or None if any verb or object is outside the vocabulary."""
    try:
        steps = [{"verb": s["verb"], "obj": s["object"], "target": s.get("target") or None, "optional": bool(s.get("optional"))}
                 for s in r["steps"]]
        u = r.get("undo")
        undo = {"verb": u["verb"], "obj": u["object"], "target": None, "optional": False} if u and u.get("verb") else None
    except (KeyError, TypeError):
        return None
    ok = lambda s: s["verb"] in verbs and s["obj"] in vocab and (s["target"] is None or s["target"] in vocab) and \
        (s["verb"] != "place" or s["target"] is not None)
    if not steps or all(s["optional"] for s in steps) or not all(map(ok, steps)) or (undo and not ok(undo)):
        return None
    return {"steps": steps, "undo": undo}


def objects_of(r: dict) -> set:
    """O(v): every object and target the step list names."""
    return {x for s in r["steps"] for x in (s["obj"], s["target"]) if x}


# -- S1: one particle filter per graph ---------------------------------------------------------------------------------
class Filter:
    """P(g, k | G_k, o_1:t) by M particles for each graph G_k, each with b = P(n | its history), its plan's
    distribution, kept exactly; and P(G_k | o_1:t) prop. P(G_k) Z_k, where the evidence Z_k multiplies each
    observation's predictive probability under G_k."""

    def __init__(self, P: Params, R: list[dict], graphs: list[frozenset], log_prior_: list[float], seed: int):
        self.P, self.R, self.N, self.rng = P, R, len(R), np.random.default_rng(seed)
        self.graphs, self.log_prior = list(graphs), np.array(log_prior_, float)
        self.anc = np.stack([ancestors(self.N, g) for g in graphs])  # [K, N]
        self.Kv = np.array([len(r["steps"]) for r in R])  # steps per subgoal
        self.Kreq = np.array([required(r["steps"]) for r in R])  # ... that must be done
        self.O = [objects_of(r) for r in R]
        K, M = len(graphs), P.particles
        self.g, self.b = np.full((K, M), -1), np.zeros((K, M, self.N))
        self.k = np.zeros((K, M, self.N), np.int64)
        self.logW, self.logZ = np.full((K, M), -np.log(M)), np.zeros(K)
        self.C = self.Q = 0
        self.v = np.arange(self.N)
        self._masks()
        w = self.options(self.S)  # the person starts on some subgoal, and has a next one in mind
        self.g = np.where(w.sum(-1) > 0, self._draw(w), -1)
        self._replan(np.ones_like(self.g, bool))

    # read-outs
    def graph_weights(self) -> np.ndarray:
        w = np.exp(self.log_prior + self.logZ - (self.log_prior + self.logZ).max())
        return w / w.sum()

    def joint(self) -> np.ndarray:
        """W(G, i): the weight of each (graph, particle)."""
        W = np.exp(self.logW - self.logW.max(1, keepdims=True))
        return self.graph_weights()[:, None] * W / W.sum(1, keepdims=True)

    def marginal(self, x: np.ndarray) -> np.ndarray:
        """P(x = v) for a particle variable x (g)."""
        J, ok = self.joint(), x >= 0
        return np.bincount(x[ok], J[ok], minlength=self.N)

    def plan_probs(self) -> np.ndarray:
        """P(n = v): the planned next subgoal, over graphs and particles."""
        return np.einsum("km,kmn->n", self.joint(), self.b)

    def done_probs(self) -> np.ndarray:
        J, Dc = self.joint(), self.D | self.C
        return np.array([(J * has(Dc, v)).sum() for v in range(self.N)])

    def edge_probs(self) -> np.ndarray:
        """E[u, v] = P(u must be done before v), over the graphs."""
        return np.einsum("k,kuv->uv", self.graph_weights(), (self.anc[:, None, :] >> self.v[:, None]) & 1)

    def evidence(self) -> float:
        """log P(o_1:t) over the pool: log sum_G P(G) Z_G, with P(G) normalized over the pool."""
        a = self.log_prior - np.logaddexp.reduce(self.log_prior)
        return float(np.logaddexp.reduce(a + self.logZ))

    # the person model
    def options(self, S: np.ndarray, also=None) -> np.ndarray:
        """pi_G(v), unnormalized [K, M, N]:  0 if v is done; rho if v was started; 1 if every predecessor of v in G is
        done; delta otherwise; times kappa if the robot claimed v. `also` (a subgoal being worked on) counts as done."""
        Dc = self.D | self.C
        sat = Dc | (bit(also) if also is not None else 0)
        ready = (self.anc[:, None, :] & ~sat[..., None]) == 0
        w = np.where(has(Dc[..., None], self.v), 0.0,
                     np.where(has(S[..., None], self.v), self.P.resume, np.where(ready, 1.0, self.P.delta)))
        return np.where(has(self.Q, self.v), self.P.claimed * w, w)

    def _reads(self, Lv: np.ndarray, obj, rec, n_vis: int):
        """The clip's likelihood under each step of each subgoal [N, max K_v], under each undo [N], and under no step:
        P(clip | verb c_s) P(object read | b_s) P(receptacle read | f_s); the reads are uniform over what is in view
        for a step that names no such thing, and for no step, whose verb is "other" or drawn from the background."""
        acc, n = self.P.obj_acc, max(2, n_vis)
        uni = (1 / n) ** ((obj is not None) + (rec is not None))
        conf = lambda read, true: 1.0 if read is None else (acc if read == true else (1 - acc) / (n - 1)) if true else 1 / n
        like = lambda s: Lv[self.P.verbs.index(s["verb"])] * conf(obj, s["obj"]) * conf(rec, s["target"])
        Lst = np.zeros((self.N, max(1, self.Kv.max())))
        for u, r in enumerate(self.R):
            Lst[u, :len(r["steps"])] = [like(s) for s in r["steps"]]
        Lund = np.array([like(r["undo"]) if r["undo"] else 0.0 for r in self.R])
        L0 = Lv[-1] if self.P.background is None else float(Lv @ np.asarray(self.P.background))
        return Lst, Lund, L0 * uni

    def _outcomes(self, Lst: np.ndarray, L0: float):
        """For subgoal u at progress p, what the clip did on u: (probability, likelihood, new progress) for no step
        (1 - a), the next step (a (1 - a2)), or the next two (a a2); a next step that is optional may be skipped."""
        P, N = self.P, self.N
        prob, like, new = np.zeros((N, self.Kv.max() + 1, 7)), np.zeros((N, self.Kv.max() + 1, 7)), \
            np.zeros((N, self.Kv.max() + 1, 7), np.int64)
        for u, r in enumerate(self.R):
            opt, K = [s["optional"] for s in r["steps"]], len(r["steps"])
            nxt = lambda p: ([p] + ([p + 1] if p + 1 < K and opt[p] else [])) if p < K else []
            for p in range(K + 1):
                one = nxt(p)
                two = [(j, j2) for j in one for j2 in nxt(j + 1)]
                a, a2 = (P.adv if one else 0.0), (P.adv2 if two else 0.0)
                outs = [(1 - a, L0, p)] + [(a * (1 - a2) / len(one), Lst[u, j], j + 1) for j in one] + \
                       [(a * a2 / len(two), Lst[u, j2], j2 + 1) for _, j2 in two]
                for o, (pr, li, nw) in enumerate(outs):
                    prob[u, p, o], like[u, p, o], new[u, p, o] = pr, li, nw
        return prob, like, new

    def clip(self, Lv: np.ndarray, obj=None, rec=None, n_vis: int = 2) -> np.ndarray:
        """A clip with Lv[c] prop. P(clip | verb c) and the detector's reads.
        Transition  P(g' = u, advance | g, b, k) = [(1 - s) [u = g] + s sw(u)] x P(the clip did on u: no step, one, or
        two),  sw(u) = lambda b'(u) + (1 - lambda m) pi_G(u),
        b' the plan's probabilities on subgoals still open (not done, claimed or g) and m their sum: the plan n
        marginalized exactly. s = s_done if g's required steps are done, else s_mid (1 with no current subgoal); with
        probability upsilon the clip instead undoes a subgoal the person did, putting it one required step back. Each
        particle draws its move from T(m) L(m) and is weighed by sum_m T(m) L(m). A new g resets b to pi_G; a planned
        subgoal that is done or claimed leaves b, and an empty b is reset."""
        P, g, N, K, M = self.P, self.g, self.N, *self.g.shape
        Lst, Lund, L0 = self._reads(Lv, obj, rec, n_vis)
        prob, like, new = self._outcomes(Lst, L0)
        Dc = self.D | self.C
        s = np.where(g < 0, 1.0, np.where(has(Dc, g), P.s_done, P.s_mid))
        w = np.where(self.v == g[..., None], 0.0, self.options(self.S))
        pi = w / np.maximum(w.sum(-1, keepdims=True), 1e-300)
        open_b = self._open(self.b, g)
        switch = P.plan * open_b + (1 - P.plan * open_b.sum(-1, keepdims=True)) * pi
        stay = (self.v == g[..., None]).astype(float)
        on = np.where((w.sum(-1) > 0)[..., None], s[..., None] * switch + (1 - s)[..., None] * stay, stay)  # P(g' = u)
        person = self.k >= self.Kreq  # subgoals the person did, which a clip may undo
        mine = person.any(-1)
        T = on[..., None] * prob[self.v, self.k] * np.where(mine, 1 - P.undo, 1.0)[..., None, None]  # [K, M, N, 7]
        q = np.concatenate([(T * like[self.v, self.k]).reshape(K, M, -1),
                            np.where(person, P.undo / np.maximum(person.sum(-1, keepdims=True), 1), 0.0) * Lund], -1)
        idle = on.sum(-1) <= 0  # no subgoal and none left: the clip is off-task
        pred = self._weigh(np.where(idle, L0, q.sum(-1)))
        m = np.where(idle, 0, self._draw(np.where(idle[..., None], 1.0, q)))
        undo, u, o = m >= N * 7, np.minimum(m // 7, N - 1), m % 7
        u = np.where(undo, m - N * 7, u)
        p = np.take_along_axis(self.k, u[..., None], -1)[..., 0]
        k2 = np.where(undo, self.Kreq[u] - 1, new[u, p, o])
        g2 = np.where(undo | idle, g, u)
        np.put_along_axis(self.k, u[..., None], np.where(idle, p, k2)[..., None], -1)
        self._masks()
        self._replan(g2 != g, g2)  # a new current subgoal: a new plan
        self.b = self._open(self.b, g2)
        mass = self.b.sum(-1, keepdims=True)
        self.b = self.b / np.maximum(mass, 1e-300)
        self._replan(mass[..., 0] <= 0, g2)  # nothing of the plan left open
        self.g = g2
        self._resample()
        return pred

    def fixation(self, x: str, visible: list[str], sal=None) -> np.ndarray:
        """A fixation on x among the visible X:
        P(x | g, n, X) = (1 - eps) s(x) e^(gamma_cur [x in O(g)] + gamma_next [x in O(n)]) / sum_x' (same) + eps / |X|,
        s the saliency of each visible object (uniform without a saliency model). Each particle is weighed by
        sum_n b(n) P(x | g, n, X), and its plan updated, b(n) prop. b(n) P(x | g, n, X): evidence for the next subgoal
        accumulates while the person is on the current one."""
        P, X = self.P, list(dict.fromkeys([*visible, x]))
        s = np.ones(len(X)) if sal is None else np.asarray(sal, float)
        inO = np.array([[y in O for y in X] for O in self.O] + [[False] * len(X)])  # [N + 1, |X|]; row N: no subgoal
        Sv, pair = inO @ s, (inO[:, None, :] & inO[None, :, :]) @ s
        e1, e2 = np.exp(P.gamma_cur), np.exp(P.gamma_next)
        g, i = self.g, X.index(x)
        num = s[i] * np.exp(P.gamma_cur * inO[g, i])[..., None] * np.exp(P.gamma_next * inO[:, i])  # [K, M, N + 1]
        den = s.sum() + (e1 - 1) * Sv[g][..., None] + (e2 - 1) * Sv + (e1 - 1) * (e2 - 1) * pair[g]
        like = (1 - P.eps) * num / den + P.eps / len(X)
        bp = np.concatenate([self.b, 1 - self.b.sum(-1, keepdims=True)], -1)  # the last column: no plan
        inc = (bp * like).sum(-1)
        self.b = (bp * like / np.maximum(inc[..., None], 1e-300))[..., :-1]
        pred = self._weigh(inc)
        self._resample()
        return pred

    def robot_done(self, v: int) -> None:
        self.C |= 1 << v
        self.Q &= ~(1 << v)

    # particle mechanics
    def _masks(self) -> None:
        bits = np.int64(1) << self.v
        self.S = ((self.k > 0) * bits).sum(-1)
        self.D = ((self.k >= self.Kreq) * bits).sum(-1)

    def _replan(self, redraw: np.ndarray, g=None) -> None:
        """b = pi_G with the current subgoal counted as done (all zero if none is possible), where `redraw`."""
        g = self.g if g is None else g
        w = np.where(self.v == g[..., None], 0.0, self.options(self.S, also=g))
        self.b = np.where(redraw[..., None], w / np.maximum(w.sum(-1, keepdims=True), 1e-300), self.b)

    def _open(self, b: np.ndarray, g: np.ndarray) -> np.ndarray:
        """b on the subgoals a plan can still name: not done, not claimed by the robot, not g."""
        shut = has((self.D | self.C)[..., None], self.v) | has(self.Q, self.v) | (self.v == g[..., None])
        return np.where(shut, 0.0, b)

    def _draw(self, W: np.ndarray) -> np.ndarray:
        c = W.cumsum(-1)
        return np.minimum((c < self.rng.random(W.shape[:-1])[..., None] * c[..., -1:]).sum(-1), W.shape[-1] - 1)

    def _weigh(self, inc: np.ndarray) -> np.ndarray:
        """w_i *= inc_i; Z_k *= sum_i w_i inc_i (returned per graph)."""
        W = np.exp(self.logW - self.logW.max(1, keepdims=True))
        pred = (W / W.sum(1, keepdims=True) * inc).sum(1)
        self.logZ += np.log(np.maximum(pred, 1e-300))
        self.logW += np.log(np.maximum(inc, 1e-300))
        return pred

    def _resample(self) -> None:
        """Systematic resampling of a graph's particles when fewer than M/2 are effective."""
        W = np.exp(self.logW - self.logW.max(1, keepdims=True))
        W /= W.sum(1, keepdims=True)
        M = W.shape[1]
        for k in np.flatnonzero(1 / (W ** 2).sum(1) < M / 2):
            i = np.minimum(np.searchsorted(W[k].cumsum(), (self.rng.random() + np.arange(M)) / M), M - 1)
            for x in (self.g, self.b, self.k):
                x[k] = x[k, i]
            self.logW[k] = -np.log(M)
        self._masks()


# -- S2: Gemini writes the prior, the step lists, and revisions ---------------------------------------------------------
MEANING = ("A required ordering A > B means subgoal A must be complete before B can start: the instruction asks for "
           "that order (for example \"first ... then ...\"), or the task cannot be done otherwise. A habit or a "
           "convenient order is not required.")
STEP_RULES = ("Write each subgoal as the short list of steps a recognizer can see, in the order a person does them. "
              "Every step is one verb ({verbs}) with one object, and a place also names its target. "
              "Mark a step optional if it happens only sometimes. Give the undo: the step that would reverse the "
              "subgoal once it is done. Use only the object names listed.")


def gemini(text: str, schema: dict, P: Params) -> tuple[dict | None, float]:
    """Gemini's JSON reply (None if it fails) and the call's latency in s."""
    from helper.llm.gemini import Gemini, generate
    reply, _, latency = generate(text, schema=schema, cfg=Gemini(model=P.model, thinking=""))
    try:
        return json.loads(reply), latency
    except ValueError:
        return None, latency


def edge_schema(ids: list[str], with_p: bool) -> dict:
    sid = {"type": "string", "enum": ids}
    props = {"before": sid, "after": sid} | ({"p": {"type": "number"}} if with_p else {})
    return {"type": "array", "items": {"type": "object", "required": list(props), "properties": props}}


def steps_schema(ids: list[str], vocab: list[str], verbs=VERBS[:4]) -> dict:
    obj = {"type": "string", "enum": vocab}
    step = {"type": "object", "required": ["verb", "object", "optional"], "properties": {
        "verb": {"type": "string", "enum": list(verbs)}, "object": obj, "target": obj, "optional": {"type": "boolean"}}}
    return {"type": "array", "items": {"type": "object", "required": ["id", "steps", "undo"], "properties": {
        "id": {"type": "string", "enum": ids}, "steps": {"type": "array", "items": step},
        "undo": {"type": "object", "required": ["verb", "object"], "properties": {
            "verb": {"type": "string", "enum": list(verbs)}, "object": obj}}}}}


class S2:
    """Gemini reads the task (subgoals under shuffled IDs, so their listing order leaks nothing): once for the prior
    p_uv and once for the step lists, both cached; on surprise, for revised graphs or step lists."""

    def __init__(self, task: dict, P: Params):
        self.task, self.P = task, P
        order = random.Random(f"{P.seed}-{task['title']}").sample(range(len(task["nodes"])), len(task["nodes"]))
        self.id = {v: f"S{k + 1}" for k, v in enumerate(order)}  # subgoal index -> shuffled ID
        self.back = {s: v for v, s in self.id.items()}
        self.ids = sorted(self.back, key=lambda s: int(s[1:]))
        self.vocab = sorted({x for v in task["nodes"] for x in task["objects"][v]} | set(task.get("vocab", ())))
        self.verbs = P.verbs[:-1]
        self.rules = STEP_RULES.format(verbs="pick, place, open or close" if self.verbs == VERBS[:4] else "one of the verbs listed")
        self.names = f"Verbs: {', '.join(self.verbs)}\n" * (self.verbs != VERBS[:4]) + f"Object names: {', '.join(self.vocab)}"

    def header(self) -> list[str]:
        return (["A robot assistant watches a person do a household task through the person's smart glasses.",
                 f"Instruction: {self.task['title']}", "Subgoals (each is completed once):"]
                + [f"{s}: {self.task['labels'][self.back[s]]}" for s in self.ids])

    def path(self, name: str) -> Path:
        return Path(self.P.cache) / self.P.model / f"{self.task['title']}{name}_{self.P.seed}.json".replace(" ", "_")

    def prior(self) -> dict:
        """{(u, v): p_uv}, clipped to [0.02, 0.98], cached per task, model and seed."""
        path = self.path("")
        if path.exists():
            return {(u, v): p for u, v, p in json.loads(path.read_text())}
        text = "\n".join(self.header() + [MEANING, "", "List every ordering that may be required, with p your probability "
                                          "that it is required. Include doubtful ones with a low p; leave out orderings "
                                          "that are clearly free."])
        reply, _ = gemini(text, {"type": "object", "required": ["edges"], "properties": {"edges": edge_schema(self.ids, True)}}, self.P)
        p = {(self.back[e["before"]], self.back[e["after"]]): min(0.98, max(0.02, float(e["p"])))
             for e in (reply or {}).get("edges", []) if e["before"] != e["after"]}
        if reply is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps([[u, v, q] for (u, v), q in p.items()]))
        return p

    def steps(self) -> list[dict]:
        """R_v for every subgoal; one Gemini fails the vocabulary check for, or leaves out, falls back to pick(o_v) ->
        place(o_v, f_v)."""
        path = self.path("_steps" if self.verbs == VERBS[:4] else f"_steps{len(self.verbs)}verbs")
        if path.exists():
            reply = json.loads(path.read_text())
        else:
            text = "\n".join(self.header() + ["", self.rules, self.names])
            reply, _ = gemini(text, {"type": "object", "required": ["subgoals"],
                                     "properties": {"subgoals": steps_schema(self.ids, self.vocab, self.verbs)}}, self.P)
            if reply is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(reply))
        got = {self.back[r["id"]]: checked(r, set(self.vocab), self.verbs) for r in (reply or {}).get("subgoals", []) if r.get("id") in self.back}
        return [got.get(v) or self.default(v) for v in range(len(self.task["nodes"]))]

    def default(self, v: int) -> dict:
        o, f = self.task["objects"][self.task["nodes"][v]][:2]
        return fallback(o, f, bool(self.task.get("closed", [False] * (v + 1))[v]))

    def show(self, R: list[dict], v: int) -> str:
        step = lambda s: f"{s['verb']}({s['obj']}{', ' + s['target'] if s['target'] else ''}){' optional' if s['optional'] else ''}"
        return " -> ".join(map(step, R[v]["steps"])) + (f"; undo {step(R[v]['undo'])}" if R[v]["undo"] else "")

    def revise(self, top: list[tuple[frozenset, float]], R: list[dict], done: np.ndarray, log: list[str],
               event: str) -> tuple[list[frozenset], dict, float]:
        """Corrected graphs for the leading ones, corrected step lists {v: R_v}, and the call's latency."""
        name = lambda v: self.id[v]
        lines = self.header() + [MEANING, "", "Each subgoal's steps, and the robot's P(done) for it now:"]
        lines += [f"{s}: {self.show(R, self.back[s])}; P(done) {done[self.back[s]]:.2f}" for s in self.ids]
        lines += ["", "The robot keeps several hypotheses of the required orderings. The most probable now:"]
        lines += [f"Graph {k + 1} (probability {w:.2f}): " + (", ".join(f"{name(u)} > {name(v)}" for u, v in sorted(g)) or "none")
                  for k, (g, w) in enumerate(top)]
        lines += [f"What the robot has read so far (its own reads, which may hold mistakes): {'; '.join(log[-30:]) or 'nothing yet'}",
                  f"Surprise: {event}", "", f"Return up to {self.P.revisions} corrected graphs, most plausible first, "
                  "each listing every required ordering, or corrected step lists for the subgoals whose steps do not "
                  "fit what the person does, or both. If the person started B while A was not complete, A > B is not "
                  f"required. {self.rules} {self.names}"]
        schema = {"type": "object", "properties": {
            "graphs": {"type": "array", "maxItems": self.P.revisions, "items": {"type": "object", "required": ["edges"],
                                                                               "properties": {"edges": edge_schema(self.ids, False)}}},
            "subgoals": steps_schema(self.ids, self.vocab, self.verbs)}}
        reply, latency = gemini("\n".join(lines), schema, self.P)
        graphs = [frozenset((self.back[e["before"]], self.back[e["after"]]) for e in g["edges"] if e["before"] != e["after"])
                  for g in (reply or {}).get("graphs", [])]
        steps = {self.back[r["id"]]: c for r in (reply or {}).get("subgoals", [])
                 if r.get("id") in self.back and (c := checked(r, set(self.vocab), self.verbs))}
        return graphs, steps, latency


# -- the belief: S1 on every observation, S2 on surprise ---------------------------------------------------------------
@dataclass
class Belief:
    task: dict
    P: Params = field(default_factory=Params)

    def __post_init__(self):
        N, P = len(self.task["nodes"]), self.P
        self.N, self.s2 = N, S2(self.task, P)
        self.p, self.R = self.s2.prior(), self.s2.steps()
        pool = sample_pool(P.pool, N, self.p, np.random.default_rng(P.seed))
        self.F = Filter(P, self.R, pool, [log_prior(self.p, g) for g in pool], P.seed)
        self.Ref = Filter(P, self.R, [frozenset()], [0.0], P.seed + 1)  # reference: no orderings, for the surprise
        self.history, self.log, self.calls, self.pending, self.last_call = [], [], [], None, -np.inf
        self.read, self.switches = None, []  # the leading current subgoal, and when it changed

    # observations
    def clip(self, t: float, verbs, obj: str | None = None, receptacle: str | None = None, visible=()) -> None:
        """verbs: the recognizer's calibrated P(c | clip) over P.verbs (a sequence, or a dict); divided by the prior it
        trained under, it is P(clip | c) up to a constant."""
        q = np.maximum(np.array([verbs.get(c, 0.0) for c in self.P.verbs] if isinstance(verbs, dict) else verbs, float), 1e-6)
        Lv = q / np.asarray(self.P.verb_prior)
        self._feed(t, "clip", (Lv / Lv.max(), obj, receptacle, max(2, len(set(visible) | {obj, receptacle} - {None}))))
        top = self.P.verbs[int(np.argmax(q))]
        if top != "other":
            self.log.append(f"{t:.0f} s read {top}" + (f" {obj}" if obj else "") + (f" into {receptacle}" if receptacle else ""))

    def fixation(self, t: float, obj: str, visible: list[str], saliency=None) -> None:
        self._feed(t, "fixation", (obj, list(visible), saliency))

    def robot_done(self, t: float, v: int) -> None:
        self._feed(t, "robot_done", v)

    def claim(self, t: float, steps: set[int]) -> None:
        self._feed(t, "claim", sum(1 << v for v in steps))

    def _apply(self, F: Filter, kind: str, x):
        if kind == "claim":
            F.Q = x
            return None
        return {"clip": lambda a: F.clip(*a), "fixation": lambda a: F.fixation(*a), "robot_done": F.robot_done}[kind](x)

    def _feed(self, t: float, kind: str, x) -> None:
        if self.pending and self.pending[0] <= t:  # S2's corrections arrive after its latency
            self._add(*self.pending[1:])
            self.pending = None
        w = self.F.graph_weights()
        pred, ref = self._apply(self.F, kind, x), self._apply(self.Ref, kind, x)
        self.history.append((kind, x))
        if kind != "clip":
            return
        cur = int(np.argmax(self.F.marginal(self.F.g)))
        if cur != self.read:
            self.read = cur
            self.switches.append(t)
            self.log.append(f"{t:.0f} s working on {self.s2.id[cur]}")
        # surprise = log P(clip | no orderings) - log P(clip | pool)
        surprise = float(np.log(max(ref[0], 1e-300)) - np.log(max(float(w @ pred), 1e-300)))
        if (surprise > self.P.surprise and not self.pending and t - self.last_call >= self.P.cooldown
                and len(self.calls) < self.P.max_calls):
            self._revise(t, surprise)

    def _revise(self, t: float, surprise: float) -> None:
        F, w = self.F, self.F.graph_weights()
        top = [(F.graphs[k], float(w[k])) for k in np.argsort(-w)[:3]]
        cur, done = self.read, self.done() > 0.5
        lead = F.anc[int(np.argmax(w))]
        unmet = [self.s2.id[u] for u in range(self.N) if lead[cur] >> u & 1 and not done[u]]
        event = (f"the person appears to work on {self.s2.id[cur]}" + (f" while {', '.join(unmet)} "
                 f"{'is' if len(unmet) == 1 else 'are'} not complete" if unmet else "") + "; the hypotheses explain this poorly.")
        graphs, steps, latency = self.s2.revise(top, self.R, self.done(), self.log, event)
        self.last_call = t
        self.calls.append({"t": round(t, 2), "surprise": round(surprise, 2), "latency": round(latency, 2),
                           "graphs": len(graphs), "steps": sorted(steps)})
        self.pending = (t + latency, graphs, steps)

    def _replay(self, R: list[dict], graphs: list[frozenset], seed: int) -> Filter:
        """A fresh filter over `graphs` with step lists R, run on the whole history."""
        F = Filter(self.P, R, graphs, [log_prior(self.p, g) for g in graphs], seed)
        for kind, x in self.history:
            self._apply(F, kind, x)
        return F

    def _add(self, graphs: list[frozenset], steps: dict | None = None) -> None:
        """Corrected step lists replace the old ones if the pool explains the history better with them; new acyclic
        graphs join the pool after being scored on the whole history, so every graph's weight stays P(G) Z."""
        seed = self.P.seed + 100 * (len(self.calls) + 1)
        if steps:
            R2 = [steps.get(v, r) for v, r in enumerate(self.R)]
            F2 = self._replay(R2, self.F.graphs, seed)
            if F2.evidence() > self.F.evidence():
                self.R, self.F, self.Ref = R2, F2, self._replay(R2, [frozenset()], seed + 1)
                self.log.append("step lists corrected for " + ", ".join(self.s2.id[v] for v in sorted(steps)))
        new = []
        for g in graphs:
            try:
                ancestors(self.N, g)
            except ValueError:
                continue
            if g not in self.F.graphs and g not in new:
                new.append(g)
        if not new:
            return
        G, F = self._replay(self.R, new, seed + 2), self.F
        F.graphs += new
        for a in ("log_prior", "anc", "g", "b", "k", "logW", "logZ"):
            setattr(F, a, np.concatenate([getattr(F, a), getattr(G, a)]))
        F._masks()

    # read-outs
    def current(self) -> np.ndarray:
        return self.F.marginal(self.F.g)

    def next(self) -> np.ndarray:
        return self.F.plan_probs()

    def done(self) -> np.ndarray:
        return self.F.done_probs()


# -- the decision -------------------------------------------------------------------------------------------------------
def decide(bel: Belief, t: float, blocked=()) -> int | None:
    """The subgoal to take now, or None:  U(v) = P_exec(v) [(1 - Y_v) Delta(v) - c Y_v] - c_prec (1 - P_exec(v)),
    the best v if U(v) > 0.
      P_exec(v)  P(every required predecessor of v is done)
      Y_v        P(the person works on v while the robot would): v is their current subgoal, or their planned next
                 and they get to it within `notice` s (after that they see the robot on it and leave it)
      Delta(v)   the person's time saved: v's duration d, less any wait for the slower robot once the person runs out
                 of work that does not need v
    Durations d: the mean time between changes of the leading current subgoal (20 s before two changes, at least 3 s);
    the time left on the current one: d E[left | spent], from PARTNR's people. Subgoals done, claimed, started or
    `blocked` (by the caller) are never taken."""
    F, P = bel.F, bel.P
    gaps = np.diff(bel.switches)
    d = np.full(F.N, max(3.0, float(gaps.mean())) if len(gaps) > 1 else 20.0)
    spent = (t - bel.switches[-1]) / d[0] if bel.switches else 0.0
    remaining = np.maximum(1.0, d * np.interp(spent, np.arange(len(P.rem_curve)) * P.rem_step, P.rem_curve))
    d_robot = d * P.slowdown
    J, Dc = F.joint(), F.D | F.C
    p_exec = np.einsum("km,kmn->n", J, (F.anc[:, None, :] & ~Dc[..., None]) == 0)
    soon = np.where(F.g >= 0, remaining[np.maximum(F.g, 0)], 0.0)  # time until the person leaves their current subgoal
    Y = np.array([(J * ((F.g == v) + (F.g != v) * F.b[..., v] * (soon < min(d_robot[v], P.notice)))).sum() for v in range(F.N)])
    E, on, done = F.edge_probs(), F.marginal(F.g), F.done_probs()
    work = (on * np.minimum(d, remaining) + (1 - on) * d) * (1 - done) * ~has(F.Q, F.v)  # the person's work left
    free = np.array([((1 - E[v]) * work).sum() - work[v] for v in range(F.N)])  # ... that does not need v
    Delta = d - np.maximum(0.0, d_robot - free)
    c, c_prec = P.beta_rel * d[0], P.beta_prec_rel * d[0]
    U = p_exec * ((1 - Y) * Delta - c * Y) - c_prec * (1 - p_exec)
    started = np.array([(J * has(F.S, v)).sum() for v in range(F.N)])
    U = np.where(has(F.C | F.Q, F.v) | (done > 0.5) | (started > 0.5) | np.isin(F.v, list(blocked)), -np.inf, U)
    return int(np.argmax(U)) if U.max() > 0 else None


def tell_value(p_missed: float, p_seek: float, P: Params) -> float:
    """V(tell the person where o is) = P(they missed the move) P(they look for o soon) search - talk."""
    return p_missed * p_seek * P.search - P.talk
