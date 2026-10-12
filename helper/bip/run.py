"""Streaming replay of fast-slow BIP on HD-EPIC; score.py scores the estimates it writes.

Per video: S2 proposes the prior pool, then actions, fixations and completions stream through S1 in
time order. When the pool explains an action much worse than the empty graph does, S2 is called; its
graphs join the pool once its latency has passed, scored on the whole history. Every `grid` s the
belief is written out as an estimate.

    python -m helper.bip.run name=step_llm tier=step s2=llm
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import llm, perceive, stream
from .config import Config, parse
from .model import Filter, ancestors
from .tasks import clock


def log_prior(edges: dict, g: frozenset) -> float:
    """log P(G | description): independent orderings with the S2 probabilities, others at 0.02."""
    return sum(np.log(p if e in g else 1 - p) for e, p in edges.items()) + np.log(0.02) * len(g - edges.keys())


def sample_pool(cfg: Config, n: int, edges: dict, rng) -> list[frozenset]:
    """The most probable acyclic graph and up to `pool` - 1 distinct samples from the S2 prior."""
    def build(order, keep) -> frozenset:
        g = set()
        for e in order:
            if keep(e):
                try:
                    ancestors(n, g | {e})
                    g.add(e)
                except ValueError:
                    pass
        return frozenset(g)
    pool = {build(sorted(edges, key=edges.get, reverse=True), lambda e: edges[e] >= 0.5)}
    for _ in range(cfg.pool * 20):
        if len(pool) >= cfg.pool:
            break
        order = [list(edges)[i] for i in rng.permutation(len(edges))]
        pool.add(build(order, lambda e: rng.random() < edges[e]))
    return [frozenset(map(tuple, g)) for g in pool]


def repair(n: int, g: frozenset, starts: list[tuple[int, int]]) -> frozenset:
    """Drop the orderings into each observed start whose source was not yet satisfied (no model call)."""
    g = set(g)
    for v, sat in starts:
        anc = ancestors(n, g)
        g -= {(a, b) for a, b in g if b == v and (not sat >> a & 1 or anc[a] & ~sat)}
    return frozenset(g)


class Replay:
    def __init__(self, cfg: Config, st: dict):
        self.cfg, self.st, self.N = cfg, st, len(st["nodes"])
        self.rng = np.random.default_rng(cfg.seed)
        self.ids = llm.ids_for(cfg, st["nodes"], st["video"])
        self.index = {v: i for i, v in enumerate(st["nodes"])}
        p = llm.prior(cfg, st, self.ids)
        self.prior_call = {k: p.get(k) for k in ("latency", "usage", "error")}
        self.edges = {(self.index[u], self.index[v]): q for u, v, q in p["edges"]}
        if cfg.tier == "real":  # objects are detector names from Gemini's vocabulary
            vocab = llm.vocab(cfg, st, self.ids)
            self.gaze_set = [set(vocab.get(v, [])) for v in st["nodes"]]
            self.real = perceive.read(cfg, st["video"])
        else:  # objects are HD-EPIC's tracked instances, grouped by the names Gemini picked
            names = {i: set(p["objects"].get(v, [])) for i, v in enumerate(st["nodes"])}
            self.gaze_set = [{o for o, nm in st["objects"].items() if nm in names[i]} for i in range(self.N)]
        pool = sample_pool(cfg, self.N, self.edges, self.rng)
        self.F = Filter(cfg, self.N, pool, [log_prior(self.edges, g) for g in pool], cfg.seed)
        self.R = Filter(cfg, self.N, [frozenset()], [0.0], cfg.seed + 1)  # reference: anything goes
        self.history, self.log, self.starts, self.seen, self.read = [], [], [], set(), None
        self.calls, self.pending, self.last_call = [], None, -np.inf

    # -- events ------------------------------------------------------------------------------------
    def handled(self, H: list[str], X: list[str]) -> np.ndarray:
        """Likelihood of the handled objects H under each subgoal: w_hand U_O(v) + (1 - w_hand) U_X per object."""
        X, L = set(X) | set(H), np.ones(self.N)
        for x in H:
            L *= np.array([self.cfg.w_hand * float(x in O) / max(1, len(O & X)) + (1 - self.cfg.w_hand) / len(X)
                           for O in self.gaze_set])
        return L

    def events(self) -> list[tuple]:
        cfg, st, out = self.cfg, self.st, []
        if cfg.tier == "real":
            out += [(w["t"], 1, "action", self.handled(w["handled"], w["seen"]), w["handled"]) for w in self.real["windows"]]
        for k, s in enumerate(st["steps"] if cfg.tier != "real" else []):
            if cfg.tier == "step":
                if s["owner"] is None:
                    continue
                L = np.full(self.N, cfg.eps_step / max(1, self.N - 1))
                L[self.index[s["owner"]]] = 1 - cfg.eps_step
            else:
                sim = np.array(st["sim"][k])
                L = np.exp(cfg.beta_action * (sim - sim.max()))
            out.append((s["t"], 1, "action", L, k))
        if cfg.gaze:
            for f in (self.real if cfg.tier == "real" else st)["fixations"]:
                if f["t"] < st["t0"]:
                    continue
                X = set(f["visible"])
                u = np.array([float(f["object"] in O) / max(1, len(O & X)) for O in self.gaze_set])
                out.append((f["t"], 2, "fixation", (u, 1 / len(X)), f["object"]))
        if cfg.completion == "oracle":
            out += [(c["t"], 0, "completion", self.index[c["node"]], c["node"]) for c in st["completions"]]
        elif cfg.completion == "noisy":  # a detector that finds det_rate of the completions, det_delay s late on average
            rng = np.random.default_rng([cfg.seed, 7])
            for c in st["completions"]:
                if rng.random() < cfg.det_rate:
                    out.append((c["t"] + rng.exponential(cfg.det_delay), 0, "detected", self.index[c["node"]], c["node"]))
        return sorted(out, key=lambda e: (e[0], e[1]))

    def apply(self, F: Filter, kind: str, x):
        if kind == "action":
            return F.action(x)
        if kind == "fixation":
            return F.fixation(*x)
        if kind == "detected":
            return F.detected(x)
        return F.completion(x)

    # -- S2 -------------------------------------------------------------------------------------------
    def add(self, graphs: list[frozenset], t: float) -> int:
        have = set(self.F.graphs)
        new = []
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

    def fire(self, t: float, surprise: float) -> None:
        cfg, F = self.cfg, self.F
        self.last_call = t
        w = F.graph_weights()
        top = [(F.graphs[k], w[k]) for k in np.argsort(-w)[:3]]
        call = {"t": round(t, 2), "surprise": round(surprise, 2)}
        if cfg.s2 == "repair":
            graphs, latency = [repair(self.N, g, self.starts) for g, _ in top], 0.0
        elif cfg.s2 == "llm":
            cur = int(np.argmax(F.marginal(F.g)))
            unmet = [self.ids[self.st["nodes"][u]] for u in range(self.N)
                     if (F.anc[np.argmax(w)][cur] >> u & 1) and not (F.done_mask() >> u & 1)]
            event = (f"the person appears to work on {self.ids[self.st['nodes'][cur]]}"
                     + (f" while {', '.join(unmet)} {'is' if len(unmet) == 1 else 'are'} not complete" if unmet else "")
                     + "; the hypotheses explain this poorly.")
            graphs, usage, latency = llm.revise(cfg, self.st, self.ids, t, top, self.log, event, self.st["nodes"])
            call |= {"prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"],
                     "thinking_tokens": usage.get("thinking_tokens", 0)}
        else:
            return self.calls.append(call | {"graphs": 0})
        call |= {"latency": round(latency, 2), "t_ready": round(t + latency, 2), "graphs": len(graphs)}
        self.calls.append(call)
        self.pending = (t + latency, graphs, call)

    # -- loop -----------------------------------------------------------------------------------------
    def estimate(self) -> dict:
        F, nodes = self.F, self.st["nodes"]
        cur, nxt, E = F.marginal(F.g), F.marginal(F.n), F.edge_probs()
        return {"done": sorted(nodes[v] for v in range(self.N) if F.done_mask() >> v & 1),
                "current": nodes[int(np.argmax(cur))] if cur.sum() > 0 else None,
                "next": [nodes[v] for v in np.argsort(-nxt)[:3] if nxt[v] > 0],
                "edges": [[nodes[u], nodes[v], round(float(E[u, v]), 3)]
                          for u in range(self.N) for v in range(self.N) if E[u, v] >= 0.05]}

    def note(self, t: float) -> None:
        """Log S1's read of the current subgoal when it changes, for S2 and for repairs."""
        F, nodes = self.F, self.st["nodes"]
        cur = int(np.argmax(F.marginal(F.g)))
        if cur == self.read:
            return
        self.read = cur
        word = "back to" if cur in self.seen else "started"
        self.log.append(f"{clock(t - self.st['t0'])} {word} {self.ids[nodes[cur]]}")
        if cur not in self.seen:
            sat = F.done_mask() | sum(1 << v for v in self.seen) if self.cfg.rule == "started" else F.done_mask()
            self.starts.append((cur, sat))
            self.seen.add(cur)

    def run(self) -> dict:
        cfg, st = self.cfg, self.st
        t0, t1 = st["t0"], st["t1"]
        records = [{"t": t0, "prior": True, "latency": 0.0, "t_avail": t0, "prompt_tokens": 0, "completion_tokens": 0,
                    "ok": True, "estimate": self.estimate()}]
        grid, s1, trace = t0 + cfg.grid, [], []
        snap = lambda t, kind, info: trace.append({"t": round(t, 2), "kind": kind, "info": info,
                                                   "cur": self.F.marginal(self.F.g).round(3).tolist(),
                                                   "next": self.F.marginal(self.F.n).round(3).tolist(),
                                                   "edges": self.F.edge_probs().round(3).tolist()})
        for t, _, kind, x, info in self.events() + [(np.inf, 9, "end", None, None)]:
            while grid <= min(t, t1 + cfg.grid - 1e-9):
                g = min(grid, t1)
                records.append({"t": round(g, 2), "prior": False, "latency": 0.0, "t_avail": round(g, 2),
                                "prompt_tokens": 0, "completion_tokens": 0, "ok": True, "estimate": self.estimate()})
                grid += cfg.grid
            if kind == "end":
                break
            if self.pending and self.pending[0] <= t:
                self.pending[2]["added"] = self.add(self.pending[1], t)
                if cfg.trace:
                    snap(self.pending[0], "s2_added", self.pending[2])
                self.pending = None
            w = self.F.graph_weights()
            start = time.perf_counter()
            pred = self.apply(self.F, kind, x)
            s1.append(time.perf_counter() - start)
            ref = self.apply(self.R, kind, x)
            self.history.append((kind, x))
            if kind in ("completion", "detected"):
                self.log.append(f"{clock(t - t0)} {self.ids[st['nodes'][x]]} complete")
            if kind == "action":
                self.note(t)
                surprise = float(np.log(max(1e-300, ref[0])) - np.log(max(1e-300, w @ pred)))
                if surprise > cfg.surprise and not self.pending and t - self.last_call >= cfg.cooldown:
                    self.fire(t, surprise)
                    if cfg.trace:
                        snap(t, "s2_fired", self.calls[-1])
            if cfg.trace:
                snap(t, kind, info)
        return ({"trace": trace} if cfg.trace else {}) | {"video": st["video"], "name": cfg.name, "model": cfg.model, "ids": self.ids, "records": records,
                "s2_calls": self.calls, "prior_call": self.prior_call, "pool": len(self.F.graphs),
                "s1_ms": {"median": round(1e3 * float(np.median(s1)), 2), "p90": round(1e3 * float(np.percentile(s1, 90)), 2),
                          "events": len(s1)}}


def main() -> None:
    cfg = parse(sys.argv[1:])
    vids = stream.split(cfg)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    (cfg.run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=1))
    todo = [v for v in vids if not (cfg.run_dir / f"{v}.json").exists()]
    print(f"{cfg.name}: {len(todo)} of {len(vids)} videos to run, {cfg.streams} at a time", flush=True)

    def one(vid: str) -> None:
        out = Replay(cfg, stream.read(cfg, vid)).run()
        for path in (cfg.run_dir / f"{vid}.json", Path(cfg.episodes) / vid / f"pred_bip_{cfg.name}.json"):
            if path.parent.exists():
                path.write_text(json.dumps(out))
        print(f"{vid}: {len(out['records'])} estimates, {len(out['s2_calls'])} S2 calls, pool {out['pool']}, "
              f"S1 {out['s1_ms']['median']} ms", flush=True)

    with ThreadPoolExecutor(cfg.streams) as pool:
        for f in [pool.submit(one, v) for v in todo]:
            f.result()


if __name__ == "__main__":
    main()
