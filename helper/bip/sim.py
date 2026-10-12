"""Closed-loop simulation of the robot assistant (RSS plan, simulation at scale).

A synthetic person does each HD-EPIC test task by its annotated graph: subgoals take their recorded
durations, the next one is drawn uniformly from those executable and not taken by the robot, and a
question costs them a pause. The robot sees a noisy step label every 3 s, gaze fixations drawn from
the mixture fitted on dev (fit.py), and completions; S1 tracks the belief, S2 repairs it on surprise
after the measured Gemini latency, and decide.py picks execute, ask or wait each second.

    python -m helper.bip.sim name=sim condition=ours seeds=3
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace

import numpy as np

from . import llm, stream
from .config import Config, parse
from .decide import Costs, decide
from .model import Filter, ancestors
from .run import log_prior, repair, sample_pool
from .tasks import load

CONDITIONS = {  # name: (robot acts, ask rule, gaze, anticipation of the next subgoal)
    "none": (False, "never", True, True),
    "ours": (True, "voi", True, True),
    "ours_nogaze": (True, "voi", False, True),
    "never_ask": (True, "never", True, True),
    "entropy_ask": (True, "entropy", True, True),
    "no_anticipation": (True, "never", True, False),
}
SIM = {"acc": 0.9, "fix_rate": 0.3, "lead": 2.0, "w_lead": 0.3, "react": 2.0, "pause": 3.0, "abort": 5.0,
       "ask_gap": 20.0, "s2_latency": 5.6, "horizon": 7200}


def durations(vid: str, nodes: list[str]) -> np.ndarray:
    task = load(3)[vid]
    d = dict.fromkeys(nodes, 0.0)
    for a, b, s in task.timeline.spans:
        if s:
            d[s] += b - a
    return np.clip([d[v] for v in nodes], 10, 600)


def episode(cfg: Config, vid: str, condition: str, seed: int, costs: Costs, trace: bool = False, acc: float = SIM["acc"]) -> dict:
    acts, ask_rule, use_gaze, anticipate = CONDITIONS[condition]
    st = stream.read(cfg, vid)
    nodes, N = st["nodes"], len(st["nodes"])
    index = {v: i for i, v in enumerate(nodes)}
    true_anc = ancestors(N, [(index[u], index[v]) for u, v in st["edges"]])
    d = durations(vid, nodes)
    p = llm.prior(cfg, st, llm.ids_for(cfg, nodes, vid))
    prior = {(index[u], index[v]): q for u, v, q in p["edges"]}
    names = sorted({nm for v in nodes for nm in p["objects"].get(v, [])}) or ["object"]
    O = [np.array([nm in p["objects"].get(v, []) for nm in names]) for v in nodes]
    world = np.random.default_rng([seed, 1, sum(map(ord, vid))])  # the person, shared across conditions
    rng = np.random.default_rng([seed, 2, sum(map(ord, vid))])
    pool = sample_pool(cfg, N, prior, rng)
    F = Filter(cfg, N, pool, [log_prior(prior, g) for g in pool], seed)
    R = Filter(cfg, N, [frozenset()], [0.0], seed + 1)
    history, starts, seen = [], [], set()

    def apply(G, kind, x):
        if kind == "action":
            return G.action(x)
        if kind == "fixation":
            return G.fixation(*x)
        if kind == "completion":
            return G.completion(x)
        if kind == "answer":
            return G.answer(*x)
        if kind == "robot_done":
            return G.robot_done(x)
        if kind == "detected":
            return G.detected(x)
        G.claimed = x  # "claimed": the robot's current subgoals

    def event(kind, x):
        history.append((kind, x))
        return apply(F, kind, x), apply(R, kind, x)

    # person state
    done = np.zeros(N, bool)
    progress = np.zeros(N)
    robot = {"v": -1, "left": 0.0, "since": 0.0, "fail": False}
    out = {"conflicts": 0, "violations": 0, "questions": 0, "s2_calls": 0, "robot_subgoals": 0, "redundant": 0}
    detections = []  # (time, subgoal) reports of a noisy completion detector

    def options(exclude: int = -1) -> list[int]:
        ok = [v for v in range(N) if not done[v] and v != exclude and not (true_anc[v] & ~int(sum(1 << u for u in range(N) if done[u])))]
        return [v for v in ok if not (v == robot["v"] and t - robot["since"] >= SIM["react"])]

    def pick(exclude: int = -1) -> int:
        o = options(exclude)
        return int(world.choice(o)) if o else -1

    def seen_on(v: int) -> None:
        """A conflict shows the robot what the person is doing."""
        L = np.full(N, 0.01 / max(1, N - 1))
        L[v] = 0.99
        event("action", L)

    def collide(g: int, pause: float) -> float:
        """The person starts the robot's subgoal before noticing the robot: both stop."""
        if g >= 0 and g == robot["v"]:
            out["conflicts"] += 1
            robot["v"] = -1
            event("claimed", 0)
            seen_on(g)
            return SIM["abort"]
        return pause

    t, pause, g = 0.0, 0.0, -1
    g = pick()
    n = pick(g)
    elapsed, pending, last_s2, last_ask, read = np.zeros(N), None, -np.inf, -np.inf, None
    ticks, right = [], []
    while not done.all() and t < SIM["horizon"]:
        t += 1
        decision = None
        # the person
        if pause > 0:
            pause -= 1
        elif g >= 0:
            progress[g] += 1
            if progress[g] >= d[g]:
                done[g] = True
                if cfg.completion == "oracle":
                    event("completion", g)
                elif cfg.completion == "noisy" and world.random() < cfg.det_rate:
                    detections.append((t + world.exponential(cfg.det_delay), g))
                g = n if n >= 0 and n in options() else pick()
                n = pick(g)
                pause = collide(g, pause)
        else:
            g = pick()
            n = pick(g)
            pause = collide(g, pause)
        # the robot's own subgoal
        if robot["v"] >= 0:
            robot["left"] -= 1
            if robot["left"] <= 0:
                v, robot["v"] = robot["v"], -1
                event("claimed", 0)
                if robot["fail"]:
                    out["violations"] += 1
                elif done[v]:
                    out["redundant"] += 1
                else:
                    done[v] = True
                    out["robot_subgoals"] += 1
                    event("robot_done", v)
                    if n == v:
                        n = pick(g)
        # observations
        for when, v in [x for x in detections if x[0] <= t]:
            event("detected", v)
            detections.remove((when, v))
        if g >= 0 and pause <= 0 and t % 3 == 0:
            o = g if world.random() < acc else int(world.choice([v for v in range(N) if v != g] or [g]))
            L = np.full(N, (1 - acc) / max(1, N - 1))
            L[o] = acc
            w = F.graph_weights()
            pred, ref = event("action", L)
            cur = int(np.argmax(F.marginal(F.g)))
            elapsed[cur] += 3
            if cur != read:
                read = cur
                if cur not in seen:
                    starts.append((cur, F.done_mask()))
                    seen.add(cur)
            surprise = float(np.log(ref[0]) - np.log(max(1e-300, w @ pred)))
            if acts and surprise > cfg.surprise and pending is None and t - last_s2 >= cfg.cooldown:
                wts = F.graph_weights()
                top = [F.graphs[k] for k in np.argsort(-wts)[:3]]
                pending, last_s2 = (t + SIM["s2_latency"], [repair(N, g2, starts) for g2 in top]), t
                out["s2_calls"] += 1
        if use_gaze and g >= 0 and world.random() < SIM["fix_rate"]:
            lead = progress[g] >= d[g] - SIM["lead"] and n >= 0
            w_next = SIM["w_lead"] if lead else cfg.w_next
            r = world.random()
            comp = O[g] if r < cfg.w_cur else O[n] if (r < cfg.w_cur + w_next and n >= 0) else np.ones(len(names), bool)
            x = int(world.choice(np.flatnonzero(comp))) if comp.any() else int(world.integers(len(names)))
            u = np.array([float(O[v][x]) / max(1, O[v].sum()) for v in range(N)])
            event("fixation", (u, 1 / len(names)))
        if pending and pending[0] <= t:
            add(F, pending[1], history, prior, cfg, N, apply)
            pending = None
        # the robot decides
        if acts and robot["v"] < 0 and pause <= 0:
            remaining = np.maximum(1.0, d - elapsed)
            rule = ask_rule if t - last_ask >= SIM["ask_gap"] else "never"
            kind, x, info = decide(F, d, remaining, costs, rule, anticipate)
            decision = [kind, x, round(info["value"], 1), round(info.get("voi", 0.0), 1)]
            if kind == "execute":
                ready = not (true_anc[x] & ~int(sum(1 << u for u in range(N) if done[u])))
                robot.update(v=x, left=d[x] * costs.robot_slowdown, since=t, fail=not ready)
                event("claimed", 1 << x)
                if x == g:
                    out["conflicts"] += 1
                    robot["v"], pause = -1, SIM["abort"]
                    event("claimed", 0)
                    seen_on(g)
            elif kind == "ask":
                out["questions"] += 1
                last_ask, pause = t, SIM["pause"]
                truthful = world.random() > costs.eps_answer
                a = n if truthful else int(world.integers(N))
                event("answer", (a if a in x else None, x, costs.eps_answer))
                decision.append(a if a in x else None)
        if g >= 0 and t % 3 == 0:
            right.append(int(np.argmax(F.marginal(F.g))) == g)
        if trace:
            ticks.append({"t": t, "g": g, "n": n, "robot": robot["v"], "fail": robot["fail"], "pause": pause,
                          "decision": decision, "belief": int(np.argmax(F.marginal(F.g))), "done": done.tolist(),
                          **{k: out[k] for k in ("conflicts", "violations", "questions")}})
    return {"video": vid, "condition": condition, "seed": seed, "time": t, "finished": bool(done.all()),
            "person_time": float(d.sum()), "belief_acc": float(np.mean(right)) if right else None, **out} | ({"trace": ticks, "durations": d.tolist()} if trace else {})


def add(F: Filter, graphs, history, prior, cfg, N, apply) -> None:
    new = [g for g in dict.fromkeys(graphs) if g not in set(F.graphs)]
    if not new:
        return
    G = Filter(cfg, N, new, [log_prior(prior, g) for g in new], len(F.graphs))
    for kind, x in history:
        apply(G, kind, x)
    F.graphs += new
    F.log_prior = np.concatenate([F.log_prior, G.log_prior])
    F.anc = np.concatenate([F.anc, G.anc])
    for a in ("g", "n", "S", "D", "logW", "logZ"):
        setattr(F, a, np.concatenate([getattr(F, a), getattr(G, a)]))


def one(args) -> dict:
    return episode(*args)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith(("condition=", "seeds=", "slowdown=", "acc="))]
    acc = float(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("acc=")), SIM["acc"]))
    cfg = replace(parse(args), pool=64, particles=32)
    conds = next((a.split("=")[1].split(",") for a in sys.argv[1:] if a.startswith("condition=")), list(CONDITIONS))
    seeds = int(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("seeds=")), 3))
    costs = Costs(robot_slowdown=float(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("slowdown=")), 1.5)))
    jobs = [(cfg, v, c, s, costs, False, acc) for v in stream.split(cfg) for c in conds for s in range(seeds)]
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(32) as pool:
        rows = list(pool.map(one, jobs, chunksize=1))
    (cfg.run_dir / "episodes.json").write_text(json.dumps({"config": asdict(cfg), "sim": SIM | {"acc": acc}, "costs": asdict(costs),
                                                           "episodes": rows}))
    print(f"{'condition':16s} {'time vs alone':>13s} {'conflicts':>9s} {'violations':>10s} {'questions':>9s} {'S2':>5s} {'robot did':>9s} {'belief acc':>10s}")
    alone = {(r["video"], r["seed"]): r["time"] for r in rows if r["condition"] == "none"}
    for c in conds:
        rs = [r for r in rows if r["condition"] == c]
        ratio = np.mean([r["time"] / alone[r["video"], r["seed"]] for r in rs]) if alone else np.nan
        print(f"{c:16s} {ratio:13.3f} {np.mean([r['conflicts'] for r in rs]):9.2f} {np.mean([r['violations'] for r in rs]):10.2f} "
              f"{np.mean([r['questions'] for r in rs]):9.2f} {np.mean([r['s2_calls'] for r in rs]):5.2f} {np.mean([r['robot_subgoals'] for r in rs]):9.2f} "
              f"{np.mean([r['belief_acc'] for r in rs if r['belief_acc'] is not None]):10.3f}")


if __name__ == "__main__":
    main()
