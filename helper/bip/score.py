"""Scores the online estimates in <runs>/<name> by share of the task seen (BIP runs and RQ1 VLM runs alike).

Edges count as stated when p >= 0.5 and are compared on transitive closure (A > C is right when the
truth has A > B > C), pooled over updates. `truth`: the annotated orderings, those the person never
broke ("consistent"), or every ordering the person followed ("followed": u complete before v started).
Baselines: the prior held fixed, and a chain of the true order of first starts.

    python -m helper.bip.score name=t_act_llm split=test truth=annotated
    python -m helper.bip.score runs=runs/rq1 name=all_27b split=test        # an RQ1 VLM run
"""

import json
import sys
from collections import defaultdict

import numpy as np

from .config import parse
from .stream import split
from .tasks import load

BINS = 10


def closure(nodes: list, edges) -> set[tuple]:
    succ = {n: set() for n in nodes}
    for u, v in edges:
        succ[u].add(v)
    out = set()
    for n in nodes:
        stack, seen = list(succ[n]), set()
        while stack:
            m = stack.pop()
            if m not in seen:
                seen.add(m)
                stack.extend(succ[m])
        out |= {(n, m) for m in seen if m != n}
    return out


def stated(est: dict) -> list[tuple[str, str]]:
    return [(u, v) for u, v, p in est["edges"] if p >= 0.5]


def prf(c: dict) -> dict:
    p = c["tp"] / max(1, c["tp"] + c["fp"])
    r = c["tp"] / max(1, c["tp"] + c["fn"])
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / max(1e-9, p + r), 3)}


def truth_edges(task, truth: str) -> list[tuple[str, str]]:
    return {"annotated": task.edges,
            "consistent": [(u, v) for u, v in task.edges if task.first[v] >= task.last[u]],
            "followed": [(u, v) for u in task.nodes for v in task.nodes if task.last[u] <= task.first[v]]}[truth]


def main() -> None:
    cfg = parse(sys.argv[1:])
    tasks, keep = load(1), set(split(cfg))
    runs = [json.loads(p.read_text()) for p in sorted(cfg.run_dir.glob("*-*.json")) if p.stem in keep]
    count = lambda: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    edges = {k: count() for k in ("model", "prior", "chain")}
    rate = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # metric -> bin -> [hits, total]
    latency, tokens_in, tokens_out, changes = [], [], [], []
    for run in runs:
        task = tasks[run["video"]]
        (t0, t1), truth = task.extent, closure(task.nodes, truth_edges(task, cfg.truth))
        prior = closure(task.nodes, stated(run["records"][0]["estimate"] or {"edges": []}))
        previous = prior
        for rec in run["records"]:
            est = rec["estimate"] or {"done": [], "current": None, "next": [], "edges": []}
            b = "prior" if rec["prior"] else min(BINS - 1, int(BINS * (rec["t"] - t0) / max(1e-9, t1 - t0)))
            guess = closure(task.nodes, stated(est))
            chain = closure(task.nodes, zip(task.order(rec["t"]), task.order(rec["t"])[1:]))
            for name, g in (("model", guess), ("prior", prior), ("chain", chain)):
                c = edges[name][b]
                c["tp"], c["fp"], c["fn"] = c["tp"] + len(g & truth), c["fp"] + len(g - truth), c["fn"] + len(truth - g)
            if rec["prior"]:
                continue
            latency.append(rec["latency"])
            tokens_in.append(rec["prompt_tokens"])
            tokens_out.append(rec["completion_tokens"])
            changes.append(len(guess ^ previous))
            previous = guess
            done = task.done(rec["t"])
            for v in task.nodes:
                rate["progress"][b][0] += (v in est["done"]) == (v in done)
                rate["progress"][b][1] += 1
            if (cur := task.current(rec["t"])) is not None:
                rate["current"][b][0] += est["current"] == cur
                rate["current"][b][1] += 1
            if (nxt := task.next(rec["t"])) is not None:
                for k in (1, 3):
                    rate[f"next_top{k}"][b][0] += nxt in est["next"][:k]
                    rate[f"next_top{k}"][b][1] += 1
    order = ["prior"] + list(range(BINS))
    pooled = lambda d: {k: sum(c[k] for b, c in d.items() if b != "prior") for k in ("tp", "fp", "fn")}
    summary = {
        "name": cfg.name, "split": cfg.split, "truth": cfg.truth, "videos": len(runs), "updates": len(latency),
        "edges": {k: {"online": prf(pooled(d)), "prior": prf(d["prior"]), "last_tenth": prf(d[BINS - 1])} for k, d in edges.items()},
        "rates": {m: round(sum(h for h, _ in d.values()) / max(1, sum(n for _, n in d.values())), 3) for m, d in rate.items()},
        "latency_s": {"median": round(float(np.median(latency)), 3), "p90": round(float(np.percentile(latency, 90)), 3),
                      "max": round(float(np.max(latency)), 3)} if latency else {},
        "tokens": {"prompt_mean": round(float(np.mean(tokens_in))), "completion_mean": round(float(np.mean(tokens_out)))} if latency else {},
        "edge_changes_per_update": round(float(np.mean(changes)), 2) if changes else 0,
        "unparsed": sum(not r["ok"] for run in runs for r in run["records"]),
        "s2_calls_per_video": round(sum(len(r.get("s2_calls", [])) for r in runs) / max(1, len(runs)), 2),
        "curves": {"bins": [str(b) for b in order],
                   "edge_f1": {k: [prf(d[b])["f1"] for b in order] for k, d in edges.items()},
                   **{m: [round(d[b][0] / d[b][1], 3) if d[b][1] else None for b in order] for m, d in rate.items()}},
    }
    (cfg.run_dir / f"summary_{cfg.split}_{cfg.truth}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "curves"}, indent=1))
    print("edge F1 by tenth of the task (prior first):")
    for k, v in summary["curves"]["edge_f1"].items():
        print(f"  {k:6s}", " ".join(f"{x:.2f}" for x in v))


if __name__ == "__main__":
    main()
