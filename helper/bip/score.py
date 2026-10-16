"""Scores the estimates of a replay run against each stream's truth.

Current subgoal: the subgoal whose segment holds t (a pause keeps it while it is incomplete). Next: the
next different subgoal to start. Progress: per subgoal, whether "done" matches the truth. Edges count as
stated at p >= 0.5 and are compared on transitive closure, pooled over updates; "prior" is the first record.
Chance picks uniformly among the subgoals not yet complete; *_4plus rates use tasks with 4 or more subgoals.

    python -m helper.bip.score name=oracle
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from .config import Config, parse
from .graph import closure


def prf(c: dict) -> dict:
    p, r = c["tp"] / max(1, c["tp"] + c["fp"]), c["tp"] / max(1, c["tp"] + c["fn"])
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / max(1e-9, p + r), 3)}


class Truth:
    def __init__(self, st: dict):
        self.nodes, self.edges = st["nodes"], [tuple(e) for e in st["truth"]["edges"]]
        self.segments = sorted(st["truth"]["segments"])
        self.done_at = {c["node"]: c["t"] for c in st["truth"]["completions"]}

    def done(self, t: float) -> set:
        return {v for v, c in self.done_at.items() if c <= t}

    def current(self, t: float):
        last = None
        for a, b, v, *_ in self.segments:
            if a <= t < b:
                return v
            if b <= t:
                last = v
        return last if last is not None and last not in self.done(t) else None

    def next(self, t: float):
        cur = self.current(t)
        return next((v for a, _, v in self.segments if a > t and v != cur and v not in self.done(t)), None)


def summarize(cfg: Config, streams: dict[str, dict]) -> dict:
    count = lambda: {"tp": 0, "fp": 0, "fn": 0}
    edges, prior_edges = count(), count()
    rate, chance = defaultdict(lambda: [0, 0]), defaultdict(list)
    runs = [json.loads(p.read_text()) for p in sorted(cfg.run_dir.glob("*.json")) if p.stem in streams]
    for run in runs:
        st = streams[run["id"]]
        T = Truth(st)
        truth = closure(T.nodes, T.edges)
        for rec in run["records"]:
            est = rec["estimate"]
            guess = closure(T.nodes, [(u, v) for u, v, p in est["edges"] if p >= 0.5])
            c = prior_edges if rec["prior"] else edges
            c["tp"], c["fp"], c["fn"] = c["tp"] + len(guess & truth), c["fp"] + len(guess - truth), c["fn"] + len(truth - guess)
            if rec["prior"]:
                continue
            t, done = rec["t"], T.done(rec["t"])
            for v in T.nodes:
                rate["progress"][0] += (v in est["done"]) == (v in done)
                rate["progress"][1] += 1
            open_ = len(T.nodes) - len(done)  # subgoals not yet complete: chance picks among them
            big = "" if len(T.nodes) < 4 else "_4plus"
            if (cur := T.current(t)) is not None:
                for m in ("current", "current" + big) if big else ("current",):
                    rate[m][0] += est["current"] == cur
                    rate[m][1] += 1
                chance["current"].append(1 / max(1, open_))
            if (nxt := T.next(t)) is not None:
                for k in (1, 3):
                    for m in (f"next_top{k}", f"next_top{k}" + big) if big else (f"next_top{k}",):
                        rate[m][0] += nxt in est["next"][:k]
                        rate[m][1] += 1
                    chance[f"next_top{k}"].append(min(1.0, k / max(1, open_)))  # a random order of the open ones
    calls = [c for r in runs for c in r["s2_calls"]]
    return {"name": cfg.name, "episodes": len(runs),
            "rates": {m: round(h / max(1, n), 3) for m, (h, n) in sorted(rate.items())},
            "chance": {m: round(sum(v) / max(1, len(v)), 3) for m, v in sorted(chance.items())},
            "edge_f1": {"online": prf(edges)["f1"], "prior": prf(prior_edges)["f1"]},
            "s2_calls_per_episode": round(len(calls) / max(1, len(runs)), 2),
            "s2_tokens": {k: sum(c.get(k, 0) for c in calls) for k in ("prompt_tokens", "completion_tokens", "thinking_tokens")},
            "s1_ms": round(sum(r["s1_ms"] for r in runs) / max(1, len(runs)), 2)}


def main() -> None:
    cfg = parse(sys.argv[1:])
    streams = {p.stem: json.loads(p.read_text()) for p in Path(cfg.streams_dir).glob("*.json")}
    out = summarize(cfg, streams)
    (cfg.run_dir / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
