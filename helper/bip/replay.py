"""Offline main loop: a recorded stream's events through the belief, in time order; score.py scores the output.

A stream (data/bip_partnr*/streams/<id>.json, written by helper/partnr/record.py) holds the task (id, title,
nodes, labels, objects, spec, furniture, map), the truth for scoring (edges, segments, completions), and the
observations of every perception tier (actions, held, fixations, completions, views, acts, frames).
observe.events() turns the observations of cfg's tier into S1 events; every `grid` s the belief's estimate is
written out.

    python -m helper.bip.replay name=scene action_tier=scene completion=scene streams_dir=data/bip_partnr_dev/streams
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import observe
from .belief import Belief
from .config import Config, parse


def run(cfg: Config, st: dict) -> dict:
    """Replay one stream: estimates on a grid, S2 calls, and S1's time per event."""
    B = Belief(cfg, st)
    t0, t1 = st["t0"], st["t1"]
    records = [{"t": t0, "prior": True, "estimate": B.estimate()}]
    grid, s1, trace = t0 + cfg.grid, [], []
    for t, _, kind, x, info in observe.events(cfg, st) + [(np.inf, 9, "end", None, None)]:
        while grid <= min(t, t1 + cfg.grid - 1e-9):
            records.append({"t": round(min(grid, t1), 2), "prior": False, "estimate": B.estimate()})
            grid += cfg.grid
        if kind == "end":
            break
        start = time.perf_counter()
        B.feed(t, kind, x)
        s1.append(time.perf_counter() - start)
        if cfg.trace:
            trace.append(B.snapshot(t, kind, info))
    return ({"trace": trace} if cfg.trace else {}) | {
        "id": st["id"], "name": cfg.name, "model": cfg.model, "records": records, "s2_calls": B.calls,
        "prior_call": B.prior_call, "pool": len(B.F.graphs),
        "s1_ms": round(1e3 * float(np.median(s1)), 2) if s1 else 0.0}


def streams(cfg: Config) -> list[Path]:
    paths = sorted(Path(cfg.streams_dir).glob("*.json"))
    return paths[: cfg.episodes] if cfg.episodes else paths


def main() -> None:
    cfg = parse(sys.argv[1:])
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    (cfg.run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=1))

    def one(path: Path) -> None:
        (cfg.run_dir / path.name).write_text(json.dumps(run(cfg, json.loads(path.read_text()))))

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(one, streams(cfg)))
    print(f"{cfg.name}: {len(streams(cfg))} streams replayed", flush=True)


if __name__ == "__main__":
    main()
