"""Metrics of collaboration episodes, against the person alone on the same task and seed:
    time_rel     time with the robot / time alone (both finished)
    robot_steps  steps the robot did
    robot_waste  robot trips for a step the person had done or was doing when the trip began, or did during it, unless
                 the trip did it (count, seconds)
    person_waste person trips for a step the robot had done or was doing when the trip began, or did during it, unless
                 the trip did it
    utterances   the robot's and the person's

    python -m helper.collab.score <run_dir> [<run_dir> ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def waste(mine: list, theirs: list, done: list, other: str) -> tuple[int, float]:
    """Trips in `mine` ([start, end, step]) for a step `other` had done before the trip began, was on then, or did
    during it, unless the trip did the step (the other yielded it)."""
    me = {"person": "robot", "robot": "person"}[other]
    n, secs = 0, 0.0
    for start, end, v in mine:
        had = any(t <= start and w == other and s == v for t, s, w in done)
        doing = any(s <= start < e and sv == v for s, e, sv in theirs)
        during = any(start < t <= end + 1.0 and w == other and s == v for t, s, w in done)
        did = any(start <= t <= end + 1.0 and w == me and s == v for t, s, w in done)
        if (had or doing or during) and not did:
            n, secs = n + 1, secs + (end - start)
    return n, secs


def trips(st: dict) -> list:
    """The robot's trips [start, end, step], with one still open when the episode ended (its last step done)."""
    segs = [list(x) for x in st["robot_segments"]]
    starts = [line.split() for line in st.get("robot_log", []) if line.split()[1:2] == ["start"]]
    if len(starts) > len(segs):
        segs.append([float(starts[-1][0]), float(st["seconds"]), int(starts[-1][2])])
    return segs


def score(st: dict, alone: dict | None) -> dict:
    rw = waste(trips(st), st["person_segments"], st["done"], "person")
    pw = waste(st["person_segments"], trips(st), st["done"], "robot")
    both = alone is not None and st["finished"] and alone["finished"]
    return {"policy": st["policy"], "seed": st["seed"], "seconds": st["seconds"], "finished": st["finished"],
            "time_rel": round(st["seconds"] / alone["seconds"], 3) if both else None,
            "robot_steps": sum(w == "robot" for w in st["by"].values()),
            "robot_waste": rw[0], "robot_waste_s": round(rw[1], 1), "person_waste": pw[0], "person_waste_s": round(pw[1], 1),
            "robot_utt": sum(u["who"] == "robot" for u in st["utterances"]),
            "person_utt": sum(u["who"] == "person" for u in st["utterances"]), "crashed": st.get("crashed")}


def main() -> None:
    eps = {}
    for d in sys.argv[1:]:
        for p in sorted(Path(d, "streams").glob("*.json")):
            st = json.loads(p.read_text())
            eps[(p.stem.rsplit("_", 2)[0], st["seed"], st["policy"])] = st
    rows = [dict(score(st, eps.get((task, seed, "alone"))), task=task) for (task, seed, pol), st in sorted(eps.items()) if pol != "alone"]
    for r in rows:
        print(json.dumps(r))
    for task in [None] + sorted({r["task"] for r in rows}):  # all tasks, then each
        print(f"== {task or 'all tasks'}")
        for pol in sorted({r["policy"] for r in rows}):
            rs = [r for r in rows if r["policy"] == pol and task in (None, r["task"])]
            crashed = sum(r["crashed"] is not None for r in rs)
            rs = [r for r in rs if r["crashed"] is None]  # the simulator's failures, not the policy's: counted apart
            if not rs:
                continue
            tr = [r["time_rel"] for r in rs if r["time_rel"] is not None]
            print(f"{pol}: {len(rs)} episodes ({crashed} crashed apart), finished {np.mean([r['finished'] for r in rs]):.0%}, time_rel "
                  f"{np.mean(tr) if tr else float('nan'):.2f} (sd {np.std(tr) if tr else float('nan'):.2f}), robot steps "
                  f"{np.mean([r['robot_steps'] for r in rs]):.1f}, robot waste {np.mean([r['robot_waste'] for r in rs]):.1f}, "
                  f"person waste {np.mean([r['person_waste'] for r in rs]):.1f}, utterances robot "
                  f"{np.mean([r['robot_utt'] for r in rs]):.1f} person {np.mean([r['person_utt'] for r in rs]):.1f}")


if __name__ == "__main__":
    main()
