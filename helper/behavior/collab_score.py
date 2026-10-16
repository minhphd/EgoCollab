"""Scores collaborative BEHAVIOR episodes (helper/behavior/collab.py) against the person alone on the same task and
seed:
    time_rel          task time with the robot / time alone (both finished)
    robot_redundant   robot trips for a step the person had done or was doing when the trip began or while it ran:
                      count, and their seconds
    person_redundant  person trips for a step the robot had done or was doing when the trip began: count, seconds
    robot_utt, person_utt   utterances
A trip is one segment of an agent's log: from choosing a step to finishing, failing or dropping it.

    python -m helper.behavior.collab_score <run_dir> [<run_dir> ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def load(dirs) -> dict:
    out = {}
    for d in dirs:
        for f in sorted(Path(d, "streams").glob("*.json")):
            st = json.loads(f.read_text())
            out[(st["activity"], st["seed"], st["policy"])] = st
    return out


def done_by(st: dict) -> list[tuple[float, str, str]]:
    """(time, step, who) of every completion, re-completions included."""
    return [(e[0], e[2], e[3]) for e in st["events"] if e[1] == "done"]


def redundant(mine: list, theirs: list, done: list, who_else: str) -> tuple[int, float]:
    """Trips in `mine` ([start, end, step, object]) for a step the other agent had done before the trip began, or was
    on (a trip of theirs on the same step under way) when it began."""
    n, secs = 0, 0.0
    for start, end, v, _ in mine:
        had = any(t <= start and w == who_else and s == v for t, s, w in done)
        doing = any(s <= start < e and sv == v for s, e, sv, _ in theirs)
        if had or doing:
            n, secs = n + 1, secs + (end - start)
    return n, secs


def score(st: dict, alone: dict | None) -> dict:
    done = done_by(st)
    rr = redundant(st["robot_segments"], st["person_segments"], done, "person")
    pr = redundant(st["person_segments"], st["robot_segments"], done, "robot")
    both = alone is not None and st["finished"] and alone["finished"]
    return {"activity": st["activity"], "seed": st["seed"], "policy": st["policy"], "seconds": st["seconds"],
            "finished": st["finished"], "time_rel": round(st["seconds"] / alone["seconds"], 3) if both else None,
            "robot_redundant": rr[0], "robot_redundant_s": round(rr[1], 1),
            "person_redundant": pr[0], "person_redundant_s": round(pr[1], 1),
            "robot_utt": len(st.get("utterances", {}).get("robot", [])),
            "person_utt": len(st.get("utterances", {}).get("person", [])),
            "robot_share": round(np.mean([w == "robot" for _, _, w in done]) if done else 0.0, 2)}


def main() -> None:
    eps = load(sys.argv[1:])
    rows = [score(st, eps.get((a, s, "alone"))) for (a, s, p), st in sorted(eps.items()) if p != "alone"]
    for r in rows:
        print(json.dumps(r))
    by = {}
    for r in rows:
        by.setdefault(r["policy"], []).append(r)
    for pol, rs in by.items():
        tr = [r["time_rel"] for r in rs if r["time_rel"] is not None]
        print(f"{pol}: {len(rs)} episodes, finished {np.mean([r['finished'] for r in rs]):.0%}, "
              f"time_rel {np.mean(tr) if tr else float('nan'):.2f}, robot redundant {np.mean([r['robot_redundant'] for r in rs]):.1f} "
              f"({np.mean([r['robot_redundant_s'] for r in rs]):.1f} s), person redundant {np.mean([r['person_redundant'] for r in rs]):.1f} "
              f"({np.mean([r['person_redundant_s'] for r in rs]):.1f} s), utterances robot {np.mean([r['robot_utt'] for r in rs]):.1f} "
              f"person {np.mean([r['person_utt'] for r in rs]):.1f}")


if __name__ == "__main__":
    main()
