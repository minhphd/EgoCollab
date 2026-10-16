"""Scores closed-loop PARTNR runs against the person working alone (paired by episode and seed).

Per run folder (runs/<name>/streams/*.json) and the person-alone recording:
    time ratio      task time with the robot / time alone (lower is better), on episodes both finished
    finished        share of episodes where every subgoal got done
    clean           finished with no ordering violation (the task done as instructed)
    PARTNR success  share with all of PARTNR's success conditions met at the end
    conflicts       per episode, times the person and the robot worked on the same subgoal at once
    clashes         conflicts where the person had already acted on it (a navigation or pick done), not only chosen it
    violations      per episode, subgoals the robot completed before a required predecessor was done
    robot share     share of subgoals the robot completed
    robot wrong     per episode, objects the robot placed that no subgoal asks for

    python -m helper.partnr.loop_score <alone_streams_dir> <run_dir> [<run_dir> ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def load(d: Path) -> dict:
    return {p.stem: json.loads(p.read_text()) for p in Path(d).glob("*.json")}


def overlaps(a: list, b: list) -> int:
    """Times both worked on the same subgoal, or on the same object (segments carry it as a 4th entry)."""
    same = lambda x, y: x[2] == y[2] or (len(x) > 3 and len(y) > 3 and x[3] is not None and x[3] == y[3])
    return sum(1 for x in a for y in b if same(x, y) and min(x[1], y[1]) > max(x[0], y[0]))


def clashes(st: dict) -> int:
    """Conflicts after the person's first completed action on the subgoal; the rest are unobservable intent."""
    same = lambda x, y: x[2] == y[2] or (len(x) > 3 and len(y) > 3 and x[3] is not None and x[3] == y[3])
    log = [(float(l.split()[0]), l) for l in st.get("person_log", []) if "Navigate[" in l or "Pick[" in l]
    return sum(1 for x in st["truth"]["segments"] for y in st.get("robot_segments", [])
               if same(x, y) and min(x[1], y[1]) > max(x[0], y[0])
               and any(x[0] < t <= min(x[1], y[0]) + 0.05 for t, _ in log))


def violations(st: dict) -> int:
    done_at = {c["node"]: c["t"] for c in st["completions"]}
    mine = set(st["result"]["done_by_robot"])
    return sum(1 for u, v in st["truth"]["edges"] if v in mine and v in done_at and done_at.get(u, np.inf) > done_at[v])


def finished(st: dict) -> bool:
    return len({c["node"] for c in st["completions"]}) == len(st["nodes"])


def score(alone: dict, run: dict) -> dict:
    both = [k for k in run if k in alone and finished(run[k]) and finished(alone[k])]
    ratio = [run[k]["result"]["seconds"] / max(1e-9, alone[k]["result"]["seconds"]) for k in both]
    n_sub = sum(len(s["nodes"]) for s in run.values())
    return {"episodes": len(run), "paired": len(both),
            "time_ratio": round(float(np.mean(ratio)), 3) if ratio else None,
            "time_ratio_median": round(float(np.median(ratio)), 3) if ratio else None,
            "finished": round(np.mean([finished(s) for s in run.values()]), 3),
            "clean": round(np.mean([finished(s) and violations(s) == 0 for s in run.values()]), 3),
            "partnr_success": round(np.mean([s["result"]["percent_complete"] == 1.0 for s in run.values()]), 3),
            "conflicts": round(np.mean([overlaps(s["truth"]["segments"], s.get("robot_segments", [])) for s in run.values()]), 2),
            "clashes": round(np.mean([clashes(s) for s in run.values()]), 2),
            "violations": round(np.mean([violations(s) for s in run.values()]), 2),
            "robot_share": round(sum(len(s["result"]["done_by_robot"]) for s in run.values()) / max(1, n_sub), 3),
            "robot_failed_skills": round(np.mean([s["result"].get("robot_failed", 0) for s in run.values()]), 2),
            "s2_calls": round(np.mean([len(s.get("robot_s2", [])) for s in run.values()]), 2),
            "robot_wrong": round(np.mean([s["result"].get("robot_wrong", 0) for s in run.values()]), 2),
            "questions": round(np.mean([s["result"].get("questions", 0) for s in run.values()]), 2)}


def main() -> None:
    alone = load(Path(sys.argv[1]))
    for d in sys.argv[2:]:
        out = {"name": Path(d).name} | score(alone, load(Path(d) / "streams"))
        (Path(d) / "loop_summary.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out))


if __name__ == "__main__":
    main()
