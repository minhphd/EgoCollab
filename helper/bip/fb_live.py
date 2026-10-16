"""The live loop (live.py) replayed on recorded false-belief tasks: the same code path a robot runs. A source turns a
recording into io.py inputs; the check is whether the robot tells the person where X is when they hold a false belief.

    python -m helper.bip.fb_live <streams_dir> <out.json> [slow=gemini-3.5-flash]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .config import Config
from .io import Act, EgoFrame, SceneObs
from .live import Live
from .mind import MindConfig


def acts(st: dict) -> list[Act]:
    out = []
    for agent, skills in (("H", st["person_skills"]), ("R", st["robot_skills"])):
        for t, tool, arg in skills:
            a = [x.strip() for x in arg.split(",")]
            if tool == "Pick":
                out.append(Act(t, agent, "Pick", (a[0],)))
            elif tool == "Place":
                out.append(Act(t, agent, "Place", (a[0], "in" if a[1] == "within" else "on", a[2], None)))
            elif tool in ("Open", "Close"):
                out.append(Act(t, agent, tool, (a[0],)))
    return out


def source(st: dict, src: Path) -> list:
    """io.py inputs in time order: the scene graph's start, both agents' actions, the glasses' frames."""
    start = [(o, *p.split(" ", 1)) for o, p in st["world0"].items() if " " in p]
    code = {"X": st["X"], "A": st["A"], "B": st["B"]}
    imgs = {int(p.stem): p for p in (src.parent / "frames" / st["task"]).glob("*.jpg")}
    views, k, frames = st["views"], 0, []
    for r in st["log"]:
        while k + 1 < len(views) and views[k + 1]["t"] <= r["t"]:
            k += 1
        vis = {n: (0.5, 0.5, 0.0) for n in views[k]["seen"]} if views else {}
        vis.update({code[c]: tuple(v) for c, v in r["vis"].items()})
        vis = {n: v for n, v in vis.items() if n != st["X"] or "X" in r["vis"]}
        img = imgs.get(int(round(r["t"] * 10)))
        frames.append(EgoFrame(r["t"], vis, tuple(r["gaze"]), img.read_bytes() if img else None, fixated=r["read"] or ""))
    return sorted([SceneObs(0.0, [(o, rel.replace("within", "in"), f) for o, rel, f in start])] + acts(st) + frames,
                  key=lambda x: (x.t, not isinstance(x, SceneObs), not isinstance(x, Act)))


def run(st: dict, src: Path, slow: str) -> dict:
    task = {"id": st["task"], "nodes": [], "labels": [], "objects": {}, "spec": {}, "furniture": sorted(set(st["spots"]) | {st["A"], st["B"], st["Q"], st["park"], st["X_from"]}), "map": []}
    live = Live(Config(premap=False), task, MindConfig(), slow_model=slow)
    q, moved, told = st["flags"]["query"], st["flags"]["moved"], None
    for x in source(st, src):
        if x.t > q + 1.0:
            break
        if moved <= x.t and st["X"] not in live.expected:
            live.expect(st["X"])  # the person will look for X: live, S1's next subgoal says so
        for c in live.on(x):
            if c.kind == "tell" and c.target == st["X"] and told is None:
                told = {"t": round(c.t, 2), "text": c.text, "p_fb": round(live.mind.false_belief(st["X"]), 3)}
    live.pool.shutdown(wait=True)
    return {"task": st["task"], "cond": st["cond"], "fb": st["false_belief"], "told": told,
            "p_fb_query": round(live.mind.false_belief(st["X"]), 3)}


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    args = dict(a.split("=", 1) for a in sys.argv[3:])
    sts = [json.loads(p.read_text()) for p in sorted(src.glob("*.json"))]
    sts = [s for s in sts if not s.get("reject") and "query" in s.get("flags", {})]
    rows = [run(s, src, args.get("slow", "")) for s in sts]
    fb = [r for r in rows if r["fb"]]
    tb = [r for r in rows if not r["fb"]]
    summary = {"n": len(rows), "told_when_false": round(float(np.mean([r["told"] is not None for r in fb])), 3) if fb else None,
               "told_when_true": round(float(np.mean([r["told"] is not None for r in tb])), 3) if tb else None}
    Path(out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
