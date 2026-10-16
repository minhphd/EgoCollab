"""False-belief detection by the partner-belief model (mind.py) on recorded tasks (helper/partnr/fb_record.py).

Replays each kept task through the same inputs a live robot gives it: the scene graph's moves (who moved the
object, from the agents' actions), and per frame the entities in the person's view and the one their gaze falls on.
The search for X starts when the robot finishes its move (live: S1's next subgoal uses X). At the query: P(false
belief), the believed place, and the truth.

    python -m helper.bip.fb_eval <streams_dir> <out.json> [variants=full,nogaze,history,first_look,full_slow] [sweep=1]
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from .mind import Mind, MindConfig

VARIANTS = {
    "full": {},
    "nogaze": {"gaze": False},
    "history": {"seek_evidence": False},  # gaze during the move only
    "first_look": {"history_gaze": False},  # gaze only in the first look of the search
    "full_slow": {},  # full, and the slow path when unsure at the search
}


def moves(st: dict) -> list[tuple]:
    """(t, object, place, actor) from the ground-truth places over time; a move belongs to whoever last picked it up."""
    out, holder = [], "H"
    for t, loc in st["x_track"]:
        holder = {"person": "H", "robot": "R"}.get(loc, holder)
        out.append((t, st["X"], loc, holder))
    return out


def frames(st: dict) -> list[tuple]:
    """(t, visible, fixated) per logged frame (15 fps); visible joins the logged X, A, B with the 2 Hz view lists."""
    code = {"X": st["X"], "A": st["A"], "B": st["B"]}
    views, k, out = st["views"], 0, []
    for r in st["log"]:
        while k + 1 < len(views) and views[k + 1]["t"] <= r["t"]:
            k += 1
        seen = set(views[k]["seen"]) - set(code.values()) if views else set()
        out.append((r["t"], seen | {code[c] for c in r["vis"]}, r["read"]))
    return out


def slow(st: dict, src: Path, model: str) -> tuple[float, dict]:
    """The slow path: Gemini on the person's view while the robot moved X (frames every 0.5 s, gaze drawn)."""
    from . import s2
    mv = [(t, p) for t, o, p, a in moves(st) if a == "R"]
    t0, t1 = mv[0][0] - 1.0, mv[-1][0] + 1.0
    d = src.parent / "frames" / st["task"]
    imgs = [f.read_bytes() for f in sorted(d.glob("*.jpg")) if t0 <= int(f.stem) / 10 <= t1]
    log = [f"t = {t:.1f} s: the robot moved the {st['X']}; it is now {p}" for t, p in mv]
    p, usage, latency = s2.saw(log, st["X"], mv[-1][1], imgs, model)
    return p, {"p_saw": round(p, 3), "frames": len(imgs), "latency": round(latency, 2), **usage}


def run(st: dict, cfg: MindConfig, src: Path | None = None, model: str = "") -> dict:
    m, call = Mind(cfg), None
    q, moved = st["flags"]["query"], st["flags"]["moved"]
    ev = [(t, 0, "move", (o, p, a)) for t, o, p, a in moves(st)] + [(t, 1, "view", (v, f)) for t, v, f in frames(st)]
    ev.append((moved, 2, "seek", st["X"]))
    for t, _, kind, x in sorted(ev, key=lambda e: (e[0], e[1])):
        if t > q:
            break
        if kind == "move":
            m.move(t, *x)
        elif kind == "view":
            m.view(t, *x)
        elif m.seek(t, x) and model:  # unsure as the search starts: ask the slow path
            p, call = slow(st, src, model)
            m.evidence(x, p)
    b = m.believed(st["X"])
    top = next(iter(b), None)
    truth = f"on {st['A']}" if st["belief"] == "A" else f"in {st['B']}"
    return {"task": st["task"], "cond": st["cond"], "design": st["design"], "p_fb": round(m.false_belief(st["X"]), 4),
            "fb": st["false_belief"], "believed": top, "truth_place": truth, "place_right": top == truth,
            "first_look": st["first_look"], "slow": call}


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def score(rows: list[dict]) -> dict:
    y, s = np.array([r["fb"] for r in rows], int), np.array([r["p_fb"] for r in rows])
    pred = (s >= 0.5).astype(int)
    by = {}
    for c in sorted({r["cond"] for r in rows}):
        rs = [r for r in rows if r["cond"] == c]
        by[c] = round(float(np.mean([(r["p_fb"] >= 0.5) == r["fb"] for r in rs])), 3)
    return {"n": len(rows), "acc": round(float(np.mean(pred == y)), 3),
            "balanced_acc": round(float(np.mean([np.mean(pred[y == k] == k) for k in (0, 1)])), 3),
            "auroc": round(auroc(y, s), 3), "brier": round(float(np.mean((s - y) ** 2)), 3),
            "place_acc": round(float(np.mean([r["place_right"] for r in rows])), 3), "by_condition": by}


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    args = dict(a.split("=", 1) for a in sys.argv[3:])
    sts = [json.loads(p.read_text()) for p in sorted(src.glob("*.json"))]
    sts = [s for s in sts if not s.get("reject") and "query" in s.get("flags", {})]
    res = {}
    for name in args.get("variants", ",".join(VARIANTS)).split(","):
        model = args.get("model", "gemini-3.5-flash") if name.endswith("_slow") else ""
        rows = [run(s, MindConfig(**VARIANTS[name]), src, model) for s in sts]
        res[name] = {"score": score(rows), "rows": rows}
        print(name, json.dumps(res[name]["score"]), flush=True)
    if args.get("sweep") == "1":  # the model's settings away from the simulator's
        grid = []
        for q_fix in (0.5, 0.7, 0.9):
            for q_vis in (0.01, 0.05, 0.2):
                for w_seek in (0.3, 0.6, 0.9):
                    rows = [run(s, MindConfig(q_fix=q_fix, q_vis=q_vis, w_seek=w_seek)) for s in sts]
                    grid.append({"q_fix": q_fix, "q_vis": q_vis, "w_seek": w_seek, **score(rows)})
        res["sweep"] = grid
        accs = [g["balanced_acc"] for g in grid]
        print("sweep balanced accuracy: min %.3f, median %.3f, max %.3f" % (min(accs), np.median(accs), max(accs)), flush=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
