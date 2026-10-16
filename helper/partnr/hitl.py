"""Statistics of real people doing PARTNR tasks (the human-in-the-loop traces), to calibrate the scripted person.

A trace holds sparse frames: per user the held and hovered (cursor) object and pick and place events, the
objects' positions, the agents' positions and the task's percent complete. Per single-person episode:
    choice      at every first pick of a task object: the candidates still to place, their rank in the
                instruction (proposition order) and their distance; fits P(v) prop. to exp(-a rank - b dist)
    replaced    task-object placements after which the same object is picked up again (mistakes, temporary spots)
    idle        the share of an episode spent standing still with empty hands
    look-ahead  whether the cursor rested on the next object picked before the current object was placed
    pauses      idle time before a pick, as a share of the time between placing and the next pick
    yielding    two-person traces: after the partner picks the object a user's cursor was on, the share of
                users who switch to another object instead of waiting for it
    residual    the expected time left on a subgoal given the time already spent, both in units of the
                episode's mean gap between successful placements: r(x) = E[D - x | D > x]
    python -m helper.partnr.hitl data/partnr_hitl data/bip_partnr/hitl_calibration.json
"""

from __future__ import annotations

import glob
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

EPISODES = Path(__file__).resolve().parents[2] / "data/partnr/versioned_data/partnr_episodes/v0_0"


def tasks(split: str) -> dict:
    return {e["episode_id"]: e for e in json.load(gzip.open(EPISODES / f"{split}.json.gz"))["episodes"]}


def frames(path: str) -> tuple[dict, list]:
    d = json.load(gzip.open(path))
    return d["episode"], [f for f in d["frames"] if f]


def task_objects(ep: dict) -> list[str]:
    """Object handles of the placement propositions, in instruction (proposition) order."""
    out = []
    for p in ep["evaluation_propositions"]:
        for h in p["args"].get("object_handles", []):
            if h not in out:
                out.append(h)
    return out


def episode_stats(ep: dict, fr: list, user: int = 0) -> dict:
    objs = task_objects(ep)
    rank = {h: i for i, h in enumerate(objs)}
    pos, ids, choices, places, picks, hovers = {}, {}, [], [], [], []
    placed_ok, pct_prev, prev, total, still = set(), 0.0, None, 0.0, 0.0
    last_place_t, idle = None, []
    for f in fr:
        t = f["t"]
        for o in f.get("object_states", []):
            pos[o["object_handle"]] = np.array(o["position"])
            ids[o["object_handle"]] = o.get("object_id")
        u = f["users"][user] if user < len(f["users"]) else None
        if u is None:
            continue
        me = np.array(f["agent_states"][user]["position"]) if user < len(f["agent_states"]) else None
        if me is not None and prev is not None:  # standing still with empty hands counts as idle
            dt = t - prev[0]
            moving = np.linalg.norm((me - prev[1])[[0, 2]]) / max(dt, 1e-6) > 0.1
            total += dt
            still += dt if not moving and u.get("held_object") is None else 0.0
        prev = (t, me) if me is not None else prev
        if u.get("hovered_object") is not None:
            hovers.append((t, u["hovered_object"]))
        for e in u.get("events", []):
            if e["type"] == "pick":
                h = e["obj_handle"]
                picks.append((t, h))
                left = [o for o in objs if o not in placed_ok]
                if h in rank and h not in {p for _, p in picks[:-1]} and len(left) > 1 and me is not None:
                    d = {o: float(np.linalg.norm((pos[o] - me)[[0, 2]])) for o in left if o in pos}
                    if h in d:
                        choices.append({"chosen": h, "rank": {o: rank[o] for o in d}, "dist": d})
                if last_place_t is not None:
                    idle.append(t - last_place_t)
            elif e["type"] == "place":
                h = e["obj_handle"]
                pct = f.get("task_percent_complete", pct_prev)
                ok = h in rank and pct > pct_prev + 1e-9
                places.append({"t": t, "obj": h, "task": h in rank, "raised": ok})
                if ok:
                    placed_ok.add(h)
                last_place_t = t
        pct_prev = f.get("task_percent_complete", pct_prev)
    # look-ahead: between a pick of object A and its place, did the cursor rest on the object picked next?
    ahead = []
    for (t0, a), (t1, b) in zip(picks, picks[1:]):
        t_place = next((p["t"] for p in places if p["obj"] == a and t0 <= p["t"] <= t1), None)
        if t_place is not None and b != a:
            ahead.append(any(t0 <= th <= t_place and ho == ids.get(b) for th, ho in hovers))
    replaced = [any(t > p["t"] and h == p["obj"] for t, h in picks) for p in places if p["task"]]
    return {"choices": choices, "places": places, "picks": picks, "ahead": ahead, "idle": idle,
            "replaced": replaced, "idle_share": still / total if total > 0 else None}


def fit_choice(choices: list[dict]) -> dict:
    """Grid MLE of P(v) prop. to exp(-a rank - b dist) over the observed first picks."""
    best = None
    for a in np.linspace(0, 3, 31):
        for b in np.linspace(0, 6, 61):
            ll = 0.0
            for c in choices:
                cand = list(c["dist"])
                s = np.array([-a * c["rank"][o] - b * c["dist"][o] for o in cand])
                s -= s.max()
                ll += s[cand.index(c["chosen"])] - np.log(np.exp(s).sum())
            if best is None or ll > best[0]:
                best = (ll, a, b)
    uniform = -sum(np.log(len(c["dist"])) for c in choices)
    first = np.mean([c["rank"][c["chosen"]] == min(c["rank"].values()) for c in choices])
    nearest = np.mean([c["dist"][c["chosen"]] == min(c["dist"].values()) for c in choices])
    return {"a_rank": round(float(best[1]), 2), "b_dist": round(float(best[2]), 2), "loglik": round(best[0], 1),
            "loglik_uniform": round(float(uniform), 1), "n": len(choices),
            "share_instruction_order": round(float(first), 3), "share_nearest": round(float(nearest), 3)}


def residual(allstats: list[dict], grid: np.ndarray = np.arange(0, 3.01, 0.25)) -> dict:
    """Mean residual life of the gaps between an episode's successful task placements, normalized per episode."""
    gaps = []
    for s in allstats:
        g = np.diff(sorted(p["t"] for p in s["places"] if p["raised"]))
        if len(g) >= 2 and g.mean() > 0:
            gaps += list(g / g.mean())
    D = np.array(gaps)
    return {"x": [round(float(x), 2) for x in grid], "r": [round(float((D[D > x] - x).mean()), 3) for x in grid],
            "n": len(D)}


def yielding(paths: list[str], window: float = 5.0) -> dict:
    """Two-person traces: after the partner picks the object a user's cursor was on within `window` s, does the
    user pick a different object next (switches) rather than the same one later (waits)?"""
    switch = wait = 0
    for path in paths:
        ep, fr = frames(path)
        hov = defaultdict(list)
        picks = []
        for f in fr:
            for i, u in enumerate(f["users"]):
                if u.get("hovered_object") is not None:
                    hov[i].append((f["t"], u["hovered_object"]))
                for e in u.get("events", []):
                    if e["type"] == "pick":
                        picks.append((f["t"], i, e["obj_handle"], e.get("obj_id")))
        for t, i, h, oid in picks:
            for j in hov:
                if j == i:
                    continue
                if any(t - window <= th <= t and (ho == oid or str(ho) == h) for th, ho in hov[j]):
                    nxt = next(((tt, hh) for tt, jj, hh, _ in picks if jj == j and tt > t), None)
                    if nxt is not None:
                        switch += nxt[1] != h
                        wait += nxt[1] == h
    n = switch + wait
    return {"switch": round(switch / max(1, n), 3), "n": n}


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    by_split = {"p5_single_val": "val", "p5_single_train_2k": "train_2k"}
    allstats = []
    for sub, split in by_split.items():
        T = tasks(split)
        for path in sorted(glob.glob(str(root / sub / "*" / "*.json.gz"))):
            if path.endswith("session.json.gz"):
                continue
            ep, fr = frames(path)
            if str(ep["episode_id"]) in T and fr:
                allstats.append(episode_stats(T[str(ep["episode_id"])], fr))
    choices = [c for s in allstats for c in s["choices"]]
    places = [p for s in allstats for p in s["places"] if p["task"]]
    wrong = [p for p in places if not p["raised"]]
    redo = sum(1 for s in allstats for p in s["places"] if p["task"] and not p["raised"]
               and any(t > p["t"] and h == p["obj"] for t, h in s["picks"]))
    ahead = [a for s in allstats for a in s["ahead"]]
    idle = np.array([x for s in allstats for x in s["idle"] if x > 0])
    multi = [p for p in glob.glob(str(root / "p5_multi_val" / "*" / "*.json.gz")) if not p.endswith("session.json.gz")]
    cal = {"episodes": len(allstats), "choice": fit_choice(choices),
           "replaced": {"share": round(float(np.mean([r for s in allstats for r in s["replaced"]])), 3),
                        "task_places": len(places)},
           "idle_share": {"median": round(float(np.median([s["idle_share"] for s in allstats if s["idle_share"] is not None])), 3),
                          "mean": round(float(np.mean([s["idle_share"] for s in allstats if s["idle_share"] is not None])), 3)},
           "look_ahead": {"share": round(float(np.mean(ahead)), 3) if ahead else None, "n": len(ahead)},
           "pauses": {"median_s": round(float(np.median(idle)), 2) if len(idle) else None,
                      "p90_s": round(float(np.percentile(idle, 90)), 2) if len(idle) else None, "n": int(len(idle))},
           "yielding": yielding(multi), "residual": residual(allstats)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cal, indent=1))
    print(json.dumps(cal, indent=1))


if __name__ == "__main__":
    main()
