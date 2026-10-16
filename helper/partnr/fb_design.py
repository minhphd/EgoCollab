"""False-belief task designs from PARTNR layouts (helper/partnr/fb_layout.py).

A design is one object X, a surface A and a closed container B in the same room R, a spot P in R from which the
person sees both A and B, a furniture Q in another room (where the person goes when away) and a furniture where
the robot parks after the move. Each design runs in every condition of helper/partnr/fb_record.py.

    python -m helper.partnr.fb_design <layout.json> <designs.json> [per_episode=3]
    python -m helper.partnr.fb_design tasks <designs.json> <tasks.json> designs=d000,d003   # every condition
"""

from __future__ import annotations

import json
import re
import sys

import numpy as np

SURFACES = ("table", "counter", "shelves", "stand", "bench")
CONTAINERS = ("cabinet", "chest_of_drawers", "wardrobe", "fridge")
AWAY = ("chair", "couch", "bed", "table", "bench", "stool")
SKIP = ("floor", "wall", "unknown")


def kind(name: str) -> str:
    return re.sub(r"_\d+$", "", name)


def xz(p) -> np.ndarray:
    return np.array([p[0], p[2]])


def angle(at, a, b) -> float:
    u, v = xz(a) - xz(at), xz(b) - xz(at)
    return float(np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9), -1, 1))))


def designs(ep: str, lay: dict, per_episode: int) -> list[dict]:
    fur = {f: v for f, v in lay["furniture"].items() if v["pos"] and v["room"] and not kind(f).startswith(SKIP)}
    out = []
    for x, ov in sorted(lay["objects"].items()):
        if ov["rel"] != "on" or ov["furniture"] not in fur or fur[ov["furniture"]]["openable"] or not ov["pos"]:
            continue
        for room in sorted({v["room"] for v in fur.values()}):
            here = {f: v for f, v in fur.items() if v["room"] == room}
            surf = [f for f, v in here.items() if kind(f) in SURFACES and not v["openable"] and f != ov["furniture"]]
            cont = [f for f, v in here.items() if kind(f) in CONTAINERS and v["openable"]]
            for a in surf:
                for b in cont:
                    dab = float(np.linalg.norm(xz(fur[a]["pos"]) - xz(fur[b]["pos"])))
                    if not 1.0 <= dab <= 3.5:
                        continue
                    mid = (np.array(fur[a]["pos"]) + np.array(fur[b]["pos"])) / 2
                    spots = [(abs(float(np.linalg.norm(xz(v["pos"]) - xz(mid))) - 3.0), p) for p, v in here.items()
                             if p not in (a, b) and 2.0 <= np.linalg.norm(xz(v["pos"]) - xz(mid)) <= 4.5
                             and angle(v["pos"], fur[a]["pos"], fur[b]["pos"]) <= 55]
                    away = [(float(np.linalg.norm(xz(v["pos"]) - xz(mid))), q) for q, v in fur.items()
                            if v["room"] != room and kind(q) in AWAY and np.linalg.norm(xz(v["pos"]) - xz(mid)) >= 5]
                    if not spots or not away:
                        continue
                    park = min((float(np.linalg.norm(xz(v["pos"]) - xz(lay["robot"]))), f) for f, v in fur.items()
                               if v["room"] != room)[1]
                    dx = float(np.linalg.norm(xz(ov["pos"]) - xz(fur[a]["pos"])))
                    score = abs(dab - 2.0) + min(spots)[0] + 0.1 * dx + 0.05 * min(away)[0]
                    near = sorted((float(np.linalg.norm(xz(v["pos"]) - xz(mid))), f) for f, v in here.items()
                                  if f not in (a, b, ov["furniture"]) and 1.2 <= np.linalg.norm(xz(v["pos"]) - xz(mid)) <= 6)
                    out.append({"episode": ep, "scene": lay["scene"], "room": room, "X": x, "X_from": ov["furniture"],
                                "spots": [f for _, f in near][:6],
                                "A": a, "B": b, "P": [p for _, p in sorted(spots)][:3], "Q": min(away)[1], "park": park,
                                "score": round(score, 3)})
    out.sort(key=lambda d: d["score"])
    picked, used = [], set()
    for d in out:  # distinct objects and rooms first
        if (d["X"], d["room"]) in used or (d["A"], d["B"]) in {(p["A"], p["B"]) for p in picked}:
            continue
        picked.append(d)
        used.add((d["X"], d["room"]))
        if len(picked) == per_episode:
            break
    return picked


CONDITIONS = ("left", "turned", "watched", "missed", "back_watched", "back_missed")


def tasks(designs: list[dict], keep: list[str]) -> list[dict]:
    by = {d["design"]: d for d in designs}
    return [{**by[k], "task": f"{k}_{c}", "cond": c, "seed": int(k[1:])} for k in keep for c in CONDITIONS]


def main() -> None:
    if sys.argv[1] == "tasks":
        args = dict(a.split("=", 1) for a in sys.argv[4:])
        rows = tasks(json.load(open(sys.argv[2])), args["designs"].split(","))
        json.dump(rows, open(sys.argv[3], "w"), indent=1)
        print(f"{len(rows)} tasks")
        return
    lay = json.load(open(sys.argv[1]))
    args = dict(a.split("=", 1) for a in sys.argv[3:])
    rows = [d for ep, v in lay.items() for d in designs(ep, v, int(args.get("per_episode", 3)))]
    for i, d in enumerate(rows):  # bounce: another episode of the scene, for clean resets
        d["design"] = f"d{i:03d}"
        d["bounce"] = next(e for e, v in lay.items() if v["scene"] == d["scene"] and e != d["episode"])
    json.dump(rows, open(sys.argv[2], "w"), indent=1)
    scenes = {d["scene"] for d in rows}
    print(f"{len(rows)} designs over {len({d['episode'] for d in rows})} episodes and {len(scenes)} scenes")


if __name__ == "__main__":
    main()
