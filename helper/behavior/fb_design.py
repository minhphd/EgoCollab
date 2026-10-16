"""False-belief task designs in BEHAVIOR-1K home scenes (the task is in docs/false_belief_tasks.md).

A design is a surface A and an openable container B in the same room, 1 to 3.5 m apart, an object X of a kind that
belongs in that room (spawned on A from the BEHAVIOR object library), and furniture Q in another room where the
person goes when away. The watching spot and the checks come from validation in OmniGibson.

    python -m helper.behavior.fb_design <datasets_dir> <designs.json> [per_room=2]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HOMES = ("Beechwood", "Benevolence", "Ihlen", "Merom", "Pomaria", "Rs", "Wainscott", "house", "hotel")
SURFACES = ("breakfast_table", "coffee_table", "countertop", "desk", "console_table", "shelf", "nightstand")
CONTAINERS = ("bottom_cabinet", "top_cabinet", "fridge", "wardrobe", "cedar_chest", "bottom_cabinet_no_top")
AWAY = ("sofa", "bed", "armchair", "straight_chair", "desk", "breakfast_table", "coffee_table")
OBJECTS = {  # kinds of X per room type, spawned from the BEHAVIOR object library
    "kitchen": ("mug", "bowl", "apple", "coffee_cup", "kettle"),
    "dining_room": ("mug", "bowl", "plate", "coffee_cup"),
    "living_room": ("hardback", "vase", "toy_figure", "teddy_bear"),
    "bedroom": ("hardback", "alarm_clock", "notebook", "teddy_bear"),
    "childs_room": ("teddy_bear", "toy_figure", "hardback"),
    "private_office": ("notebook", "hardback", "mug"),
}


def scene_objects(path: Path) -> dict:
    """{name: (category, room, xy)} for the scene's objects with a room and a pose."""
    d = json.loads(path.read_text())
    init, reg = d["objects_info"]["init_info"], d["state"]["registry"]["object_registry"]
    out = {}
    for name, v in init.items():
        rooms, pose = v["args"].get("in_rooms") or [], reg.get(name, {}).get("root_link", {}).get("pos")
        if rooms and pose:
            out[name] = (v["args"]["category"], rooms[0], np.array(pose[:2]))
    return out


def designs(scene: str, objs: dict, per_room: int) -> list[dict]:
    out = []
    for room in sorted({r for _, r, _ in objs.values()}):
        kind = room.rsplit("_", 1)[0]
        if kind not in OBJECTS:
            continue
        here = {n: v for n, v in objs.items() if v[1] == room}
        pairs = sorted((abs(float(np.linalg.norm(here[a][2] - here[b][2])) - 2.0), a, b)
                       for a in here if here[a][0] in SURFACES for b in here if here[b][0] in CONTAINERS
                       if 1.0 <= np.linalg.norm(here[a][2] - here[b][2]) <= 3.5)
        mid = lambda a, b: (here[a][2] + here[b][2]) / 2
        used = set()
        for _, a, b in pairs:
            if a in used or b in used or len([d for d in out if d["room"] == room]) >= per_room:
                continue
            away = sorted((float(np.linalg.norm(v[2] - mid(a, b))), n) for n, v in objs.items()
                          if v[1] != room and v[0] in AWAY and np.linalg.norm(v[2] - mid(a, b)) >= 5)
            if not away:
                continue
            k = len([d for d in out if d["room"] == room])
            out.append({"scene": scene, "room": room, "A": a, "B": b, "Q": away[0][1],
                        "X_kind": OBJECTS[kind][k % len(OBJECTS[kind])], "A_kind": here[a][0], "B_kind": here[b][0],
                        "dist_AB": round(float(np.linalg.norm(here[a][2] - here[b][2])), 2)})
            used |= {a, b}
    return out


def main() -> None:
    root, dst = Path(sys.argv[1]), Path(sys.argv[2])
    args = dict(a.split("=", 1) for a in sys.argv[3:])
    scenes = sorted(p.name for p in (root / "behavior-1k-assets/scenes").iterdir()
                    if p.name.split("_")[0] in HOMES and "garden" not in p.name)
    rows = []
    for s in scenes:
        rows += designs(s, scene_objects(root / f"behavior-1k-assets/scenes/{s}/json/{s}_best.json"), int(args.get("per_room", 2)))
    prefix = args.get("prefix", "b")
    for i, d in enumerate(rows):
        d["design"] = f"{prefix}{i:03d}"
    dst.write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} designs over {len({d['scene'] for d in rows})} scenes and {len({(d['scene'], d['room']) for d in rows})} rooms")


if __name__ == "__main__":
    main()
