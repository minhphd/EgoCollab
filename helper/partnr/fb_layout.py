"""Layouts of PARTNR episodes for designing false-belief tasks: rooms, furniture and objects with positions.

    python -m helper.partnr.fb_layout <out.json> [per_scene=2] [split=val]     # GPU node
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

from helper.bip.config import Config, parse

from .env import PARTNR, Sim
from .perceive import relation, sim_position
from .person import openable


def pick_episodes(split: str, per_scene: int) -> list[str]:
    eps = json.load(gzip.open(PARTNR / f"data/datasets/partnr_episodes/v0_0/{split}.json.gz"))["episodes"]
    by = defaultdict(list)
    for e in eps:
        by[e["scene_id"]].append(e["episode_id"])
    return [i for ids in by.values() for i in ids[:per_scene]]


def layout(sim) -> dict:
    wg = sim.env.full_world_graph
    room = lambda n: (lambda r: r.name if r is not None else None)(_try(lambda: wg.get_room_for_entity(wg.get_node_from_name(n))))
    pos = lambda n: [round(float(x), 3) for x in p] if (p := sim_position(sim, n)) is not None else None
    fur = {f.name: {"room": room(f.name), "pos": pos(f.name), "openable": openable(sim, f.name)}
           for f in wg.get_all_furnitures()}
    objs = {}
    for o in wg.get_all_objects():
        r = relation(sim, o.name)
        objs[o.name] = {"furniture": r[1] if r else None, "rel": r[0] if r else None, "pos": pos(o.name),
                        "room": fur.get(r[1], {}).get("room") if r else None}
    agent = lambda uid: [round(float(x), 3) for x in sim.sim.agents_mgr[uid].articulated_agent.base_pos]
    return {"scene": sim.episode.scene_id, "furniture": fur, "objects": objs, "robot": agent(0), "person": agent(1)}


def _try(f):
    try:
        return f()
    except Exception:
        return None


def main() -> None:
    out = Path(sys.argv[1]).resolve()  # Sim changes directory
    args = dict(a.split("=", 1) for a in sys.argv[2:])
    cfg = parse([f"split={args.get('split', 'val')}", "max_steps=10"], Config)
    ids = pick_episodes(cfg.split, int(args.get("per_scene", 2)))
    sim = Sim(cfg, ids, out.parent / "partnr_logs")
    rows = {}
    for i in ids:
        try:
            sim.reset(i)
            rows[i] = layout(sim)
            print(i, rows[i]["scene"], len(rows[i]["furniture"]), "furniture", len(rows[i]["objects"]), "objects", flush=True)
        except Exception as e:
            print(i, "FAILED", type(e).__name__, e, flush=True)
    out.write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
