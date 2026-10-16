"""Body clearance in recorded BEHAVIOR tasks: for every logged position of the person and the robot, the distance to
the nearest obstacle in the scene's floor map (floor_trav_0.png, 1 cm per pixel, furniture included). Under the body's
radius means the body overlaps furniture or a wall.

    python -m helper.behavior.clearance <streams_dir> <maps_dir> [radius=0.25]
maps_dir holds <scene>/floor_trav_no_door_0.png (doors open; else floor_trav_0.png), copied from
BEHAVIOR-1K/datasets/behavior-1k-assets/scenes/<scene>/layout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RES = 0.01  # m per pixel


def clearance_map(png: Path) -> np.ndarray:
    import cv2
    m = (cv2.imread(str(png), cv2.IMREAD_GRAYSCALE) == 255).astype(np.uint8)
    return cv2.distanceTransform(m, cv2.DIST_L2, 5) * RES


def at(dist: np.ndarray, xy) -> float:
    """OmniGibson's world_to_map: row from y, column from x, the origin at the map's centre."""
    n = dist.shape[0]
    r, c = int(xy[1] / RES + n / 2), int(xy[0] / RES + n / 2)
    return float(dist[r, c]) if 0 <= r < n and 0 <= c < n else 0.0


def check(st: dict, dist: np.ndarray, radius: float) -> dict:
    out = {}
    for who in ("person", "robot"):
        if who == "robot" and st.get("policy") == "alone":  # parked off the map
            continue
        rows = [(r["t"], r[who]) for r in st.get("log", []) if r.get(who) is not None]
        if not rows:
            continue
        d = np.array([at(dist, p[:2]) for _, p in rows])
        bad = [t for (t, _), x in zip(rows, d) if x < radius]
        out[who] = {"min_m": round(float(d.min()), 3), "share_under": round(len(bad) / len(d), 3),
                    "first_under_s": bad[:3]}
    return out


def main() -> None:
    streams, maps = Path(sys.argv[1]), Path(sys.argv[2])
    args = dict(a.split("=", 1) for a in sys.argv[3:])
    radius, cache = float(args.get("radius", 0.25)), {}
    for f in sorted(streams.glob("*.json")):
        st = json.loads(f.read_text())
        scene = st["scene"]
        if scene not in cache:
            png = maps / scene / "floor_trav_no_door_0.png"  # the runs open every door
            cache[scene] = clearance_map(png if png.exists() else maps / scene / "floor_trav_0.png")
        print(f.stem, json.dumps(check(st, cache[scene], radius)))


if __name__ == "__main__":
    main()
