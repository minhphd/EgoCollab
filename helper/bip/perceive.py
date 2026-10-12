"""Real perception, step 2: turns detections (detect.py) and the Aria 2D gaze into S1's observations,
written to data/bip/real/<video>.json. Objects are detector class names, not tracked instances.

- windows: every `window` s, the objects a hand touched (the hand box grown by `hand_margin` overlaps
  the object box; objects larger than `max_area` never count) and every object seen, reported at the
  window's end
- fixations: gaze held on one object for `fix_min` s; the fixated object is the smallest box holding the
  gaze point in the latest detection frame available at that time

Causal as a live system: a frame's detections count only `det_latency` s after the frame.

    python -m helper.bip.perceive build [workers=16]
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..datasets.hd_epic import load_framewise
from . import detect, stream
from .config import Config, parse

GAZE_HZ = 30


def path(cfg: Config, vid: str) -> Path:
    return Path(cfg.cache) / ("real" if cfg.det_fps == cfg.fps else f"real_{cfg.det_fps:g}fps") / f"{vid}.json"


def read(cfg: Config, vid: str) -> dict:
    return json.loads(path(cfg, vid).read_text())


def touched(cfg: Config, objects: list, hands: list, names: list[str]) -> set[str]:
    out = set()
    for _, x1, y1, x2, y2 in hands:
        mx, my = (x2 - x1) * cfg.hand_margin, (y2 - y1) * cfg.hand_margin
        hx1, hy1, hx2, hy2 = x1 - mx, y1 - my, x2 + mx, y2 + my
        best = (0.0, None)
        for c, _, a1, b1, a2, b2 in objects:
            if (a2 - a1) * (b2 - b1) > cfg.max_area:
                continue
            inter = max(0.0, min(hx2, a2) - max(hx1, a1)) * max(0.0, min(hy2, b2) - max(hy1, b1))
            if inter > best[0]:
                best = (inter, names[c])
        if best[1]:
            out.add(best[1])
    return out


def observe(args: tuple[Config, str]) -> dict:
    cfg, vid = args
    det, st = json.loads(detect.path(cfg, vid).read_text()), stream.read(cfg, vid)
    names, fps, first = det["names"], det["fps"], det["first"]
    n = len(det["objects"])
    frame = lambda t: min(n - 1, max(0, int(np.floor((t - cfg.det_latency) * fps + 1e-9)) - first))  # latest available
    seen = [sorted({names[o[0]] for o in objs}) for objs in det["objects"]]
    hit = [touched(cfg, objs, hands, names) for objs, hands in zip(det["objects"], det["hands"])]
    windows, t = [], st["t0"]
    while t < st["t1"]:
        ks = range(frame(t) + 1, frame(t + cfg.window) + 1)  # frames that became available in (t, t + window]
        windows.append({"t": round(t + cfg.window, 2), "handled": sorted(set().union(*(hit[k] for k in ks))),
                        "seen": sorted(set().union(*(seen[k] for k in ks)))})
        t += cfg.window
    gaze, fixations, run = load_framewise(vid)[0], [], None
    step = round(GAZE_HZ / cfg.gaze_hz)
    for i in range(max(0, round((st["t0"] - 5) * GAZE_HZ)), min(len(gaze), round(st["t1"] * GAZE_HZ)), step):
        t, g, obj = i / GAZE_HZ, gaze[i], None
        if g:
            k = frame(t)
            inside = [o for o in det["objects"][k] if o[2] <= g[0] <= o[4] and o[3] <= g[1] <= o[5]]
            if inside:
                obj = names[min(inside, key=lambda o: (o[4] - o[2]) * (o[5] - o[3]))[0]]
        if run and obj == run[1]:
            if not run[2] and t - run[0] >= cfg.fix_min:
                fixations.append({"t": round(t, 2), "object": obj, "visible": sorted(set(seen[frame(t)]) | {obj})})
                run[2] = True
            continue
        run = [t, obj, False] if obj else None
    return {"video": vid, "names": names, "windows": windows, "fixations": fixations}


def build(cfg: Config, workers: int) -> None:
    vids = [v for v in stream.split(cfg) if detect.path(cfg, v).exists()]
    path(cfg, "x").parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers) as pool:
        for out in pool.map(observe, [(cfg, v) for v in vids]):
            path(cfg, out["video"]).write_text(json.dumps(out))
            print(out["video"], len(out["windows"]), "windows,", sum(bool(w["handled"]) for w in out["windows"]),
                  "with a handled object,", len(out["fixations"]), "fixations", flush=True)


if __name__ == "__main__" and sys.argv[1] == "build":
    extra = [a for a in sys.argv[2:] if a.startswith("workers=")]
    build(parse([a for a in sys.argv[2:] if a not in extra] + ["split=all"]), int(extra[0].split("=")[1]) if extra else 16)
