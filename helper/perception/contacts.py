"""Hand-object contacts and gaze fixations from detections (helper.perception.detect), causal as a live system.

A frame's detections count only `latency` s after the frame, and only frames already available are used.

- windows: every `window` s, the objects a hand touched (the hand box grown by `margin` of its size
  overlaps the object box; objects larger than `max_area` of the frame never count) and every object
  seen, reported at the window's end.
- fixations: gaze held `min_s` s on one object; the fixated object is the smallest box holding the gaze
  point in the latest detection frame available.

Objects are detector class names, not tracked instances.

    from helper.perception.contacts import windows, fixations
    w = windows(det, t0=0, t1=60)            # [{"t": 3.0, "handled": ["spatula"], "seen": [...]}, ...]
    f = fixations(det, gaze, t0=0, t1=60)    # [{"t": 4.2, "object": "plate", "visible": [...]}, ...]
"""

import numpy as np


def touched(objects: list, hands: list, names: list[str], margin: float = 0.15, max_area: float = 0.2) -> set[str]:
    """Names of the objects touched by any hand in one frame (one object per hand, the largest overlap)."""
    out = set()
    for _, x1, y1, x2, y2 in hands:
        mx, my = (x2 - x1) * margin, (y2 - y1) * margin
        hx1, hy1, hx2, hy2 = x1 - mx, y1 - my, x2 + mx, y2 + my
        best = (0.0, None)
        for c, _, a1, b1, a2, b2 in objects:
            if (a2 - a1) * (b2 - b1) > max_area:
                continue
            inter = max(0.0, min(hx2, a2) - max(hx1, a1)) * max(0.0, min(hy2, b2) - max(hy1, b1))
            if inter > best[0]:
                best = (inter, names[c])
        if best[1]:
            out.add(best[1])
    return out


def latest(det: dict, latency: float):
    """Index of the latest detection frame available at time t."""
    n, fps, first = len(det["objects"]), det["fps"], det["first"]
    return lambda t: min(n - 1, max(0, int(np.floor((t - latency) * fps + 1e-9)) - first))


def windows(det: dict, t0: float, t1: float, window: float = 3.0, latency: float = 0.05,
            margin: float = 0.15, max_area: float = 0.2) -> list[dict]:
    names, frame = det["names"], latest(det, latency)
    seen = [{names[o[0]] for o in objs} for objs in det["objects"]]
    hit = [touched(objs, hands, names, margin, max_area) for objs, hands in zip(det["objects"], det["hands"])]
    out, t = [], t0
    while t < t1:
        ks = range(frame(t) + 1, frame(t + window) + 1)  # frames that became available in (t, t + window]
        out.append({"t": round(t + window, 2), "handled": sorted(set().union(*(hit[k] for k in ks))),
                    "seen": sorted(set().union(*(seen[k] for k in ks)))})
        t += window
    return out


def fixations(det: dict, gaze: list, t0: float, t1: float, gaze_hz: float = 30.0, hz: float = 10.0,
              min_s: float = 0.3, latency: float = 0.05) -> list[dict]:
    """`gaze` holds one normalized [x, y] (or None) per sample at `gaze_hz`, as hd_epic.load_framewise returns."""
    names, frame = det["names"], latest(det, latency)
    out, run, step = [], None, max(1, round(gaze_hz / hz))
    for i in range(max(0, round(t0 * gaze_hz)), min(len(gaze), round(t1 * gaze_hz)), step):
        t, g, obj = i / gaze_hz, gaze[i], None
        k = frame(t)
        if g:
            inside = [o for o in det["objects"][k] if o[2] <= g[0] <= o[4] and o[3] <= g[1] <= o[5]]
            if inside:
                obj = names[min(inside, key=lambda o: (o[4] - o[2]) * (o[5] - o[3]))[0]]
        if run and obj == run[1]:
            if not run[2] and t - run[0] >= min_s:
                out.append({"t": round(t, 2), "object": obj,
                            "visible": sorted({names[o[0]] for o in det["objects"][k]} | {obj})})
                run[2] = True
            continue
        run = [t, obj, False] if obj else None
    return out
