"""Gaze noise: false-belief detection and anticipation by the partner-belief model when the gaze readout is wrong, with
and without a fixation filter (MindConfig.min_fix).

    python -m helper.behavior.fb_noise <streams_dir> <out.json> [seeds=3]

offset  a calibration error: the gaze point moves a fixed distance (image widths) in a random direction per task
jitter  extra gaze noise per frame (image widths)
swap    tracking glitches: this share of 0.2 s windows reads a random entity in view, or nothing
aoi     the robot's readout radius: X is read when the gaze point is within aoi of its centre (the recorder's: 0.04)
Furniture reads stay, since furniture spans far more than these offsets. One image width is 63 degrees.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from helper.behavior.fb_analysis import adapt, predicted
from helper.behavior.fb_record import FB
from helper.bip.fb_eval import frames, run, score
from helper.bip.mind import MindConfig

NOISE = {"clean": {}, "offset_2deg": {"offset": 0.032}, "offset_4deg": {"offset": 0.063}, "jitter_2deg": {"jitter": 0.032},
         "glitch_10pct": {"swap": 0.10}, "glitch_20pct": {"swap": 0.20}, "offset_2deg_glitch_10pct": {"offset": 0.032, "swap": 0.10}}
AOI = (0.04, 0.095)  # 2.5 and 6 degrees
MIN_FIX = (0.0, 0.2)


def noisy(st: dict, rng, offset: float = 0.0, jitter: float = 0.0, swap: float = 0.0, aoi: float = FB.aoi,
          window: int = 3) -> dict:
    log, fr = [dict(r) for r in st["log"]], frames(st)
    a = rng.uniform(0, 2 * np.pi)
    off = offset * np.array([np.cos(a), np.sin(a)])
    glitch = None
    for i, r in enumerate(log):
        if offset or jitter or aoi != FB.aoi:
            g, x = np.asarray(r["gaze"]) + off + jitter * rng.standard_normal(2), r["vis"].get("X")
            if x is not None and np.hypot(g[0] - x[0], g[1] - x[1]) <= aoi:
                r["read"] = st["X"]
            elif r["read"] == st["X"]:
                r["read"] = None
        if swap and i % window == 0:  # a glitch holds for the window, on one entity
            glitch = None
            if rng.random() < swap:
                pool = sorted(fr[i][1]) + [None]
                glitch = ("on", pool[rng.integers(len(pool))])
        if glitch:
            r["read"] = glitch[1]
    return {**st, "log": log}


def anticipation(sts: list[dict], cfg: MindConfig) -> float:
    """Share of tasks where the model's predicted destination for X at the query is where the person went."""
    right = []
    for st in sts:
        f = st["flags"]
        went = st["A"] if f.get("went_for") == "A" and "redirected" not in f else st["B"]
        top = [(t, d) for t, d in predicted(st, "gaze", cfg) if t >= f["query"]]
        if top:
            right.append(top[0][1] == went)
    return round(float(np.mean(right)), 3)


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    args = dict(a.split("=", 1) for a in sys.argv[3:])
    sts = [adapt(json.loads(p.read_text())) for p in sorted(src.glob("*.json"))]
    sts = [s for s in sts if not s.get("reject") and "query" in s.get("flags", {})]
    res = {"n": len(sts), "grid": []}
    for name, kw in NOISE.items():
        for aoi in AOI:
            for mf in MIN_FIX:
                cfg, rows = MindConfig(min_fix=mf), []
                for seed in range(int(args.get("seeds", 3)) if kw else 1):
                    ns = [noisy(s, np.random.default_rng(seed), aoi=aoi, **kw) for s in sts]
                    sc = score([run(s, cfg) for s in ns])
                    rows.append({**{k: sc[k] for k in ("balanced_acc", "auroc", "brier", "by_condition")},
                                 "anticipation": anticipation(ns, cfg)})
                m = lambda k: round(float(np.mean([r[k] for r in rows])), 3)
                res["grid"].append({"noise": name, "aoi": aoi, "min_fix": mf, "balanced_acc": m("balanced_acc"),
                                    "auroc": m("auroc"), "brier": m("brier"), "anticipation": m("anticipation"),
                                    "by_condition": {c: round(float(np.mean([r["by_condition"][c] for r in rows])), 3)
                                                     for c in rows[0]["by_condition"]}})
                print(json.dumps({k: v for k, v in res["grid"][-1].items() if k != "by_condition"}), flush=True)
    dst.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
