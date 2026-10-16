"""BIP (bip.py) on HD-EPIC video, every input real. A clip event every `clip` s carries the verb recognizer's calibrated
probabilities (helper/action_cls/stream.py's bank of test clips) and the object a detector sees touching a hand (YOLOE
boxes at 2 fps, data/bip/detect, against the hand boxes of hands_5fps); fixations are real Aria gaze on those boxes at
2 fps. Subgoals, their labels and objects come from the annotated task streams (data/bip/stream, data/bip/vocab); Gemini
writes the prior and the step lists (cached per video). Scored per clip against the annotated current subgoal, next
subgoal and done set (helper/datasets/hd_epic/tasks.py), and at the end the orderings against the annotated ones.

    python -m helper.collab.hdepic name=real bank=runs/action_cls/stream3/bank.npz [verbs=oracle] [objects=oracle] [gaze=0] [limit=n] [seed=k]
    python -m helper.collab.hdepic name=real1 bank=runs/action_cls/stream1/bank.npz clip=1

With verbs=oracle a clip's verb is its true one, read with confidence 0.9; with objects=oracle its object and receptacle
are the nouns of the last step the narrations say ends in it, matched to the detector's vocabulary. With
verbs=oracle_all the verb set is every HD-EPIC verb class (take, put and insert named pick, place and place) and a
clip's verb is that of the last narrated action ending in it; Gemini writes step lists over that set. With
background=1, a clip that completes no listed step reads verbs as often as clips of the training participants do. With bip=old, the
BIP from before the proposal (runs/collab_snapshots/2026-10-15_before_proposal) reads every narrated action's nouns
instead of clips, as in its own oracle setting; it is scored at the same clip times. With clip=1, the bank's clips are
1 s long and the person model is the one fitted on 1 s clips (data/collab/fit/person_clip1.json).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from helper.action_cls.config import Config
from helper.bip.io import fixated
from helper.datasets.hd_epic.data import load_framewise, load_narrations
from helper.datasets.hd_epic.tasks import load

from . import bip

ROOT = Path(__file__).resolve().parents[2]
DATA, HANDS = ROOT / "data/bip", ROOT / "data/hd_epic/features/hands_5fps"


def boxes(det: dict, i: int, min_score: float = 0.3) -> dict:
    """name -> (x1, y1, x2, y2) at detection frame i, the most confident box per name."""
    out = {}
    for c, s, *b in sorted(det["objects"][i], key=lambda d: d[1]) if 0 <= i < len(det["objects"]) else []:
        if s >= min_score:
            out[det["names"][c]] = b
    return out


def touched(B: dict, hands: np.ndarray) -> str | None:
    """The object whose box the hands cover most (at least a fifth of it); boxes over 30% of the frame are fixtures."""
    best, name = 0.2, None
    for n, (x1, y1, x2, y2) in B.items():
        area = max(1e-6, (x2 - x1) * (y2 - y1))
        if area > 0.3:
            continue
        for conf, cx, cy, w, h in hands:
            if conf > 0:
                ix = max(0.0, min(x2, cx + w / 2) - max(x1, cx - w / 2))
                iy = max(0.0, min(y2, cy + h / 2) - max(y1, cy - h / 2))
                if ix * iy / area > best:
                    best, name = ix * iy / area, n
    return name


def named(noun: str | None, vocab: list[str]) -> str | None:
    """The vocabulary name for a narration noun: itself, else the shortest name containing it or contained in it."""
    if not noun:
        return None
    hits = [v for v in vocab if v == noun] or sorted((v for v in vocab if noun in v or v in noun), key=len)
    return hits[0] if hits else None


@cache
def verb_set(every: bool, top: int = 80) -> tuple[tuple, dict]:
    """(the verb set, "other" last; verb class -> its name there): pick, place, open and close, or the `top` HD-EPIC
    verb classes most frequent on the training participants (Gemini's schema takes fewer than about 90), by key, with
    take, put and insert named as in the 4-verb set. A rarer verb counts as other."""
    names = dict(Config().verb_map)
    if every:
        vc = pd.read_csv(ROOT / "data/hd_epic/annotations/narrations-and-action-segments/HD_EPIC_verb_classes.csv")
        narr = load_narrations()
        freq = narr[~narr.participant_id.isin(Config().split["test"])].main_action_classes.map(
            lambda m: m[0][0] if len(m) else -1).value_counts()
        keep = [int(c) for c in freq.index if c >= 0][:top]
        names |= {int(i): k for i, k in zip(vc.id, vc.key) if int(i) in keep and int(i) not in names}
    return tuple(dict.fromkeys(list(bip.VERBS[:4]) + list(names.values()))) + ("other",), names


def narrated(vid: str, vocab: list[str], narr, names: dict) -> list[tuple[float, str, str | None, str | None]]:
    """(end, verb, object, receptacle) of every narrated action whose verb class is in `names`, by end."""
    rows = narr[narr.video_id == vid]
    verb = rows.main_action_classes.map(lambda m: m[0][0] if len(m) else -1)
    out = [(float(e), names[c], named(n[0] if len(n) else None, vocab),
            named(n[1] if len(n) > 1 else None, vocab) if names[c] == "place" else None)
           for e, n, c in zip(rows.end_timestamp, rows.nouns, verb) if c in names]
    return sorted(out, key=lambda r: r[0])


def background(narr, verbs: tuple, names: dict, clip: float) -> tuple:
    """P(verb read) on the clips of the training participants' videos: the verb of the last action in the set ending
    in the clip, else other. It stands in for the reads of a clip that completes no listed step."""
    counts = np.zeros(len(verbs))
    for _, rows in narr[~narr.participant_id.isin(Config().split["test"])].groupby("video_id"):
        verb = rows.main_action_classes.map(lambda m: names.get(m[0][0]) if len(m) else None)
        keep = verb.notna().to_numpy()
        ends, verb = rows.end_timestamp.to_numpy(float)[keep], verb.to_numpy()[keep]
        order = np.argsort(ends, kind="stable")
        ends, verb = ends[order], verb[order]
        grid = np.arange(rows.start_timestamp.min() + clip, rows.end_timestamp.max() + 1e-9, clip)
        for t, i in zip(grid, np.searchsorted(ends, grid, side="right") - 1):
            counts[verbs.index(verb[i]) if i >= 0 and ends[i] > t - clip else -1] += 1
    return tuple(counts / counts.sum())


SNAPSHOTS = {"old": "2026-10-15_before_proposal/collab/bip.py", "pre_acc": "2026-10-16_before_accumulator/bip.py"}


def snapshot(name: str):
    """An earlier bip.py from runs/collab_snapshots: old (before the proposal) or pre_acc (before the plan accumulator)."""
    spec = importlib.util.spec_from_file_location(f"bip_{name}", ROOT / "runs/collab_snapshots" / SNAPSHOTS[name])
    mod = sys.modules[f"bip_{name}"] = importlib.util.module_from_spec(spec)  # registered first: its dataclasses look it up
    spec.loader.exec_module(mod)
    return mod


def actions_read(vid: str, vocab: list[str], narr) -> list[tuple[float, list[str]]]:
    """(end, nouns in the vocabulary) of every narrated action that names one."""
    rows = narr[narr.video_id == vid]
    out = [(float(e), [m for m in (named(x, vocab) for x in n) if m]) for e, n in zip(rows.end_timestamp, rows.nouns)]
    return sorted((r for r in out if r[1]), key=lambda r: r[0])


def two(objs: list[str], names: list[str]) -> list[str]:
    """A subgoal's objects, at least two (bip.Belief takes [object, target]; the fallback step list needs both)."""
    return list(objs) + [x for x in names if x not in objs][:max(0, 2 - len(objs))]


def closure(nodes: list[str], edges) -> set:
    anc = bip.ancestors(len(nodes), [(nodes.index(u), nodes.index(v)) for u, v in edges])
    return {(u, v) for v in range(len(nodes)) for u in range(len(nodes)) if anc[v] >> u & 1}


def video(vid: str, clips: list[tuple[float, np.ndarray, int]], args: dict, task, narr=None, extra=None) -> dict | None:
    """extra: Params fields set for every video (the verb set, its prior and the background)."""
    st = json.loads((DATA / "stream" / f"{vid}.json").read_text())
    vocab = json.loads((DATA / "vocab/gemini-3.5-flash" / f"{vid}.json").read_text())
    det = json.loads((DATA / "detect" / f"{vid}.json").read_text())
    hands, (gaze, _) = np.load(HANDS / f"{vid}.npy").astype(np.float32), load_framewise(vid)
    nodes = st["nodes"]
    spec = {"title": st["title"], "nodes": nodes, "labels": st["labels"], "vocab": det["names"],
            "objects": {n: two(vocab.get(n, []), det["names"]) for n in nodes}}
    old, clip = args.get("bip") == "old", float(args.get("clip", 3.0))
    mod = snapshot(args["bip"]) if args.get("bip") else bip
    fitted = {} if old else {k: v for k, v in json.loads(  # the person model fitted on the clip length, never on test videos
        (ROOT / f"data/collab/fit/person_clip{clip:g}.json").read_text()).items() if k in bip.Params.__dataclass_fields__ and k != "clip"}
    P = mod.Params(seed=int(args.get("seed", 0)), cache=str(ROOT / "data/collab/prior_hdepic" / vid), **({"clip": clip} if not old else {}),
                   **fitted, **(extra or {}))
    bel = mod.Belief(spec, P)
    t0, t1 = st["t0"], st["t1"]
    events = [(t, "clip", (p, y)) for t, p, y in clips if t0 + clip <= t <= t1]
    if old:  # the old BIP reads each narrated action; the clips only mark when it is scored
        events += [(t, "action", objs) for t, objs in actions_read(vid, det["names"], narr) if t0 <= t <= t1]
    if args.get("gaze", "1") != "0" and gaze:
        events += [((det["first"] + i) / det["fps"], "fixation", i) for i in range(len(det["objects"]))
                   if t0 <= (det["first"] + i) / det["fps"] <= t1]
    hand = lambda t: hands[min(len(hands) - 1, round(t * 5))]
    every = args.get("verbs") == "oracle_all"
    verbs, names = verb_set(every)
    told = narrated(vid, det["names"], narr, names) if args.get("objects") == "oracle" or every else None
    res = {"current": [], "current3": [], "next1": [], "next3": [], "done": [], "done_none": []}  # done_none: "nothing is done"
    for t, kind, x in sorted(events, key=lambda e: (e[0], e[1] == "clip")):
        i = int(round(t * det["fps"])) - det["first"]
        B = boxes(det, i)
        if kind == "action":
            bel.action(t, x)
            continue
        if kind == "fixation":
            g = gaze[min(len(gaze) - 1, int(round(t * 30)))]
            visible = {n: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, (b[2] - b[0]) * (b[3] - b[1])) for n, b in B.items()}
            obj = fixated(tuple(g) if g else None, visible, {n for n, v in visible.items() if v[2] > 0.1})
            if obj:
                bel.fixation(t, obj, list(visible))
            continue
        p, y = x
        last = [r for r in told if t - clip < r[0] <= t] if told is not None else []  # what the narrations end in the clip
        if args.get("verbs") == "oracle":  # the true verb, read with confidence
            p = np.full(len(bip.VERBS), 0.025)
            p[y] = 0.9
        if every:
            p = np.full(len(verbs), 0.1 / (len(verbs) - 1))
            p[verbs.index(last[-1][1]) if last else -1] = 0.9
        if args.get("objects") == "oracle":
            obj, rec = last[-1][2:] if last else (None, None)
        else:
            obj, rec = touched(B, hand(t)), None
        if not old:
            bel.clip(t, p, obj, rec, list(B))
        truth = task.current(t)
        if truth in nodes and t > t0 + 2 * clip:
            res["current"].append(int(np.argmax(bel.current())) == nodes.index(truth))
            res["current3"].append(nodes.index(truth) in np.argsort(-bel.current())[:3])
            nxt = task.next(t)
            if nxt in nodes:
                res["next1"].append(int(np.argmax(bel.next())) == nodes.index(nxt))
                res["next3"].append(nodes.index(nxt) in np.argsort(-bel.next())[:3])
            done, d = task.done(t), bel.done()
            res["done"].append(np.mean([(d[k] > 0.5) == (n in done) for k, n in enumerate(nodes) if n in task.nodes]))
            res["done_none"].append(np.mean([n not in done for n in nodes if n in task.nodes]))
    if not res["current"]:
        return None
    E, true = bel.F.edge_probs(), closure(nodes, [e for e in st["edges"] if e[0] in nodes and e[1] in nodes])
    pred = {(u, v) for u in range(len(nodes)) for v in range(len(nodes)) if E[u, v] >= 0.5}
    tp = len(pred & true)
    f1 = 2 * tp / max(1, len(pred) + len(true)) if pred or true else 1.0
    return {"video": vid, "clips": len(res["current"]), **{k: float(np.mean(v)) if v else None for k, v in res.items()},
            "edge_f1": f1, "chance_current": 1 / len(nodes), "gemini_calls": len(bel.calls)}


def main():
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    b = np.load(args["bank"])
    tasks, out, narr = load(1), [], load_narrations()
    every, clip = args.get("verbs") == "oracle_all", float(args.get("clip", 3.0))
    verbs, names = verb_set(every)
    extra = {"verbs": verbs, "verb_prior": (1 / len(verbs),) * len(verbs)} if every else {}
    if args.get("background") == "1":
        extra["background"] = background(narr, verbs, names, clip)
        print("background:", {v: round(q, 3) for v, q in zip(verbs, extra["background"]) if q >= 0.01}, flush=True)
    for vid in sorted(set(b["video"].tolist()))[:int(args.get("limit", 10 ** 6))]:
        if vid not in tasks or not all((DATA / d / f"{vid}.json").exists() for d in ("stream", "detect", "vocab/gemini-3.5-flash")):
            continue
        m = b["video"] == vid
        r = video(vid, list(zip(b["t"][m], b["p"][m], b["y"][m])), args, tasks[vid], narr, extra)
        if r:
            out.append(r)
            print(f"{vid}: {r['clips']} clips, current {r['current']:.2f} (chance {r['chance_current']:.2f}), "
                  f"next top 3 {r['next3'] if r['next3'] is None else round(r['next3'], 2)}, done {r['done']:.2f}, edge F1 {r['edge_f1']:.2f}",
                  flush=True)
    w = np.array([r["clips"] for r in out], float)
    mean = lambda k: float(np.average([r[k] for r in out if r[k] is not None], weights=[r["clips"] for r in out if r[k] is not None]))
    summary = {"videos": len(out), "clips": int(w.sum()), **{k: mean(k) for k in ("current", "current3", "next1", "next3", "done", "done_none", "chance_current")},
               "edge_f1": float(np.mean([r["edge_f1"] for r in out]))}
    path = ROOT / "runs/collab/hdepic" / f"{args.get('name', 'real')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"args": args, "summary": summary, "videos": out}, indent=1))
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
