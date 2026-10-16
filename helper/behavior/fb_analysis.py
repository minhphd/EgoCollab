"""Analyses of the BEHAVIOR-1K false-belief recordings (helper/behavior/fb_record.py).

    python -m helper.behavior.fb_analysis <streams_dir> <out.json> [vlm=gemini-3.5-flash]

policies     what each way of talking costs and saves the person, from the scene's walking distances: silent;
             announce before moving (delegation); ask before moving (clarification); tell after the move by BIP's
             value (correction), with and without gaze; always tell. The closed-loop recordings check the model.
beliefs      the partner-belief model (helper/bip/mind.py) offline: with gaze, without, gaze in the move only, first
             look only, and without the person's view (their position only); gaze is the entity each frame's gaze
             falls on, accumulated as dwell
anticipation where the person will go for X, as the robot predicts it while they look and walk: the share right at
             the query and when they start walking, and how long before arrival the prediction settles on the truth
vlm          Gemini on frames (no gaze drawn) and the world's true state, then with the person's gaze as dwell per
             entity, in order
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from helper.behavior.sim import WALK
from helper.bip.fb_eval import VARIANTS, run as mind_run, score
from helper.bip.decide import Costs, tell_value
from helper.bip.mind import MindConfig

OPEN_TAKE = 1.3  # s to open the container and take X
LOOK = 1.5  # s at the empty place before searching
TALK = {"announce": 2.5, "ask": 4.5, "tell": 2.5}  # s of the person's time: hear it (and answer a question)


def adapt(st: dict) -> dict:
    """A BEHAVIOR stream in the fields helper/bip's evaluations read (2 Hz view lists, fixations, the world)."""
    st = dict(st)
    st["views"] = [{"t": r["t"], "seen": r["seen"]} for r in st["log"] if r.get("seen") is not None]
    fixes, cur, since = [], None, 0.0
    for r in st["log"]:
        if r["read"] != cur:
            if cur is not None and r["t"] - since >= 0.3:
                fixes.append([since, r["t"], cur])
            cur, since = r["read"], r["t"]
    st["fixations"] = fixes
    st["world0"] = {st["X"]: f"on {st['X_from']}"}
    track, loc = [[0.0, f"on {st['X_from']}"]], None  # where X is, from the agents' picks and places
    for e in st["events"]:
        if e[3] != st["X"]:
            continue
        loc = e[1] if e[2] == "Pick" else f"{e[4]} {e[5]}" if e[2] == "Place" else None
        if loc and loc != track[-1][1]:
            track.append([e[0], loc])
    st["x_track"] = track
    return st


def presence(st: dict, r_m: float = 4.5) -> dict:
    """The stream as a robot without the person's view has it: their position only. Near the two places they count as
    seeing X unless X is inside a container; there is no gaze."""
    mid, track = np.asarray(st["mid"][:2]), st["x_track"]
    place = lambda t: [loc for t0, loc in track if t0 <= t][-1]
    log = []
    for r in st["log"]:
        near = float(np.hypot(*(np.asarray(r["person"][:2]) - mid))) <= r_m
        vis = {c: 0 for c in ("A", "B")} | ({"X": 0} if not place(r["t"]).startswith("in ") else {}) if near else {}
        log.append({**r, "vis": vis, "read": None})
    return {**st, "log": log, "views": []}


def retrieval(st: dict, believed: str, tell_t: float | None) -> float:
    """Seconds from setting off for X to holding it, from the walking distances; a tell before reaching A redirects."""
    d, go = st["dist"], st["flags"].get("go", 0.0)
    if believed == "B":
        return d["spot_B"] / WALK + OPEN_TAKE
    t_at_a = d["spot_A"] / WALK
    if tell_t is not None and tell_t - go < t_at_a:  # heard it on the way: turn to B from wherever they are
        walked = max(0.0, tell_t - go) * WALK
        return max(0.0, tell_t - go) + (walked + d["spot_B"]) / WALK + OPEN_TAKE  # back the way they came, at most
    return t_at_a + LOOK + st["fb"]["search_s"] + d["A_B"] / WALK + OPEN_TAKE


def tell_time(st: dict, name: str) -> float | None:
    """When each correcting policy tells: BIP's live tell; the same rule on the model without gaze; right after the move."""
    q = st["flags"]
    if name == "tell_value":
        return st["tells"][0]["t"] if st["tells"] else None
    if name == "tell_value_nogaze":
        p = mind_run(st, MindConfig(gaze=False))["p_fb"]
        return (st["tells"][0]["t"] if st["tells"] else q["moved"] + 0.5) if tell_value(p, 1.0, Costs()) > 0 else None
    return q["moved"] if name == "always_tell" else None


def policies(sts: list[dict]) -> dict:
    out = {}
    for name in ("silent", "announce", "ask", "tell_value", "tell_value_nogaze", "always_tell"):
        rows = []
        for st in sts:
            fb, q = st["false_belief"], st["flags"]
            tell = tell_time(st, name)
            knows = name in ("announce", "ask") or (tell is not None and tell <= q.get("go", 0.0))
            believed = "B" if knows or not fb else "A"
            r = retrieval(st, believed, tell if fb and not knows else None)
            talk = TALK["announce"] if name == "announce" else TALK["ask"] if name == "ask" else TALK["tell"] if tell is not None else 0.0
            wrong = believed == "A" and not (tell is not None and fb)
            rows.append({"retrieval": r, "talk": talk, "wrong_trip": wrong, "utterance": talk > 0, "needless": talk > 0 and not fb})
        a = lambda k: round(float(np.mean([r[k] for r in rows])), 3)
        out[name] = {"retrieval_s": a("retrieval"), "talk_s": a("talk"), "person_s": round(a("retrieval") + a("talk"), 3),
                     "wrong_trips": a("wrong_trip"), "utterances": a("utterance"), "needless_utterances": a("needless")}
    measured = [st["flags"]["got"] - st["flags"]["go"] for st in sts if "got" in st["flags"] and "go" in st["flags"]]
    model = [retrieval(st, "B" if (st["tells"] and st["tells"][0]["t"] <= st["flags"].get("go", 0)) or not st["false_belief"] else "A",
                       st["tells"][0]["t"] if st["tells"] else None) for st in sts if "got" in st["flags"]]
    out["check"] = {"closed_loop_measured_s": round(float(np.mean(measured)), 2) if measured else None,
                    "model_for_the_same_s": round(float(np.mean(model)), 2) if model else None, "n": len(measured)}
    return out


def predicted(st: dict, how: str, cfg: MindConfig | None = None) -> list[tuple]:
    """(t, predicted furniture the person goes to for X) from the move on, by the partner-belief model with gaze, with
    the first look only, without gaze, or with the person's position only (replayed), or by their walking heading."""
    from helper.bip.fb_eval import frames, moves
    from helper.bip.mind import Mind
    f = st["flags"]
    if how == "presence":
        return predicted(presence(st), "nogaze")
    if how == "heading":
        out = []
        for r in st["log"]:
            if r["t"] < f["moved"]:
                continue
            x, y, yaw = r["person"]
            ang = lambda p: abs((np.arctan2(p[1] - y, p[0] - x) - yaw + np.pi) % (2 * np.pi) - np.pi)
            out.append((r["t"], st["A"] if ang(st["A_top"]) <= ang(st["B_top"]) else st["B"]))
        return out
    m = Mind(cfg or MindConfig(**{"gaze": {"gaze": True}, "first_look": {"history_gaze": False}}.get(how, {"gaze": False})))
    ev = [(t, 0, "move", (o, p_, a_)) for t, o, p_, a_ in moves(st)] + [(t, 1, "view", (v, x)) for t, v, x in frames(st)]
    ev += [(f["moved"], 2, "seek", st["X"])] + [(tl["t"], 3, "told", st["X"]) for tl in st["tells"][:1] if "told" in f]
    out = []
    for t, _, kind, x in sorted(ev, key=lambda e: (e[0], e[1])):
        if kind == "move":
            m.move(t, *x)
        elif kind == "view":
            m.view(t, *x)
            if t >= f["moved"]:
                dest = [k for k in m.destination(st["X"]) if k in (st["A"], st["B"])]
                out.append((t, dest[0] if dest else None))
        elif kind == "seek":
            m.seek(t, x)
        else:
            m.inform(t, x)
    return out


def anticipation_by(sts: list[dict]) -> dict:
    """The three predictors of where the person goes for X, on the same recordings."""
    res = {}
    for how in ("gaze", "first_look", "nogaze", "presence", "heading"):
        rows = []
        for st in sts:
            f = st["flags"]
            went = st["A"] if f.get("went_for") == "A" and "redirected" not in f else st["B"]
            arrive = f.get("at_A") if went == st["A"] else f.get("at_B")
            if arrive is None:
                continue
            top = [(t, d) for t, d in predicted(st, how) if t <= arrive]
            at = lambda t0: next((d for t, d in top if t >= t0), None)
            settle = next((t for i, (t, d) in enumerate(top) if all(dd == went for _, dd in top[i:])), arrive)
            rows.append({"query": at(f["query"]) == went, "go": at(f.get("go", f["query"])) == went, "lead": arrive - settle,
                         "fb": st["false_belief"]})
        a = lambda k, rs=rows: round(float(np.mean([r[k] for r in rs])), 3) if rs else None
        res[how] = {"n": len(rows), "right_at_query": a("query"), "right_at_setting_off": a("go"), "lead_s": a("lead"),
                    "right_at_query_false_belief": a("query", [r for r in rows if r["fb"]]),
                    "right_at_query_true_belief": a("query", [r for r in rows if not r["fb"]])}
    return res


def anticipation(sts: list[dict]) -> dict:
    """The robot's predicted destination for X (the furniture of the believed place) against where the person went."""
    rows = []
    for st in sts:
        f = st["flags"]
        went = st["A"] if f.get("went_for") == "A" and "redirected" not in f else st["B"]
        arrive = f.get("at_A") if went == st["A"] else f.get("at_B")
        if arrive is None:
            continue
        log = [r for r in st["log"] if f["moved"] <= r["t"] <= arrive and r.get("dest")]
        top = [(r["t"], next(iter(r["dest"]))) for r in log]
        at = lambda t0: next((d for t, d in top if t >= t0), None)
        settle = next((t for i, (t, d) in enumerate(top) if all(dd == went for _, dd in top[i:])), arrive)
        rows.append({"query_right": at(f["query"]) == went, "go_right": at(f.get("go", f["query"])) == went,
                     "lead_s": arrive - settle, "walk_s": arrive - f.get("go", f["query"])})
    a = lambda k: round(float(np.mean([r[k] for r in rows])), 3) if rows else None
    return {"n": len(rows), "right_at_query": a("query_right"), "right_at_setting_off": a("go_right"),
            "lead_s": a("lead_s"), "walk_s": a("walk_s")}


def beliefs(sts: list[dict]) -> dict:
    out = {name: score([mind_run(st, MindConfig(**VARIANTS[name])) for st in sts]) for name in ("full", "nogaze", "history", "first_look")}
    out["presence"] = score([mind_run(presence(st), MindConfig(gaze=False)) for st in sts])
    return out


def vlm(sts: list[dict], src: Path, model: str) -> dict:
    from concurrent.futures import ThreadPoolExecutor
    from helper.bip import fb_vlm
    out = {}
    for variant in ("frames", "gaze_list"):
        with ThreadPoolExecutor(8) as pool:
            rows = list(pool.map(lambda s: fb_vlm.ask(s, src, variant, model, gaze_drawn=False), sts))
        out[variant] = {"score": score(rows), "rows": rows}
    return out


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    args = dict(a.split("=", 1) for a in sys.argv[3:])
    sts = [adapt(json.loads(p.read_text())) for p in sorted(src.glob("*.json"))]
    sts = [s for s in sts if not s.get("reject") and "query" in s.get("flags", {})]
    res = {"n": len(sts), "policies": policies(sts), "anticipation": anticipation(sts),
           "anticipation_by": anticipation_by(sts), "beliefs": beliefs(sts)}
    print(json.dumps({k: v for k, v in res.items()}, indent=1), flush=True)
    if args.get("vlm"):
        res["vlm"] = vlm(sts, src, args["vlm"])
        print(json.dumps({k: v["score"] for k, v in res["vlm"].items()}), flush=True)
    dst.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
