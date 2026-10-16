"""VLM baselines for false-belief detection on recorded tasks (helper/partnr/fb_record.py).

The VLM gets what the partner-belief model gets: the person's view (one frame per second, the gaze drawn as a red
circle) and the ground truth of the world (every object's place over time, and who moved it, from a fixed action
set). With gaze_list it also gets the ordered list of what the person looked at. It answers at the query: where the
person thinks X is, and P(false belief).

    python -m helper.bip.fb_vlm <streams_dir> <out.json> [variant=frames|gaze_list] [model=gemini-3.5-flash] [workers=8]
"""

from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from helper.llm.gemini import Gemini, generate

from .fb_eval import moves, score

GUIDE = ("A false belief: the person thinks an object is somewhere it no longer is, because it was moved while they "
         "did not notice. The person knows only what they themselves saw: an object in their field of view may still "
         "go unnoticed if they were looking elsewhere, and they always know where they put things themselves.")
CIRCLE = "The red circle in each image marks where the person's eyes point."


def words(name: str) -> str:
    return re.sub(r"_\d+$", "", name).replace("_", " ")


def place_text(loc: str) -> str:
    fixed = {"person": "in the person's hand", "robot": "in the robot's gripper", "unknown": "somewhere unknown"}
    if loc in fixed or " " not in loc:
        return fixed.get(loc, loc)
    rel, f = loc.split(" ", 1)
    return f"{rel} the {words(f)} ({f})"


def world_log(st: dict) -> list[str]:
    """Every object's place at the start and every change, with the agent, up to the query."""
    q = st["flags"]["query"]
    out = [f"t = 0.0 s: {o} is {place_text(p)}" for o, p in sorted(st["world0"].items())]
    who = {"H": "the person", "R": "the robot"}
    for t, o, p, a in moves(st)[1:]:
        if t <= q:
            out.append(f"t = {t:.1f} s: {who[a]} moved {o}; it is now {place_text(p)}")
    return out


def gaze_list(st: dict) -> list[str]:
    q = st["flags"]["query"]
    return [f"{a:.1f} to {b:.1f} s: {x}" for a, b, x in st["fixations"] if a <= q]


def frames(st: dict, src: Path) -> tuple[list[bytes], list[float]]:
    """One frame per second up to the query: the kept frame nearest each whole second (frames are named t × 10)."""
    d, q = src.parent / "frames" / st["task"], st["flags"]["query"]
    saved = {int(p.stem) / 10: p for p in sorted(d.glob("*.jpg")) if int(p.stem) / 10 <= q}
    keep = sorted({min(saved, key=lambda t: abs(t - s)) for s in range(int(q) + 1)}) if saved else []
    return [saved[t].read_bytes() for t in keep], keep


def ask(st: dict, src: Path, variant: str, model: str, gaze_drawn: bool = True) -> dict:
    imgs, ts = frames(st, src)
    a, b = f"on {st['A']}", f"in {st['B']}"
    lines = [f"A robot watches a person through the person's smart glasses. The {len(imgs)} images are the person's "
             f"view at t = {', '.join(f'{t:g}' for t in ts)} s, in order.", GUIDE + (" " + CIRCLE if gaze_drawn else ""), "",
             "Ground truth of where things are (the robot's scene graph):", *world_log(st)]
    if variant == "gaze_list":
        lines += ["", "What the person's gaze rested on, in order, with how long (the object the glasses' gaze falls on):",
                  *gaze_list(st)]
    lines += ["", f"At t = {st['flags']['query']:.1f} s the person comes back for {st['X']}.",
              f"Where does the person think {st['X']} is: \"{place_text(a)}\" or \"{place_text(b)}\"? "
              f"Give p_false_belief, your probability that they think it is somewhere other than where it really is."]
    schema = {"type": "object", "required": ["reasoning", "believed", "p_false_belief"],
              "properties": {"reasoning": {"type": "string"}, "believed": {"type": "string", "enum": ["A", "B"]},
                             "p_false_belief": {"type": "number"}}}
    text = "\n".join(lines).replace(f'"{place_text(a)}" or "{place_text(b)}"', f'A ("{place_text(a)}") or B ("{place_text(b)}")')
    reply, usage, latency = generate(text, imgs, schema, cfg=Gemini(model=model, thinking="low"))
    try:
        r = json.loads(reply)
        p, top = min(1.0, max(0.0, float(r["p_false_belief"]))), (a if r["believed"] == "A" else b)
    except (ValueError, KeyError, TypeError):
        p, top, r = 0.5, None, {"error": reply[:200]}
    truth = a if st["belief"] == "A" else b
    return {"task": st["task"], "cond": st["cond"], "design": st["design"], "p_fb": round(p, 4), "fb": st["false_belief"],
            "believed": top, "truth_place": truth, "place_right": top == truth, "reasoning": r.get("reasoning", "")[:600],
            "usage": usage, "latency": round(latency, 2), "frames": len(imgs)}


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    args = dict(a.split("=", 1) for a in sys.argv[3:])
    variant, model = args.get("variant", "frames"), args.get("model", "gemini-3.5-flash")
    sts = [json.loads(p.read_text()) for p in sorted(src.glob("*.json"))]
    sts = [s for s in sts if not s.get("reject") and "query" in s.get("flags", {})]
    with ThreadPoolExecutor(int(args.get("workers", 8))) as pool:
        rows = list(pool.map(lambda s: ask(s, src, variant, model), sts))
    res = {"variant": variant, "model": model, "score": score(rows), "rows": rows}
    out.write_text(json.dumps(res, indent=1))
    print(variant, model, json.dumps(res["score"]), flush=True)


if __name__ == "__main__":
    main()
