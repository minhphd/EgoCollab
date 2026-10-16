"""Clips of false-belief tasks (helper/partnr/fb_record.py): the person's view with gaze, both third-person views,
where the object is and where the person thinks it is, at 2x speed, with a card per task.

    python -m helper.partnr.fb_clip <out.mp4> <record_dir> <task> [<task> ...] [intro=1]
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .movie import BG, DONE, FPS, H, INK, MUTED, W, card, encode, font, wrap

TRUE_C, FALSE_C, GAZE_C, X_C = DONE, (196, 40, 40), (226, 112, 20), (30, 90, 200)
EGO, THIRD = 840, 500  # pixel sizes on the canvas
STORY = {
    "left": "is in another room",
    "turned": "is in the room but faces away",
    "watched": "watches it",
    "missed": "has the move in view but looks elsewhere",
    "back_watched": "is away for the pick-up and comes back to watch the placement",
    "back_missed": "is away for the pick-up and comes back during the placement but looks elsewhere",
}


def words(name: str | None) -> str:
    """A readable name: "chest_of_drawers_52" -> "chest of drawers", BEHAVIOR's "bottom_cabinet_rvpunw_0" -> "bottom
    cabinet", a spawned "X_mug" -> "mug"."""
    n = re.sub(r"^X_", "", name or "nothing")
    n = re.sub(r"_[a-z]{6}_\d+$", "", n)
    return re.sub(r"_\d+$", "", n).replace("_", " ")


def belief_track(st: dict) -> list:
    """The person's belief about X over time, by the generator's rule: X's place when they attend X in view; seeing X
    go into the container (attended within 1.5 s before) counts."""
    track, loc = [], lambda t: max((r for r in st["x_track"] if r[0] <= t), key=lambda r: r[0])[1]
    cur, last_att = None, -9.0
    for r in st["log"]:
        t = r["t"]
        if loc(t) == "person" or (r["attend"] == st["X"] and "X" in r["vis"]):  # a holder knows what they hold
            cur, last_att = loc(t), t
        elif cur == "person":  # they know where they put it down
            cur = loc(t)
        elif cur == "robot" and loc(t).startswith("in ") and t - last_att <= 1.5:
            cur = loc(t)
        if not track or track[-1][1] != cur:
            track.append([t, cur])
    return track


def phase(st: dict, t: float) -> str:
    f, x, a, b = st["flags"], words(st["X"]), words(st["A"]), words(st["B"])
    q, look1 = f.get("query", 1e9), st["fb"]["look1"]
    if t < f.get("placed", 1e9):
        return f"The person puts the {x} on the {a}."
    if t < f.get("approach", 1e9):
        return "The person " + {"left": "leaves the room.", "turned": "turns away.", "back_watched": "leaves the room.",
                                "back_missed": "leaves the room."}.get(st["cond"], "steps back.")
    if t < f.get("moved", 1e9):
        return f"The robot moves the {x} into the {b}."
    if t < q - look1:
        return f"The person comes back for the {x}."
    if t < q:
        return f"First look: the {words(st['A'] if st['first_look'] == 'A' else st['B'])}."
    return f"The person goes to the {a if st['belief'] == 'A' else b}."


def describe(loc: str | None) -> str:
    if loc is None:
        return "not yet seen"
    if loc in ("robot", "person"):
        return f"in the {loc}'s hand"
    if " " not in loc:  # "unknown": they saw it taken and lost track of it
        return "they don't know"
    rel, f = loc.split(" ", 1)
    return f"{rel} the {words(f)}"


def task_card(st: dict, k: int, n: int) -> dict:
    x, a, b = words(st["X"]), words(st["A"]), words(st["B"])
    fb = "a false belief" if st["false_belief"] else "a true belief"
    return {"title": f"Task {k} of {n}: {st['cond'].replace('_', ' ')}", "secs": 7, "size": 30,
            "subtitle": f"{st['task']} · scene {st['scene']} · {st['room'].replace('_', ' ')}",
            "bullets": [f"The person puts the {x} on the {a}. The robot then moves it into the {b} while the person "
                        f"{STORY[st['cond']]}.",
                        f"Coming back, the person first looks at the {a if st['first_look'] == 'A' else b}.",
                        f"Ground truth: the person thinks the {x} is on the {a if st['belief'] == 'A' else b}, {fb}.",
                        "The query comes at the end of that first look. Video at 2x speed."]}


def frames(st: dict, rec: Path, k: int, n: int):
    head = cv2.VideoCapture(str(rec / "videos" / f"{st['task']}_head.mp4"))
    third = cv2.VideoCapture(str(rec / "videos" / f"{st['task']}_third.mp4"))
    belief, track, log = belief_track(st), st["x_track"], st["log"]
    q = st["flags"].get("query")
    x = words(st["X"])
    trail = []
    for i, row in enumerate(log):
        ok1, h = head.read()
        ok2, th = third.read()
        if not (ok1 and ok2):
            break
        t = row["t"]
        h = cv2.resize(cv2.cvtColor(h, cv2.COLOR_BGR2RGB), (EGO, EGO))
        trail = (trail + [row["gaze"]])[-8:]
        for j, g in enumerate(trail):
            cv2.circle(h, (int(g[0] * EGO), int(g[1] * EGO)), 6 if j < len(trail) - 1 else 16, GAZE_C, 2 if j < len(trail) - 1 else 4)
        img = Image.new("RGB", (W, H), BG)
        img.paste(Image.fromarray(h), (40, 150))
        th = cv2.cvtColor(th, cv2.COLOR_BGR2RGB)
        side = cv2.resize(th, (2 * THIRD, THIRD)) if th.shape[1] >= 1.5 * th.shape[0] else cv2.resize(th, (THIRD, THIRD))
        img.paste(Image.fromarray(side), (W - 40 - 2 * THIRD, 150))
        d = ImageDraw.Draw(img)
        d.text((40, 30), f"Task {k} of {n}: {st['cond'].replace('_', ' ')}", font=font(40, True, True), fill=INK)
        d.text((40, 88), phase(st, t), font=font(30), fill=INK)
        d.text((40, 150 + EGO + 8), "The person's view, gaze in orange", font=font(22), fill=MUTED)
        d.text((W - 40 - 2 * THIRD, 150 + THIRD + 8), "Behind the person, behind the robot" if th.shape[1] >= 1.5 * th.shape[0]
               else "The room", font=font(22), fill=MUTED)
        real = max((r for r in track if r[0] <= t), key=lambda r: r[0])[1]
        thinks = max((r for r in belief if r[0] <= t), key=lambda r: r[0])[1]
        wrong = thinks is not None and thinks != real
        y = 150 + THIRD + 60
        for label, value, color in ((f"Where the {x} is", describe(real), X_C),
                                    ("Where the person thinks it is", describe(thinks), FALSE_C if wrong else TRUE_C),
                                    ("The person looks at", words(row["read"]), GAZE_C)):
            d.text((W - 40 - 2 * THIRD, y), label, font=font(24), fill=MUTED)
            d.text((W - 40 - 2 * THIRD + 380, y), value, font=font(28, True), fill=color)
            y += 52
        x0, x1, yb = W - 40 - 2 * THIRD, W - 40, H - 70  # the episode's timeline, query marked
        d.rectangle([x0, yb, x1, yb + 10], fill=(220, 220, 214))
        t_end = log[-1]["t"]
        d.rectangle([x0, yb, x0 + (x1 - x0) * t / t_end, yb + 10], fill=MUTED)
        if q is not None:
            xq = x0 + (x1 - x0) * q / t_end
            d.line([xq, yb - 14, xq, yb + 24], fill=FALSE_C, width=4)
            d.text((xq - 30, yb + 26), "query", font=font(20), fill=FALSE_C)
        frame = np.asarray(img)
        yield frame
        nxt = log[i + 1]["t"] if i + 1 < len(log) else 1e9
        if q is not None and t < q <= nxt:  # hold the end of the first look for 2 s
            stop = Image.fromarray(frame.copy())
            sd = ImageDraw.Draw(stop)
            ans = f"on the {words(st['A'])}" if st["belief"] == "A" else f"in the {words(st['B'])}"
            msg = [f"Query: where does the person think the {x} is?",
                   f"Truth: {ans}, " + ("a false belief" if st["false_belief"] else "a true belief")]
            sd.rectangle([40, 150 + EGO - 130, 40 + EGO, 150 + EGO], fill=(255, 255, 255))
            for j, line in enumerate(msg):
                sd.text((56, 150 + EGO - 118 + 54 * j), line, font=font(28, True), fill=FALSE_C if j else INK)
            for _ in range(2 * FPS):
                yield np.asarray(stop)


INTRO = {"title": "False belief in a simulated home", "size": 30, "secs": 14,
         "subtitle": "A Sally-Anne task in PARTNR: the person puts an object down, the robot later moves it into a closed "
                     "container, and the person comes back for it.",
         "bullets": ["The person learns where the object is only by looking at it while it is in view.",
                     "Coming back, the first look goes to where they think it is (anticipatory looking) 80% of the time.",
                     "The query: at the end of that first look, where does the person think the object is?"],
         "table": {"header": ["Condition", "During the move", "Belief"],
                   "rows": [["left", "in another room", "false"], ["turned", "in the room, facing away", "false"],
                            ["watched", "in view, gaze follows the object", "true"],
                            ["missed", "in view, gaze elsewhere (same video as watched)", "false"],
                            ["back watched", "away for the pick, back for the placement, gaze follows", "true"],
                            ["back missed", "as back watched, gaze elsewhere (same video)", "false"]],
                   "widths": [0.2, 0.62, 0.18]}, "table_size": 26}


def main() -> None:
    out, rec = Path(sys.argv[1]), Path(sys.argv[2])
    names = [a for a in sys.argv[3:] if "=" not in a]
    args = dict(a.split("=", 1) for a in sys.argv[3:] if "=" in a)
    sts = [json.loads((rec / "streams" / f"{t}.json").read_text()) for t in names]
    gen = (fr for k, st in enumerate(sts, 1)
           for g in ((card(task_card(st, k, len(sts))) for _ in range(7 * FPS)), frames(st, rec, k, len(sts))) for fr in g)
    if args.get("intro", "1") == "1":
        intro = card(INTRO)
        gen = (fr for g in ((intro for _ in range(INTRO["secs"] * FPS)), gen) for fr in g)
    encode(gen, out)
    small = out.with_name(out.stem + "_720p.mp4")
    subprocess.run([str(Path.home() / ".conda/envs/egocollab/bin/ffmpeg"), "-v", "error", "-y", "-i", str(out), "-vf",
                    "scale=1280:-2", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast", str(small)], check=True)
    print(out, small)


if __name__ == "__main__":
    main()
