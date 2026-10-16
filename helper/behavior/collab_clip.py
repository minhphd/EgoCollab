"""Clips of collaborative BEHAVIOR episodes (helper/behavior/collab.py): the person's view with gaze, the room view,
and the task's subgoals with who has done or is doing each, at 2x speed.

    python -m helper.behavior.collab_clip <out.mp4> <run_dir> <episode>[@<until s>] [<episode> ...]
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

from helper.partnr.fb_clip import GAZE_C
from helper.partnr.movie import BG, DONE, FPS, H, INK, MUTED, ROBOT, W, card, encode, font

VIEW = 600
PERSON_C, FREE_C = (226, 112, 20), (170, 170, 170)
POLICY = {"alone": "the person alone", "oracle": "with a robot that knows what the person is doing",
          "graph": "with a robot that takes a random ready step", "bip": "with BIP, silent",
          "bip+announce": "with BIP announcing each step it takes", "bip+ask": "with BIP asking before each step",
          "bip+tell": "with BIP telling what the person's gaze missed",
          "bip+tellnogaze": "with BIP telling what was out of the person's view",
          "bip+always": "with BIP telling every move"}


def name(n: str, names: dict) -> str:
    """The object's category from the stream, else from its name (a BEHAVIOR model id is 6 letters then an index)."""
    return names.get(n) or re.sub(r"_\d+$", "", re.sub(r"_[a-z]{6}_\d{1,2}$", "", n)).replace("_", " ")


def labels(st: dict) -> list[str]:
    """One label per step; steps that would read the same are numbered ("toy figure 2 in toy box")."""
    names, rel = st.get("names", {}), dict(ontop="on", inside="in", nextto="next to", under="under")
    base = [(name(s["obj"], names), rel.get(s["rel"], s["rel"]), name(s["target"], names)) for s in st["subgoals"]]
    seen = {}
    out = []
    for o, r, f in base:
        if base.count((o, r, f)) > 1:
            seen[(o, r, f)] = seen.get((o, r, f), 0) + 1
            o = f"{o} {seen[(o, r, f)]}"
        out.append(f"{o} {r} {f}")
    return out


def frames(st: dict, run: Path, until: float = float("inf")):
    eid = f"{st['activity']}_{st['policy']}_{st['seed']}"
    head = cv2.VideoCapture(str(run / "videos" / f"{eid}_head.mp4"))
    third = cv2.VideoCapture(str(run / "videos" / f"{eid}_third.mp4"))
    subs, names = st["subgoals"], labels(st)
    done_at = {e[2]: (e[0], e[3]) for e in st["events"] if e[1] == "done"}
    says = [(u["t"], who, u["text"]) for who in ("robot", "person") for u in st.get("utterances", {}).get(who, [])]
    trail = []
    for row in st["log"]:
        if row["t"] > until:
            break
        ok1, h = head.read()
        ok2, th = third.read()
        if not (ok1 and ok2):
            break
        t = row["t"]
        h = cv2.resize(cv2.cvtColor(h, cv2.COLOR_BGR2RGB), (VIEW, VIEW))
        trail = (trail + [row["gaze"]])[-8:]
        for j, g in enumerate(trail):
            last = j == len(trail) - 1
            cv2.circle(h, (int(g[0] * VIEW), int(g[1] * VIEW)), 14 if last else 5, GAZE_C, 4 if last else 2)
        img = Image.new("RGB", (W, H), BG)
        img.paste(Image.fromarray(h), (40, 140))
        img.paste(Image.fromarray(cv2.resize(cv2.cvtColor(th, cv2.COLOR_BGR2RGB), (VIEW, VIEW))), (40 + VIEW + 20, 140))
        d = ImageDraw.Draw(img)
        d.text((40, 28), f"{st['activity'].replace('_', ' ').capitalize()}: {POLICY.get(st['policy'], st['policy'])}",
               font=font(40, True, True), fill=INK)
        d.text((40, 84), f"{st['scene'].replace('_', ' ')} · {t:5.1f} s · video at 2x", font=font(24), fill=MUTED)
        d.text((40, 140 + VIEW + 8), "The person's view, gaze in orange", font=font(22), fill=MUTED)
        d.text((40 + VIEW + 20, 140 + VIEW + 8), "The room", font=font(22), fill=MUTED)
        x0, y = 40 + 2 * VIEW + 50, 140
        d.text((x0, y), "Steps", font=font(28, True), fill=INK)
        y += 46
        for i, s in enumerate(subs):
            v = f"v{i}"
            if v in done_at and done_at[v][0] <= t:
                who = done_at[v][1]
                mark, col = f"done by the {who}", ROBOT if who == "robot" else DONE
            elif row.get("cur") == v:
                mark, col = "the person is on it", PERSON_C
            elif row.get("robot_cur") == v:
                mark, col = "the robot is on it", ROBOT
            else:
                mark, col = "", FREE_C
            d.text((x0, y), names[i], font=font(22, bool(mark)), fill=INK if mark else MUTED)
            d.text((x0 + 20, y + 28), mark, font=font(20), fill=col)
            y += 62
        said = [(tt, who, text) for tt, who, text in says if tt <= t < tt + 4.0]
        for k, (_, who, text) in enumerate(said[-2:]):
            d.text((40, 140 + VIEW + 50 + 44 * k), f"{who.capitalize()}: \"{text}\"", font=font(28, True),
                   fill=ROBOT if who == "robot" else PERSON_C)
        yield np.asarray(img)


def main() -> None:
    out, run = Path(sys.argv[1]), Path(sys.argv[2])
    eps = [(e.split("@")[0], float(e.split("@")[1]) if "@" in e else float("inf")) for e in sys.argv[3:]]
    sts = [(json.loads((run / "streams" / f"{e}.json").read_text()), until) for e, until in eps]

    def gen():
        for st, until in sts:
            done = sum(1 for e in st["events"] if e[1] == "done" and e[0] <= until)
            span = f"the first {until:.0f} s" if until < st["seconds"] else f"{st['seconds']:.0f} s"
            spec = {"title": f"{st['activity'].replace('_', ' ').capitalize()}", "size": 30, "secs": 5,
                    "subtitle": f"{st['scene'].replace('_', ' ')} · {POLICY.get(st['policy'], st['policy'])}",
                    "bullets": [f"{len(st['subgoals'])} steps; {done} done in {span}"
                                + ("" if st["finished"] or until < st["seconds"] else " (stopped before the end)") + ".",
                                "The person works from what they see; the robot works alongside. Video at 2x speed."]}
            yield from (card(spec) for _ in range(spec["secs"] * FPS))
            yield from frames(st, run, until)

    encode(gen(), out)
    small = out.with_name(out.stem + "_720p.mp4")
    subprocess.run([str(Path.home() / ".conda/envs/egocollab/bin/ffmpeg"), "-v", "error", "-y", "-i", str(out), "-vf",
                    "scale=1280:-2", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast", str(small)], check=True)
    print(out, small)


if __name__ == "__main__":
    main()
