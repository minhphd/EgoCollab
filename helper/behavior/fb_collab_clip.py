"""Side-by-side clips of one false-belief task from two passes: the robot silent (left) and BIP telling the person
where the object is (right), aligned at the query, with the person's view, the room, what the robot says and the
person's time to get the object. Ends with a card of the pass-wide numbers.

    python -m helper.behavior.fb_collab_clip <out.mp4> <silent_dir> <heard_dir> <task> [<task> ...] [summary=<b1k_main.json>]
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from helper.partnr.fb_clip import GAZE_C, STORY, words
from helper.partnr.movie import BG, DONE, FPS, H, INK, MUTED, ROBOT, W, card, encode, font, wrap

EGO, ROOM, COL = 560, 330, 940  # pixel sizes; each pass gets one column
BEFORE, AFTER = 9.0, 2.5  # s of task time shown before the query and after the last pickup
SPEECH_S = 4.0  # s a spoken line stays on screen


class Run:
    """One recorded task: its stream, its two videos and a frame reader by task time."""

    def __init__(self, rec: Path, task: str):
        self.st = json.loads((rec / "streams" / f"{task}.json").read_text())
        self.f = self.st["flags"]
        self.times = np.array([r["t"] for r in self.st["log"]])
        self.caps = {k: cv2.VideoCapture(str(rec / "videos" / f"{task}_{k}.mp4")) for k in ("head", "third")}
        self.i, self.cache = -1, {}

    def at(self, t: float) -> tuple[int, dict, dict]:
        """Index, log row and frames ({head, third}) at task time t, holding the last frame past the end."""
        j = int(np.clip(np.searchsorted(self.times, t), 0, len(self.times) - 1))
        while self.i < j:
            got = {k: c.read() for k, c in self.caps.items()}
            if not all(ok for ok, _ in got.values()):
                break
            self.i, self.cache = self.i + 1, {k: fr for k, (_, fr) in got.items()}
        return j, self.st["log"][j], self.cache

    def status(self, t: float) -> str:
        f, x, a, b = self.f, words(self.st["X"]), words(self.st["A"]), words(self.st["B"])
        steps = [("picked", f"The robot moves the {x} into the {b}."), ("moved", f"The person comes back for the {x}."),
                 ("go", f"The person goes to the {a if f.get('went_for') == 'A' else b}."),
                 ("at_A", f"The {a} is empty: the person looks around."), ("searched", f"The person goes to the {b}."),
                 ("got", f"The person has the {x}.")]
        out = f"The person {STORY[self.st['cond']]}." if t >= f.get("approach", 0) else f"The person puts the {x} on the {a}."
        for k, s in steps:
            if k in f and t >= f[k]:
                out = s
        return out


def column(img: Image.Image, run: Run, t: float, x0: int, title: str, heard: bool) -> None:
    d = ImageDraw.Draw(img)
    f, q = run.f, run.f["query"]
    i, row, fr = run.at(t)
    d.text((x0, 112), title, font=font(34, True), fill=ROBOT if heard else INK)
    if fr:
        h = cv2.resize(cv2.cvtColor(fr["head"], cv2.COLOR_BGR2RGB), (EGO, EGO))
        for j, r in enumerate(run.st["log"][max(0, i - 7):i + 1][::-1]):
            g = r["gaze"]
            cv2.circle(h, (int(g[0] * EGO), int(g[1] * EGO)), 14 if j == 0 else 5, GAZE_C, 4 if j == 0 else 2)
        img.paste(Image.fromarray(h), (x0, 165))
        img.paste(Image.fromarray(cv2.resize(cv2.cvtColor(fr["third"], cv2.COLOR_BGR2RGB), (ROOM, ROOM))), (x0 + EGO + 20, 165))
    d.text((x0, 165 + EGO + 6), "The person's view, gaze in orange", font=font(20), fill=MUTED)
    d.text((x0 + EGO + 20, 165 + ROOM + 6), "The room", font=font(20), fill=MUTED)
    told = run.st["tells"][0]["t"] if run.st.get("tells") else None
    y = 165 + ROOM + 50
    if told is None or t < told:  # BIP's live estimate, before any line is spoken
        d.text((x0 + EGO + 20, y), "BIP: P(false belief)", font=font(22), fill=MUTED)
        d.text((x0 + EGO + 20, y + 30), f"{row['p_fb']:.2f}", font=font(34, True), fill=INK)
    y = 165 + EGO + 50
    if heard and told is not None and told <= t < told + SPEECH_S:
        say = f"Robot: \"The {words(run.st['X'])} is in the {words(run.st['B'])}.\""
        d.rounded_rectangle([x0, y - 10, x0 + COL - 40, y + 50], radius=14, fill=(227, 236, 247))
        d.text((x0 + 16, y), say, font=font(28, True), fill=ROBOT)
    elif not heard and told is not None and t >= told and t < told + SPEECH_S:
        d.text((x0 + 16, y), "Robot: (says nothing)", font=font(28), fill=MUTED)
    for k, line in enumerate(wrap(run.status(t), font(28), COL - 40)[:2]):
        d.text((x0, y + 76 + 38 * k), line, font=font(28), fill=INK)
    dt = t - q
    got = f.get("got", 1e9) - q
    if dt >= 0:
        done = dt >= got
        msg = f"Had the {words(run.st['X'])} {got:.1f} s after the query" if done else f"Since the query: {dt:.1f} s"
        d.text((x0, y + 170), msg, font=font(32, True), fill=DONE if done else INK)


def pair(silent: Run, heard: Run, k: int, n: int):
    st = heard.st
    fb = "a false belief" if st["false_belief"] else "a true belief"
    yield from (card({"title": f"Task {k} of {n}: {st['cond'].replace('_', ' ')}", "size": 30,
                      "subtitle": f"{st['task']} · scene {st['scene']} · {st['room'].replace('_', ' ')}",
                      "bullets": [f"The person puts the {words(st['X'])} on the {words(st['A'])}. The robot moves it into the "
                                  f"{words(st['B'])} while the person {STORY[st['cond']]}: {fb}.",
                                  "Left, the robot stays silent. Right, BIP tells the person when it is worth it.",
                                  "The two recordings are aligned at the query, the end of the person's first look on "
                                  "coming back. Video at 2x speed."]}) for _ in range(6 * FPS))
    end = max(r.f["got"] - r.f["query"] for r in (silent, heard)) + AFTER
    for dt in np.arange(-BEFORE, end, 1 / 15):  # recordings run at 15 fps: one task frame per output frame is 2x
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((40, 24), f"Task {k} of {n}: {st['cond'].replace('_', ' ')}, {fb}", font=font(40, True, True), fill=INK)
        d.text((40, 76), "query" if abs(dt) < 0.04 else ("before the query" if dt < 0 else "after the query"),
               font=font(24), fill=(196, 40, 40) if abs(dt) < 0.5 else MUTED)
        column(img, silent, silent.f["query"] + dt, 40, "Robot silent", False)
        column(img, heard, heard.f["query"] + dt, 40 + COL, "BIP tells the person", True)
        d.line([W // 2 - 10, 110, W // 2 - 10, H - 30], fill=(216, 214, 206), width=2)
        yield np.asarray(img)


def summary(path: Path) -> dict:
    m = json.loads(path.read_text())
    p = m["policies"]
    return {"title": "Correcting false beliefs in the closed loop", "size": 30, "secs": 10,
            "subtitle": f"BEHAVIOR-1K, {m['n']} tasks in which the person hears what the robot says",
            "bullets": [f"The person's time to get the object, plus listening: {p['tell_value']['person_s']:.1f} s with BIP's "
                        f"corrections, {p['silent']['person_s']:.1f} s with the robot silent, {p['announce']['person_s']:.1f} s "
                        f"announcing every move, {p['ask']['person_s']:.1f} s asking first.",
                        f"BIP never speaks needlessly; announcing, asking or telling without gaze speak needlessly after "
                        f"{100 * p['announce']['needless_utterances']:.0f}% of moves (the ones the person saw).",
                        "Gaze read as entities is what tells a seen move from a missed one."]}


def main() -> None:
    out, silent_dir, heard_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    names = [a for a in sys.argv[4:] if "=" not in a]
    args = dict(a.split("=", 1) for a in sys.argv[4:] if "=" in a)

    def gen():
        for k, task in enumerate(names, 1):
            yield from pair(Run(silent_dir, task), Run(heard_dir, task), k, len(names))
        if "summary" in args:
            spec = summary(Path(args["summary"]))
            yield from (card(spec) for _ in range(spec["secs"] * FPS))

    encode(gen(), out)
    small = out.with_name(out.stem + "_720p.mp4")
    subprocess.run([str(Path.home() / ".conda/envs/egocollab/bin/ffmpeg"), "-v", "error", "-y", "-i", str(out), "-vf",
                    "scale=1280:-2", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast", str(small)], check=True)
    print(out, small)


if __name__ == "__main__":
    main()
