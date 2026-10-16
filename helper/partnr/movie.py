"""One video of a closed-loop comparison: text cards, then every task as a card with its numbers and a 2 x 2
grid of the conditions (the person alone and each robot) played together at 2x speed.

Each grid cell shows the person's view with the gaze trail, a third-person view and the subgoals: the person's
current one in orange, the robot's boxed in blue, done ones in green, with the robot's belief as bars.
    python -m helper.partnr.movie <out.mp4> cards=<cards.json> alone=<dir> graph=<dir> bip=<dir> oracle=<dir> [workers=10]
cards.json: {"tasks": [episode ids], "intro": [card], "outro": [card]}; a card is {"title", "subtitle",
"bullets", "table": {"header", "rows", "widths", "hl"}, "secs"}. Each <dir> holds streams/ and videos/.
"""

from __future__ import annotations

import json
import subprocess
import sys
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from helper.bip.config import Config

from .clip import active, at
from .loop_score import finished, load, overlaps, violations

W, H, FPS = 1920, 1080, 30  # the recordings run at 15 fps, so one source frame per output frame is 2x speed
HEAD, CW, CH, BAR = 72, 960, 480, 24  # header, cell body, cell label bar
HOLD = 10.0  # s of task time a spoken line stays on screen
BG, INK, MUTED, HL = (246, 245, 240), (20, 33, 61), (62, 74, 92), (227, 236, 247)
DONE, CUR, ROBOT = (46, 125, 50), (226, 112, 20), (30, 90, 200)
COLOR = {"alone": (110, 110, 110), "graph": (123, 63, 160), "bip": (30, 90, 200), "oracle": (46, 125, 50),
         "bipv2": (200, 110, 30), "given": (30, 90, 200), "talker_once": (226, 112, 20), "talker_min": (240, 163, 94),
         "instruction": (142, 155, 179), "asks": (226, 112, 20)}
TTF = Path(matplotlib.get_data_path()) / "fonts" / "ttf"


@lru_cache(None)
def font(size: int, bold: bool = False, serif: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(TTF / f"DejaVu{'Serif' if serif else 'Sans'}{'-Bold' if bold else ''}.ttf"), size)


def wrap(text: str, f, width: int) -> list[str]:
    lines, cur = [], ""
    for word in str(text).split():
        if cur and f.getlength(cur + " " + word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    return lines + [cur] if cur else lines


def fit(text: str, f, width: int) -> str:
    while text and f.getlength(text) > width:
        text = text[:-2] + "…"
    return text


# -- cards ------------------------------------------------------------------------------------------
def card(spec: dict) -> np.ndarray:
    img = Image.new("RGB", (W, H), BG)
    d, x, width, s = ImageDraw.Draw(img), 120, W - 240, spec.get("size", 32)
    y = 90
    for line in wrap(spec["title"], font(58, True, True), width):
        d.text((x, y), line, font=font(58, True, True), fill=INK)
        y += 72
    for line in wrap(spec.get("subtitle", ""), font(s + 2), width):
        d.text((x, y + 6), line, font=font(s + 2), fill=MUTED)
        y += int(1.4 * (s + 2))
    y += 30
    for b in spec.get("bullets", []):
        d.text((x, y), "•", font=font(s), fill=INK)
        for line in wrap(b, font(s), width - 40):
            d.text((x + 40, y), line, font=font(s), fill=INK)
            y += int(1.38 * s)
        y += s // 2
    if spec.get("table"):
        table(d, x, y + 10, width, spec["table"], spec.get("table_size", s - 4))
    return np.asarray(img)


def table(d, x: int, y: int, width: int, t: dict, s: int) -> None:
    widths = [int(w * width) for w in t.get("widths", [1 / len(t["header"])] * len(t["header"]))]
    for r, row in enumerate([t["header"]] + t["rows"]):
        f = font(s, bold=r == 0 or r - 1 in t.get("hl", []))
        cells = [wrap(c, f, w - 24) for c, w in zip(row, widths)]
        h = max(len(c) for c in cells) * int(1.3 * s) + 16
        if r - 1 in t.get("hl", []):
            d.rectangle([x, y, x + width, y + h], fill=HL)
        cx = x
        for lines, w in zip(cells, widths):
            for i, line in enumerate(lines):
                d.text((cx + 8, y + 8 + i * int(1.3 * s)), line, font=f, fill=INK)
            cx += w
        d.line([x, y + h, x + width, y + h], fill=(200, 200, 195), width=2 if r == 0 else 1)
        y += h


def task_card(k: int, n: int, runs: dict, eid: str) -> dict:
    st = runs["alone"][eid]
    base = st["result"]["seconds"]
    edges = st["truth"]["edges"]
    rows = []
    for name, run in runs.items():
        s = run[eid]
        sec, rob = s["result"]["seconds"], len(s["result"].get("done_by_robot", []))
        rows.append([name, f"{sec:.1f} s" + ("" if finished(s) else " (not finished)"), f"{sec / base:.2f}",
                     str(overlaps(s["truth"]["segments"], s.get("robot_segments", []))), str(violations(s)),
                     f"{rob} of {len(s['nodes'])}", f"{100 * s['result']['percent_complete']:.0f}%"])
    return {"title": f"Task {k} of {n}: {eid}", "subtitle": st["title"], "size": 24, "secs": 9,
            "bullets": [f"{v}: {label}" for v, label in zip(st["nodes"], st["labels"])]
                       + ["Required orderings: " + (", ".join(f"{u} before {v}" for u, v in edges) or "none")],
            "table": {"header": ["", "Time", "vs alone", "Conflicts", "Violations", "Robot's subgoals", "PARTNR"],
                      "rows": rows, "widths": [0.16, 0.2, 0.12, 0.12, 0.12, 0.16, 0.12], "hl": [2]}, "table_size": 24}


# -- the grid ---------------------------------------------------------------------------------------
class Cell:
    """One condition's recording, frame by frame, as a 960 x 504 image (label bar and body)."""

    def __init__(self, name: str, st: dict, base: float):
        vids = Path(st["frames"]).parents[1] / "videos"
        self.head = cv2.VideoCapture(str(vids / f"{st['id']}_head.mp4"))
        self.third = cv2.VideoCapture(str(vids / f"{st['id']}_third.mp4"))
        self.name, self.st, self.fps = name, st, st["gaze"]["fps"]
        self.pts, self.trail, self.belief = st["gaze"]["points"], [], st.get("robot_belief", [])
        self.done_t = {c["node"]: c["t"] for c in st["completions"]}
        self.panels, self.last, self.ended = {}, None, False
        self.bar = self.label(base)

    def label(self, base: float) -> np.ndarray:
        img = Image.new("RGB", (CW, BAR), (235, 233, 226))
        d, s = ImageDraw.Draw(img), self.st
        d.rectangle([0, 0, 10, BAR], fill=COLOR.get(self.name, INK))
        d.text((18, 2), self.name, font=font(17, True), fill=INK)
        sec = s["result"]["seconds"]
        info = f"{'finished' if finished(s) else 'stopped, not finished,'} at {sec:.1f} s ({sec / base:.2f} of alone)"
        if self.name != "alone":
            info += (f"   conflicts {overlaps(s['truth']['segments'], s.get('robot_segments', []))}   violations"
                     f" {violations(s)}   robot did {len(s['result']['done_by_robot'])} of {len(s['nodes'])}"
                     + (f"   wrong {s['result']['robot_wrong']}" if s["result"].get("robot_wrong") else ""))
        d.text((18 + font(17, True).getlength(self.name) + 16, 3), info, font=font(15), fill=MUTED)
        return np.asarray(img)

    def panel(self, t: float) -> np.ndarray:
        s = self.st
        done = frozenset(v for v, tt in self.done_t.items() if tt <= t)
        cur, rob, b = active(s["truth"]["segments"], t), active(s.get("robot_segments", []), t), at(self.belief, t)
        key = (done, cur, rob, None if b is None else b["t"])
        if key not in self.panels:
            img = Image.new("RGB", (CW // 2, CH // 2), (252, 251, 248))
            d, n = ImageDraw.Draw(img), len(s["nodes"])
            top = 4
            if b is not None and "w" in b:  # bipv2: the posterior over recipes first
                for r, p in sorted(b["w"].items()):
                    true = r == s.get("recipe")
                    d.text((8, top), fit(r.replace("_", ", "), font(12, true), 200), font=font(12, true), fill=INK)
                    d.rectangle([215, top + 4, 215 + int(240 * p), top + 11], fill=ROBOT if true else (160, 165, 175))
                    top += 16
                seg = next((x for x in s.get("robot_segments", []) if x[0] <= t < x[1]), None)
                wanted = {o for v in s["nodes"] for o in ((s["spec"].get(v) or {}).get("objects") or [(s["spec"].get(v) or {}).get("object")])}
                if seg is not None and len(seg) > 3 and seg[3] and seg[3] not in wanted:  # the object it handles serves no subgoal
                    d.text((8, top), f"robot fetching {seg[3]}: not in the task", font=font(12, True), fill=(200, 40, 40))
                    top += 16
            rh = min(30, (CH // 2 - 4 - top) // n)
            for i, (v, label) in enumerate(zip(s["nodes"], s["labels"])):
                y = top + i * rh
                color = DONE if v in done else CUR if v == cur else INK
                d.text((8, y + 1), fit(f"{v}  {label}", font(13, v == cur), 320), font=font(13, v == cur), fill=color)
                if v == rob:
                    d.rectangle([3, y - 1, CW // 2 - 4, y + rh - 3], outline=ROBOT, width=2)
                if b is not None:  # the robot's belief: P(current) and P(done)
                    pc, pd = (b["keys"].get(v, (0, 0, 0))[1:] if "keys" in b else (b["cur"][i], b["done"][i]))
                    d.rectangle([335, y + 3, 335 + int(130 * pc), y + 7], fill=(240, 165, 90))
                    d.rectangle([335, y + 10, 335 + int(130 * pd), y + 14], fill=(140, 200, 140))
            self.panels[key] = np.asarray(img)
        return self.panels[key]

    def frame(self, k: int) -> np.ndarray:
        if self.ended:
            return self.last
        ok, a = self.head.read()
        ok2, b = self.third.read()
        if not ok:
            self.ended = True
            self.last = self.freeze()
            return self.last
        t = k / self.fps
        a = cv2.cvtColor(cv2.resize(a, (CH, CH)), cv2.COLOR_BGR2RGB)
        if k < len(self.pts):
            self.trail = (self.trail + [self.pts[k]])[-8:]
            for j, (x, y) in enumerate(self.trail):
                cv2.circle(a, (int(x * CH), int(y * CH)), 3 + j, (230, 40, 40), 1 if j < len(self.trail) - 1 else 3)
        top = cv2.cvtColor(cv2.resize(b, (CW // 2, CH // 2)), cv2.COLOR_BGR2RGB) if ok2 else np.zeros((CH // 2, CW // 2, 3), np.uint8)
        body = np.concatenate([a, np.concatenate([top, self.panel(t)], 0)], 1)
        self.last = self.bubbles(np.concatenate([self.bar, body], 0), t)
        return self.last

    def bubbles(self, frame: np.ndarray, t: float) -> np.ndarray:
        """Speech bubbles for what the robot and the person said in the last HOLD seconds: the robot's on the
        right in blue, the person's on the left in orange."""
        said = [x for x in self.st.get("dialogue", []) if x[0] <= t < x[0] + HOLD][-2:]
        if not said:
            return frame
        img = Image.fromarray(frame.copy())
        d, y = ImageDraw.Draw(img), BAR + 10
        for x_ in said:
            who, text = x_[1], x_[2]
            lines = wrap(f"{who}: {text}", font(17), 580)[:4]
            h, x = 24 * len(lines) + 16, CW - 612 if who == "Robot" else 12
            d.rounded_rectangle([x, y, x + 600, y + h], 12, fill=(226, 235, 250) if who == "Robot" else (252, 232, 214),
                                outline=ROBOT if who == "Robot" else CUR, width=2)
            for k, line in enumerate(lines):
                d.text((x + 12, y + 8 + 24 * k), line, font=font(17), fill=INK)
            y += h + 8
        return np.asarray(img)

    def freeze(self) -> np.ndarray:
        """The last frame, dimmed, with when the episode ended."""
        base = self.last if self.last is not None else np.zeros((BAR + CH, CW, 3), np.uint8)
        body = (0.45 * base[BAR:] + 0.55 * 250).astype(np.uint8)
        img = Image.fromarray(np.concatenate([base[:BAR], body], 0))
        d, sec = ImageDraw.Draw(img), self.st["result"]["seconds"]
        text = f"{'finished' if finished(self.st) else 'ended, not finished,'} at {sec:.1f} s"
        w, x, y = font(34, True).getlength(text), CW / 2, BAR + CH / 2
        d.rounded_rectangle([x - w / 2 - 24, y - 34, x + w / 2 + 24, y + 30], 12, fill=(255, 255, 255), outline=MUTED)
        d.text((x - w / 2, y - 22), text, font=font(34, True), fill=INK)
        return np.asarray(img)


def grid(k: int, n: int, runs: dict, eid: str):
    base = runs["alone"][eid]["result"]["seconds"]
    cells = [Cell(name, run[eid], base) for name, run in runs.items()]
    top = Image.new("RGB", (W, HEAD), BG)
    ImageDraw.Draw(top).text((24, 8), fit(f"Task {k} of {n}   {eid}   {runs['alone'][eid]['title']}", font(22, True), W - 48),
                             font=font(22, True), fill=INK)
    top = np.asarray(top)
    legend = ("orange: the person's subgoal    blue box: the robot's subgoal    green: done    "
              "bars: the robot's belief, P(current) and P(done)    red circles: gaze")
    f = 0
    while not all(c.ended for c in cells):
        frames = [c.frame(f) for c in cells]
        head = Image.fromarray(top.copy())
        ImageDraw.Draw(head).text((24, 40), f"t = {f / cells[0].fps:5.1f} s   (2x speed)    {legend}", font=font(18), fill=MUTED)
        canvas = np.full((H, W, 3), 250, np.uint8)
        canvas[:HEAD] = np.asarray(head)
        for i, fr in enumerate(frames):
            y, x = HEAD + (i // 2) * (BAR + CH), (i % 2) * CW
            canvas[y:y + BAR + CH, x:x + CW] = fr
        yield canvas
        f += 1
    for _ in range(FPS):  # hold the final state for a second
        yield canvas


# -- encoding ---------------------------------------------------------------------------------------
def encode(frames, out: Path) -> None:
    p = subprocess.Popen([Config().ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                          "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
    p.stdin.close()
    p.wait()


def still(spec: dict):
    img = card(spec)
    for _ in range(int(spec.get("secs", 12) * FPS)):
        yield img


def part(job: tuple) -> str:
    kind, out, payload = job
    if kind == "card":
        encode(still(payload), out)
    else:
        k, n, dirs, eid = payload
        runs = {name: load(Path(d) / "streams") for name, d in dirs.items()}
        encode((fr for g in (still(task_card(k, n, runs, eid)), grid(k, n, runs, eid)) for fr in g), out)
    return str(out)


def main() -> None:
    out = Path(sys.argv[1])
    args = dict(a.split("=", 1) for a in sys.argv[2:])
    cards, workers = json.loads(Path(args.pop("cards")).read_text()), int(args.pop("workers", 10))
    parts = out.parent / f"{out.stem}_parts"
    parts.mkdir(parents=True, exist_ok=True)
    n = len(cards["tasks"])
    jobs = [("card", parts / f"a{i:02d}.mp4", c) for i, c in enumerate(cards["intro"])]
    jobs += [("task", parts / f"b{k:02d}.mp4", (k + 1, n, args, eid)) for k, eid in enumerate(cards["tasks"])]
    jobs += [("card", parts / f"c{i:02d}.mp4", c) for i, c in enumerate(cards.get("outro", []))]
    with Pool(workers) as pool:
        done = pool.map(part, jobs, chunksize=1)
    (parts / "list.txt").write_text("".join(f"file '{Path(p).resolve()}'\n" for p in done))
    subprocess.run([Config().ffmpeg, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(parts / "list.txt"),
                    "-c", "copy", str(out)], check=True)
    print(out)


if __name__ == "__main__":
    main()
