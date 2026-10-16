"""Clips of PARTNR experiments: the person's view with gaze, both third-person views, and the belief.

The panel lists the subgoals with P(current) (bar), P(done) (grey bar), the truth (the person's current
subgoal in orange, done in green) and the robot's subgoal (blue). The belief comes from a closed-loop run
(the stream's robot_belief) or from an offline replay run with trace=True.

    python -m helper.partnr.clip <stream.json> <out.mp4> [replay_run.json]
"""

from __future__ import annotations

import bisect
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import cv2
import numpy as np

from helper.bip.config import Config

S, W_PANEL, FONT = 640, 640, cv2.FONT_HERSHEY_SIMPLEX


def at(rows: list[dict], t: float) -> dict | None:
    i = bisect.bisect_right([r["t"] for r in rows], t) - 1
    return rows[i] if i >= 0 else None


def active(segments: list, t: float):
    return next((seg[2] for seg in segments if seg[0] <= t < seg[1]), None)


def panel(st: dict, t: float, belief: dict | None, h: int) -> np.ndarray:
    img = np.full((h, W_PANEL, 3), 250, np.uint8)
    y = 22
    for line in textwrap.wrap(st["title"], 80)[:3]:
        cv2.putText(img, line, (10, y), FONT, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
        y += 18
    cv2.putText(img, f"t = {t:5.1f} s   robot: {st.get('robot', 'idle')}", (10, y + 4), FONT, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    y += 22
    done = {c["node"] for c in st["completions"] if c["t"] <= t}
    cur, rob = active(st["truth"]["segments"], t), active(st.get("robot_segments", []), t)
    row = max(14, min(40, (h - y - 24) // max(1, len(st["nodes"]))))
    for i, (v, label) in enumerate(zip(st["nodes"], st["labels"])):
        yy = y + i * row
        color = (60, 160, 40) if v in done else (20, 120, 230) if v == cur else (60, 60, 60)  # BGR
        tag = " [robot]" if v == rob else ""
        cv2.putText(img, (label + tag)[:80], (10, yy + 12), FONT, 0.4, color, 1, cv2.LINE_AA)
        if v == rob:
            cv2.rectangle(img, (6, yy), (W_PANEL - 6, yy + row - 4), (220, 140, 40), 1)
        if belief:
            pc, pd = belief["cur"][i], belief["done"][i]
            cv2.rectangle(img, (10, yy + 16), (10 + int(300 * pc), yy + 22), (230, 120, 40), -1)
            cv2.rectangle(img, (320, yy + 16), (320 + int(300 * pd), yy + 22), (170, 170, 170), -1)
    cv2.putText(img, "blue: P(current)   grey: P(done)   orange: truth current   green: done", (10, h - 10), FONT, 0.36,
                (90, 90, 90), 1, cv2.LINE_AA)
    return img


def make(st: dict, out: Path, belief: list[dict], cfg: Config = Config()) -> None:
    vids = Path(st["frames"]).parents[1] / "videos"
    head = cv2.VideoCapture(str(vids / f"{st['id']}_head.mp4"))
    third = cv2.VideoCapture(str(vids / f"{st['id']}_third.mp4"))
    fps, pts = st["gaze"]["fps"], st["gaze"]["points"]
    tmp = out.with_suffix(".raw.mp4")
    writer, k, trail = None, 0, []
    while True:
        ok, a = head.read()
        ok2, b = third.read()
        if not ok:
            break
        t = k / fps
        a = cv2.resize(a, (S, S))
        if k < len(pts):
            trail = (trail + [pts[k]])[-8:]
            for j, (x, y) in enumerate(trail):
                cv2.circle(a, (int(x * S), int(y * S)), 4 + j, (40, 40, 230), 1 if j < len(trail) - 1 else 3)
        top = cv2.resize(b, (S, S // 2)) if ok2 else np.zeros((S // 2, S, 3), np.uint8)
        pan = panel(st, t, at(belief, t), S // 2)
        frame = np.concatenate([a, np.concatenate([top, pan], 0)], 1)
        if writer is None:
            writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        k += 1
    if writer is not None:
        writer.release()
        subprocess.run([cfg.ffmpeg, "-v", "error", "-y", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24",
                        str(out)], check=True)
        tmp.unlink()


def main() -> None:
    st = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 3:
        run = json.loads(Path(sys.argv[3]).read_text())
        belief = [{"t": r["t"], "cur": r["cur"], "done": r["done"]} for r in run.get("trace", [])]
    else:
        belief = st.get("robot_belief", [])
    make(st, out, belief)
    print(out)


if __name__ == "__main__":
    main()
