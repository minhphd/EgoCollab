"""Frames of an egocentric video: a JPEG cache at a fixed rate, and frames drawn with the gaze as a red glow.

<folder>/<video>.jpg.npy holds the JPEG bytes of every frame back to back and <video>.end.npy the end
offset of each; frame k is at k / fps seconds. data/frames448 holds every HD-EPIC video at 2 fps, 448 px.

    python -m helper.perception.frames --shard 0/8        # caches every 8th HD-EPIC video; skips cached ones

    from helper.perception.frames import Frames, View
    from helper.datasets.hd_epic import load_framewise
    view = View(Frames("data/frames448", vid), load_framewise(vid)[0])
    img = view.image(12.5)                     # RGB, 448 x 448, gaze of the last second as a glow
    jpgs = view.jpegs(12.5, n=4, gap=3.0)      # 4 JPEGs 3 s apart ending at 12.5 s
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "frames448"


def cache(video: Path, folder: Path, vid: str, fps: float = 2.0, px: int = 448, quality: int = 90) -> None:
    """Decode `video` at `fps`, square-resize to `px` and store the JPEGs as <folder>/<vid>.{jpg,end}.npy."""
    cmd = [str(Path(sys.executable).with_name("ffmpeg")), "-v", "error", "-threads", "8", "-i", str(video),
           "-vf", f"fps={fps},scale={px}:{px}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    chunks, ends, size = [], [], 0
    while len(raw := proc.stdout.read(px * px * 3)) == px * px * 3:
        jpg = cv2.imencode(".jpg", np.frombuffer(raw, np.uint8).reshape(px, px, 3), [cv2.IMWRITE_JPEG_QUALITY, quality])[1].ravel()
        chunks.append(jpg)
        size += len(jpg)
        ends.append(size)
    if proc.wait():
        raise RuntimeError(f"ffmpeg failed on {video}")
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / f"{vid}.jpg.npy", np.concatenate(chunks))
    np.save(folder / f"{vid}.end.npy", np.array(ends, np.int64))  # written last: marks the video done


def decode(video: Path, start: float, end: float, fps: float, px: int = 448, batch: int = 64):
    """BGR frames of `video` from `start` to `end` s at `fps`, in batches, straight from ffmpeg."""
    cmd = [str(Path(sys.executable).with_name("ffmpeg")), "-v", "error", "-threads", "8", "-ss", f"{start:.3f}",
           "-i", str(video), "-t", f"{end - start:.3f}", "-vf", f"fps={fps},scale={px}:{px}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc, out = subprocess.Popen(cmd, stdout=subprocess.PIPE), []
    while len(raw := proc.stdout.read(px * px * 3)) == px * px * 3:
        out.append(np.frombuffer(raw, np.uint8).reshape(px, px, 3))
        if len(out) == batch:
            yield out
            out = []
    if out:
        yield out
    proc.wait()


class Frames:
    """RGB frames of one cached video; indices past either end are clamped."""

    def __init__(self, folder: str | Path, vid: str):
        self.buf = np.load(Path(folder) / f"{vid}.jpg.npy", mmap_mode="r")
        self.ends = np.load(Path(folder) / f"{vid}.end.npy")

    def __len__(self) -> int:
        return len(self.ends)

    def __getitem__(self, k: int) -> np.ndarray:
        k = max(0, min(k, len(self.ends) - 1))
        start = self.ends[k - 1] if k else 0
        return cv2.imdecode(np.asarray(self.buf[start: self.ends[k]]), cv2.IMREAD_COLOR)[..., ::-1].copy()


class View:
    """The frame at t with the gaze of the last `window` s painted as a red glow.

    `gaze` holds one normalized [x, y] (or None) per video frame at `gaze_hz`, as hd_epic.load_framewise returns.
    """

    def __init__(self, frames: Frames, gaze: list, fps: float = 2.0, px: int = 448, gaze_hz: float = 30.0,
                 window: float = 1.0, sigma: float = 0.04, alpha: float = 0.55, show_gaze: bool = True):
        self.frames, self.gaze, self.fps, self.px, self.gaze_hz = frames, gaze, fps, px, gaze_hz
        self.window, self.sigma, self.alpha, self.show_gaze = window, sigma, alpha, show_gaze

    def image(self, t: float) -> np.ndarray:
        px = self.px
        img = cv2.resize(self.frames[round(t * self.fps)], (px, px), interpolation=cv2.INTER_AREA)
        idx = [round((t - k / self.gaze_hz) * self.gaze_hz) for k in range(int(self.window * self.gaze_hz))]
        points = [self.gaze[i] for i in idx if 0 <= i < len(self.gaze) and self.gaze[i]]
        if self.show_gaze and points:
            yy, xx = np.mgrid[0:px, 0:px] / px
            heat = sum(np.exp(-((xx - q[0]) ** 2 + (yy - q[1]) ** 2) / (2 * self.sigma ** 2)) for q in points)
            a = (self.alpha * heat / heat.max())[..., None]
            img = (img * (1 - a) + np.array([255, 0, 0]) * a).astype(np.uint8)
        return img

    def jpegs(self, t: float, n: int = 4, gap: float = 3.0) -> list[bytes]:
        """`n` JPEG frames `gap` s apart ending at t, oldest first."""
        times = [max(0.0, t - gap * (n - 1 - j)) for j in range(n)]
        return [cv2.imencode(".jpg", self.image(s)[..., ::-1])[1].tobytes() for s in times]


def main() -> None:
    from ..datasets.hd_epic import DATA
    ap = argparse.ArgumentParser(description="Cache HD-EPIC videos as JPEG frames.")
    ap.add_argument("--shard", default="0/1", help="i/n: every n-th video starting at i")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--px", type=int, default=448)
    ap.add_argument("--out", type=Path, default=CACHE)
    args = ap.parse_args()
    vids = sorted((DATA / "videos").glob("*/*.mp4"))
    i, n = map(int, args.shard.split("/"))
    for video in vids[i::n]:
        if not (args.out / f"{video.stem}.end.npy").exists():
            start = time.time()
            cache(video, args.out, video.stem, args.fps, args.px)
            print(f"{video.stem}: {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
