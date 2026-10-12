"""Frames of the person's view for S2 and the clip: a 2 fps JPEG cache, drawn with the gaze as a red glow.

<frame_cache>/<video>.jpg.npy holds the JPEG bytes of every frame back to back and <video>.end.npy the
end offset of each; frame k is at k / fps seconds.

    python -m helper.bip.view --shard 0/8      # caches every 8th HD-EPIC video; skips cached ones
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from ..datasets.hd_epic import DATA, load_framewise
from .config import Config, parse

GAZE_HZ = 30  # HD-EPIC gaze is per video frame


def video_path(vid: str) -> Path:
    return DATA / "videos" / vid[:3] / f"{vid}.mp4"


def cache(cfg: Config, vid: str) -> None:
    folder, s = Path(cfg.frame_cache), cfg.image_px
    cmd = [str(Path(sys.executable).with_name("ffmpeg")), "-v", "error", "-threads", "8", "-i", str(video_path(vid)),
           "-vf", f"fps={cfg.fps},scale={s}:{s}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    chunks, ends, size = [], [], 0
    while len(raw := proc.stdout.read(s * s * 3)) == s * s * 3:
        jpg = cv2.imencode(".jpg", np.frombuffer(raw, np.uint8).reshape(s, s, 3), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].ravel()
        chunks.append(jpg)
        size += len(jpg)
        ends.append(size)
    if proc.wait():
        raise RuntimeError(f"ffmpeg failed on {vid}")
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / f"{vid}.jpg.npy", np.concatenate(chunks))
    np.save(folder / f"{vid}.end.npy", np.array(ends, np.int64))  # written last: marks the video done


class Frames:
    """RGB frames of one cached video; indices past either end are clamped."""

    def __init__(self, cfg: Config, vid: str):
        self.buf = np.load(Path(cfg.frame_cache) / f"{vid}.jpg.npy", mmap_mode="r")
        self.ends = np.load(Path(cfg.frame_cache) / f"{vid}.end.npy")

    def __getitem__(self, k: int) -> np.ndarray:
        k = max(0, min(k, len(self.ends) - 1))
        start = self.ends[k - 1] if k else 0
        return cv2.imdecode(np.asarray(self.buf[start: self.ends[k]]), cv2.IMREAD_COLOR)[..., ::-1].copy()


class View:
    """The frame at t with the gaze of the last `gaze_window` s painted as a red glow (if cfg.gaze)."""

    def __init__(self, cfg: Config, vid: str):
        self.cfg, self.frames, self.gaze = cfg, Frames(cfg, vid), load_framewise(vid)[0]

    def image(self, t: float) -> np.ndarray:
        cfg, px = self.cfg, self.cfg.image_px
        img = cv2.resize(self.frames[round(t * cfg.fps)], (px, px), interpolation=cv2.INTER_AREA)
        idx = [round((t - k / GAZE_HZ) * GAZE_HZ) for k in range(int(cfg.gaze_window * GAZE_HZ))]
        points = [self.gaze[i] for i in idx if 0 <= i < len(self.gaze) and self.gaze[i]]
        if cfg.gaze and points:
            yy, xx = np.mgrid[0:px, 0:px] / px
            heat = sum(np.exp(-((xx - q[0]) ** 2 + (yy - q[1]) ** 2) / (2 * cfg.gaze_sigma ** 2)) for q in points)
            a = (cfg.gaze_alpha * heat / heat.max())[..., None]
            img = (img * (1 - a) + np.array([255, 0, 0]) * a).astype(np.uint8)
        return img

    def jpegs(self, t: float, n: int, gap: float) -> list[bytes]:
        """`n` frames `gap` s apart ending at t, oldest first."""
        times = [max(0.0, t - gap * (n - 1 - j)) for j in range(n)]
        return [cv2.imencode(".jpg", self.image(s)[..., ::-1])[1].tobytes() for s in times]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1", help="i/n: every n-th video starting at i")
    ap.add_argument("overrides", nargs="*", help="config key=value")
    args = ap.parse_args()
    cfg = parse(args.overrides)
    vids = sorted(p.stem for p in (DATA / "videos").glob("*/*.mp4"))
    i, n = map(int, args.shard.split("/"))
    for vid in vids[i::n]:
        if not (Path(cfg.frame_cache) / f"{vid}.end.npy").exists():
            start = time.time()
            cache(cfg, vid)
            print(f"{vid}: {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
