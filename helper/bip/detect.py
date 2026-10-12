"""Real perception, step 1: detections on the 2 fps frames of each task, written to data/bip/detect/<video>.json.

Objects come from YOLOE (open-vocabulary YOLO) prompted with the object names Gemini lists from the task
text (llm.vocab), and hands from a YOLO hand detector. Boxes are [x1, y1, x2, y2] in [0, 1].

    CUDA_VISIBLE_DEVICES=0 python -m helper.bip.detect --shard 0/4      # one GPU per shard
    CUDA_VISIBLE_DEVICES=0 python -m helper.bip.detect --shard 0/4 det_fps=10   # decoded from the video
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import llm, stream
from .config import ROOT, Config, parse
from .view import Frames, video_path

WEIGHTS = ROOT / "models" / "yolo"


def path(cfg: Config, vid: str) -> Path:
    return Path(cfg.cache) / ("detect" if cfg.det_fps == cfg.fps else f"detect_{cfg.det_fps:g}fps") / f"{vid}.json"


def boxes(r, px: int, cls: bool) -> list[list]:
    xy = np.round(r.boxes.xyxy.cpu().numpy() / px, 3).tolist()
    score = np.round(r.boxes.conf.cpu().numpy(), 2).tolist()
    if not cls:
        return [[s, *b] for s, b in zip(score, xy)]
    return [[int(c), s, *b] for c, s, b in zip(r.boxes.cls.cpu().numpy(), score, xy)]


def batches(cfg: Config, vid: str, first: int, last: int, size: int = 64):
    """BGR frames first..last at det_fps: from the frame cache at its own rate, else decoded from the video."""
    if cfg.det_fps == cfg.fps:
        frames = Frames(cfg, vid)
        last = min(last, len(frames.ends) - 1)
        for i in range(first, last + 1, size):
            yield [frames[k][..., ::-1] for k in range(i, min(last + 1, i + size))]
        return
    s = cfg.image_px
    cmd = [str(Path(sys.executable).with_name("ffmpeg")), "-v", "error", "-threads", "8", "-ss", f"{first / cfg.det_fps:.3f}", "-i", str(video_path(vid)),
           "-t", f"{(last - first + 1) / cfg.det_fps:.3f}", "-vf", f"fps={cfg.det_fps},scale={s}:{s}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc, batch = subprocess.Popen(cmd, stdout=subprocess.PIPE), []
    while len(raw := proc.stdout.read(s * s * 3)) == s * s * 3:
        batch.append(np.frombuffer(raw, np.uint8).reshape(s, s, 3))
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
    proc.wait()


def detect(cfg: Config, vid: str, objects, hands) -> dict:
    st = stream.read(cfg, vid)
    names = sorted({n for ns in llm.vocab(cfg, st, llm.ids_for(cfg, st["nodes"], vid)).values() for n in ns})
    objects.set_classes(names, objects.get_text_pe(names))
    first = max(0, int((st["t0"] - 5) * cfg.det_fps))
    out = {"names": names, "fps": cfg.det_fps, "first": first, "objects": [], "hands": []}
    for imgs in batches(cfg, vid, first, int((st["t1"] + 1) * cfg.det_fps)):
        for ro, rh in zip(objects.predict(imgs, imgsz=cfg.det_px, conf=cfg.det_conf, verbose=False),
                          hands.predict(imgs, imgsz=cfg.det_px, conf=cfg.hand_conf, verbose=False)):
            out["objects"].append(boxes(ro, cfg.image_px, True))
            out["hands"].append(boxes(rh, cfg.image_px, False))
    return out


def main() -> None:
    from ultralytics import YOLO, YOLOE, settings
    settings.update({"weights_dir": str(WEIGHTS)})
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = parse(args.overrides + ["split=all"])
    objects, hands = YOLOE(str(WEIGHTS / cfg.det_model)), YOLO(str(WEIGHTS / cfg.hand_model))
    i, n = map(int, args.shard.split("/"))
    for vid in stream.split(cfg)[i::n]:
        if path(cfg, vid).exists():
            continue
        start = time.time()
        out = detect(cfg, vid, objects, hands)
        path(cfg, vid).parent.mkdir(parents=True, exist_ok=True)
        path(cfg, vid).write_text(json.dumps(out, separators=(",", ":")))
        print(f"{vid}: {len(out['objects'])} frames, {len(out['names'])} names, {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
