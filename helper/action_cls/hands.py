"""Hand boxes in every frame of the frame cache at `fps` (`hand_frames`), from the YOLO hand detector BIP's perception
uses (`hand_model`): per frame, the two most confident hands ordered left to right, each [confidence, cx, cy, w, h]
in [0, 1]; a lone hand takes the slot of its image half, and an empty slot is zeros. One float16 array [frames, 2, 5]
per video in `hands`; row k is the frame at k / fps s.

    CUDA_VISIBLE_DEVICES=0 python -m helper.action_cls.hands
"""
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from helper.perception.frames import Frames

from .config import parse


class Video(Dataset):
    def __init__(self, folder, vid):
        self.frames = Frames(folder, vid)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, k):
        return torch.from_numpy(self.frames[k])  # RGB, uint8 [H, W, 3]


def slots(det):
    """Detections [n, 5] (confidence, cx, cy, w, h) -> [2, 5]: the two most confident, left to right."""
    out = np.zeros((2, 5), np.float32)
    top = det[np.argsort(-det[:, 0])[:2]]
    if len(top) == 2:
        out[:] = top[np.argsort(top[:, 1])]
    elif len(top) == 1:
        out[int(top[0, 1] >= 0.5)] = top[0]
    return out


def main():
    cfg = parse(sys.argv[1:])
    from ultralytics import YOLO
    model = YOLO(str(cfg.hand_model))
    cfg.hands.mkdir(parents=True, exist_ok=True)
    vids = sorted(p.name[:-len(".end.npy")] for p in cfg.hand_frames.glob("*.end.npy"))
    for vid in vids:
        if (cfg.hands / f"{vid}.npy").exists():
            continue
        t0, rows = time.time(), []
        for x in DataLoader(Video(cfg.hand_frames, vid), batch_size=256, num_workers=cfg.workers):
            x = x.cuda().permute(0, 3, 1, 2).float() / 255  # a tensor in [0, 1] skips YOLO's letterboxing
            for r in model.predict(x, imgsz=x.shape[-1], conf=cfg.hand_conf, half=True, verbose=False):
                rows.append(slots(torch.cat([r.boxes.conf[:, None], r.boxes.xywhn], 1).cpu().numpy()))
        np.save(cfg.hands / f"{vid}.npy", np.stack(rows).astype(np.float16))
        print(f"{vid}: {len(rows)} frames, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
