"""Open-vocabulary objects and hands on egocentric frames, with YOLO.

Objects come from YOLOE-26L prompted with object names (helper.llm.vocab can list them from a task text),
hands from a YOLOv9c hand detector (Bingsu/adetailer, hand_yolov9c.pt); YOLOE's own "hand" prompt misses
most hands in egocentric views. Weights live in models/yolo/. Both detectors together take 21.7 ms per
frame on one RTX PRO 4000 (53 ms at worst). Boxes are [x1, y1, x2, y2] in [0, 1] of the frame.

    from helper.perception.detect import Detector
    det = Detector()
    det.set_names(["frying pan", "spatula", "plate"])
    for objects, hands in det(batch_of_bgr_frames):     # objects: [cls, score, x1, y1, x2, y2]; hands: [score, x1, y1, x2, y2]
        ...

    python -m helper.perception.detect --video P04-20240414-175337 --names "frying pan,spatula,plate" \\
        --start 0 --end 60 --fps 2 --out dosa.json          # GPU node only
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "models" / "yolo"


class Detector:
    def __init__(self, objects: str = "yoloe-26l-seg.pt", hands: str = "hand_yolov9c.pt", px: int = 640,
                 conf: float = 0.2, hand_conf: float = 0.3):
        from ultralytics import YOLO, YOLOE, settings
        settings.update({"weights_dir": str(WEIGHTS)})  # the YOLOE text encoder downloads here once
        self.objects, self.hands = YOLOE(str(WEIGHTS / objects)), YOLO(str(WEIGHTS / hands))
        self.px, self.conf, self.hand_conf, self.names = px, conf, hand_conf, []

    def set_names(self, names: list[str]) -> None:
        self.names = list(names)
        self.objects.set_classes(self.names, self.objects.get_text_pe(self.names))

    def __call__(self, frames: list[np.ndarray]) -> list[tuple[list, list]]:
        """Per BGR frame: (objects, hands), boxes normalized by the frame size."""
        if not frames:
            return []
        h, w = frames[0].shape[:2]
        scale = np.array([w, h, w, h])
        out = []
        for ro, rh in zip(self.objects.predict(frames, imgsz=self.px, conf=self.conf, verbose=False),
                          self.hands.predict(frames, imgsz=self.px, conf=self.hand_conf, verbose=False)):
            boxes = lambda r: np.round(r.boxes.xyxy.cpu().numpy() / scale, 3).tolist()
            scores = lambda r: np.round(r.boxes.conf.cpu().numpy(), 2).tolist()
            out.append(([[int(c), s, *b] for c, s, b in zip(ro.boxes.cls.cpu().numpy(), scores(ro), boxes(ro))],
                        [[s, *b] for s, b in zip(scores(rh), boxes(rh))]))
        return out


def run(det: Detector, batches, names: list[str], fps: float, first: int) -> dict:
    """Detections over frame batches, in the format helper.perception.contacts reads.
    `first` is the index (at `fps`) of the first frame, so frame k is at (first + k) / fps seconds."""
    det.set_names(names)
    out = {"names": names, "fps": fps, "first": first, "objects": [], "hands": []}
    for batch in batches:
        for objects, hands in det(batch):
            out["objects"].append(objects)
            out["hands"].append(hands)
    return out


def main() -> None:
    from ..datasets.hd_epic import DATA
    from .frames import decode
    ap = argparse.ArgumentParser(description="YOLO objects and hands on one HD-EPIC video.")
    ap.add_argument("--video", required=True, help="HD-EPIC video id")
    ap.add_argument("--names", required=True, help="comma-separated object names")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=60.0)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    first = int(args.start * args.fps)
    batches = decode(DATA / "videos" / args.video[:3] / f"{args.video}.mp4", first / args.fps, args.end, args.fps)
    out = run(Detector(), batches, [n.strip() for n in args.names.split(",")], args.fps, first)
    args.out.write_text(json.dumps(out, separators=(",", ":")))
    print(f"{len(out['objects'])} frames -> {args.out}")


if __name__ == "__main__":
    main()
