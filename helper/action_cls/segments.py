"""HD-EPIC action segments labelled pick, place, open, close or other, with their split by participant, and the
video times whose features the windows need."""
import json

import numpy as np
import pandas as pd

from .config import Config


def segments(cfg: Config) -> pd.DataFrame:
    """One row per segment: video_id, participant_id, start, end (s), label, y (class index), split."""
    df = pd.read_pickle(cfg.hd_epic / "annotations/narrations-and-action-segments/HD_EPIC_Narrations.pkl")
    main = lambda r: (r.main_action_classes[0][0] if len(r.main_action_classes) else  # the main action's verb
                      r.verb_classes[0] if len(r.verb_classes) else -1)
    df = df.assign(start=df.start_timestamp.astype(float), end=df.end_timestamp.astype(float), verb=df.apply(main, axis=1))
    df = df[df.end > df.start]
    df["label"] = df.verb.map(cfg.verb_map).fillna("other")
    others = df[df.label == "other"]
    others = others.sample(n=min(cfg.other_max, len(others)), random_state=cfg.seed)
    df = pd.concat([df[df.label != "other"], others]).sort_values(["video_id", "start"])
    df["split"] = df.participant_id.map({p: s for s, ps in cfg.split.items() for p in ps})
    df["y"] = df.label.map({c: i for i, c in enumerate(cfg.classes)})
    return df[["video_id", "participant_id", "start", "end", "label", "y", "split", "narration"]].reset_index(drop=True)


def movements(cfg: Config) -> dict:
    """video -> [n, 2]: start and end (s) of every annotated object movement, from pick-up to placement, while the
    object is in hand. HD-EPIC annotates every movement, slight pushes too."""
    a = json.loads((cfg.hd_epic / "annotations/scene-and-object-movements/assoc_info.json").read_text())
    return {v: np.array([t["time_segment"] for o in objs.values() for t in o["tracks"]], float).reshape(-1, 2)
            for v, objs in a.items()}


def frame_times(cfg: Config, seg: pd.DataFrame) -> dict:
    """video -> the sorted frame times (s, on the fps grid) some window needs: windows end in
    [start + end_lo * duration, end + end_hi] and reach window / fps back. With `all_frames`, every frame of the video,
    as many as the frame cache at fps (`hand_frames`) holds."""
    if cfg.all_frames:
        return {v: np.arange(len(np.load(cfg.hand_frames / f"{v}.end.npy"))) / cfg.fps for v in sorted(seg.video_id.unique())}
    out = {}
    for vid, g in seg.groupby("video_id"):
        lo = np.floor((g.start + cfg.end_lo * (g.end - g.start)) * cfg.fps) - cfg.window
        hi = np.ceil((g.end + cfg.end_hi) * cfg.fps) + 1
        ks = set()
        for a, b in zip(lo.astype(int), hi.astype(int)):
            ks.update(range(max(0, a), b + 1))
        out[vid] = np.array(sorted(ks)) / cfg.fps
    return out


if __name__ == "__main__":
    import sys

    from .config import parse
    cfg = parse(sys.argv[1:])
    seg = segments(cfg)
    print(seg.groupby(["split", "label"]).size().unstack(fill_value=0))
    ft = frame_times(cfg, seg)
    n = sum(len(t) for t in ft.values())
    per_frame = (cfg.grid ** 2 * (cfg.pca_dim or 768) + 768 + cfg.grid ** 2 * 3) * 2  # patches, CLS, flow; float16
    print(f"videos {len(ft)}, frames {n} at {cfg.fps:g} fps, ~{n * per_frame / 1e9:.0f} GB of features")
