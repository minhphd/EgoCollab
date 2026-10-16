"""Features of every frame a window needs, one file per video, from the whole frame (never cropped): DINOv2's CLS
token and its patches pooled to a grid and projected by PCA, the optical flow between consecutive sampled frames
pooled to the same grid (where the hands move), and the gaze point. First fit the PCA, then one process per GPU:

    python -m helper.action_cls.extract pca=1
    CUDA_VISIBLE_DEVICES=0 python -m helper.action_cls.extract part=0/2
"""
import json
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .config import parse
from .segments import frame_times, segments

MEAN, STD = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def gaze_of(cfg, vid: str) -> dict:
    """frame index -> (x, y) in [0, 1] on the frame, for frames with gaze."""
    path = cfg.hd_epic / "framewise" / vid[:3] / vid / "framewise_info.jsonl"
    out = {}
    if path.exists():
        for line in path.open():
            r = json.loads(line)
            if r.get("gaze_centre_in_pixels"):
                x, y = r["gaze_centre_in_pixels"][:2]
                out[r["frame_index"]] = (x / cfg.gaze_px, y / cfg.gaze_px)
    return out


class Videos(torch.utils.data.IterableDataset):
    """Decodes the videos one by one (each worker its own share) and yields chunks of the needed frames in order:
    (video, times, images [n, H, W, 3], gaze [n, 3], last chunk of the video)."""

    def __init__(self, cfg, jobs, chunk=128):
        self.cfg, self.jobs, self.chunk = cfg, jobs, chunk

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        jobs = self.jobs[info.id::info.num_workers] if info else self.jobs
        cfg, size = self.cfg, (self.cfg.image, self.cfg.image)
        for vid, times in jobs:
            gz = gaze_of(cfg, vid)
            want = {int(round(t * cfg.video_fps)): t for t in times}
            cap, frames, ts, gaze, last = cv2.VideoCapture(str(cfg.hd_epic / "videos" / vid[:3] / f"{vid}.mp4")), [], [], [], max(want)
            sent = False
            for fi in range(last + 1):
                if not cap.grab():
                    break
                if fi in want:
                    ok, img = cap.retrieve()
                    if not ok:
                        continue
                    frames.append(cv2.cvtColor(cv2.resize(img, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
                    ts.append(want[fi])
                    g = gz.get(fi)
                    gaze.append((g[0], g[1], 1.0) if g else (0.5, 0.5, 0.0))
                if len(frames) == self.chunk or (fi == last and frames):
                    sent = fi == last
                    yield vid, np.array(ts), np.stack(frames), np.array(gaze, np.float32), sent
                    frames, ts, gaze = [], [], []
            if not sent:  # the video ended early: what is left, and the end
                empty = np.zeros((0, cfg.image, cfg.image, 3), np.uint8)
                yield vid, np.array(ts), np.stack(frames) if frames else empty, np.array(gaze, np.float32).reshape(-1, 3), True
            cap.release()


def keep(item):
    return item


def dino_patches(cfg, dino, x):
    """CLS [n, 768] and the patches pooled to the grid [n, grid^2, 768] of images x [n, 3, H, W] in [0, 1]."""
    side = cfg.image // 14
    h = dino(pixel_values=((x - MEAN.cuda()) / STD.cuda()).half()).last_hidden_state
    p = h[:, 1:1 + side * side].transpose(1, 2).reshape(len(x), -1, side, side)
    return h[:, 0], F.adaptive_avg_pool2d(p.float(), cfg.grid).flatten(2).transpose(1, 2)


@torch.no_grad()
def main():
    flags = dict(a.split("=", 1) for a in sys.argv[1:] if a.startswith(("part=", "only=", "pca=")))
    k, n = map(int, flags.get("part", "0/1").split("/"))
    cfg = parse([a for a in sys.argv[1:] if not a.startswith(("part=", "only=", "pca="))])
    cfg.features.mkdir(parents=True, exist_ok=True)
    ft = frame_times(cfg, segments(cfg))
    from transformers import AutoModel
    dino = AutoModel.from_pretrained(cfg.dino, dtype=torch.float16).cuda().eval()
    if flags.get("pca") == "1":  # fit the patch projection on each participant's shortest video
        by_p = {}
        for v, t in sorted(ft.items(), key=lambda kv: len(kv[1])):
            by_p.setdefault(v[:3], (v, t))
        loader = torch.utils.data.DataLoader(Videos(cfg, list(by_p.values())), batch_size=None, num_workers=cfg.workers, collate_fn=keep)
        s1, s2, m = torch.zeros(768, dtype=torch.float64, device="cuda"), torch.zeros(768, 768, dtype=torch.float64, device="cuda"), 0
        for vid, ts, frames, gaze, last in loader:
            if len(ts):
                _, p = dino_patches(cfg, dino, torch.from_numpy(frames).cuda().permute(0, 3, 1, 2).float() / 255)
                p = p.flatten(0, 1).double()
                s1, s2, m = s1 + p.sum(0), s2 + p.T @ p, m + len(p)
        mean = s1 / m
        evals, evecs = torch.linalg.eigh(s2 / m - torch.outer(mean, mean))
        comps = evecs[:, torch.argsort(evals, descending=True)[:cfg.pca_dim]]
        kept = float(evals.sort(descending=True).values[:cfg.pca_dim].sum() / evals.sum())
        torch.save({"mean": mean.float().cpu(), "comps": comps.float().cpu(), "variance_kept": kept}, cfg.features / "pca.pt")
        print(f"PCA from {m} patch vectors of {len(by_p)} videos: {cfg.pca_dim} dims keep {kept:.3f} of the variance", flush=True)
        return
    pca = torch.load(cfg.features / "pca.pt") if cfg.pca_dim else None
    jobs = sorted(ft.items(), key=lambda kv: -len(kv[1]))[k::n]  # the biggest first, shared round-robin
    jobs = [(v, t) for v, t in jobs if not (cfg.features / f"{v}.pt").exists() and flags.get("only", v) == v]
    print(f"part {k}/{n}: {len(jobs)} videos, {sum(len(t) for _, t in jobs)} frames", flush=True)
    from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
    raft = raft_small(weights=Raft_Small_Weights.DEFAULT).cuda().eval() if cfg.raft else None
    loader = torch.utils.data.DataLoader(Videos(cfg, jobs), batch_size=None, num_workers=cfg.workers, collate_fn=keep)
    acc, prev, cell = {}, {}, cfg.image // cfg.grid
    for vid, ts, frames, gaze, last in loader:
        if len(ts):
            x = torch.from_numpy(frames).cuda().permute(0, 3, 1, 2).float() / 255
            cls, patch = dino_patches(cfg, dino, x)
            if pca is not None:
                patch = (patch - pca["mean"].cuda()) @ pca["comps"].cuda()
            flow = torch.zeros(len(x), cfg.grid ** 2, 3, device="cuda")
            if raft is not None:  # flow from the previous sampled frame, where it is one step earlier
                pt, px = prev.get(vid, (None, None))
                before = torch.cat([px, x[:-1]]) if px is not None else torch.cat([x[:1], x[:-1]])
                tprev = np.concatenate([[pt if pt is not None else -1.0], ts[:-1]])
                ok = torch.from_numpy(np.abs(ts - tprev - 1 / cfg.fps) < 0.5 / cfg.fps).cuda()
                f = torch.cat([raft(before[i:i + 32] * 2 - 1, x[i:i + 32] * 2 - 1)[-1] for i in range(0, len(x), 32)])
                f = f - f.flatten(2).median(-1).values[..., None, None]
                f = F.avg_pool2d(f, cell).flatten(2).transpose(1, 2) / cfg.image  # [n, grid^2, 2], image widths per step
                flow = torch.cat([f, f.norm(dim=-1, keepdim=True)], -1) * ok[:, None, None]
                prev[vid] = (ts[-1], x[-1:])
            a = acc.setdefault(vid, {"t": [], "cls": [], "patch": [], "flow": [], "gaze": []})
            for key, val in (("t", torch.from_numpy(ts)), ("cls", cls.cpu()), ("patch", patch.half().cpu()),
                             ("flow", flow.half().cpu()), ("gaze", torch.from_numpy(gaze))):
                a[key].append(val)
        if last:
            a = acc.pop(vid, None)
            if a:
                torch.save({key: torch.cat(v) for key, v in a.items()}, cfg.features / f"{vid}.pt")
            prev.pop(vid, None)
            print(f"{vid} done", flush=True)


if __name__ == "__main__":
    main()
