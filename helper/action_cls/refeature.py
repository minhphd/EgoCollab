"""DINO features of every frame again with another backbone (`dino`, e.g. facebook/dinov2-large), read from the 5 fps,
448 px frame cache (`hand_frames`). Each frame's flow and gaze are copied from an existing feature set (`from=`), so
the files keep extract.py's layout {t, cls, patch, flow, gaze}. The patches are projected by a PCA fit first on each
participant's shortest video (`pca=1`), as in extract.py.

    CUDA_VISIBLE_DEVICES=1 python -m helper.action_cls.refeature pca=1 dino=facebook/dinov2-large \
        features=data/hd_epic/features/action_cls_448_large from=data/hd_epic/features/action_cls_448_all
    CUDA_VISIBLE_DEVICES=1 python -m helper.action_cls.refeature dino=facebook/dinov2-large \
        features=data/hd_epic/features/action_cls_448_large from=data/hd_epic/features/action_cls_448_all
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from helper.perception.frames import Frames

from .config import parse
from .extract import dino_patches


class Rows(Dataset):
    """The cached frames of one video at the given 5 fps keys, RGB uint8."""

    def __init__(self, folder, vid, keys):
        self.frames, self.keys = Frames(folder, vid), keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        return torch.from_numpy(self.frames[int(self.keys[i])])


@torch.no_grad()
def encode(cfg, dino, vid, keys):
    """CLS [n, D] and grid-pooled patches [n, grid^2, D] of the frames at `keys`."""
    cls, patch = [], []
    for x in DataLoader(Rows(cfg.hand_frames, vid, keys), batch_size=64, num_workers=cfg.workers):
        c, p = dino_patches(cfg, dino, x.cuda().permute(0, 3, 1, 2).float() / 255)
        cls.append(c.cpu())
        patch.append(p.cpu())
    return torch.cat(cls), torch.cat(patch)


def main():
    flags = dict(a.split("=", 1) for a in sys.argv[1:] if a.startswith(("pca=", "from=", "part=")))
    cfg = parse([a for a in sys.argv[1:] if not a.startswith(("pca=", "from=", "part="))])
    src, (k, n) = Path(flags["from"]), map(int, flags.get("part", "0/1").split("/"))
    cfg.features.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModel
    dino = AutoModel.from_pretrained(cfg.dino, dtype=torch.float16).cuda().eval()
    vids = sorted(p.stem for p in src.glob("P*.pt"))
    keys = lambda v: np.round(torch.load(src / f"{v}.pt", mmap=True)["t"].numpy() * cfg.fps).astype(int)
    if flags.get("pca") == "1":  # fit the projection on each participant's shortest video
        by_p = {}
        for v in sorted(vids, key=lambda v: (src / f"{v}.pt").stat().st_size):
            by_p.setdefault(v[:3], v)
        s1 = s2 = 0.0
        m = 0
        for v in by_p.values():
            p = encode(cfg, dino, v, keys(v))[1].flatten(0, 1).double().cuda()
            s1, s2, m = s1 + p.sum(0), s2 + p.T @ p, m + len(p)
        mean = s1 / m
        evals, evecs = torch.linalg.eigh(s2 / m - torch.outer(mean, mean))
        comps = evecs[:, torch.argsort(evals, descending=True)[:cfg.pca_dim]]
        kept = float(evals.sort(descending=True).values[:cfg.pca_dim].sum() / evals.sum())
        torch.save({"mean": mean.float().cpu(), "comps": comps.float().cpu(), "variance_kept": kept}, cfg.features / "pca.pt")
        print(f"PCA from {m} patch vectors of {len(by_p)} videos: {cfg.pca_dim} dims keep {kept:.3f} of the variance", flush=True)
        return
    pca = torch.load(cfg.features / "pca.pt")
    todo = [v for v in sorted(vids, key=lambda v: -(src / f"{v}.pt").stat().st_size)[k::n] if not (cfg.features / f"{v}.pt").exists()]
    print(f"part {k}/{n}: {len(todo)} videos", flush=True)
    for v in todo:
        old = torch.load(src / f"{v}.pt", mmap=True)
        cls, patch = encode(cfg, dino, v, np.round(old["t"].numpy() * cfg.fps).astype(int))
        patch = (patch - pca["mean"]) @ pca["comps"]
        torch.save({"t": old["t"].clone(), "cls": cls, "patch": patch.half(), "flow": old["flow"].clone(),
                    "gaze": old["gaze"].clone()}, cfg.features / f"{v}.pt")
        print(f"{v} done", flush=True)


if __name__ == "__main__":
    main()
