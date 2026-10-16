"""The verb recognizer BIP reads (helper/collab/bip.py, the verb term): train.py's network on clips of the video
stream instead of windows anchored to actions. Each clip of `clip` s gets one label, the verb of the last pick, place,
open or close that ends in it, or other when none does; its input is the `window` frames ending with it (6 s, since
gaze leads the hands). Training draws clip ends at any frame, class-balanced, so the network's prior is uniform and
its probabilities are likelihoods up to a constant; validation and test cut every video into consecutive clips. A
temperature is then fit on validation (class-balanced negative log-likelihood), and the test clips' calibrated
probabilities are saved as the bank the simulator draws its recognizer outputs from (bank.npz: p [n, 5], y [n],
and each clip's video and end time).

    CUDA_VISIBLE_DEVICES=0 python -m helper.action_cls.stream name=stream3 features=data/hd_epic/features/action_cls_448_all
"""
import json
import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import parse
from .dataset import Windows
from .model import ActionNet
from .segments import segments
from .train import cuda


class Clips(Windows):
    """Clips ending at every frame (training) or cut one after another (evaluation), each labelled by its last step."""

    def __init__(self, cfg, seg, split, train):
        super().__init__(cfg, seg, split, train)
        steps, other, every = seg[seg.label != "other"].sort_values("end"), cfg.classes.index("other"), round(cfg.clip * cfg.fps)
        self.items = []
        for v, keys in self.keys.items():
            ends = keys if train else np.arange(every, keys[-1] + 1, every)
            t, g = ends / cfg.fps, steps[steps.video_id == v]
            et, ey = g.end.to_numpy(), g.y.to_numpy()
            i = np.maximum(np.searchsorted(et, t, side="right") - 1, 0)
            y = np.where((len(et) > 0) & (et[i] <= t) & (et[i] > t - cfg.clip), ey[i], other) if len(et) else np.full(len(t), other)
            self.items += [(v, int(k), int(c)) for k, c in zip(ends, y)]
        self.y = np.array([c for _, _, c in self.items])

    def __getitem__(self, j):
        v, k, y = self.items[j]
        return {**self.window(v, k), "y": y}

    def class_weights(self):
        return torch.from_numpy(1.0 / np.bincount(self.y, minlength=len(self.cfg.classes)).astype(float)[self.y])


def logits_of(model, loader):
    model.eval()
    out, ys = [], []
    with torch.no_grad():
        for b in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(model(cuda(b))["logits"].float().cpu())
            ys.append(b["y"])
    return torch.cat(out), torch.cat(ys)


def balanced_nll(logits, y, C, T=1.0) -> float:
    nll = -(logits / T).log_softmax(-1)[torch.arange(len(y)), y]
    w = 1.0 / torch.bincount(y, minlength=C).clamp(min=1).float()[y]
    return float((w * nll).sum() / w.sum())


def scores(logits, y, C, T=1.0) -> dict:
    """Accuracy, macro F1 and the confusion matrix at the natural class mix; the negative log-likelihood (natural and
    class-balanced); and, per step verb, the mean log P(verb) / P(other) on its clips: the evidence a clip gives BIP."""
    logp = (logits / T).log_softmax(-1)
    pred = logp.argmax(-1)
    conf = torch.zeros(C, C, dtype=torch.long).index_put_((y, pred), torch.ones_like(y), accumulate=True)
    f1 = (2 * conf.diag().float() / (conf.sum(0) + conf.sum(1)).clamp(min=1)).tolist()
    llr = [float((logp[y == c, c] - logp[y == c, C - 1]).mean()) if (y == c).any() else float("nan") for c in range(C - 1)]
    return {"acc": (pred == y).float().mean().item(), "macro_f1": float(np.mean(f1)), "f1": f1, "confusion": conf.tolist(),
            "nll": float(-logp[torch.arange(len(y)), y].mean()), "nll_balanced": balanced_nll(logits, y, C, T),
            "llr_vs_other": llr, "T": T}


def main():
    cfg = parse(sys.argv[1:])
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    out = cfg.runs / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    seg, C = segments(cfg), len(cfg.classes)
    train, val, test = (Clips(cfg, seg, s, s == "train") for s in ("train", "val", "test"))
    print(f"clips: train {len(train)} ends, val {len(val)}, test {len(test)}; test labels {np.bincount(test.y, minlength=C).tolist()}",
          flush=True)
    load = dict(batch_size=cfg.batch, num_workers=cfg.workers, pin_memory=True)
    tl = DataLoader(train, sampler=WeightedRandomSampler(train.class_weights(), cfg.windows_per_epoch), drop_last=True,
                    persistent_workers=True, **load)
    model = ActionNet(cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * (cfg.windows_per_epoch // cfg.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / cfg.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    best, log = -1.0, []
    for ep in range(cfg.epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for b in tl:
            b = cuda(b)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(b, mask_ratio=cfg.mask_ratio if cfg.inpaint > 0 else 0.0)
            loss = F.cross_entropy(o["logits"].float(), b["y"], label_smoothing=cfg.label_smoothing) + \
                cfg.inpaint * o.get("inpaint", torch.zeros((), device="cuda")).float()
            if "holding" in o:
                loss = loss + cfg.hold_w * F.cross_entropy(o["holding"].float()[b["valid"]], b["holding"][b["valid"]])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot, nb = tot + float(loss.detach()), nb + 1
        r = scores(*logits_of(model, DataLoader(val, **load)), C)
        log.append({"epoch": ep, "loss": tot / max(1, nb), "val": r, "s": round(time.time() - t0, 1)})
        print(f"epoch {ep} {time.time() - t0:.0f}s loss {tot / max(1, nb):.3f} val acc {r['acc']:.3f} macro f1 {r['macro_f1']:.3f} "
              f"nll balanced {r['nll_balanced']:.3f}", flush=True)
        if r["macro_f1"] >= best:
            best = r["macro_f1"]
            torch.save(model.state_dict(), out / "best.pt")
    model.load_state_dict(torch.load(out / "best.pt"))
    lv, yv = logits_of(model, DataLoader(val, **load))
    grid = np.exp(np.linspace(np.log(0.25), np.log(4.0), 121))
    T = float(min(grid, key=lambda T: balanced_nll(lv, yv, C, T)))
    lt, yt = logits_of(model, DataLoader(test, **load))
    res = {"val": scores(lv, yv, C, T), "test": scores(lt, yt, C, T), "test_T1": scores(lt, yt, C), "prior": [1 / C] * C}
    np.savez(out / "bank.npz", p=(lt / T).softmax(-1).numpy().astype(np.float32), y=yt.numpy(),
             video=np.array([v for v, _, _ in test.items]), t=np.array([k / cfg.fps for _, k, _ in test.items]))
    cfg_out = {k: (str(v) if not isinstance(v, (int, float, bool, str)) else v) for k, v in vars(cfg).items()}
    (out / "metrics.json").write_text(json.dumps({"config": cfg_out, "best_val_macro_f1": best, **res, "log": log}, indent=1))
    t = res["test"]
    print(f"TEST {cfg.name}: T {T:.2f} acc {t['acc']:.3f} macro f1 {t['macro_f1']:.3f} nll balanced {t['nll_balanced']:.3f} "
          f"(T=1: {res['test_T1']['nll_balanced']:.3f})", flush=True)
    print("TEST f1:", dict(zip(cfg.classes, np.round(t["f1"], 3).tolist())), "| log P(verb)/P(other) on its clips:",
          dict(zip(cfg.classes[:-1], np.round(t["llr_vs_other"], 2).tolist())), flush=True)


if __name__ == "__main__":
    main()
