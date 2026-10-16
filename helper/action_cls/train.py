"""Trains the action classifier on HD-EPIC windows, keeps the epoch with the best validation macro F1, and tests it
on held-out participants: accuracy and macro F1 mid-action and just after the action ends, and completion (AUROC of
done against not done, from the progress head; the progress error).

    CUDA_VISIBLE_DEVICES=0 python -m helper.action_cls.train name=full
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


def auroc(score, label):
    """P(a done window scores above a not-done one), by ranks."""
    pos, neg = score[label > 0.5], score[label <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.empty(len(pos) + len(neg))
    ranks[np.argsort(np.concatenate([pos, neg]), kind="stable")] = np.arange(1, len(ranks) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def cuda(b):
    return {k: v.cuda(non_blocking=True) for k, v in b.items()}


@torch.no_grad()
def evaluate(model, loader, cfg):
    model.eval()
    out, hold = [], []
    for b in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=getattr(model, "amp", True)):
            o = model(cuda(b))
        out.append((o["logits"].float().softmax(-1).cpu(), o["progress"].float().sigmoid().cpu(), b["y"], b["progress"],
                    b["done"], b["point"]))
        if "holding" in o:  # every valid frame of every window
            hold.append((o["holding"].float().softmax(-1).cpu()[b["valid"]], b["holding"][b["valid"]]))
    prob, prog, y, tprog, done, point = (torch.cat(x) for x in zip(*out))
    C, res = len(cfg.classes), {}
    for name, idx in (("mid", 1), ("end", 3)):  # windows ending at 60% of the action, and 0.2 s after its end
        m = point == idx
        pred, yy = prob[m].argmax(-1), y[m]
        conf = torch.zeros(C, C, dtype=torch.long).index_put_((yy, pred), torch.ones_like(yy), accumulate=True)
        tp = conf.diag().float()
        f1 = (2 * tp / (conf.sum(0) + conf.sum(1)).clamp(min=1)).tolist()
        res[name] = {"acc": (pred == yy).float().mean().item(), "macro_f1": float(np.mean(f1)),
                     "f1": dict(zip(cfg.classes, f1)), "confusion": conf.tolist()}
    res["completion"] = {"auroc": auroc(prog.numpy(), done.numpy()), "progress_mae": (prog - tprog).abs().mean().item(),
                         "auroc_by_class": {c: auroc(prog[y == i].numpy(), done[y == i].numpy()) for i, c in enumerate(cfg.classes)}}
    if hold:
        p, t = (torch.cat(x) for x in zip(*hold))
        res["holding"] = {"acc": (p.argmax(-1) == t).float().mean().item(),  # the count of objects in hand
                          "auroc_any": auroc((1 - p[:, 0]).numpy(), (t > 0).float().numpy())}  # some object in hand
    return res


def main():
    cfg = parse(sys.argv[1:])
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    out = cfg.runs / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    seg = segments(cfg)
    train, val, test = (Windows(cfg, seg, s, s == "train") for s in ("train", "val", "test"))
    print(f"windows: train {len(train)} segments, val {len(val)}, test {len(test)}", flush=True)
    weights = train.class_weights() if cfg.balanced else torch.ones(len(train))
    load = dict(batch_size=cfg.batch, num_workers=cfg.workers, pin_memory=True)
    tl = DataLoader(train, sampler=WeightedRandomSampler(weights, cfg.windows_per_epoch), drop_last=True,
                    persistent_workers=True, **load)
    vl = DataLoader(val, **load)
    model = ActionNet(cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * (cfg.windows_per_epoch // cfg.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / cfg.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    best, log = -1.0, []
    for ep in range(cfg.epochs):
        model.train()
        t0, tot = time.time(), {"cls": 0.0, "progress": 0.0, "inpaint": 0.0, "hold": 0.0, "n": 0}
        for b in tl:
            b = cuda(b)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(b, mask_ratio=cfg.mask_ratio if cfg.inpaint > 0 else 0.0)
            l_cls = F.cross_entropy(o["logits"].float(), b["y"], label_smoothing=cfg.label_smoothing)
            l_prog = F.binary_cross_entropy_with_logits(o["progress"].float(), b["progress"].float())
            l_inp = o.get("inpaint", torch.zeros((), device="cuda")).float()
            l_hold = F.cross_entropy(o["holding"].float()[b["valid"]], b["holding"][b["valid"]]) \
                if "holding" in o else torch.zeros((), device="cuda")
            loss = l_cls + cfg.progress_w * l_prog + cfg.inpaint * l_inp + cfg.hold_w * l_hold
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            for k, v in (("cls", l_cls), ("progress", l_prog), ("inpaint", l_inp), ("hold", l_hold)):
                tot[k] += float(v.detach())
            tot["n"] += 1
        res = evaluate(model, vl, cfg) if len(val) else {"end": {"macro_f1": 0.0, "acc": 0.0}, "mid": {"macro_f1": 0.0},
                                                          "completion": {"auroc": float("nan")}}
        losses = {k: tot[k] / max(1, tot["n"]) for k in ("cls", "progress", "inpaint", "hold")}
        log.append({"epoch": ep, "loss": losses, "val": res, "s": round(time.time() - t0, 1)})
        print(f"epoch {ep} {time.time() - t0:.0f}s loss {losses['cls']:.3f}/{losses['progress']:.3f}/{losses['inpaint']:.3f}/{losses['hold']:.3f} "
              f"val end f1 {res['end']['macro_f1']:.3f} acc {res['end']['acc']:.3f} mid f1 {res['mid']['macro_f1']:.3f} "
              f"done auroc {res['completion']['auroc']:.3f}" + (f" holding acc {res['holding']['acc']:.3f}" if "holding" in res else ""),
              flush=True)
        if res["end"]["macro_f1"] >= best:
            best = res["end"]["macro_f1"]
            torch.save(model.state_dict(), out / "best.pt")
    model.load_state_dict(torch.load(out / "best.pt"))
    if not len(test):
        print("no test windows", flush=True)
        return
    res = evaluate(model, DataLoader(test, **load), cfg)
    cfg_out = {k: (str(v) if not isinstance(v, (int, float, bool, str)) else v) for k, v in vars(cfg).items()}
    (out / "metrics.json").write_text(json.dumps({"config": cfg_out, "best_val_end_macro_f1": best, "test": res, "log": log}, indent=1))
    print(f"TEST {cfg.name}: end f1 {res['end']['macro_f1']:.3f} acc {res['end']['acc']:.3f} | mid f1 {res['mid']['macro_f1']:.3f} "
          f"| done auroc {res['completion']['auroc']:.3f} progress mae {res['completion']['progress_mae']:.3f}", flush=True)
    print("TEST f1 by class (end):", {k: round(v, 3) for k, v in res["end"]["f1"].items()}, flush=True)
    if "holding" in res:
        print("TEST holding:", {k: round(v, 3) for k, v in res["holding"].items()}, flush=True)


if __name__ == "__main__":
    main()
