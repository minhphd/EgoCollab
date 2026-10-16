"""Which subgoal does a person start next? A learned choice over the candidate subgoals, made while the person is still
on the current one, from text (the subgoals' labels), images (the egocentric view: the whole frame, and the patch
around gaze) and the task's history. SigLIP embeds images and text in one space, so the view can be compared with each
candidate's label. Trained on HD-EPIC's annotated switches: P01 to P06, model selection on P07, test on P08 and P09.

The history is the annotated one (the current subgoal, what is started and done, from the narrations), so this measures
the choice model alone, as a ceiling; inside BIP the history would be inferred. Gemini's prior is gemini-3.5-flash's.

    PYTHONPATH=<sentencepiece> HF_HOME=<cache> python -m helper.collab.nextgoal embed=1   # SigLIP embeddings, once
    python -m helper.collab.nextgoal                                                       # train, select, test
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from helper.action_cls.config import Config
from helper.datasets.hd_epic.data import load_framewise
from helper.datasets.hd_epic.tasks import load
from helper.perception.frames import Frames

ROOT = Path(__file__).resolve().parents[2]
OUT, PRIOR = ROOT / "data/collab/nextgoal", ROOT / "data/bip/prior/gemini-3.5-flash"
LAGS = (0.5, 1.0, 2.0, 4.0)  # s before the switch at which the choice is predicted
HIST = ("started", "recency", "actions", "ready", "p_after_current", "p_before_current")


def records() -> list[dict]:
    """One per (switch, lag) at which the person is still on the subgoal they leave: the candidates (not done, not the
    current one), their history features, and which one comes next. Through a pause between actions the person stays
    on the last subgoal they worked on, as in fit.py."""
    out = []
    for vid, task in load(2).items():
        if not (PRIOR / f"{vid}.json").exists():
            continue
        p = {(u, v): q for u, v, q in json.loads((PRIOR / f"{vid}.json").read_text())["edges"]}
        spans = task.timeline.spans
        t0, t1 = task.extent
        times, cur, last = np.arange(t0, t1, 0.2), [], None
        for t in times:
            last = task.current(t) or last
            cur.append(last)
        for i in range(1, len(times)):
            g, n = cur[i - 1], cur[i]
            if g is None or n is None or g == n:
                continue
            for lag in LAGS:
                j = i - int(round(lag / 0.2))
                t = times[i] - lag
                if j < 0 or cur[j] != g:
                    continue
                done = task.done(t)
                cands = [u for u in task.nodes if u not in done and u != g]
                if n not in cands or len(cands) < 2:
                    continue
                hist = []
                for u in cands:
                    ends = [b for a, b, o in spans if o == u and b <= t]
                    ready = np.prod([1 - p.get((w, u), 0.0) for w in task.nodes if w not in done and w != u])
                    hist.append([float(task.first[u] < t), np.log1p((t - max(ends)) / 10) if ends else 0.0,
                                 np.log1p(len(ends)), ready, p.get((g, u), 0.0), p.get((u, g), 0.0)])
                out.append({"vid": vid, "t": float(t), "lag": lag, "g": g, "n": n, "cands": cands, "hist": hist,
                            "title": task.title, "labels": {u: task.label[u] for u in [g] + cands}})
    return out


@torch.no_grad()
def embed(recs: list[dict]) -> dict:
    """SigLIP embeddings: each label; and for each record, the whole frame (mean over t, t - 1 s, t - 2 s) and the
    224 px patch around gaze at t (zeros where gaze is missing)."""
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    name = "google/siglip-base-patch16-224"
    model, proc = AutoModel.from_pretrained(name).cuda().eval(), AutoProcessor.from_pretrained(name)
    feats = lambda o: F.normalize(getattr(o, "pooler_output", o).float(), dim=-1).cpu()
    labels = sorted({s for r in recs for s in r["labels"].values()})
    txt = torch.cat([feats(model.get_text_features(**{k: v.cuda() for k, v in proc(
        text=labels[i:i + 256], return_tensors="pt", padding="max_length").items()})) for i in range(0, len(labels), 256)])
    cfg, frames, gazes = Config(), {}, {}
    full, gaze, has = [], [], []
    for r in recs:
        if r["vid"] not in frames:
            frames[r["vid"]], gazes[r["vid"]] = Frames(cfg.hand_frames, r["vid"]), load_framewise(r["vid"])[0]
        fr, gz = frames[r["vid"]], gazes[r["vid"]]
        imgs = [fr[int(round((r["t"] - d) * cfg.fps))] for d in (0.0, 1.0, 2.0)]
        pt = gz[min(len(gz) - 1, int(round(r["t"] * 30)))] if gz else None
        if pt:
            x, y = (int(min(448 - 112, max(112, c * 448))) for c in pt)
            crop = imgs[0][y - 112:y + 112, x - 112:x + 112]
        else:
            crop = np.zeros((224, 224, 3), np.uint8)
        e = feats(model.get_image_features(pixel_values=proc(images=[Image.fromarray(a) for a in imgs + [crop]],
                                                                return_tensors="pt")["pixel_values"].cuda()))
        full.append(e[:3].mean(0))
        gaze.append(e[3])
        has.append(float(pt is not None))
    return {"labels": labels, "text": txt, "full": torch.stack(full), "gaze": torch.stack(gaze), "has_gaze": torch.tensor(has)}


class Chooser(nn.Module):
    """score(u) = MLP([history of u, cos(view, label u), cos(gaze patch, label u), learned view x label, gaze patch x
    label and current label x label interactions, a label bias]); softmax over the candidates."""

    def __init__(self, use: set, dim=768, rank=16, hidden=32):
        super().__init__()
        self.use = use
        self.pi, self.pg, self.pt, self.pc, self.pu = (nn.Linear(dim, rank, bias=False) for _ in range(5))
        self.bias = nn.Linear(dim, 1)
        n_in = len(HIST) * ("history" in use) + 6 * ("text" in use)
        self.mlp = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, 1))

    def forward(self, b):
        x = []
        if "history" in self.use:
            x.append(b["hist"])
        if "text" in self.use:  # every text feature here reads a label: alone, against the view, against the current one
            e, img = b["cand"], "image" in self.use
            v, gz, hg = b["full"][:, None], b["gaze"][:, None], b["has_gaze"][:, None, None]
            zero = torch.zeros_like(e[..., :1])
            x += [10 * (v * e).sum(-1, keepdim=True) if img else zero, 10 * hg * (gz * e).sum(-1, keepdim=True) if img else zero,
                  (self.pi(v) * self.pt(e)).sum(-1, keepdim=True) if img else zero,
                  hg * (self.pg(gz) * self.pt(e)).sum(-1, keepdim=True) if img else zero,
                  (self.pc(b["cur"][:, None]) * self.pu(e)).sum(-1, keepdim=True), self.bias(e)]
        s = self.mlp(torch.cat(x, -1)).squeeze(-1)
        return s.masked_fill(~b["mask"], -1e9)


def batch(recs, emb, idx) -> dict:
    """Records -> padded tensors: history [B, C, 6], candidate labels [B, C, 768], current label, view, gaze patch."""
    row = {s: i for i, s in enumerate(emb["labels"])}
    C = max(len(recs[i]["cands"]) for i in idx)
    out = {"hist": torch.zeros(len(idx), C, len(HIST)), "cand": torch.zeros(len(idx), C, emb["text"].shape[1]),
           "mask": torch.zeros(len(idx), C, dtype=torch.bool), "y": torch.zeros(len(idx), dtype=torch.long)}
    for j, i in enumerate(idx):
        r = recs[i]
        k = len(r["cands"])
        out["hist"][j, :k] = torch.tensor(r["hist"], dtype=torch.float)
        out["cand"][j, :k] = emb["text"][[row[r["labels"][u]] for u in r["cands"]]]
        out["mask"][j, :k] = True
        out["y"][j] = r["cands"].index(r["n"])
    out["cur"] = emb["text"][[row[recs[i]["labels"][recs[i]["g"]]] for i in idx]]
    for k in ("full", "gaze", "has_gaze"):
        out[k] = emb[k][list(idx)]
    return out


def scores(s: torch.Tensor, b: dict, recs, idx) -> dict:
    """Top 1 and top 3 with ties broken at random (the expected hit): the true candidate's place is uniform among
    the candidates scored the same."""
    true = s.gather(1, b["y"][:, None])
    above, tied = (s > true).sum(1).float(), (s == true).sum(1).float()  # tied counts the true one
    hit = lambda k: ((k - above).clamp(0) .minimum(tied) / tied)
    lag = torch.tensor([recs[i]["lag"] for i in idx])
    res = {"top1": float(hit(1).mean()), "top3": float(hit(3).mean()), "nll": float(F.cross_entropy(s, b["y"])), "n": len(idx)}
    res["top1_by_lag"] = {f"{l:g}": float(hit(1)[lag == l].mean()) for l in LAGS}
    return res


def fit(use: set, recs, emb, split, seed=0) -> dict:
    torch.manual_seed(seed)
    tr, va, te = (batch(recs, emb, split[s]) for s in ("train", "val", "test"))
    model = Chooser(use)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.05)
    best, state = float("inf"), None
    for ep in range(300):
        model.train()
        loss = F.cross_entropy(model(tr), tr["y"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            v = float(F.cross_entropy(model(va), va["y"]))
        if v < best:
            best, state = v, {k: x.clone() for k, x in model.state_dict().items()}
    model.load_state_dict(state)
    with torch.no_grad():
        return {"val_nll": best, **scores(model(te), te, recs, split["test"])}


def main():
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    OUT.mkdir(parents=True, exist_ok=True)
    if args.get("embed") == "1":
        recs = records()
        (OUT / "records.json").write_text(json.dumps(recs))
        torch.save(embed(recs), OUT / "emb.pt")
        print(f"{len(recs)} records, {len({(r['vid'], r['t'] + r['lag']) for r in recs})} switches embedded", flush=True)
        return
    recs, emb, cfg = json.loads((OUT / "records.json").read_text()), torch.load(OUT / "emb.pt"), Config()
    split = {s: [i for i, r in enumerate(recs) if r["vid"][:3] in ps] for s, ps in cfg.split.items()}
    print({s: len(v) for s, v in split.items()}, "decision points;", "candidates per point, median",
          int(np.median([len(r["cands"]) for r in recs])), flush=True)
    te = batch(recs, emb, split["test"])
    hist = te["hist"]
    rho, delta = 3.0, 0.40  # bip.Params: pi_G with the prior's chance that every predecessor is done
    pi = torch.where(hist[..., 0] > 0, rho, hist[..., 3] + (1 - hist[..., 3]) * delta).log().masked_fill(~te["mask"], -1e9)
    view = (te["full"][:, None] * te["cand"]).sum(-1).masked_fill(~te["mask"], -1e9)
    look = (te["gaze"][:, None] * te["cand"]).sum(-1).masked_fill(~te["mask"], -1e9)
    res = {"uniform": {"top1": float((1 / te["mask"].sum(1).float()).mean()),
                       "top3": float((3 / te["mask"].sum(1).float()).clamp(max=1).mean())},
           "pi_G (hand-set)": scores(pi, te, recs, split["test"]),
           "zero-shot: view vs label": scores(view, te, recs, split["test"]),
           "zero-shot: gaze patch vs label": scores(look, te, recs, split["test"])}
    for name, use in (("learned: history", {"history"}), ("learned: history + text", {"history", "text"}),
                      ("learned: text + image", {"text", "image"}), ("learned: history + text + image", {"history", "text", "image"})):
        res[name] = fit(use, recs, emb, split)
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    print(f"{'model':34s} {'top 1':>6s} {'top 3':>6s}   top 1 at 0.5 / 1 / 2 / 4 s before the switch")
    for name, r in res.items():
        lags = " / ".join(f"{r['top1_by_lag'][f'{l:g}']:.2f}" for l in LAGS) if "top1_by_lag" in r else ""
        print(f"{name:34s} {r['top1']:6.3f} {r['top3']:6.3f}   {lags}", flush=True)


if __name__ == "__main__":
    main()
