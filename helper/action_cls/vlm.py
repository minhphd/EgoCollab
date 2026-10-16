"""The smallest multi-image VLM (SmolVLM2, 256M), fine-tuned end to end on the classifier's windows: the window's
frames (`vlm_frames`, the gaze drawn as a ring, a point and nothing more) and a question go in; a class head and a
progress head on its last token's hidden state come out. Same segments, split, window ends and metrics as train.py.

    CUDA_VISIBLE_DEVICES=1 python -m helper.action_cls.vlm name=vlm6
"""
import functools
import json
import math
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler

from helper.perception.frames import Frames

from .config import parse
from .dataset import EVAL_POINTS
from .segments import segments
from .train import cuda, evaluate

QUESTION = ("These are {n} frames, {dt:g} s apart, from a person's head-mounted camera; the orange ring marks where "
            "they look. Which action is the person doing at the end: pick, place, open, close, or other? How far "
            "through it are they?")


class Clips(torch.utils.data.Dataset):
    def __init__(self, cfg, seg, split, train):
        self.cfg, self.train = cfg, train
        self.seg = seg[seg.split == split].reset_index(drop=True)
        self.gz = {}  # video -> (5 fps frame keys, gaze [n, 3]) from the cached features (`features`)
        for v in self.seg.video_id.unique():
            f = torch.load(cfg.features / f"{v}.pt", mmap=True)
            self.gz[v] = (np.round(f["t"].numpy() * cfg.fps).astype(int), f["gaze"].numpy().copy())
        self.items = [(i, None) for i in range(len(self.seg))] if train else \
            [(i, p) for i in range(len(self.seg)) for p in EVAL_POINTS]

    def __len__(self):
        return len(self.items)

    @functools.lru_cache(maxsize=8)
    def frames(self, vid):
        return Frames(self.cfg.vlm_frames, vid)

    def gaze(self, vid, t):
        """The cached gaze nearest to time t (within 1 / fps), or None."""
        keys, g = self.gz[vid]
        i = min(int(np.searchsorted(keys, round(t * self.cfg.fps))), len(keys) - 1)
        return g[i] if abs(keys[i] - t * self.cfg.fps) <= 1 and g[i, 2] > 0 else None

    def __getitem__(self, j):
        cfg = self.cfg
        i, point = self.items[j]
        r = self.seg.iloc[i]
        if point is None:
            t_end = np.random.uniform(r.start + cfg.end_lo * (r.end - r.start), r.end + cfg.end_hi)
        else:
            t_end = r.start + point[1] * (r.end - r.start) if point[0] == "frac" else r.end + point[1]
        k_end = int(math.floor(t_end * cfg.vlm_fps + 1e-6))
        frames = []
        for k in range(k_end - cfg.vlm_n + 1, k_end + 1):
            img = np.ascontiguousarray(self.frames(r.video_id)[max(k, 0)])
            g = self.gaze(r.video_id, max(k, 0) / cfg.vlm_fps)
            if cfg.vlm_gaze and g is not None:
                h, w = img.shape[:2]
                cv2.circle(img, (int(g[0] * w), int(g[1] * h)), max(6, w // 40), (255, 140, 0), max(2, w // 150))
            frames.append(img)
        t = k_end / cfg.vlm_fps
        return {"frames": np.stack(frames), "y": int(r.y), "progress": float(np.clip((t - r.start) / (r.end - r.start), 0, 1)),
                "done": float(t >= r.end - 1e-6), "point": EVAL_POINTS.index(point) if point else -1}

    def class_weights(self):
        y = self.seg.y.to_numpy()
        return torch.from_numpy(1.0 / np.bincount(y, minlength=len(self.cfg.classes)).astype(float)[y])


class Collate:
    """Stacks the windows' frames (uint8) and targets. Every window has the same prompt, so its tokens, with the image
    tokens the processor expands, are made once."""

    def __init__(self, processor, n, dt):
        msg = [{"role": "user", "content": [{"type": "image"}] * n + [{"type": "text", "text": QUESTION.format(n=n, dt=dt)}]}]
        prompt = processor.apply_chat_template(msg, add_generation_prompt=True)
        px = processor.image_processor.max_image_size["longest_edge"]
        self.ids = processor(text=[prompt], images=[[Image.new("RGB", (px, px))] * n], return_tensors="pt")["input_ids"]

    def __call__(self, items):
        batch = {"input_ids": self.ids.repeat(len(items), 1), "frames": torch.from_numpy(np.stack([it["frames"] for it in items]))}
        for k, dtype in (("y", torch.long), ("progress", torch.float), ("done", torch.float), ("point", torch.long)):
            batch[k] = torch.tensor([it[k] for it in items], dtype=dtype)
        return batch


class VLMNet(nn.Module):
    def __init__(self, cfg, image_processor):
        super().__init__()
        from transformers import AutoModelForImageTextToText
        self.vlm = AutoModelForImageTextToText.from_pretrained(cfg.vlm, dtype=torch.float32).model  # no LM head
        merge = self.vlm.inputs_merger  # under autocast the image features come out bf16: match the text embeddings
        self.vlm.inputs_merger = lambda input_ids, inputs_embeds, image_hidden_states: merge(
            input_ids, inputs_embeds, image_hidden_states.to(inputs_embeds.dtype))
        self.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})  # many frames
        if not cfg.vlm_train_vision:
            self.vlm.vision_model.requires_grad_(False)
        self.px = image_processor.max_image_size["longest_edge"]
        self.register_buffer("mean", torch.tensor(image_processor.image_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(image_processor.image_std).view(1, 3, 1, 1), persistent=False)
        h = self.vlm.config.text_config.hidden_size
        self.head, self.prog = nn.Linear(h, len(cfg.classes)), nn.Linear(h, 1)

    def pixels(self, frames):
        """uint8 frames [B, N, H, W, 3] -> the vision encoder's input, as the processor makes it: square, normalized."""
        x = frames.flatten(0, 1).permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=(self.px, self.px), mode="bicubic", antialias=True)
        return ((x - self.mean) / self.std).unflatten(0, frames.shape[:2])

    def forward(self, batch, mask_ratio=0.0):
        out = self.vlm(input_ids=batch["input_ids"], pixel_values=self.pixels(batch["frames"]), use_cache=False)
        last = out.last_hidden_state[:, -1]  # one prompt for every window, so no padding: the last position ends it
        return {"logits": self.head(last), "progress": self.prog(last).squeeze(-1)}


def main():
    cfg = parse(sys.argv[1:])
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    out = cfg.runs / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(cfg.vlm)
    processor.image_processor.do_image_splitting = False  # one image, one tile of tokens
    collate = Collate(processor, cfg.vlm_n, 1 / cfg.vlm_fps)
    seg = segments(cfg)
    train, val, test = (Clips(cfg, seg, s, s == "train") for s in ("train", "val", "test"))
    load = dict(batch_size=cfg.vlm_batch, num_workers=cfg.workers, collate_fn=collate, pin_memory=True)
    tl = DataLoader(train, sampler=WeightedRandomSampler(train.class_weights(), cfg.vlm_windows), drop_last=True, **load)
    model = VLMNet(cfg, processor.image_processor).cuda()
    opt = torch.optim.AdamW([{"params": [p for p in model.vlm.parameters() if p.requires_grad], "lr": cfg.vlm_lr},
                             {"params": list(model.head.parameters()) + list(model.prog.parameters()), "lr": cfg.lr}],
                            weight_decay=0.01)
    steps = cfg.vlm_epochs * (cfg.vlm_windows // cfg.vlm_batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    best, log = -1.0, []
    for ep in range(cfg.vlm_epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for b in tl:
            b = cuda(b)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(b)
            loss = F.cross_entropy(o["logits"].float(), b["y"], label_smoothing=cfg.label_smoothing) + \
                cfg.progress_w * F.binary_cross_entropy_with_logits(o["progress"].float(), b["progress"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot, nb = tot + float(loss.detach()), nb + 1
            if nb % 100 == 0:
                print(f"  step {nb} loss {tot / nb:.3f} {time.time() - t0:.0f}s", flush=True)
        res = evaluate(model, DataLoader(val, **load), cfg)
        log.append({"epoch": ep, "loss": tot / max(1, nb), "val": res, "s": round(time.time() - t0, 1)})
        print(f"epoch {ep} {time.time() - t0:.0f}s loss {tot / max(1, nb):.3f} val end f1 {res['end']['macro_f1']:.3f} "
              f"acc {res['end']['acc']:.3f} mid f1 {res['mid']['macro_f1']:.3f} done auroc {res['completion']['auroc']:.3f}", flush=True)
        if res["end"]["macro_f1"] >= best:
            best = res["end"]["macro_f1"]
            torch.save({k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}, out / "best.pt")
    model.load_state_dict({k: v.float() for k, v in torch.load(out / "best.pt").items()})
    res = evaluate(model, DataLoader(test, **load), cfg)
    cfg_out = {k: (str(v) if not isinstance(v, (int, float, bool, str)) else v) for k, v in vars(cfg).items()}
    (out / "metrics.json").write_text(json.dumps({"config": cfg_out, "best_val_end_macro_f1": best, "test": res, "log": log}, indent=1))
    print(f"TEST {cfg.name}: end f1 {res['end']['macro_f1']:.3f} acc {res['end']['acc']:.3f} | mid f1 {res['mid']['macro_f1']:.3f} "
          f"| done auroc {res['completion']['auroc']:.3f} progress mae {res['completion']['progress_mae']:.3f}", flush=True)
    print("TEST f1 by class (end):", {k: round(v, 3) for k, v in res["end"]["f1"].items()}, flush=True)


if __name__ == "__main__":
    main()
