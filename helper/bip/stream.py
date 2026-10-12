"""What the assistant observes in one HD-EPIC video, cached as data/bip/stream/<video>.json.

Steps are the narrated actions inside the task (start, end, text, owning subgoal), completions are
the end of each subgoal's last action, fixations come from gaze.py, and `sim` holds the cosine
similarity of each narration to each subgoal label (all-MiniLM-L6-v2), the S1 action scorer.

    python -m helper.bip.stream build [workers=16]
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..datasets.hd_epic import load_narrations
from . import gaze
from .config import Config, parse
from .tasks import load

EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"


def path(cfg: Config, vid: str) -> Path:
    return Path(cfg.cache) / "stream" / f"{vid}.json"


def read(cfg: Config, vid: str) -> dict:
    return json.loads(path(cfg, vid).read_text())


def split(cfg: Config) -> list[str]:
    vids = list(load(cfg.min_nodes))
    vids = {"dev": vids[: cfg.dev], "test": vids[cfg.dev:], "all": vids}[cfg.split]
    vids = [v for v in vids if v in cfg.only_videos] if cfg.only_videos else vids
    return vids[: cfg.videos or None]


def observe(args: tuple[Config, str]) -> dict:
    cfg, vid = args
    task = load(cfg.min_nodes)[vid]
    t0, t1 = task.extent
    owner = {round(a, 3): s for a, _, s in task.timeline.spans}
    narr = load_narrations()
    rows = narr[(narr.video_id == vid) & (narr.start_timestamp >= t0) & (narr.start_timestamp <= t1)]
    steps = [{"t": r.start_timestamp, "end": r.end_timestamp, "text": r.narration.strip(),
              "owner": owner.get(round(r.start_timestamp, 3))} for r in rows.sort_values("start_timestamp").itertuples()]
    objects = {o["id"]: o["name"] for o in gaze.resting(vid)}
    fix = [f for f in gaze.fixations(vid, cfg) if t0 - 5 <= f["t"] <= t1]
    return {"video": vid, "title": task.title, "nodes": task.nodes, "labels": [task.label[v] for v in task.nodes],
            "edges": task.edges, "t0": t0, "t1": t1, "steps": steps,
            "completions": sorted(({"t": task.last[v], "node": v} for v in task.nodes), key=lambda c: c["t"]),
            "fixations": fix, "objects": objects}


def embed(texts: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok, model = AutoTokenizer.from_pretrained(EMBEDDER), AutoModel.from_pretrained(EMBEDDER).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 256):
            batch = tok(texts[i:i + 256], padding=True, truncation=True, max_length=64, return_tensors="pt")
            h = model(**batch).last_hidden_state
            m = batch["attention_mask"][..., None].float()
            e = (h * m).sum(1) / m.sum(1)
            out.append(torch.nn.functional.normalize(e, dim=-1).numpy())
    return np.concatenate(out)


def build(cfg: Config, workers: int) -> None:
    vids = split(cfg)
    with ProcessPoolExecutor(workers) as pool:
        streams = list(pool.map(observe, [(cfg, v) for v in vids]))
    texts = sorted({s["text"] for st in streams for s in st["steps"]} | {l for st in streams for l in st["labels"]})
    vec = dict(zip(texts, embed(texts)))
    path(cfg, "x").parent.mkdir(parents=True, exist_ok=True)
    for st in streams:
        L = np.array([vec[l] for l in st["labels"]])
        st["sim"] = [np.round(L @ vec[s["text"]], 3).tolist() for s in st["steps"]]
        path(cfg, st["video"]).write_text(json.dumps(st))
        print(st["video"], len(st["steps"]), "steps", len(st["fixations"]), "fixations", flush=True)


if __name__ == "__main__" and sys.argv[1] == "build":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    extra = [a for a in sys.argv[2:] if a.startswith("workers=")]
    build(parse([a for a in sys.argv[2:] if a not in extra] + ["split=all"]),
          int(extra[0].split("=")[1]) if extra else 16)
