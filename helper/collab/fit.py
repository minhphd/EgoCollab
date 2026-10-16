"""The person model's parameters (bip.Params) as ratios of counts on HD-EPIC. Each video is cut into clips of `clip` s;
at each clip end the person is on the subgoal of the latest action (helper/datasets/hd_epic/tasks.py, pauses inside a
subgoal kept, and the last subgoal kept until another starts), and a subgoal's steps are its actions whose main verb is
pick, place, open or close, in order of their ends. Only the training and validation participants are counted. Fitted:
    s_done, s_mid  P(the next clip is on another subgoal), once the current one's steps are all done / before
    adv, adv2      P(a clip completes a step of the subgoal it is on, given one remains); P(two, given one)
    resume, delta  the pi weights of a started subgoal and of one whose predecessors are not done (a ready one: 1),
                   by maximum likelihood over the switches, with the annotated orderings as G
lambda (the plan is not annotated), upsilon and kappa are not fitted. With gaze=1, the gaze boosts instead:
    gamma_cur, gamma_next, eps  bip.Filter.fixation with uniform saliency, by maximum likelihood on the training
                   participants' fixations: real gaze on the YOLOE boxes at 2 fps (data/bip/detect), each subgoal's
                   objects from Gemini's vocabulary (data/bip/vocab), the annotated current and next subgoals

    python -m helper.collab.fit clip=3.0
    python -m helper.collab.fit gaze=1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from helper.action_cls.config import Config
from helper.datasets.hd_epic import posthoc_graph
from helper.datasets.hd_epic.data import load_narrations
from helper.datasets.hd_epic.tasks import load

OUT = Path(__file__).resolve().parents[2] / "data/collab/fit"


def step_ends(vid: str, task, narr, verb_map: dict) -> dict:
    """subgoal -> the sorted end times of its pick, place, open and close actions."""
    owner = posthoc_graph.action_owner(posthoc_graph.find(vid))
    rows = narr[narr.video_id == vid]
    verb = rows.main_action_classes.map(lambda m: m[0][0] if len(m) else -1)
    out = {v: [] for v in task.nodes}
    for i, end, c in zip(rows["index"], rows.end_timestamp.astype(float), verb):
        v = owner.get(f"{vid}-{i}")
        if v in out and c in verb_map:
            out[v].append(end)
    return {v: np.sort(e) for v, e in out.items()}


def events(task, ends: dict, clip: float) -> list[dict]:
    """One record per clip: the subgoal before and after, its steps done before, and the steps completed in it."""
    t0, t1 = task.extent
    out, prev = [], None
    for t in np.arange(t0 + clip, t1 + clip, clip):
        g = task.current(t) or prev
        if prev is not None and g is not None:
            k = int(np.sum(ends[g] <= t - clip))  # g's steps done when the clip began
            out.append({"t": t, "from": prev, "to": g, "K_from": len(ends[prev]), "k_from": int(np.sum(ends[prev] <= t - clip)),
                        "K": len(ends[g]), "k": k, "n": int(np.sum((ends[g] > t - clip) & (ends[g] <= t)))})
        prev = g
    return out


def switches(task, ev: list[dict], clip: float) -> list[tuple[str, list[str], dict]]:
    """At each switch: the subgoal chosen, the candidates (not done, not the one left), and each candidate's class,
    all as of the clip's start."""
    pred = {v: {u for u, w in task.edges if w == v} for v in task.nodes}
    out = []
    for e in ev:
        if e["to"] == e["from"]:
            continue
        t = e["t"] - clip
        done = task.done(t)
        cand = [v for v in task.nodes if v not in done and v != e["from"]]
        kind = {v: "started" if task.first[v] < t else "ready" if pred[v] <= done else "blocked" for v in cand}
        out.append((e["to"], cand, kind))
    return out


def gaze_rows(tasks) -> np.ndarray:
    """Per fixation: [x in O(g), x in O(n), |X|, |O(g) & X|, |O(n) & X|, |O(g) & O(n) & X|]."""
    from helper.bip.io import fixated
    from helper.datasets.hd_epic.data import load_framewise
    from .hdepic import DATA, boxes
    train, rows = Config().split["train"], []
    for vid, task in tasks.items():
        paths = [DATA / d / f"{vid}.json" for d in ("detect", "vocab/gemini-3.5-flash")]
        if vid[:3] not in train or not all(p.exists() for p in paths):
            continue
        det, vocab = (json.loads(p.read_text()) for p in paths)
        gaze, _ = load_framewise(vid)
        for i in range(len(det["objects"]) if gaze else 0):
            t = (det["first"] + i) / det["fps"]
            g, n = task.current(t), task.next(t)
            if g is None:
                continue
            B = boxes(det, i)
            vis = {k: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, (b[2] - b[0]) * (b[3] - b[1])) for k, b in B.items()}
            pt = gaze[min(len(gaze) - 1, int(round(t * 30)))]
            x = fixated(tuple(pt) if pt else None, vis, {k for k, v in vis.items() if v[2] > 0.1})
            if x is None:
                continue
            X, Og, On = set(vis), set(vocab.get(g, [])), set(vocab.get(n, [])) if n else set()
            rows.append([x in Og, x in On, len(X), len(Og & X), len(On & X), len(Og & On & X)])
    return np.array(rows, float)


def gaze_fit(rows: np.ndarray) -> dict:
    import torch
    a, b, n, cg, cn, cgn = torch.tensor(rows, dtype=torch.float64).T
    th = torch.tensor([1.0, 0.3, 0.0], dtype=torch.float64, requires_grad=True)  # gamma_cur, gamma_next, logit eps

    def nll():
        e = torch.sigmoid(th[2])
        den = n + torch.expm1(th[0]) * cg + torch.expm1(th[1]) * cn + torch.expm1(th[0]) * torch.expm1(th[1]) * cgn
        return -torch.log((1 - e) * torch.exp(th[0] * a + th[1] * b) / den + e / n).sum()
    opt = torch.optim.LBFGS([th], max_iter=200, line_search_fn="strong_wolfe")

    def step():
        opt.zero_grad()
        loss = nll()
        loss.backward()
        return loss
    opt.step(step)
    fun, x = float(nll()), th.detach().numpy()
    base = -np.log(1 / rows[:, 2]).sum()  # every visible object equally likely
    return {"gamma_cur": float(x[0]), "gamma_next": float(x[1]), "eps": float(1 / (1 + np.exp(-x[2]))),
            "fixations": len(rows), "nats_per_fixation_over_uniform": float((base - fun) / len(rows)),
            "on_current": float(rows[:, 0].mean()), "on_next": float(rows[:, 1].mean()),
            "current_share_of_view": float((rows[:, 3] / rows[:, 2]).mean()), "next_share_of_view": float((rows[:, 4] / rows[:, 2]).mean())}


def main():
    cfg = Config()
    args = dict(a.split("=") for a in sys.argv[1:])
    if args.get("gaze") == "1":
        fit = gaze_fit(gaze_rows(load(2)))
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "gaze.json").write_text(json.dumps(fit, indent=1))
        print(json.dumps(fit, indent=1))
        return
    clip = float(args.get("clip", 3.0))
    narr, tasks = load_narrations(), load(2)
    ev, sw, held_out = [], [], set(cfg.split["test"])
    tasks = {v: t for v, t in tasks.items() if v[:3] not in held_out}  # never the test participants
    for vid, task in tasks.items():
        e = events(task, step_ends(vid, task, narr, cfg.verb_map), clip)
        ev += e
        sw += switches(task, e, clip)
    left = lambda done: [e["to"] != e["from"] for e in ev if e["K_from"] > 0 and (e["k_from"] >= e["K_from"]) == done]
    on = [e for e in ev if e["to"] == e["from"] and e["k"] < e["K"]]
    two = [e for e in on if e["n"] >= 1 and e["K"] - e["k"] >= 2]
    chosen = [(c, kind) for c, cand, kind in sw if c in kind]
    grid = [(r, d) for r in np.exp(np.linspace(np.log(0.05), np.log(20), 61)) for d in np.exp(np.linspace(np.log(1e-3), np.log(1), 61))]

    def loglik(r, d):
        w = {"started": r, "ready": 1.0, "blocked": d}
        return sum(np.log(w[kind[c]] / sum(w[x] for x in kind.values())) for c, kind in chosen)
    r, d = max(grid, key=lambda rd: loglik(*rd))
    fit = {"clip": clip, "videos": len(tasks), "clips": len(ev),
           "s_done": float(np.mean(left(True))), "n_s_done": len(left(True)),
           "s_mid": float(np.mean(left(False))), "n_s_mid": len(left(False)),
           "adv": float(np.mean([e["n"] >= 1 for e in on])), "n_adv": len(on),
           "adv2": float(np.mean([e["n"] >= 2 for e in two])), "n_adv2": len(two),
           "resume": float(r), "delta": float(d), "switches": len(sw),
           "switch_to_done": float(1 - len(chosen) / max(1, len(sw))),
           "switch_kinds": {k: int(sum(kind[c] == k for c, kind in chosen)) for k in ("started", "ready", "blocked")}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"person_clip{clip:g}.json").write_text(json.dumps(fit, indent=1))
    print(json.dumps(fit, indent=1))


if __name__ == "__main__":
    main()
