"""Does an action fit a subgoal? Gemini judges each action the robot can read (a verb and an object) against the task's
subgoals, P(subgoal | action) with "none" for actions outside them, in a few calls per task, cached. On HD-EPIC's test
videos every narrated action the annotations assign to a subgoal is scored against it, next to exact step-list
matching and zero-shot text similarity (SigLIP) between the action and the subgoal labels, whose temperature is fitted
on the training participants.

Actions are named two ways: as the robot would read them (a verb of the 79-verb set and the detector's name for the
object, "an unrecognized object" when it has none), and with the narration's own noun.

    PYTHONPATH=<sentencepiece> HF_HOME=<cache> python -m helper.collab.judge
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

from helper.action_cls.config import Config
from helper.datasets.hd_epic import posthoc_graph
from helper.datasets.hd_epic.data import load_narrations

from . import bip
from .hdepic import DATA, ROOT, named, two, verb_set

OUT, CHUNK, MODELS = ROOT / "data/collab/judge", 60, ("gemini-3.5-flash-lite", "gemini-3.5-flash")
RULES = ("The robot recognizes each action the person does as a verb and an object. For each action below, give the "
         "probability that the person does it as part of each subgoal, or as part of none of them (\"none\"). An action "
         "can serve several subgoals: spread the probability when unsure. List the subgoals with probability 0.05 or "
         "more; the probabilities of each action sum to 1.")


def task(vid: str) -> tuple[dict, list[str]]:
    """BIP's task spec for a video (hdepic.video's) and the detector's object names ([] without detections)."""
    st = json.loads((DATA / "stream" / f"{vid}.json").read_text())
    det_path, voc_path = DATA / "detect" / f"{vid}.json", DATA / "vocab/gemini-3.5-flash" / f"{vid}.json"
    names = json.loads(det_path.read_text())["names"] if det_path.exists() else []
    vocab = json.loads(voc_path.read_text()) if voc_path.exists() else {}
    return {"title": st["title"], "nodes": st["nodes"], "labels": st["labels"], "vocab": names,
            "objects": {n: two(vocab.get(n, []), names) for n in st["nodes"]}}, names


def actions(vid: str, spec: dict, names: list[str], narr, verbs: dict) -> list[dict]:
    """Every narrated action the annotations assign to one of the task's subgoals: its verb, the detector's name for its
    object, the narrated noun, and the subgoal (index)."""
    owner = posthoc_graph.action_owner(posthoc_graph.find(vid))
    rows, out = narr[narr.video_id == vid], []
    for i, m, nouns in zip(rows["index"], rows.main_action_classes, rows.nouns):
        v, c = owner.get(f"{vid}-{i}"), (m[0][0] if len(m) else -1)
        if v in spec["nodes"] and c in verbs:
            noun = nouns[0] if len(nouns) else None
            out.append({"verb": verbs[c], "det": named(noun, names), "noun": noun, "owner": spec["nodes"].index(v)})
    return out


def text(a: dict, rep: str) -> str:
    return f"{a['verb'].replace('-', ' ')} {a['det' if rep == 'detector' else 'noun'] or 'an unrecognized object'}"


def judge(spec: dict, texts: list[str], model: str, path) -> dict[str, np.ndarray]:
    """{action: P(subgoal) over the subgoals, then none}, from Gemini in chunks of CHUNK actions; cached at path."""
    if path.exists():
        return {k: np.array(v) for k, v in json.loads(path.read_text()).items()}
    P = bip.Params(seed=0, model=model)
    s2, N, out = bip.S2(spec, P), len(spec["nodes"]), {}
    for j in range(0, len(texts), CHUNK):
        part = texts[j:j + CHUNK]
        ids = [f"A{k + 1}" for k in range(len(part))]
        entry = {"type": "object", "required": ["subgoal", "p"], "properties": {
            "subgoal": {"type": "string", "enum": s2.ids + ["none"]}, "p": {"type": "number"}}}
        schema = {"type": "object", "required": ["actions"], "properties": {"actions": {"type": "array", "items": {
            "type": "object", "required": ["id", "p"], "properties": {
                "id": {"type": "string", "enum": ids}, "p": {"type": "array", "items": entry}}}}}}
        prompt = "\n".join(s2.header() + ["", RULES, "Actions:"] + [f"{i}: {t}" for i, t in zip(ids, part)])
        reply, _ = bip.gemini(prompt, schema, P)
        for r in (reply or {}).get("actions", []):
            p = np.zeros(N + 1)
            for e in r.get("p", []):
                p[N if e["subgoal"] == "none" else s2.back[e["subgoal"]]] += max(0.0, float(e["p"]))
            if r["id"] in ids and p.sum() > 0:
                out[part[ids.index(r["id"])]] = p / p.sum()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: v.tolist() for k, v in out.items()}))
    return out


@torch.no_grad()
def siglip(texts: list[str]) -> dict[str, torch.Tensor]:
    from transformers import AutoModel, AutoProcessor
    name = "google/siglip-base-patch16-224"
    model, proc = AutoModel.from_pretrained(name).eval(), AutoProcessor.from_pretrained(name)
    out = {}
    for i in range(0, len(texts), 256):
        x = proc(text=[t.lower() for t in texts[i:i + 256]], return_tensors="pt", padding="max_length", truncation=True)
        e = model.get_text_features(input_ids=x["input_ids"])
        out |= dict(zip(texts[i:i + 256], F.normalize(getattr(e, "pooler_output", e).float(), dim=-1)))
    return out


def softmax(s: np.ndarray, T: float) -> np.ndarray:
    e = np.exp((s - s.max()) / T)
    return e / e.sum()


def score(q: np.ndarray, owner: int) -> dict:
    """Top 1 and top 3 of P over the N subgoals (ties broken at random), and P(owner) N."""
    N, q = len(q), q / q.sum()
    above, tied = (q > q[owner]).sum(), (q == q[owner]).sum()
    hit = lambda k: min(max(k - above, 0), tied) / tied
    return {"top1": hit(1), "top3": hit(3), "ratio": float(q[owner] * N)}


def evidence(rows: list[dict]) -> tuple[float, float]:
    """The mean log likelihood ratio against uniform of P' = (1 - eps) P + eps / N, at the eps that maximizes it: what
    one action tells BIP once its floor is set."""
    nats = {e: np.mean([np.log((1 - e) * r["ratio"] + e) for r in rows]) for e in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)}
    best = max(nats, key=nats.get)
    return float(nats[best]), best


def main():
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    narr, cfg = load_narrations(), Config()
    _, verbs = verb_set(True)
    vids = sorted(p.stem for p in (DATA / "stream").glob("*.json"))
    test = [v for v in vids if v[:3] in cfg.split["test"] and (DATA / "detect" / f"{v}.json").exists()][:int(args.get("limit", 99))]
    train = [v for v in vids if v[:3] not in cfg.split["test"] and posthoc_graph.find(v)]
    data = {}
    for v in test + train:
        spec, names = task(v)
        data[v] = (spec, names, actions(v, spec, names, narr, verbs))
    # SigLIP: the temperature that best explains the training participants' actions, nouns as narrated
    texts = sorted({text(a, r) for v in data for a in data[v][2] for r in ("detector", "noun")} |
                   {s for v in data for s in data[v][0]["labels"]})
    emb = siglip(texts)
    sims = lambda v, rep: [(torch.stack([emb[text(a, rep)]]) @ torch.stack([emb[s] for s in data[v][0]["labels"]]).T)[0].numpy()
                           for a in data[v][2]]
    tr = [(s, a["owner"]) for v in train for s, a in zip(sims(v, "noun"), data[v][2])]
    temp = min((0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5), key=lambda T: -np.mean([np.log(softmax(s, T)[o] + 1e-12) for s, o in tr]))
    jobs = [(v, m, rep) for v in test for m in MODELS[:int(args.get("models", len(MODELS)))] for rep in ("detector", "noun")]
    with ThreadPoolExecutor(8) as ex:  # the calls wait on the network
        got = ex.map(lambda j: judge(data[j[0]][0], sorted({text(a, j[2]) for a in data[j[0]][2]}), j[1], OUT / j[1] / f"{j[0]}_{j[2]}.json"), jobs)
        tables = dict(zip(jobs, got))
    res, rows = {}, {}
    for v in test:
        spec, names, acts = data[v]
        N = len(spec["nodes"])
        cache = str(ROOT / "data/collab/prior_hdepic" / v)
        lists = {"exact 4-verb lists": bip.S2(spec, bip.Params(seed=0, cache=cache)).steps(),
                 "exact 79-verb lists": bip.S2(spec, bip.Params(seed=0, cache=cache, verbs=verb_set(True)[0],
                                                                verb_prior=(1 / 80,) * 80)).steps()}
        for rep in ("detector", "noun"):
            for a, s in zip(acts, sims(v, rep)):
                rows.setdefault(("SigLIP text similarity", rep), []).append(score(softmax(s, temp), a["owner"]))
        for name, R in lists.items():
            for a in acts:
                fit = np.array([any(s["verb"] == a["verb"] and s["obj"] == a["det"] for s in r["steps"]) for r in R], float)
                rows.setdefault((name, "detector"), []).append(score(fit if fit.any() else np.ones(N), a["owner"]))
        for (vid, m, rep), tab in tables.items():
            for a in acts if vid == v else ():
                p = tab.get(text(a, rep))
                rows.setdefault((m, rep), []).append({**score(p[:N] if p is not None and p[:N].sum() > 0 else np.ones(N), a["owner"]),
                                                       "none": float(p[N]) if p is not None else None, "missing": p is None})
        for a in acts:
            rows.setdefault(("uniform", "-"), []).append(score(np.ones(N), a["owner"]))
    print(f"{len(test)} test videos, {sum(len(data[v][2]) for v in test)} actions; SigLIP temperature {temp} (training videos)")
    print(f"{'method':24s} {'actions as':9s} {'top 1':>6s} {'top 3':>6s} {'nats':>6s} {'eps':>5s}  P(none)  missing")
    for (name, rep), r in rows.items():
        mean = lambda k: np.mean([x[k] for x in r if x.get(k) is not None]) if any(x.get(k) is not None for x in r) else float("nan")
        nats, eps = evidence(r)
        res[f"{name} | {rep}"] = {k: float(mean(k)) for k in ("top1", "top3", "none")} | {
            "nats": nats, "eps": eps, "missing": int(sum(x.get("missing", 0) for x in r))}
        print(f"{name:24s} {rep:9s} {mean('top1'):6.3f} {mean('top3'):6.3f} {nats:6.2f} {eps:5.2f}  {mean('none'):6.2f}  {res[f'{name} | {rep}']['missing']}")
    (ROOT / "runs/collab/judge.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
