"""S2: Gemini proposes task graphs, once from the description and again on surprise.

The prior call is text only: task, subgoals and the scene's object names. It returns required
orderings with probabilities and the objects of each subgoal (the gaze sets O(v)). A revision also
sees the person's view of the last seconds with gaze, the leading graphs and what S1 has read so
far, and returns corrected graphs. S2 only proposes: S1 scores every graph it returns.
"""

import json
import random
from pathlib import Path

from .config import Config
from .gemini import generate
from .tasks import clock
from .view import View

MEANING = ("A required ordering A > B means subgoal A must be complete before B can start, because of how the task "
           "works (an object must be fetched, a container opened, a device switched on, an ingredient prepared), "
           "not just the person's habit.")


def ids_for(cfg: Config, nodes: list[str], vid: str) -> dict[str, str]:
    """Shuffled subgoal IDs, so the listing order leaks nothing (as in RQ1)."""
    order = random.Random(f"{cfg.seed}-{vid}").sample(nodes, len(nodes))
    return {v: f"S{i + 1}" for i, v in enumerate(order)}


def header(st: dict, ids: dict) -> list[str]:
    label = dict(zip(st["nodes"], st["labels"]))
    lines = ["A robot assistant watches a person do a household task through the person's smart glasses.",
             f"Task: {st['title']}", "Subgoals (each is completed once):"]
    return lines + [f"{s}: {label[v]}" for v, s in sorted(ids.items(), key=lambda kv: int(kv[1][1:]))] + [MEANING]


def edge_schema(sid: dict, p: bool) -> dict:
    props = {"before": sid, "after": sid} | ({"p": {"type": "number"}} if p else {})
    return {"type": "array", "items": {"type": "object", "properties": props, "required": list(props)}}


def prior(cfg: Config, st: dict, ids: dict) -> dict:
    """{"edges": [[u, v, p]], "objects": {node: [names]}, "latency", "usage"}, cached per model and video."""
    path = Path(cfg.cache) / "prior" / cfg.model / f"{st['video']}.json"
    if path.exists():
        return json.loads(path.read_text())
    names = sorted(set(st["objects"].values()))
    lines = header(st, ids) + [f"Objects tracked in the scene: {'; '.join(names)}.", "",
                               "1. List every ordering that may be required, with p your probability that it is "
                               "required. Include doubtful ones with a low p.",
                               "2. For every subgoal, list the scene objects a person handles or looks at while "
                               "doing it, using the exact names above."]
    sid = {"type": "string", "enum": sorted(ids.values(), key=lambda s: int(s[1:]))}
    schema = {"type": "object", "required": ["edges", "objects"], "properties": {
        "edges": edge_schema(sid, True),
        "objects": {"type": "array", "items": {"type": "object", "required": ["subgoal", "names"], "properties": {
            "subgoal": sid, "names": {"type": "array", "items": {"type": "string", "enum": names}}}}}}}
    text, usage, latency = generate(cfg, "\n".join(lines), schema=schema)
    back = {s: v for v, s in ids.items()}
    try:
        obj = json.loads(text)
        edges = {(back[e["before"]], back[e["after"]]): min(0.98, max(0.02, float(e["p"])))
                 for e in obj["edges"] if e["before"] != e["after"]}
        objects = {back[o["subgoal"]]: sorted(set(o["names"])) for o in obj["objects"]}
    except (ValueError, KeyError, TypeError):
        return {"edges": [], "objects": {}, "latency": latency, "usage": usage, "error": text[:500]}
    out = {"edges": [[u, v, p] for (u, v), p in edges.items()], "objects": objects, "latency": latency, "usage": usage}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


def vocab(cfg: Config, st: dict, ids: dict) -> dict[str, list[str]]:
    """Objects of each subgoal as plain nouns a detector can find, from the task text alone (detector prompts
    and gaze sets for the real perception tier), cached per model and video."""
    path = Path(cfg.cache) / "vocab" / cfg.model / f"{st['video']}.json"
    if path.exists():
        return json.loads(path.read_text())
    lines = header(st, ids)[:-1] + ["", "For every subgoal, list up to 6 physical objects a person handles or looks at while "
                                    "doing it, as short common nouns an object detector recognizes (for example \"frying pan\", "
                                    "\"spatula\", \"plate\"). Use the same name for the same object across subgoals."]
    sid = {"type": "string", "enum": sorted(ids.values(), key=lambda s: int(s[1:]))}
    schema = {"type": "object", "required": ["objects"], "properties": {"objects": {"type": "array", "items": {
        "type": "object", "required": ["subgoal", "names"],
        "properties": {"subgoal": sid, "names": {"type": "array", "maxItems": 6, "items": {"type": "string"}}}}}}}
    text, _, _ = generate(cfg, "\n".join(lines), schema=schema)
    back = {s: v for v, s in ids.items()}
    try:
        out = {back[o["subgoal"]]: sorted({n.strip().lower() for n in o["names"] if n.strip()}) for o in json.loads(text)["objects"]}
    except (ValueError, KeyError, TypeError):
        return {}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


def frames(cfg: Config, vid: str, t: float) -> list[bytes]:
    return View(cfg, vid).jpegs(t, cfg.s2_frames, cfg.s2_frame_gap)


def revise(cfg: Config, st: dict, ids: dict, t: float, top: list[tuple[frozenset, float]], log: list[str],
           event: str, nodes: list[str]) -> tuple[list[frozenset], dict, float]:
    """Corrected graphs (sets of (u, v) node-index pairs), token usage and latency."""
    name = lambda i: ids[nodes[i]]
    lines = header(st, ids) + ["", "The robot keeps several hypotheses of the required orderings. The most probable now:"]
    for k, (g, w) in enumerate(top):
        lines.append(f"Graph {k + 1} (probability {w:.2f}): " + (", ".join(f"{name(u)} > {name(v)}" for u, v in sorted(g)) or "none"))
    lines += [f"What the robot has read so far, by time since the start (its own reads, which may hold mistakes): "
              f"{'; '.join(log[-40:]) or 'nothing yet'}",
              f"Surprise at {clock(t - st['t0'])}: {event}",
              f"The {cfg.s2_frames} images show the person's view over the last {cfg.s2_frame_gap * (cfg.s2_frames - 1):g} s, "
              "oldest first; the red glow marks where they looked.", "",
              f"Return up to {cfg.revisions} corrected graphs, most plausible first, each listing every required "
              "ordering. If the person started B while A was not complete, A > B is not required. Keep the orderings "
              "the task needs, and add any the hypotheses miss."]
    sid = {"type": "string", "enum": sorted(ids.values(), key=lambda s: int(s[1:]))}
    schema = {"type": "object", "required": ["graphs"], "properties": {"graphs": {
        "type": "array", "maxItems": cfg.revisions,
        "items": {"type": "object", "required": ["edges"], "properties": {"edges": edge_schema(sid, False)}}}}}
    text, usage, latency = generate(cfg, "\n".join(lines), frames(cfg, st["video"], t), schema)
    index = {ids[v]: i for i, v in enumerate(nodes)}
    try:
        graphs = [frozenset((index[e["before"]], index[e["after"]]) for e in g["edges"] if e["before"] != e["after"])
                  for g in json.loads(text)["graphs"]]
    except (ValueError, KeyError, TypeError):
        graphs = []
    return graphs, usage, latency


def compare(models: list[str]) -> None:
    """Edge F1 of each model's prior (orderings with p >= 0.5, on closure) on the test videos."""
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import replace

    from .score import closure, prf
    from .tasks import load
    from . import stream
    cfg, tasks = Config(split="test"), load(3)
    for model in models:
        c = replace(cfg, model=model)
        def one(vid):
            st = stream.read(c, vid)
            return vid, prior(c, st, ids_for(c, st["nodes"], vid))
        with ThreadPoolExecutor(16) as pool:
            out = dict(pool.map(one, stream.split(c)))
        n = {"tp": 0, "fp": 0, "fn": 0}
        for vid, p in out.items():
            task = tasks[vid]
            truth, guess = closure(task.nodes, task.edges), closure(task.nodes, [(u, v) for u, v, q in p["edges"] if q >= 0.5])
            n["tp"], n["fp"], n["fn"] = n["tp"] + len(guess & truth), n["fp"] + len(guess - truth), n["fn"] + len(truth - guess)
        lat = sorted(p["latency"] for p in out.values())
        print(model, prf(n), f"median latency {lat[len(lat) // 2]:.1f} s, failed {sum('error' in p and p['error'] for p in out.values())}")


if __name__ == "__main__":
    import sys
    compare(sys.argv[1:])
