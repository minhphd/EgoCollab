"""S2, the slow proposer: Gemini suggests task graphs, once from the instruction and again on surprise.

    prior(cfg, st)          P(G | l) as one probability p_uv per ordering, from the instruction and the subgoals
                            under shuffled IDs (text only, one call per task, cached)
    revise(cfg, st, ...)    on surprise: the leading graphs, S1's reads so far, the surprising event and the
                            person's last ego frames with gaze; returns corrected graphs
S2 only proposes: S1 scores every graph it returns on the whole history (belief.py). prior_edges() chooses the
prior's source by cfg.prior: Gemini, flat (every ordering at p_flat) or the true graph (an oracle).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from helper.llm.gemini import Gemini, generate

from .config import Config

MEANING = ("A required ordering A > B means subgoal A must be complete before B can start: the instruction asks for "
           "that order (for example \"first ... then ...\"), or the task cannot be done otherwise. A habit or a "
           "convenient order is not required.")


def ids_for(cfg: Config, st: dict) -> dict[str, str]:
    """Shuffled subgoal IDs, so the listing order leaks nothing."""
    order = random.Random(f"{cfg.seed}-{st['id']}").sample(st["nodes"], len(st["nodes"]))
    return {v: f"S{i + 1}" for i, v in enumerate(order)}


def header(st: dict, ids: dict) -> list[str]:
    label = dict(zip(st["nodes"], st["labels"]))
    lines = ["A robot assistant watches a person do a household task through the person's smart glasses.",
             f"Instruction: {st['title']}", "Subgoals (each is completed once):"]
    return lines + [f"{s}: {label[v]}" for v, s in sorted(ids.items(), key=lambda kv: int(kv[1][1:]))] + [MEANING]


def schema(ids: dict, p: bool) -> dict:
    sid = {"type": "string", "enum": sorted(ids.values(), key=lambda s: int(s[1:]))}
    props = {"before": sid, "after": sid} | ({"p": {"type": "number"}} if p else {})
    return {"type": "array", "items": {"type": "object", "properties": props, "required": list(props)}}


def gemini(cfg: Config) -> Gemini:
    return Gemini(model=cfg.model, thinking=cfg.thinking)


def prior(cfg: Config, st: dict) -> dict:
    """{"edges": [[u, v, p]], "latency", "usage"}, cached per model and episode."""
    model = cfg.prior_model or cfg.model
    path = Path(cfg.cache) / "prior" / model / f"{st['id']}.json"
    if path.exists():
        return json.loads(path.read_text())
    ids = ids_for(cfg, st)
    text = "\n".join(header(st, ids) + ["", "List every ordering that may be required, with p your probability that it "
                                        "is required. Include doubtful ones with a low p; leave out orderings that are "
                                        "clearly free."])
    out_schema = {"type": "object", "required": ["edges"], "properties": {"edges": schema(ids, True)}}
    reply, usage, latency = generate(text, schema=out_schema, cfg=Gemini(model=model, thinking=cfg.thinking))
    back = {s: v for v, s in ids.items()}
    try:
        edges = {(back[e["before"]], back[e["after"]]): min(0.98, max(0.02, float(e["p"])))
                 for e in json.loads(reply)["edges"] if e["before"] != e["after"]}
    except (ValueError, KeyError, TypeError):
        return {"edges": [], "latency": latency, "usage": usage, "error": reply[:300]}
    out = {"edges": [[u, v, p] for (u, v), p in edges.items()], "latency": latency, "usage": usage}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


def revise(cfg: Config, st: dict, top: list[tuple[frozenset, float]], log: list[str], event: str,
           images: list[bytes]) -> tuple[list[frozenset], dict, float]:
    """Corrected graphs (sets of (u, v) node-index pairs), token usage and latency."""
    ids, nodes = ids_for(cfg, st), st["nodes"]
    name = lambda i: ids[nodes[i]]
    lines = header(st, ids) + ["", "The robot keeps several hypotheses of the required orderings. The most probable now:"]
    for k, (g, w) in enumerate(top):
        lines.append(f"Graph {k + 1} (probability {w:.2f}): " + (", ".join(f"{name(u)} > {name(v)}" for u, v in sorted(g)) or "none"))
    lines += [f"What the robot has read so far (its own reads, which may hold mistakes): {'; '.join(log[-30:]) or 'nothing yet'}",
              f"Surprise: {event}"]
    if images:
        lines.append(f"The {len(images)} images show the person's view, oldest first; the red dot marks their gaze.")
    lines += ["", f"Return up to {cfg.revisions} corrected graphs, most plausible first, each listing every required "
              "ordering. If the person started B while A was not complete, A > B is not required."]
    out_schema = {"type": "object", "required": ["graphs"], "properties": {"graphs": {
        "type": "array", "maxItems": cfg.revisions,
        "items": {"type": "object", "required": ["edges"], "properties": {"edges": schema(ids, False)}}}}}
    reply, usage, latency = generate("\n".join(lines), images, out_schema, cfg=gemini(cfg))
    index = {ids[v]: i for i, v in enumerate(nodes)}
    try:
        graphs = [frozenset((index[e["before"]], index[e["after"]]) for e in g["edges"] if e["before"] != e["after"])
                  for g in json.loads(reply)["graphs"]]
    except (ValueError, KeyError, TypeError):
        graphs = []
    return graphs, usage, latency


def prior_edges(cfg: Config, st: dict) -> tuple[dict, dict]:
    """{(u, v) index pair: p_uv} for the graph prior, and the Gemini call's record."""
    index, N = {v: i for i, v in enumerate(st["nodes"])}, len(st["nodes"])
    if cfg.prior == "truth":
        return {(index[u], index[v]): 0.98 for u, v in st["truth"]["edges"]}, {}
    if cfg.prior == "flat":
        return {(u, v): cfg.p_flat for u in range(N) for v in range(N) if u != v}, {}
    p = prior(cfg, st)
    return {(index[u], index[v]): q for u, v, q in p["edges"]}, {k: p.get(k) for k in ("latency", "usage", "error")}


def frames(st: dict, t: float, k: int, gap: float) -> list[bytes]:
    """The last k ego frames with gaze before t, gap s apart, oldest first (JPEG bytes)."""
    if not st.get("frames"):
        return []
    out = []
    for j in range(k - 1, -1, -1):
        i = max(0, int(round((t - j * gap) * st["frame_fps"])))
        p = Path(st["frames"]) / f"{i:06d}.jpg"
        if p.exists():
            out.append(p.read_bytes())
    return out


def saw(text_log: list[str], obj: str, place: str, images: list[bytes], model: str = "gemini-3.5-flash") -> tuple[float, dict, float]:
    """The slow path for a belief: P(the person saw `obj` reach `place`), from their view during the move (the red
    circle marks their gaze), with the robot's log of what moved when. Returns (p, usage, latency)."""
    lines = ["A robot watches a person through the person's smart glasses. The images show the person's view, oldest "
             "first, while the robot moved an object; the red circle marks where the person's eyes point.",
             f"What the robot did: {'; '.join(text_log)}",
             f"Question: did the person notice that the {obj} ended up {place}? Judge from where they look, not from "
             "what is merely in the frame. Give p, your probability that they noticed."]
    out_schema = {"type": "object", "required": ["p", "reason"],
                  "properties": {"p": {"type": "number"}, "reason": {"type": "string"}}}
    reply, usage, latency = generate("\n".join(lines), images, out_schema, cfg=Gemini(model=model, thinking="low"))
    try:
        return min(1.0, max(0.0, float(json.loads(reply)["p"]))), usage, latency
    except (ValueError, KeyError, TypeError):
        return 0.5, usage | {"error": reply[:200]}, latency
