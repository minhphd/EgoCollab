"""Post-hoc task graphs: primitive actions grouped into subgoals, both with dependency edges.

One JSON file per task, possibly spanning several videos:
{
  "task": "Chopped Chickpea Salad",
  "videos": ["P07-20240529-131737", "P07-20240529-134410"],
  "nodes": [{"id": "lettuce_washed", "label": "Lettuce washed and dried", "group": "Salad",
             "actions": {"P07-20240529-131737": "40-52,57"}}],
  "edges": [{"from": "lettuce_washed", "to": "lettuce_in_bowl", "why": "Lettuce is washed before it goes in"}],
  "primitives": {"P07-20240529-131737": {"40": {"label": "turn on tap", "after": []}}}
}
Action numbers are the index suffix of HD-EPIC `unique_narration_id`. An edge (u, v) means v requires u.
A primitive's `after` lists integers (same video) or "<video_id>-<index>" strings (another video).
"""

import json
from pathlib import Path

from ..episode import Node, Segment

ANNOTATIONS = Path(__file__).resolve().parent / "annotations"
MERGE_GAP_SEC = 2.0  # actions of one node closer than this form one segment
CLIP_PAD_SEC = 1.0  # context shown before the first and after the last action of a clip


def parse_ranges(spec: str) -> list[int]:
    out = []
    for part in filter(None, (p.strip() for p in spec.split(","))):
        a, _, b = part.partition("-")
        out += range(int(a), int(b or a) + 1)
    return out


def find(video_id: str) -> dict | None:
    for path in sorted(ANNOTATIONS.glob("*.json")):
        g = json.loads(path.read_text())
        if video_id in g["videos"]:
            return g
    return None


def action_owner(g: dict) -> dict[str, str]:
    """Maps primitive id "<video>-<index>" to its subgoal id."""
    return {f"{vid}-{i}": n["id"] for n in g["nodes"] for vid, spec in n["actions"].items() for i in parse_ranges(spec)}


def primitive_edges(g: dict) -> list[tuple[str, str]]:
    edges = []
    for vid, table in g.get("primitives", {}).items():
        for idx, p in table.items():
            edges += [(ref if isinstance(ref, str) else f"{vid}-{ref}", f"{vid}-{idx}") for ref in p["after"]]
    return edges


def _cycle(succ: dict[str, list[str]]) -> str | None:
    """Returns one edge "u -> v" closing a cycle, or None."""
    state = {}
    for root in succ:
        if root in state:
            continue
        state[root], stack = "open", [(root, iter(succ[root]))]
        while stack:
            u, it = stack[-1]
            v = next(it, None)
            if v is None:
                state[u] = "closed"
                stack.pop()
            elif state.get(v) == "open":
                return f"{u} -> {v}"
            elif v not in state:
                state[v] = "open"
                stack.append((v, iter(succ.get(v, []))))
    return None


def _reach(succ: dict[str, list[str]]) -> dict[str, set[str]]:
    reach = {}
    for u in succ:
        seen, stack = set(), list(succ[u])
        while stack:
            v = stack.pop()
            if v not in seen:
                seen.add(v)
                stack.extend(succ.get(v, []))
        reach[u] = seen
    return reach


def validate(g: dict, times: dict[str, dict[int, tuple[float, float]]]) -> tuple[list[str], list[str]]:
    """Returns (errors, warnings). `times[video][index] = (start, end)` for every narration."""
    errors, warnings, owner = [], [], {}
    ids = {n["id"] for n in g["nodes"]}
    for n in g["nodes"]:
        for vid, spec in n["actions"].items():
            for i in parse_ranges(spec):
                if i not in times.get(vid, {}):
                    errors.append(f"{n['id']}: no action {vid}-{i}")
                elif (vid, i) in owner:
                    errors.append(f"{vid}-{i} in both {owner[vid, i]} and {n['id']}")
                owner[vid, i] = n["id"]
    succ = {i: [] for i in ids}
    for e in g["edges"]:
        if e["from"] not in ids or e["to"] not in ids:
            errors.append(f"edge {e['from']} -> {e['to']}: unknown node")
        else:
            succ[e["from"]].append(e["to"])
    if c := _cycle(succ):
        errors.append(f"subgoal cycle through {c}")
    if "primitives" not in g:
        return errors, warnings

    sub_of = action_owner(g)
    have = {f"{vid}-{i}" for vid, table in g["primitives"].items() for i in table}
    errors += [f"primitive {p} is not in any subgoal" for p in sorted(have - set(sub_of))]
    warnings += [f"no primitive for assigned action {p}" for p in sorted(set(sub_of) - have)]
    psucc = {p: [] for p in have}
    reach, realised = _reach(succ), set()
    for u, v in primitive_edges(g):
        if u not in have:
            errors.append(f"primitive {v}: unknown predecessor {u}")
            continue
        psucc[u].append(v)
        su, sv = sub_of.get(u), sub_of.get(v)
        if su != sv:
            realised.add((su, sv))
            if sv not in reach.get(su, set()):
                warnings.append(f"primitive edge {u} -> {v} crosses {su} -> {sv}, which the subgoal graph lacks")
    if c := _cycle(psucc):
        errors.append(f"primitive cycle through {c}")
    warnings += [f"subgoal edge {e['from']} -> {e['to']} has no primitive edge" for e in g["edges"]
                 if (e["from"], e["to"]) not in realised]
    return errors, warnings


def subgoal_episodes(g: dict, vid: str, times: dict[int, tuple[float, float]], default_labels: dict[int, str]) -> list[dict]:
    """One episode per subgoal worked on in `vid`: its primitive actions and the edges among them.

    Edges from other subgoals are dropped, so every node starts incomplete.
    """
    sub_of, labels = action_owner(g), g.get("primitives", {}).get(vid, {})
    edges = [(u, v) for u, v in primitive_edges(g) if sub_of.get(u) == sub_of.get(v)]
    out = []
    for n in g["nodes"]:
        idx = [i for i in parse_ranges(n["actions"].get(vid, "")) if i in times]
        if not idx:
            continue
        ids = {f"{vid}-{i}" for i in idx}
        start, end = min(times[i][0] for i in idx), max(times[i][1] for i in idx)
        out.append({
            "subgoal": n,
            "nodes": [Node(f"{vid}-{i}", labels.get(str(i), {}).get("label") or default_labels.get(i, f"action {i}"),
                           n["label"]) for i in idx],
            "edges": [(u, v) for u, v in edges if u in ids and v in ids],
            "segments": [Segment(f"{vid}-{i}", *times[i]) for i in idx],
            "clip": (max(0.0, start - CLIP_PAD_SEC), end + CLIP_PAD_SEC),
        })
    return out
