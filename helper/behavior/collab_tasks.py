"""Collaborative rearrangement tasks from the BEHAVIOR-1K 2026 challenge instances: each activity's BDDL goal, grounded
on the instance's objects, becomes subgoals (object, relation, target) with orderings (a subgoal whose target is
itself moved comes after the move). Writes one row per (activity, scene) template.

    python -m helper.behavior.collab_tasks <out.json> [min_subgoals=4]
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path.home() / "BEHAVIOR-1K"
DEFS = ROOT / "bddl3/bddl/activity_definitions"
INST = ROOT / "datasets/2026-challenge-task-instances"
RELS = {"ontop", "inside", "nextto", "under", "touching"}


def sexp(text: str):
    """The BDDL text as nested lists of tokens."""
    toks = re.findall(r"\(|\)|[^\s()]+", re.sub(r";.*", "", text))
    stack = [[]]
    for t in toks:
        if t == "(":
            stack.append([])
        elif t == ")":
            done = stack.pop()
            stack[-1].append(done)
        else:
            stack[-1].append(t)
    return stack[0][0]


def section(problem: list, name: str) -> list:
    return next(s for s in problem if isinstance(s, list) and s and s[0] == name)


def instances(problem: list) -> dict[str, list[str]]:
    """{category: [instance, ...]} from the :objects section."""
    out, names, toks = {}, [], section(problem, ":objects")[1:]
    for i, t in enumerate(toks):
        if t == "-" or (i and toks[i - 1] == "-"):
            if t != "-":  # the category of the names before the dash
                out.setdefault(t, []).extend(names)
                names = []
            continue
        names.append(t)
    return out


def ground(e, objs: dict, bind: dict) -> list[tuple]:
    """Atoms (rel, a, b) of goal expression e under variable bindings; the first option of a choice."""
    head = e[0]
    arg = lambda v: bind.get(v, v.lstrip("?"))
    if head == "and":
        return [a for x in e[1:] for a in ground(x, objs, bind)]
    if head in ("or",):
        return ground(e[1], objs, bind)
    if head == "not":
        return []  # negated goals are not placements
    if head in ("forall", "exists", "forn"):
        var, cat = (e[2] if head == "forn" else e[1])[0], (e[2] if head == "forn" else e[1])[2]
        body, items = e[-1], objs.get(cat, [])
        items = items[:int(e[1][0])] if head == "forn" else items[:1] if head == "exists" else items
        return [a for it in items for a in ground(body, objs, {**bind, var: it})]
    if head in ("forpairs", "fornpairs"):
        (v1, _, c1), (v2, _, c2) = e[-3], e[-2]
        pairs = list(zip(objs.get(c1, []), objs.get(c2, [])))
        pairs = pairs[:int(e[1][0])] if head == "fornpairs" else pairs
        return [a for p, q in pairs for a in ground(e[-1], objs, {**bind, v1: p, v2: q})]
    return [(head, *[arg(v) for v in e[1:]])]


def subgoals(activity: str, names: dict) -> tuple[list[dict], list[list[int]]]:
    problem = sexp((DEFS / activity / "problem0.bddl").read_text())
    atoms = ground(section(problem, ":goal")[1], instances(problem), {})
    if any(a[0] not in RELS for a in atoms):  # also needs a state change (cooked, frozen, open...)
        raise ValueError(f"not pure rearrangement: {sorted({a[0] for a in atoms} - RELS)}")
    sg = [{"obj": names.get(a[1], a[1]), "rel": a[0], "target": names.get(a[2], a[2]), "inst": [a[1], a[2]]}
          for a in atoms if a[0] in RELS and len(a) == 3]
    moved = {s["obj"]: i for i, s in enumerate(sg)}
    order = [[moved[s["target"]], j] for j, s in enumerate(sg) if s["target"] in moved and moved[s["target"]] != j]
    return sg, order


def main() -> None:
    out, args = Path(sys.argv[1]), dict(a.split("=", 1) for a in sys.argv[2:])
    rows = []
    for tpl in sorted(INST.glob("scenes/*/json/*_task_*_0_0_template.json")):
        scene, activity = re.match(r"(.+)_task_(.+)_0_0_template", tpl.stem).groups()
        if not (DEFS / activity / "problem0.bddl").exists():
            continue
        meta = json.loads(tpl.read_text())["metadata"]["task"]
        names = ast.literal_eval(meta["inst_to_name"]) if isinstance(meta["inst_to_name"], str) else meta["inst_to_name"]
        try:
            sg, order = subgoals(activity, names)
        except Exception as ex:
            print("skip", activity, repr(ex)[:120])
            continue
        if len(sg) < int(args.get("min_subgoals", 4)) or any(s["rel"] not in RELS for s in sg):
            continue
        rows.append({"activity": activity, "scene": scene, "template": str(tpl.relative_to(ROOT)), "subgoals": sg,
                     "orderings": order, "robot_start": meta.get("robot_poses", {}).get("robot", [None])[0]})
        print(activity, scene, len(sg), "subgoals", len(order), "orderings")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out, "w"), indent=1)
    print(len(rows), "tasks ->", out)


if __name__ == "__main__":
    main()
