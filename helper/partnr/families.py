"""Task families of our own on PARTNR's scenes: one instruction, several valid structures (recipes), one sampled per
episode. Spare objects of every class sit in the kitchen, so the scene does not reveal the recipe.

    set_table   "Set the table for two." Recipes: plate, fork and knife; bowl, spoon and cup; plate, fork and
                glass, for each of two places, on the dining table. Cutlery goes after its own plate or bowl
                (an ordering; next-to placements are left out, since PARTNR's placement sampler often fails them).

Episodes are PARTNR's format (generator placements, then our propositions), loadable with split=<out>. Run on
a GPU node with the partnr env, from third_party/partnr/partnr-planner (its data/ links to ours):
    python -m helper.partnr.families family=set_table out=settable_pilot per_scene=6
    python -c "from helper.partnr.families import count; count(src, dst)"    counted propositions (category level)
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np

FAMILIES = {
    "set_table": {
        "instruction": "Set the table for two.",
        "people": 2,
        "spare": {"plate": 3, "bowl": 3, "fork": 3, "knife": 3, "spoon": 3, "cup": 3, "glass": 3},
        "recipes": {"plate_fork_knife": [("plate", None), ("fork", "plate"), ("knife", "plate")],
                    "bowl_spoon_cup": [("bowl", None), ("spoon", "bowl"), ("cup", None)],
                    "plate_fork_glass": [("plate", None), ("fork", "plate"), ("glass", None)]},
        "target": ("table", "dining room"),
        "source_regions": ["kitchen"],
        "clutter": 4,
        "scenes": ["106366386_174226770", "106366410_174226806", "102344529", "102815835", "104348010_171512832",
                   "106878960_174887073"],
    },
}


def init_state(fam: dict, sid: str, eid: str) -> dict:
    """The generator's constraints: spare objects of every class anywhere in the kitchen, then clutter."""
    state = [{"name": f"spare {c}", "number": str(n), "object_classes": [c], "location": "on", "furniture_classes": [],
              "allowed_regions": fam["source_regions"]} for c, n in fam["spare"].items()]
    state.append({"name": "clutter", "number": str(fam["clutter"]), "common_sense_object_classes": True,
                  "location": "on", "furniture_names": []})
    return {"scene_id": sid, "episode_id": eid, "initial_state": state}


def targets(gen, cls: str, region: str) -> list[str]:
    """Handles of the furniture of class `cls` inside a region of category `region`."""
    sim, mi, out = gen.sim, gen.metadata_interface, []
    for mgr in (sim.get_rigid_object_manager(), sim.get_articulated_object_manager()):
        for h, obj in mgr.get_objects_by_handle_substring().items():
            if mi.get_object_instance_category(obj) == cls and any(
                    r.category.name() == region and r.contains(obj.translation) for r in sim.semantic_scene.regions):
                out.append(h)
    return sorted(out)


def evaluation(fam: dict, recipe: list, objects: dict, tables: list[str]) -> tuple[list, list]:
    """PARTNR propositions and constraints: every item on a target table; cutlery after its own plate or bowl."""
    props, edges = [], []
    for i in range(fam["people"]):
        index = {}
        for cls, ref in recipe:
            h = objects[cls][i]
            index[cls] = len(props)
            props.append({"function_name": "is_on_top", "args": {"object_handles": [h], "receptacle_handles": tables,
                                                                  "number": 1, "is_same_receptacle": False}})
            if ref is not None:
                edges.append([index[ref], len(props) - 1])
    n = len(props)
    cons = [{"type": "TemporalConstraint", "args": {"dag_edges": edges, "n_propositions": n}},
            {"type": "TerminalSatisfactionConstraint", "args": {"proposition_indices": list(range(n)), "n_propositions": None}}]
    return props, cons


def kind(name: str) -> str:
    return re.sub(r"_\d+$", "", name)


def hypotheses(task: dict, family: str = "set_table") -> dict:
    """Every recipe of the family as a category-level structure: the j-th plate, the j-th fork, ..., each served
    by any instance of its class in the (premapped) home, on the task's target furniture; every plate or bowl
    before any cutlery, as PARTNR scores it. Node names match tasks.category_subgoals ("plate1")."""
    fam = FAMILIES[family]
    tos = sorted({f for sp in task["spec"].values() if sp for f in (sp.get("tos") or [sp["to"]])})
    inst = {c: sorted(o for o in task["inventory"] if kind(o) == c) for c in fam["spare"]}
    out = {}
    for recipe, items in fam["recipes"].items():
        nodes, spec, objects, labels, edges = [], {}, {}, [], []
        for j in range(1, fam["people"] + 1):
            for cls, ref in items:
                v = f"{cls}{j}"
                nodes.append(v)
                spec[v] = {"objects": inst[cls], "rank": j, "rel": "on", "to": tos[0], "tos": tos, "ref": None}
                objects[v] = inst[cls] + tos
                labels.append(f"put {'a' if j == 1 else 'another'} {cls} on the table")
                if ref:
                    edges += [[f"{ref}{k}", v] for k in range(1, fam["people"] + 1)]
        out[recipe] = {"nodes": nodes, "spec": spec, "objects": objects, "labels": labels, "edges": edges,
                       "furniture": task.get("furniture", []), "map": task.get("map", []), "t0": 0.0}
    return out


def recipe_prior(cfg, task: dict, sid: str, family: str = "set_table") -> dict:
    """P(recipe | instruction, premapped home): Gemini's probability that the task uses each object class, a
    recipe's prior is the product over the family's classes, normalized over its recipes (cached per episode)."""
    fam = FAMILIES[family]
    path = Path(cfg.cache) / "recipe_prior" / (cfg.prior_model or cfg.model) / f"{sid}.json"
    if path.exists():
        return json.loads(path.read_text())
    from helper.llm.gemini import Gemini, generate
    classes = sorted({kind(o) for o in task["inventory"]})
    text = (f"A person does a household task. Instruction: {fam['instruction']}\n"
            f"Furniture in the home: {', '.join(task['furniture'])}.\n"
            f"Objects in the home and where they are: {'; '.join(f'{o} on {f}' for o, f in task['inventory'].items())}.\n"
            "For every object class, give p, your probability that the task requires putting objects of that class "
            "on the table.")
    schema = {"type": "object", "required": ["classes"], "properties": {"classes": {"type": "array", "items": {
        "type": "object", "required": ["cls", "p"], "properties": {"cls": {"type": "string", "enum": classes}, "p": {"type": "number"}}}}}}
    reply, usage, latency = generate(text, schema=schema, cfg=Gemini(model=cfg.prior_model or cfg.model))
    p = {c: 0.02 for c in classes} | {x["cls"]: min(0.98, max(0.02, float(x["p"]))) for x in json.loads(reply)["classes"]}
    raw = {r: float(np.prod([p[c] if c in {k for k, _ in items} else 1 - p[c] for c in fam["spare"] if c in p]))
           for r, items in fam["recipes"].items()}
    out = {r: v / sum(raw.values()) for r, v in raw.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


def count(src: str, dst: str) -> None:
    """Rewrite a family's episodes with counted propositions: "any 2 plates on a target table" per recipe class,
    over every instance of the class; an ordering between two classes stays an ordering between their
    propositions. Classes come from the object metadata."""
    import csv
    cats = {}
    for p in ("data/objects_ovmm/object_categories.csv", "data/hssd-hab/metadata/object_categories_filtered.csv"):
        for row in csv.reader(open(p)):
            if len(row) >= 2:
                cats.setdefault(row[0], row[1])
    data = json.load(gzip.open(src))
    for e in data["episodes"]:
        fam = FAMILIES[e["info"]["family"]]
        recipe, tables = fam["recipes"][e["info"]["recipe"]], e["evaluation_propositions"][0]["args"]["receptacle_handles"]
        by_cls = {}
        for h in e["name_to_receptacle"]:
            by_cls.setdefault(cats.get(re.sub(r"_:\d+$", "", h)), []).append(h)
        index, props, edges = {}, [], []
        for cls, ref in recipe:
            index[cls] = len(props)
            props.append({"function_name": "is_on_top", "args": {"object_handles": sorted(by_cls[cls]), "receptacle_handles": tables,
                                                                  "number": fam["people"], "is_same_receptacle": False}})
            if ref:
                edges.append([index[ref], index[cls]])
        n = len(props)
        e["evaluation_propositions"] = props
        e["evaluation_constraints"] = [{"type": "TemporalConstraint", "args": {"dag_edges": edges, "n_propositions": n}},
                                       {"type": "TerminalSatisfactionConstraint", "args": {"proposition_indices": list(range(n)), "n_propositions": None}}]
    with gzip.open(dst, "wt") as f:
        json.dump(data, f)
    print(len(data["episodes"]), "episodes ->", dst)


def main() -> None:
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    fam, name = FAMILIES[args.get("family", "set_table")], args.get("family", "set_table")
    per_scene = int(args.get("per_scene", 6))
    from dataset_generation.benchmark_generation.generate_episodes import (default_gen_config, generate_episode,
                                                                          initialize_generator, save_ep_dataset)
    from habitat_llm.sims.metadata_interface import default_metadata_dict
    gen = initialize_generator(dict(default_gen_config), dict(default_metadata_dict))
    eps, extra, recipes = [], [], list(fam["recipes"])
    for j, sid in enumerate(fam["scenes"]):
        for k in range(per_scene):
            recipe = recipes[(j + k) % len(recipes)]  # every scene gets every recipe equally often
            eid = f"{name}_{j}_{k}"
            ep = None
            for _ in range(3):  # placements can fail; retry with fresh samples
                ep, _details = generate_episode(gen, init_state(fam, sid, eid))
                if ep is not None:
                    break
            if ep is None:
                print("failed", eid, flush=True)
                continue
            objects = {}
            for obj in gen.ep_sampled_objects:
                objects.setdefault(gen.metadata_interface.get_object_instance_category(obj), []).append(obj.handle)
            tables = targets(gen, *fam["target"])
            if not tables or any(len(objects.get(c, [])) < fam["people"] for c, _ in fam["recipes"][recipe]):
                print("skipped", eid, "tables", len(tables), {c: len(v) for c, v in objects.items()}, flush=True)
                continue
            eps.append(ep)
            extra.append({"episode_id": eid, "recipe": recipe, "family": name,
                          "evaluation": evaluation(fam, fam["recipes"][recipe], objects, tables)})
            print("ok", eid, recipe, {c: len(v) for c, v in objects.items()}, "tables", len(tables), flush=True)
    out = Path(f"data/datasets/partnr_episodes/v0_0/{args.get('out', 'settable_pilot')}.json.gz")
    save_ep_dataset(eps, str(out))
    data = json.load(gzip.open(out))  # add what the generator leaves out: instruction, evaluation, our ids
    for e, x in zip(data["episodes"], extra):
        props, cons = x["evaluation"]
        e |= {"episode_id": x["episode_id"], "instruction": fam["instruction"], "evaluation_propositions": props,
              "evaluation_constraints": cons, "evaluation_proposition_dependencies": []}
        e["info"] = e.get("info", {}) | {"family": x["family"], "recipe": x["recipe"]}
    with gzip.open(out, "wt") as f:
        json.dump(data, f)
    print(len(data["episodes"]), "episodes ->", out)


if __name__ == "__main__":
    main()
