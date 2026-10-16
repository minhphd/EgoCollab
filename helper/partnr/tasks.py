"""Subgoals of a PARTNR episode: one per object placement, with the skill script that achieves it.

PARTNR's own resolver (PropositionResolver.solve_dag) picks the objects and receptacles that satisfy the
propositions and their argument constraints, and expands each into Navigate, (Open,) Pick, Navigate, Place.
A placement proposition (is_on_top, is_inside, is_on_floor) is one subgoal; an is_next_to proposition about
the same object joins it, and its reference becomes the Place's next_to argument (on its own the resolver
would pick a receptacle near the reference's starting place); when another subgoal places the reference, this
one uses that subgoal's receptacle. A subgoal is done when all its propositions
hold and nobody carries its object. True orderings are the TemporalConstraint edges between subgoals; a
subgoal's entities O(v) are its object, receptacle and next-to reference.
"""

from __future__ import annotations

import re
from collections import defaultdict


def resolver_class():
    from habitat_llm.planner.scripted_centralized_planner import PropositionResolver

    class Capture(PropositionResolver):
        def get_list_actions_from_rearrange(self, action, *args, **kwargs):
            out = super().get_list_actions_from_rearrange(action, *args, **kwargs)
            self.acts[action.prop_ind].append(action)
            return out

        def assign_dag(self, prop_groups, prop_indices, constraints_agent_assignment, num_agents=2):
            self.groups = (prop_groups, prop_indices)
            return {i: [] for i in range(num_agents)}

    return Capture


def words(name: str) -> str:
    return name.replace("_", " ")


PLACE = ("is_on_top", "is_inside", "is_on_floor")


def counted(ep) -> bool:
    """Whether the episode's propositions are counted placements ("any 2 plates on a dining table")."""
    return any(p.args.get("number", 1) > 1 for p in ep.evaluation_propositions)


def category_subgoals(sim) -> dict:
    """Subgoals of a counted episode: one category subgoal per rank ("plate1", "plate2"), any instance of the
    class serves; scripts are None and built when the person or the robot starts one (person.instance_script).
    An ordering between counted propositions holds for every rank (every plate before any fork), as PARTNR scores
    it: an unrolled proposition inherits all its parent's orderings."""
    from habitat_llm.agent.env.evaluation.evaluation_functions import TemporalConstraint

    env, ep = sim.env, sim.episode
    name = lambda h: env.perception.sim_handle_to_name.get(h, h)
    kind = lambda x: re.sub(r"_\d+$", "", x)
    props, nodes, spec, objects, labels, index, cls = ep.evaluation_propositions, [], {}, {}, [], {}, {}
    for i, p in enumerate(props):
        objs, tos = sorted(name(h) for h in p.args["object_handles"]), sorted(name(h) for h in p.args["receptacle_handles"])
        cls[i] = kind(objs[0])
        for j in range(1, p.args.get("number", 1) + 1):
            v = f"{cls[i]}{j}"
            nodes.append(v)
            index[v] = i
            spec[v] = {"objects": objs, "rank": j, "rel": "on", "to": tos[0], "tos": tos, "ref": None}
            objects[v] = objs + tos
            labels.append(f"put {'a' if j == 1 else 'another'} {words(cls[i])} on the {words(kind(tos[0]))}")
    edges = []
    for c in ep.evaluation_constraints:
        if isinstance(c, TemporalConstraint):
            for u, w in c.dag.edges:
                edges += [[f"{cls[int(u)]}{j}", f"{cls[int(w)]}{k}"] for j in range(1, props[int(u)].args.get("number", 1) + 1)
                          for k in range(1, props[int(w)].args.get("number", 1) + 1)]
    return {"title": ep.instruction, "nodes": nodes, "index": index, "labels": labels, "objects": objects,
            "object": {v: None for v in nodes}, "props": {v: [index[v]] for v in nodes}, "spec": spec,
            "scripts": {v: None for v in nodes}, "edges": edges,
            "inventory": {name(h): name(r.split("|")[0]) for h, r in ep.name_to_receptacle.items()},
            "recipe": (ep.info or {}).get("recipe")}


def subgoals(sim, seed: int) -> dict:
    """{title, nodes, labels, objects, object, props, scripts, edges} for the current episode."""
    from habitat_llm.agent.env.evaluation.evaluation_functions import TemporalConstraint

    env, ep = sim.env, sim.episode
    if counted(ep):
        return category_subgoals(sim)
    name = lambda h: env.perception.sim_handle_to_name.get(h, h)
    R = resolver_class()(env.perception.sim_handle_to_name, env.perception.region_id_to_name, seed=seed)
    R.acts = defaultdict(list)
    R.solve_dag(ep.evaluation_propositions, ep.evaluation_constraints, ep.evaluation_proposition_dependencies,
                env.full_world_graph, env)
    script = defaultdict(list)
    for group, indices in zip(*R.groups):
        for actions, i in zip(group, indices):
            script[i] += [(tool, arg) for tool, arg, _ in actions]
    props = ep.evaluation_propositions
    obj_of = {i: R.acts[i][0].obj for i in script if R.acts[i] and hasattr(R.acts[i][0], "obj")}
    home = {i: i for i in script}  # proposition -> the subgoal (placement proposition) it belongs to
    ref = {}
    for j, p in enumerate(props):
        if p.function_name != "is_next_to" or j not in script:
            continue
        a = name(p.args["entity_handles_a"][0])
        place = [i for i in script if props[i].function_name in PLACE and obj_of.get(i) == a]
        if place:
            home[j], ref[place[0]] = place[0], name(p.args["entity_handles_b"][0])
    nodes = sorted(i for i in script if home[i] == i)
    to = {i: R.acts[i][0].to for i in nodes if hasattr(R.acts[i][0], "to")}
    placed_by = {obj_of.get(i): i for i in nodes}
    labels, objects, scr, spec = {}, {}, {}, {}
    for i in nodes:
        a, steps = R.acts[i][0], list(script[i])
        k = placed_by.get(ref.get(i))
        if i in to and k is not None and k != i and to.get(k) and to[k] != to[i]:  # next to an object placed elsewhere
            pick = max(n for n, (t, x) in enumerate(steps) if t == "Pick")
            steps = steps[:pick + 1] + [(t, x.replace(to[i], to[k])) for t, x in steps[pick + 1:]]
            a, to[i] = type(a)(**{**a.__dict__, "to": to[k]}), to[k]  # later references follow this one
        if hasattr(a, "to"):
            nt = ref.get(i) or (a.next_to if a.next_to and a.next_to != a.to else None)
            spec[i] = {"object": a.obj, "rel": "in" if a.preposition == "within" else "on", "to": a.to, "ref": nt}
            if nt:  # place next to the reference on this subgoal's own receptacle
                steps = [(t, f"{a.obj}, {a.preposition}, {a.to}, next_to, {nt}" if t == "Place" else x) for t, x in steps]
            labels[i] = f"put the {words(a.obj)} {a.preposition} the {words(a.to)}" + (f" next to the {words(nt)}" if nt else "")
            objects[i] = sorted({a.obj, a.to} | ({nt} if nt else set()))
        else:
            labels[i], objects[i] = f"{a.attribute} the {words(a.obj)}", [a.obj]
        scr[i] = steps
    edges = set()
    for c in ep.evaluation_constraints:
        if isinstance(c, TemporalConstraint):
            edges |= {(home[int(u)], home[int(v)]) for u, v in c.dag.edges
                      if int(u) in home and int(v) in home and home[int(u)] != home[int(v)]}
    key = lambda i: f"p{i}"
    return {"title": ep.instruction, "nodes": [key(i) for i in nodes], "index": {key(i): i for i in nodes},
            "labels": [labels[i] for i in nodes], "objects": {key(i): objects[i] for i in nodes},
            "object": {key(i): obj_of.get(i) for i in nodes},
            "props": {key(i): sorted(j for j in home if home[j] == i) for i in nodes},
            "spec": {key(i): spec.get(i) for i in nodes},  # the predicate: object rel to (next to ref)
            "scripts": {key(i): scr[i] for i in nodes}, "edges": [[key(u), key(v)] for u, v in sorted(edges)],
            # ground truth for analysis only, never shown to the robot: every object's starting furniture, the recipe
            "inventory": {name(h): name(r.split("|")[0]) for h, r in ep.name_to_receptacle.items()},
            "recipe": (ep.info or {}).get("recipe")}


def drop(task: dict, gone: set) -> dict:
    """The task without the subgoals in `gone` (already satisfied when the episode starts)."""
    keep = [i for i, v in enumerate(task["nodes"]) if v not in gone]
    nodes = [task["nodes"][i] for i in keep]
    return task | {"nodes": nodes, "labels": [task["labels"][i] for i in keep],
                   "objects": {v: task["objects"][v] for v in nodes}, "scripts": {v: task["scripts"][v] for v in nodes},
                   "object": {v: task["object"][v] for v in nodes}, "props": {v: task["props"][v] for v in nodes},
                   "spec": {v: task["spec"][v] for v in nodes},
                   "edges": [e for e in task["edges"] if e[0] in nodes and e[1] in nodes]}
