"""The robot's side of the conversation. The human model decides what to ask; the slow VLM (S2) asks it.

    propose(cfg, dialogue, home, seen) -> (structures, prior)  S2 reads the conversation, the premapped home and
                                                          what the person has been seen placing, and gives up to 3
                                                          plans, each a category-level structure
    need(B, d, rem, costs, cfg) -> (value, aspect) | None where the plans disagree (a class's presence, its count,
                                                          where it goes) with the highest value of knowing the answer:
                                                          the expected best decision value over the plans each answer
                                                          leaves, minus the best now; None unless it beats ask_cost
    compose(cfg, dialogue, home, B, aspect) -> question   S2 turns that need into one natural question, describing
                                                          furniture by room and what is on it, never by ID
The robot asks again after every answer while a need is worth more than the interruption (patience:
cfg.max_questions). A plan's subgoals are named "<class><rank>" ("cup1"); where each goes is a property of the plan.
"""

from __future__ import annotations

import json
import re

import numpy as np

from .config import Config

OPENING = "Hi! What are we doing today, and how can I help?"
OPENING_ALL = "Hi! Please tell me every step of the task: each object, where it goes, and anything that has to come first."
kind = lambda x: re.sub(r"_\d+$", "", x)
words = lambda x: kind(x).replace("_", " ")


def propose(cfg: Config, dialogue: list, home: dict, seen: list = ()) -> tuple[dict, dict]:
    """The 3 most likely plans that fit the conversation and where the person has put things (("clock_0", "chest_of_drawers_62"), ...),
    as structures for StructureBelief, with their probabilities."""
    from helper.llm.gemini import Gemini, generate
    objs, furn = home["objects"], home["rooms"]
    classes, fnames = sorted({kind(o) for o in objs}), sorted(furn)
    text = ("A robot helps a person with a household task. The robot has mapped the home.\n"
            f"Furniture (name: where it is): {'; '.join(f'{f}: {describe(f, home)}' for f in furn)}.\n"
            f"Objects (name on furniture): {'; '.join(f'{o} on {f}' for o, f in objs.items())}.\n"
            "The conversation so far:\n" + "\n".join(f"{x[1]}: {x[2]}" for x in dialogue) + "\n"
            + (f"Where the person has put things so far: {'; '.join(f'{o} on {f}' for o, f in seen)}. Most are part of "
               "the task; people sometimes set an object down on the way and move it again later.\n" if seen else "")
            + "Give the 3 most likely plans for the task given the conversation, most likely first, with honest "
              "probabilities (a short request leaves much open) that sum "
            "to 1. A plan lists the whole task, including what is already done: per object class, how many, where they must end up (one furniture "
            "name; a class split between furniture is listed once per furniture), and the class it must end next to, if any; "
            "and the required "
            "orderings between classes (the first "
            "class must be in place before the second is moved). Use only the classes and furniture listed. "
            "Make the plans differ where the conversation leaves the task open: when several furniture fit what was "
            "said, put each in a different plan instead of guessing one; likewise for how many and what goes next to "
            "what. Do not make plans that differ only in the order of steps.")
    schema = {"type": "object", "required": ["plans"], "properties": {"plans": {"type": "array", "minItems": 3, "maxItems": 3, "items": {
        "type": "object", "required": ["p", "subgoals", "orderings"], "properties": {
            "p": {"type": "number"},
            "subgoals": {"type": "array", "items": {"type": "object", "required": ["cls", "count", "to", "next_to"], "properties": {
                "cls": {"type": "string", "enum": classes}, "count": {"type": "integer"},
                "to": {"type": "string", "enum": fnames},
                "next_to": {"type": "string", "enum": classes + ["none"]}}}},
            "orderings": {"type": "array", "items": {"type": "object", "required": ["before", "after"], "properties": {
                "before": {"type": "string", "enum": classes}, "after": {"type": "string", "enum": classes}}}}}}}}}
    reply, usage, latency = generate(text, schema=schema, cfg=Gemini(model=cfg.prior_model or cfg.model, thinking=cfg.s2_thinking))
    try:
        plans = json.loads(reply)["plans"]
    except (ValueError, KeyError, TypeError):
        return {}, {}
    structures, prior = {}, {}
    for i, plan in enumerate(plans):
        st = structure(plan, objs, home)
        if st["nodes"]:
            structures[f"plan{i}"], prior[f"plan{i}"] = st, max(1e-3, float(plan.get("p", 0)))
    z = sum(prior.values()) or 1.0
    return structures, {n: p / z for n, p in prior.items()}


def structure(plan: dict, objs: dict, home: dict) -> dict:
    """One plan as a category-level structure; orderings hold for every rank (every plate before any fork)."""
    nodes, spec, objects, labels, keys = [], {}, {}, [], {}
    for sg in plan["subgoals"]:
        cands = sorted(o for o in objs if kind(o) == sg["cls"])
        tos = [sg["to"]] if isinstance(sg["to"], str) else sorted(set(sg["to"]))[:1]
        if not cands or not tos:
            continue
        for j in range(1, min(max(1, int(sg["count"])), len(cands)) + 1):
            v = f"{sg['cls']}{len(keys.get(sg['cls'], [])) + 1}"  # what, not where; numbered on across repeats of a class
            nodes.append(v)
            keys.setdefault(sg["cls"], []).append(v)
            refs = sorted(o for o in objs if kind(o) == sg.get("next_to")) if sg.get("next_to") not in (None, "none") else []
            spec[v] = {"objects": cands, "rank": j, "rel": "on", "to": tos[0], "tos": tos, "ref": None, "refs": refs}
            objects[v] = cands + tos + refs
            labels.append(f"put {'a' if j == 1 else 'another'} {words(sg['cls'])} on the {words(tos[0])}")
    after = {}  # class -> classes ordered after it; an ordering that would close a loop is skipped
    reach = lambda a, b: a == b or any(reach(c, b) for c in after.get(a, ()))
    for o in plan.get("orderings", []):
        if o["before"] in keys and o["after"] in keys and not reach(o["after"], o["before"]):
            after.setdefault(o["before"], set()).add(o["after"])
    edges = [[u, v] for a, bs in after.items() for b in bs for u in keys[a] for v in keys[b]]
    return {"nodes": nodes, "spec": spec, "objects": objects, "labels": labels, "edges": edges,
            "furniture": home["furniture"], "map": home["map"], "t0": 0.0}


def cls_of(key: str) -> str:
    return re.sub(r"\d+$", "", key)


def destinations(B, key: str) -> dict:
    """Where subgoal `key` goes: furniture -> probability, over the plans that contain it."""
    w, out = B.weights(), {}
    for n in B.names:
        if key in B.idx[n]:
            f = B.st[n]["spec"][key]["to"]
            out[f] = out.get(f, 0.0) + w[n]
    z = sum(out.values()) or 1.0
    return {f: p / z for f, p in out.items()}


def aspects(B) -> list[tuple]:
    """Where the candidate plans disagree, each with the plans every answer leaves: (kind, class, {answer: plans})."""
    out = []
    for c in sorted({cls_of(k) for k in B.keys}):
        part = {"exists": {}, "count": {}, "where": {}, "next to": {}}
        for n in B.names:
            nodes = [v for v in B.st[n]["nodes"] if cls_of(v) == c]
            part["exists"].setdefault(bool(nodes), []).append(n)
            part["count"].setdefault(len(nodes), []).append(n)
            if nodes:
                sp = B.st[n]["spec"][nodes[0]]
                part["where"].setdefault(sp["to"], []).append(n)
                part["next to"].setdefault(kind(sp["refs"][0]) if sp.get("refs") else "nothing", []).append(n)
        out += [(kind, c, p) for kind, p in part.items() if len(p) > 1]
    return out


def needs(B, d: float, rem: float, costs, cfg: Config, blocked=None, skip=()) -> list[tuple[float, tuple]]:
    """Every disagreement worth something, best first, with its value of information in seconds. A plan's value is
    what the robot could do over the rest of the task: the sum of max(0, V(execute v)) over its subgoals.
    skip: (kind, class) pairs not to ask about now."""
    w = B.weights()
    best_value = lambda ws=None: sum(max(0.0, v) for v in B.values(d, rem, costs, cfg.beta_wrong_rel * d, blocked, w=ws).values())
    now, out = best_value(), []
    for kind, c, part in aspects(B):
        if (kind, c) in skip:
            continue
        after = 0.0
        for names in part.values():
            pw = sum(w[n] for n in names)
            if pw > 1e-6:
                after += pw * best_value({n: (w[n] if n in names else 0.0) for n in B.names})
        if after - now > 1e-6:
            out.append((after - now, (kind, c, part)))
    return sorted(out, key=lambda x: -x[0])


def need(B, d: float, rem: float, costs, cfg: Config, blocked=None, skip=()) -> tuple[float, tuple] | None:
    """The disagreement most worth settling, if it beats cfg.ask_cost."""
    n = needs(B, d, rem, costs, cfg, blocked, skip)
    return n[0] if n and n[0][0] >= cfg.ask_cost else None


def vlm_decide(cfg: Config, dialogue: list, home: dict, B, ranked: list, person: str) -> tuple[str, tuple] | None:
    """S2 decides whether to ask, with the human model's objective: what each open question is worth in seconds of
    the robot's help, against what asking costs the person. Returns (question, aspect) or None."""
    from helper.llm.gemini import Gemini, generate
    opts = "\n".join(f"{i + 1}. {topic(a, B, home)}: worth about {v:.0f} s" for i, (v, a) in enumerate(ranked[:4]))
    text = ("You are a robot helping a person with a household task. The conversation so far:\n"
            + "\n".join(f"{x[1]}: {x[2]}" for x in dialogue) + f"\n{person}\n"
            f"Your model of the person lists what is still open about the task and how many seconds of the task "
            f"knowing each would save, through the help you could then give:\n{opts}\n"
            f"Asking interrupts the person and costs them about {cfg.ask_cost:.0f} s. They work the task out as they "
            "go, so they may not know yet, and asking again what they could not answer is wasted.\n"
            "Decide whether to ask now. If yes, reply with the number of the topic, a colon, and one short question "
            "(refer to furniture and objects as a person would, never by an ID). If not, reply NONE.")
    reply, usage, latency = generate(text, cfg=Gemini(model=cfg.prior_model or cfg.model, thinking=cfg.s2_thinking))
    m = re.match(r"\s*(\d+)\s*[:.)-]\s*(.+)", reply.strip().split("\n")[0])
    if not m or not 1 <= int(m.group(1)) <= min(4, len(ranked)):
        return None
    return m.group(2).strip().strip('"'), ranked[int(m.group(1)) - 1][1]


def vlm_question(cfg: Config, dialogue: list, home: dict, B) -> str | None:
    """Baseline: S2 decides by itself whether to ask, from the conversation and its plans, with no objective from
    the human model. Returns a question or None."""
    from helper.llm.gemini import Gemini, generate
    w = B.weights()
    plans = "\n".join(f"Plan {k + 1} ({w[n]:.2f}): " + "; ".join(f"{v} to {describe(B.st[n]['spec'][v]['to'], home)}" for v in B.st[n]["nodes"])
                       for k, n in enumerate(sorted(B.names, key=w.get, reverse=True)))
    text = ("You are a robot helping a person with a household task; asking interrupts them. The conversation so far:\n"
            + "\n".join(f"{x[1]}: {x[2]}" for x in dialogue) + f"\nYour candidate plans:\n{plans}\n"
            "If you need more information from the person to help well, write the one short question you would ask "
            "(refer to furniture and objects as a person would, never by an ID). Otherwise write NONE.")
    reply, usage, latency = generate(text, cfg=Gemini(model=cfg.prior_model or cfg.model, thinking=cfg.s2_thinking))
    q = reply.strip().split("\n")[0].strip('"')
    return None if not q or q.upper().startswith("NONE") else q


def describe(f: str, home: dict) -> str:
    """Furniture the way a person would place it: room, neighbours, what is on it."""
    on = sorted(words(o) for o, g in home["objects"].items() if g == f)
    room = words(home["rooms"].get(f, "")) or "an unknown room"
    near = [words(x) for x in home.get("near", {}).get(f, [])]
    return (f"{words(f)} in the {room}" + (f", next to the {' and the '.join(near)}" if near else "")
            + (f", with {', '.join(on[:4])} on it" if on else ""))


def topic(aspect: tuple, B, home: dict) -> str:
    """An open question about the task in words, with how likely the robot thinks each answer is."""
    kind, c, part = aspect
    if kind == "more":
        return "what else the task needs: your plans have nothing left that you could do, but the person is still working"
    w = B.weights()
    opts = {a: (x if isinstance(x, float) else sum(w[n] for n in x)) for a, x in part.items()}
    if kind == "where":
        return f"where the {words(c)} should go. Candidates, with how likely you think each is: " + "; ".join(
            f"{describe(f, home)} ({p:.2f})" for f, p in sorted(opts.items(), key=lambda kv: -kv[1]))
    if kind == "next to":
        return f"what the {words(c)} should end up next to; you think: " + "; ".join(f"{words(r)} ({p:.2f})" for r, p in opts.items())
    if kind == "count":
        return f"how many {words(c)}s the task needs; you think: " + "; ".join(f"{n} ({p:.2f})" for n, p in opts.items())
    return f"whether the task includes the {words(c)} at all (you think: {opts.get(True, 0):.2f})"


def compose(cfg: Config, dialogue: list, home: dict, B, aspect: tuple) -> str:
    """One short natural question, written by S2, that settles the need."""
    from helper.llm.gemini import Gemini, generate
    what = topic(aspect, B, home)
    text = ("You are a robot helping a person with a household task. The conversation so far:\n"
            + "\n".join(f"{x[1]}: {x[2]}" for x in dialogue) + "\n"
            f"To decide what to do next you need to know {what}.\n"
            "Write the one short question you would ask the person to settle this. Refer to furniture and objects the "
            "way a person would (by room, by what is on it), never by an ID or number. Reply with the question only.")
    reply, usage, latency = generate(text, cfg=Gemini(model=cfg.prior_model or cfg.model, thinking=cfg.s2_thinking))
    return reply.strip().split("\n")[0].strip('"')
