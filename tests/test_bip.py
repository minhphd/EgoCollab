"""helper/collab/bip.py on synthetic episodes: a person works through a task with known orderings, one subgoal at a
time: walk, pick its object, carry it, place it on its target (in a quarter of the subgoals the pick and the place
share one 3 s clip). Each clip the recognizer reports verb probabilities (the right verb read 60% of the time) and a
detector the object (right 70%); gaze follows the current and next subgoals. Gemini is replaced by canned replies.

    python -m pytest tests/test_bip.py -q
"""

import numpy as np
import pytest

from helper.collab import bip

SUB = 6
TRUE = {(0, 1), (2, 3)}  # subgoal 1 needs subgoal 0 done; subgoal 3 needs subgoal 2
TASK = {"title": "test task", "nodes": [f"v{i}" for i in range(SUB)], "labels": [f"put o{i} on f{i}" for i in range(SUB)],
        "objects": {f"v{i}": [f"o{i}", f"f{i}"] for i in range(SUB)}}
EVERY = [x for i in range(SUB) for x in (f"o{i}", f"f{i}")] + ["chair", "lamp"]
STEPS = {}  # subgoal -> (steps, undo) the canned step-list reply gives instead of pick -> place


def pick_place(v, obj=None):
    o = obj or f"o{v}"
    return [{"verb": "pick", "object": o, "optional": False}, {"verb": "place", "object": o, "target": f"f{v}", "optional": False}], \
        {"verb": "pick", "object": o}


@pytest.fixture(autouse=True)
def canned_gemini(monkeypatch, tmp_path):
    """Gemini's prior: the true orderings at 0.8, a false one at 0.3. Step lists: pick(o) -> place(o, f), except
    subgoal 5's, which names an object outside the vocabulary (so it falls back), and any set in STEPS. Revisions
    return nothing."""
    STEPS.clear()

    def reply(text, schema, P):
        s2, props = bip.S2(TASK, P), schema["properties"]
        if "edges" in props:
            edges = [(u, v, 0.8) for u, v in TRUE] + [(4, 5, 0.3)]
            return {"edges": [{"before": s2.id[u], "after": s2.id[v], "p": p} for u, v, p in edges]}, 0.0
        if "graphs" not in props:
            lists = {v: STEPS.get(v, pick_place(v, "spaceship" if v == 5 else None)) for v in range(SUB)}
            return {"subgoals": [{"id": s2.id[v], "steps": st, "undo": un} for v, (st, un) in lists.items()]}, 0.0
        return {"graphs": []}, 0.0
    monkeypatch.setattr(bip, "gemini", reply)
    monkeypatch.setattr(bip.Params, "cache", str(tmp_path))


def verbs(rng, label, acc=0.6):
    """The recognizer's probabilities: the right verb read with probability acc, else another; 0.6 on the one read."""
    read = label if rng.random() < acc else [c for c in bip.VERBS if c != label][rng.integers(4)]
    q = np.full(len(bip.VERBS), 0.1)
    q[bip.VERBS.index(read)] = 0.6
    return q


def episode(seed: int, order=(0, 2, 1, 4, 3, 5)):
    """(t, kind, args, true subgoal, true verb) events: two fixations before each clip."""
    rng, t, out = np.random.default_rng(seed), 0.0, []
    read = lambda x: None if x is None else (x if rng.random() < 0.7 else EVERY[rng.integers(len(EVERY))])
    for k, v in enumerate(order):
        nxt = order[k + 1] if k + 1 < len(order) else None
        o, f = f"o{v}", f"f{v}"
        clips = [("other", None, None), ("place", o, f)] if rng.random() < 0.25 else \
            [("other", None, None), ("pick", o, None), ("other", o, None), ("place", o, f)]
        for label, obj, rec in clips:
            for _ in range(2):
                r = rng.random()
                pool = [o, f] if r < 0.3 else [f"o{nxt}", f"f{nxt}"] if r < 0.4 and nxt is not None else EVERY
                out.append((t, "fixation", (pool[rng.integers(len(pool))], EVERY), v, None))
                t += 1.0
            out.append((t, "clip", (verbs(rng, label), read(obj), read(rec), EVERY), v, label))
            t += 1.0
    return out


def run(bel, events, until=None):
    for t, kind, args, v, label in events:
        getattr(bel, kind)(t, *args)
        if until and until(t, kind, v, label):
            break


def test_tracks_the_current_subgoal():
    """From each subgoal's second clip on, BIP's leading current subgoal is the true one 80% of the time or more; the
    detector alone names one of its objects 70% of the time, and nothing on walking clips."""
    hits = total = 0
    for seed in range(5):
        bel, first = bip.Belief(TASK, bip.Params(seed=seed)), set()
        for t, kind, args, v, label in episode(seed):
            getattr(bel, kind)(t, *args)
            if kind == "clip":
                if v in first:
                    hits += int(np.argmax(bel.current())) == v
                    total += 1
                first.add(v)
    assert hits / total >= 0.8, hits / total


def test_completion_from_evidence():
    """One clip into the person's 4th subgoal, the subgoals behind read as done and those ahead as not."""
    order, behind, ahead = (0, 2, 1, 4, 3, 5), [], []
    for seed in range(5):
        bel, seen = bip.Belief(TASK, bip.Params(seed=seed)), []
        run(bel, episode(seed, order), lambda t, kind, v, label: kind == "clip" and v == order[3] and (seen.append(1) or len(seen) == 1))
        done = bel.done()
        behind += [done[u] for u in order[:3]]
        ahead += [done[u] for u in order[4:]]
    # ahead: about 0.1 with delta 0.40 fitted on HD-EPIC (people start subgoals out of order), so the bound is 0.15
    assert np.mean(behind) > 0.7 and np.mean(ahead) < 0.15, (np.round(behind, 2), np.round(ahead, 2))


def test_never_takes_the_persons_subgoal_or_one_not_ready():
    bel, order, checked = bip.Belief(TASK, bip.Params(seed=2)), (0, 2, 1, 4, 3, 5), 0
    for t, kind, args, v, label in episode(2, order):
        getattr(bel, kind)(t, *args)
        if kind != "clip":
            continue
        x = bip.decide(bel, t)
        done = set(order[:order.index(v)])
        if x is not None:
            checked += 1
            assert not (x == v and bel.current()[v] > 0.8), "took the person's current subgoal"
            assert all(u in done or u == v for u, w in TRUE if w == x), f"took {x} before its predecessor"
    assert checked > 0


def test_robot_done_and_claims_shape_the_belief():
    bel = bip.Belief(TASK, bip.Params(seed=3))
    bel.robot_done(1.0, 0)
    bel.claim(1.0, {2})
    assert bel.done()[0] == 1.0
    assert bip.decide(bel, 2.0) not in (0, 2)


def test_prior_comes_from_gemini():
    E = bip.Belief(TASK, bip.Params(seed=4)).F.edge_probs()
    assert E[0, 1] > 0.5 and E[2, 3] > 0.5 and E[1, 0] < 0.05


def test_step_lists_come_from_gemini_and_are_checked():
    """Gemini's lists are used as given; one naming an object outside the vocabulary falls back to pick -> place."""
    STEPS[0] = ([{"verb": "open", "object": "f0", "optional": True}] + pick_place(0)[0], {"verb": "pick", "object": "o0"})
    R = bip.Belief(TASK, bip.Params(seed=5)).R
    assert [s["verb"] for s in R[0]["steps"]] == ["open", "pick", "place"] and R[0]["steps"][0]["optional"]
    assert [(s["verb"], s["obj"], s["target"]) for s in R[5]["steps"]] == [("pick", "o5", None), ("place", "o5", "f5")]


def strong(verb):
    q = np.full(len(bip.VERBS), 0.02)
    q[bip.VERBS.index(verb)] = 0.92
    return q


def test_pick_and_place_in_one_clip_complete_a_subgoal():
    """A clip read as place(o0, f0) right away: only a pick and a place in one clip explain it, and subgoal 0 is done."""
    bel = bip.Belief(TASK, bip.Params(seed=6))
    bel.clip(1.0, strong("place"), "o0", "f0", EVERY)
    assert bel.done()[0] > 0.8, bel.done()


def test_undo_takes_a_subgoal_back():
    bel = bip.Belief(TASK, bip.Params(seed=7, undo=0.05))
    for t, (verb, o, f) in enumerate([("pick", "o0", None), ("place", "o0", "f0"), ("other", None, None)]):
        bel.clip(float(t), strong(verb), o, f, EVERY)
    before = bel.done()[0]
    bel.clip(4.0, strong("pick"), "o0", None, EVERY)
    assert before > 0.8 and bel.done()[0] < before - 0.3, (before, bel.done()[0])


def test_optional_steps_may_happen_or_not():
    """With open and close optional around the place, the subgoal is done after pick and place, with or without them."""
    door = lambda verb: {"verb": verb, "object": "f0", "optional": True}
    STEPS[0] = ([pick_place(0)[0][0], door("open"), pick_place(0)[0][1], door("close")], pick_place(0)[1])
    closed = TASK  # the same title: the canned reply shuffles IDs by it
    for clips in ([("pick", "o0", None), ("place", "o0", "f0")],
                  [("pick", "o0", None), ("open", "f0", None), ("place", "o0", "f0"), ("close", "f0", None)]):
        bel = bip.Belief(closed, bip.Params(seed=8))
        assert [s["verb"] for s in bel.R[0]["steps"]] == ["pick", "open", "place", "close"]
        for t, (verb, o, f) in enumerate(clips):
            bel.clip(float(t), strong(verb), o, f, EVERY)
        assert bel.done()[0] > 0.8, (clips, bel.done())


def test_a_corrected_step_list_is_kept_only_if_it_explains_more():
    """Subgoal 0's list is wrong (open f0 -> close f0); the person picks o0 and places it on f0. The correction to
    pick -> place is adopted; a correction to close f0 alone is not."""
    STEPS[0] = ([{"verb": "open", "object": "f0", "optional": False}, {"verb": "close", "object": "f0", "optional": False}],
                {"verb": "open", "object": "f0"})
    bel = bip.Belief(TASK, bip.Params(seed=9))
    for t, (verb, o, f) in enumerate([("pick", "o0", None), ("other", "o0", None), ("place", "o0", "f0")]):
        bel.clip(float(t), strong(verb), o, f, EVERY)
    good = bip.checked({"steps": pick_place(0)[0], "undo": pick_place(0)[1]}, set(EVERY))
    bad = bip.checked({"steps": [{"verb": "close", "object": "f0", "optional": False}], "undo": None}, set(EVERY))
    bel._add([], {0: bad})
    assert [s["verb"] for s in bel.R[0]["steps"]] == ["open", "close"]
    bel._add([], {0: good})
    assert [s["verb"] for s in bel.R[0]["steps"]] == ["pick", "place"] and bel.done()[0] > 0.8


def test_tell_value():
    P = bip.Params()
    assert bip.tell_value(0.9, 1.0, P) > 0 > bip.tell_value(0.05, 1.0, P)


def test_sees_the_person_go_for_the_robots_subgoal():
    """The robot claimed subgoal 4; the person's clips on its object make 4 their current subgoal."""
    bel = bip.Belief(TASK, bip.Params(seed=0))
    bel.claim(0.5, {4})
    for t in (1.0, 4.0, 7.0):
        bel.clip(t, strong("pick"), "o4", None, EVERY)
    assert bel.current()[4] > 0.5


def test_fixations_accumulate_evidence_for_the_next_subgoal():
    """While the person is on subgoal 0, each fixation on subgoal 3's objects raises P(next = 3 | on 0) above its prior
    (3 waits for 2). Unconditionally, many such fixations rather say the person is already on 3 (gamma_cur > gamma_next)."""
    bel = bip.Belief(TASK, bip.Params(seed=10))
    bel.clip(1.0, strong("pick"), "o0", None, EVERY)
    F = bel.F
    on0 = lambda: (F.joint() * (F.g == 0) * F.b[..., 3]).sum() / (F.joint() * (F.g == 0)).sum()
    probs = [on0()]
    for t in range(6):
        bel.fixation(2.0 + t, "o3", EVERY)
        probs.append(on0())
    assert probs[-1] > 1.5 * probs[0] and all(b >= a for a, b in zip(probs, probs[1:])), np.round(probs, 3)
