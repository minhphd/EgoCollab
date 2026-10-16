"""Robot policies for the closed loop. All share the executor (claim a subgoal, run its skill script with the
robot's oracle skills, release it when done or failed) and differ in how they choose a subgoal:
    idle    never acts
    bip     S1 belief from the person's ego stream (helper/bip) and the act-or-wait rule (decide.py)
    graph   the belief-free control: a random subgoal that is ready under S2's most likely graph and not
            observed done, ignoring what the person does
    oracle  the true graph and the person's true current subgoal: a random allowed subgoal they are not on
    bipv2   subgoals not given: the task family's recipes as candidate structures (helper/bip/structures.py),
            weighted by Gemini's prior times the evidence; the decision adds the cost of a wrong subgoal
    talker  not told the task: the human model says what it needs to know and how much that is worth; the slow
            VLM asks, and asks again after every answer while a need beats the interruption (helper/bip/dialogue.py)
A category subgoal ("a plate") has no fixed script: the executor picks the nearest free instance when it starts.
"""

from __future__ import annotations

import re

import numpy as np

from helper.bip import observe
from helper.bip.config import Config
from helper.bip.decide import Costs, decide, exec_prob, overlap, saved
from helper.bip.belief import Belief
from helper.bip.dialogue import (OPENING, OPENING_ALL, aspects, compose, destinations, needs, propose,
                                  vlm_decide, vlm_question)
from helper.bip.structures import StructureBelief
from helper.bip.track import Tracker

from .person import from_hand, instance_script, recover


class Robot:
    name = "idle"

    def __init__(self, cfg: Config, seed: int = 0):
        self.cfg, self.rng = cfg, np.random.default_rng(seed)
        self.claims, self.log, self.segments, self.done = {}, [], [], set()
        self.cur, self.queue, self.skill, self.seg_start, self.skill_end, self.next_decide = None, [], None, 0.0, -1.0, 0.0
        self.failed, self.fails, self.tries = 0, {}, {}
        self.skills_done = []  # (t, tool, arg) of every successful skill
        self.target, self.avoid = None, set()  # the instance it goes for; the instance the person goes for

    def start(self, task: dict, sid: str, person) -> None:
        self.task, self.sid, self.person = task, sid, person
        self.index = {v: i for i, v in enumerate(task["nodes"])}
        self.furniture = sorted(f for f in task.get("furniture", []) if not f.startswith("floor_"))
        self.wander, self.visited = None, set()

    def observe(self, t: float, seen: dict) -> None:
        pass

    def explore(self, t: float, sim):
        """With nothing worth doing, walk to furniture not yet visited so the robot's camera maps the objects."""
        if self.wander is None:
            todo = [f for f in self.furniture if f not in self.visited] or list(self.furniture)
            self.visited &= set(todo)
            self.wander = todo[self.rng.integers(len(todo))]
        vec, resp = sim.skill(0, "Navigate", self.wander)
        if resp:
            self.visited.add(self.wander)
            self.log.append(f"{t:.1f} looked at {self.wander}")
            self.wander = None
        return vec

    def choose(self, t: float, seen: dict) -> str | None:
        return None

    def claimed(self, t: float) -> None:
        pass

    def known_done(self, seen: dict) -> set:
        """Subgoals the robot knows are done (this executor reads the simulator; Bip overrides it)."""
        return seen["done"]

    def spec_for(self, v: str) -> dict:
        return self.task["spec"][v]

    def ready(self, t: float, sim) -> bool:
        return True

    def script_for(self, v: str, sim) -> list | None:
        """The skill script for v: the task's, or for a category subgoal one built now around a free instance."""
        script = self.task["scripts"].get(v)
        if script is not None:
            self.target = self.task["object"].get(v)
            return from_hand(script, sim.held(0))
        here = np.array(sim.sim.agents_mgr[0].articulated_agent.base_pos)
        self.target, script = instance_script(sim, self.spec_for(v), here, sim.held(0), self.avoid)
        return script if self.target is not None else None

    def _end(self, t: float, why: str) -> None:
        self.segments.append([round(self.seg_start, 2), round(t, 2), self.cur, self.target])
        self.log.append(f"{t:.1f} {why} {self.cur}")
        self.claims.pop(self.cur, None)
        self.cur, self.queue, self.skill = None, [], None
        self.claimed(t)

    def act(self, t: float, sim, seen: dict):
        self.observe(t, seen)
        busy = self.skill is not None and self.skill[0] != "Navigate"
        running = self.skill is not None or self.queue  # a finished script is credited below, not here
        if self.cur is not None and running and self.cur in self.known_done(seen) and sim.held(0) is None and not busy:
            self._end(t, "seen done")
        if self.cur is None:
            seen_done = getattr(self, "person_t", None)  # the scene tier knows the person's completions from its graph
            last = max([0.0] + (seen_done if seen_done is not None else [c["t"] for c in seen["completions"] if c["node"] not in self.done]))
            if t >= self.next_decide and t >= last + self.cfg.look_s:  # let the person's next choice show first
                self.next_decide = t + self.cfg.decide_every
                v = self.choose(t, seen)
                script = self.script_for(v, sim) if v is not None and self.fails.get(v, 0) < self.cfg.max_fails else None
                if v is not None and script is None:  # no free instance: try again later
                    self.fails[v] = self.fails.get(v, 0) + 1
                if script is not None:
                    if self.wander is not None:  # stop looking around
                        self.wander = None
                        sim.agents[0].reset()
                    self.cur, self.queue, self.seg_start = v, script, t
                    self.claims[v] = t
                    self.log.append(f"{t:.1f} start {v}")
                    self.claimed(t)
            if self.cur is None:
                return self.explore(t, sim) if self.cfg.explore else None
        if self.skill is None:
            if not self.queue:  # the script ended with a successful placement
                self.done.add(self.cur)
                self._end(t, "done")
                return None
            if not self.ready(t, sim):  # a subclass may hold the next skill (the talker asks where first)
                return None
            self.skill = self.queue.pop(0)
        vec, resp = sim.skill(0, *self.skill)
        if resp:
            ok = resp.startswith("Successful")
            if ok:
                self.skills_done.append((t, self.skill[0], self.skill[1]))
            self.log.append(f"{t:.1f} {self.skill[0]}[{self.skill[1]}] {'ok' if ok else resp[:60]}")
            failed, self.skill, self.skill_end = self.skill, None, t
            if not ok and not recover(failed, self.queue, self.tries):
                self.fails[self.cur] = self.fails.get(self.cur, 0) + 1
                self.failed += 1
                self._end(t, "failed")
        return vec


class Bip(Robot):
    name = "bip"

    def start(self, task: dict, sid: str, person) -> None:
        super().start(task, sid, person)
        self.st = {"id": sid, "title": task["title"], "nodes": task["nodes"], "labels": task["labels"],
                   "objects": task["objects"], "truth": {"edges": task["edges"]}, "frames": None, "frame_fps": 1.0}
        self.B = Belief(self.cfg, self.st)
        self.ptr = {"actions": 0, "fixations": 0, "held": 0, "completions": 0}
        self.O = [set(task["objects"][v]) for v in task["nodes"]]
        self.pending_det, self.read, self.read_t, self.belief, self.decisions = [], None, 0.0, [], []
        self.B.feed(0.0, "action", np.ones(len(task["nodes"]) + 1))  # the person starts some allowed subgoal now
        if self.cfg.action_tier == "scene":  # the scene graph from both views and both agents' actions
            self.T = Tracker(self.cfg, self.st | {"spec": task["spec"], "furniture": task["furniture"], "map": task["map"], "t0": 0.0})
            self.ptr |= {"views": 0, "acts": 0}
            self.done_t, self.person_t = [], []  # times the graph showed a subgoal done (by the person)

    def observe(self, t: float, seen: dict) -> None:
        cfg, ev = self.cfg, []
        self.st["frames"] = seen["frames"]
        new = lambda k: seen[k][self.ptr[k]:]
        if cfg.action_tier == "scene":
            obs = [(v["t"], 0, v) for v in new("views")] + [(a["t"], 1, a) for a in new("acts")]
            for tt, _, x in sorted(obs, key=lambda e: (e[0], e[1])):
                out = self.T.view(tt, x["triples"]) if "triples" in x else self.T.act(tt, x["actor"], tuple(x["action"]))
                self.done_t += [e[0] for e in out if e[2] in ("completion", "robot_done")]
                self.person_t += [e[0] for e in out if e[2] == "completion"]
                ev += [e[:4] for e in out]
        elif cfg.action_tier == "ego":
            for h in new("held"):
                x = observe.read(h["object"], h["visible"], cfg.hand_acc, cfg.hand_miss, self.rng) if h["object"] else None
                if x:
                    ev.append((h["t"], 1, "action", observe.likelihood(cfg, [x], self.O)))
        else:
            for a in new("actions"):
                ents = list(a["entities"])
                if cfg.action_tier == "noisy" and ents and self.rng.random() > cfg.action_acc:
                    others = [x for x in a["visible"] if x not in ents]
                    if others:
                        ents[0] = others[self.rng.integers(len(others))]
                ev.append((a["t"], 1, "action", observe.likelihood(cfg, ents, self.O)))
        if cfg.gaze:
            ev += [observe.fixation(cfg, f, self.O, self.rng)[:4] for f in new("fixations")]
        for c in new("completions") if cfg.action_tier != "scene" else []:  # the scene tier reads completions off the graph
            i = self.index[c["node"]]
            if c["node"] == self.cur or c["node"] in self.done:
                ev.append((c["t"], 0, "robot_done", i))
            elif cfg.completion == "oracle":  # the person moves on: an uninformative action right after
                ev += [(c["t"], 0, "completion", i), (c["t"], 0.5, "action", np.ones(len(self.O) + 1))]
            elif cfg.completion == "noisy" and self.rng.random() < cfg.det_rate:
                self.pending_det.append((c["t"] + self.rng.exponential(cfg.det_delay), i))
        ev += [e for tt, i in self.pending_det if tt <= t for e in ((tt, 0, "detected", i), (tt, 0.5, "action", np.ones(len(self.O) + 1)))]
        self.pending_det = [(tt, i) for tt, i in self.pending_det if tt > t]
        for k in self.ptr:
            self.ptr[k] = len(seen[k])
        for tt, _, kind, x in sorted(ev, key=lambda e: (e[0], e[1])):
            self.B.feed(tt, kind, x)
        F = self.B.F
        cur = int(np.argmax(F.marginal(F.g))) if F.marginal(F.g).sum() > 0 else None
        if cur != self.read:
            self.read, self.read_t = cur, t
        if ev:
            self.belief.append({"t": round(t, 2), "cur": F.marginal(F.g).round(3).tolist(), "next": F.marginal(F.n).round(3).tolist(),
                                "done": F.done_probs().round(3).tolist()})

    def durations(self, seen: dict) -> np.ndarray:
        times = sorted(self.done_t if self.cfg.action_tier == "scene" else (c["t"] for c in seen["completions"]))
        gaps = np.diff([0.0] + times)
        return np.full(len(self.task["nodes"]), float(np.mean(gaps)) if len(gaps) else 20.0)

    def choose(self, t: float, seen: dict) -> str | None:
        cfg, F = self.cfg, self.B.F
        d = self.durations(seen)
        x = np.arange(len(cfg.rem_curve)) * cfg.rem_step  # E[time left | time spent], both in units of d
        remaining = np.maximum(1.0, d * np.interp((t - self.read_t) / d.mean(), x, cfg.rem_curve))
        costs = Costs(cfg.beta_rel * d.mean(), cfg.beta_prec_rel * d.mean(), cfg.kappa, 0.05, cfg.robot_slowdown, cfg.notice_s)
        kind, x, info = decide(F, d, remaining, costs, "never", blocked=self.unknown())
        d_r = d * cfg.robot_slowdown  # the terms of V(execute v), for diagnosis
        terms = {"d": round(float(d.mean()), 2), "rem": round(float(remaining.mean()), 2), "P_exec": exec_prob(F).round(3).tolist(),
                 "Y": overlap(F, F.joint(), remaining, np.minimum(d_r, cfg.notice_s)).round(3).tolist(),
                 "Delta": saved(F, d, remaining, d_r).round(2).tolist()}
        self.decisions.append({"t": round(t, 2), "kind": kind, "x": x if kind == "execute" else None, "value": round(info["value"], 2)} | terms)
        return self.task["nodes"][x] if kind == "execute" else None

    def claimed(self, t: float) -> None:
        self.B.feed(t, "claimed", sum(1 << self.index[v] for v in self.claims))

    def known_done(self, seen: dict) -> set:
        """The scene tier knows only what its graph shows; the other tiers are oracle perception."""
        if self.cfg.action_tier != "scene":
            return seen["done"]
        return {v for i, v in enumerate(self.task["nodes"]) if self.B.F.C >> i & 1}

    def unknown(self) -> np.ndarray | None:
        """Subgoals whose object the robot has never located (scene tier): it cannot go and get them."""
        if self.cfg.action_tier != "scene":
            return None
        return np.array([not any(self.T.scene.known(o) for o in self.T.scene.candidates(self.task["spec"][v]))
                         for v in self.task["nodes"]])


class Graph(Bip):
    name = "graph"

    def choose(self, t: float, seen: dict) -> str | None:
        F = self.B.F
        ready, done, unknown = exec_prob(F), F.done_probs(), self.unknown()
        ok = [v for v, i in self.index.items() if ready[i] > 0.5 and done[i] < 0.5 and v not in self.claims
              and (unknown is None or not unknown[i])]
        return ok[self.rng.integers(len(ok))] if ok else None


class Oracle(Robot):
    name = "oracle"

    def choose(self, t: float, seen: dict) -> str | None:
        p = self.person
        ok = [v for v in p.allowed(t, seen["done"], {}) if v != p.cur and v not in self.claims]
        return ok[self.rng.integers(len(ok))] if ok else None


class BipV2(Robot):
    """BIP with the subgoals not given (helper/bip/structures.py): the family's recipes are the candidate
    structures, with Gemini's prior; keys are category subgoals ("plate1") shared across structures."""
    name = "bipv2"

    def start(self, task: dict, sid: str, person) -> None:
        super().start(task, sid, person)
        from .families import hypotheses, recipe_prior
        self.S = hypotheses(task)
        self.prior = recipe_prior(self.cfg, task, sid) if self.cfg.recipe_prior == "llm" else {n: 1 / len(self.S) for n in self.S}
        self.B = StructureBelief(self.cfg, self.S, self.prior, self.cfg.seed)
        self.ptr = {"views": 0, "acts": 0, "fixations": 0}
        self.read, self.read_t, self.belief, self.decisions = None, 0.0, [], []

    @property
    def person_t(self) -> list:
        return self.B.person_t[self.B.lead()] if self.B is not None else []

    def spec_for(self, v: str) -> dict:
        lead = self.B.lead()
        return self.S[lead if v in self.S[lead]["spec"] else next(n for n in self.S if v in self.S[n]["spec"])]["spec"][v]

    def known_done(self, seen: dict) -> set:
        if self.B is None:
            return set()
        lead = self.B.lead()
        return {v for v, i in self.B.idx[lead].items() if self.B.F[lead].C >> i & 1}

    def observe(self, t: float, seen: dict) -> None:
        new = {k: seen[k][self.ptr[k]:] for k in self.ptr}
        for k in self.ptr:
            self.ptr[k] = len(seen[k])
        if any(new.values()):
            self.observe_new(t, new)

    def observe_new(self, t: float, new: dict) -> None:
        self.B.observe(new["views"], new["acts"], new["fixations"] if self.cfg.gaze else [])
        lead = self.B.lead()
        F = self.B.F[lead]
        read = (lead, int(np.argmax(F.marginal(F.g))))
        if read != self.read:
            self.read, self.read_t = read, t
        self.belief.append({"t": round(t, 2), "w": {n: round(p, 3) for n, p in self.B.weights().items()}, "keys": self.B.classes()})

    def choose(self, t: float, seen: dict) -> str | None:
        cfg, lead = self.cfg, self.B.lead()
        gaps = np.diff([0.0] + sorted(self.B.done_t[lead]))
        d = float(np.mean(gaps)) if len(gaps) else 20.0
        rem = max(1.0, d * float(np.interp((t - self.read_t) / d, np.arange(len(cfg.rem_curve)) * cfg.rem_step, cfg.rem_curve)))
        costs = Costs(cfg.beta_rel * d, cfg.beta_prec_rel * d, cfg.kappa, 0.05, cfg.robot_slowdown, cfg.notice_s)
        unknown = lambda k: not any(self.B.T[lead].scene.known(o) for o in self.spec_for(k)["objects"])
        V = self.B.values(d, rem, costs, cfg.beta_wrong_rel * d, unknown)
        best = max(V, key=V.get) if V else None
        go = best is not None and V[best] > 0
        self.decisions.append({"t": round(t, 2), "kind": "execute" if go else "wait", "x": best if go else None,
                               "value": round(float(V[best]), 2) if best else 0.0,
                               "w": {n: round(p, 3) for n, p in self.B.weights().items()}})
        return best if go else None

    def claimed(self, t: float) -> None:
        self.B.claim(set(self.claims))


class Talker(BipV2):
    """Not told the task. It opens with a question; after every answer S2 turns the conversation into up to 3
    plans, the candidate structures, replayed on everything observed so far. Then, and at every decision, the
    human model finds where the plans disagree in a way that changes the robot's next decision; while that is
    worth more than cfg.ask_cost and patience lasts, S2 asks about it (helper/bip/dialogue.py)."""
    name = "talker"

    def start(self, task: dict, sid: str, person) -> None:
        Robot.start(self, task, sid, person)
        objs = {o: f for o, rel, f in task.get("map", []) if rel in ("on", "in")}  # what the robot mapped
        self.home = {"objects": objs, "rooms": task.get("rooms", {}), "near": task.get("near", {}),
                     "furniture": task["furniture"], "map": task["map"]}
        self.S, self.prior, self.B, self.questions, self.seen = {}, {}, None, 0, {}  # seen: object -> where the person last put it
        self.asked, self.progress = {}, 0  # (kind, class) -> subgoals done when last asked; subgoals done now
        self.consulted, self.declined = -1e9, None  # vlm_num: when S2 was last asked to decide; what it declined
        self.answered, self.settled = 0, 0  # answers to the robot's questions; those that settled what was asked
        self.ptr = {"views": 0, "acts": 0, "fixations": 0}
        self.history = {"views": [], "acts": [], "fixations": []}
        self.read, self.read_t, self.belief, self.decisions = None, 0.0, [], []
        self.dialogue, self.question, self.last_ask, self.why = [], None, -1e9, None
        start = self.cfg.talk_start
        if start == "instruction":  # told the instruction, as in PARTNR; asks only what the human model needs
            self.hear(0.0, task["say"])
        else:
            self.question, self.why = (OPENING_ALL if start == "ask_all" else OPENING), {"need": "the task"}

    def say(self, t: float, text: str) -> None:
        self.dialogue.append([round(t, 2), "Robot", text] + ([self.why] if self.why else []))
        self.questions += (self.why or {}).get("need") != "the task"
        self.last_ask, self.why = t, None

    def spec_for(self, v: str) -> dict:
        """The spec of the plan that sends v where it most likely goes."""
        dest = destinations(self.B, v)
        if not dest:  # no plan has it any more: the last spec it ran with
            return self.task["spec"].get(v) or next(iter(self.S.values()))["spec"].get(v)
        to = max(dest, key=dest.get)
        w = self.B.weights()
        n = max((n for n in self.B.names if v in self.B.idx[n] and self.B.st[n]["spec"][v]["to"] == to), key=w.get)
        return self.B.st[n]["spec"][v]

    def ready(self, t: float, sim) -> bool:
        """Before carrying an object to an uncertain destination, ask where it goes, when a wrong placement would
        cost more than the question ((1 + beta_wrong) x P(wrong) > ask_cost) and patience lasts."""
        if self.question is not None:
            return False
        q = self.queue
        if self.B is None or self.cur not in self.B.keys or len(q) < 2 or q[0][0] != "Navigate" or q[1][0] != "Place":
            return True
        dest = destinations(self.B, self.cur)
        d = self.terms(t)[0]
        value = (self.cfg.beta_wrong_rel + 1) * d * (1 - max(dest.values())) * self.p_know()  # a wrong placement saves nothing and costs beta_wrong
        bar = self.cfg.ask_floor if self.cfg.talk_policy == "vlm_num" else self.cfg.ask_cost
        if self.cfg.talk_policy not in ("model", "vlm_num") or self.questions >= self.cfg.max_questions or value <= bar:
            return True
        part = next((a for a in aspects(self.B) if a[0] == "where" and a[1] == re.sub(r"\d+$", "", self.cur)), None)
        if part is None or self.asked.get(part[:2]) == self.progress:
            return True
        self.asked[part[:2]] = self.progress
        if self.cfg.talk_policy == "vlm_num":
            r = vlm_decide(self.cfg, self.dialogue, self.home, self.B, [(value, part)], self.situation(carrying=True))
            if r is None:
                return True
            self.question = r[0]
        else:
            self.question = compose(self.cfg, self.dialogue, self.home, self.B, part)
        self.why = {"need": "where", "class": part[1], "carrying": True}
        self.decisions.append({"t": round(t, 2), "kind": "ask", "x": f"where {part[1]}", "value": 0.0})
        return False

    def retarget(self) -> None:
        """After the plans change, send the object being carried where it now most likely goes: the Place and the
        Navigate right before it."""
        if self.cur is None or self.B is None or self.cur not in self.B.keys:
            return
        to = self.spec_for(self.cur)["to"]
        k = next((i for i, (tool, _) in enumerate(self.queue) if tool == "Place"), None)
        if k is None:
            return
        parts = [x.strip() for x in self.queue[k][1].split(",")]
        parts[2] = to
        self.queue[k] = ("Place", ", ".join(parts))
        if k > 0 and self.queue[k - 1][0] == "Navigate":
            self.queue[k - 1] = ("Navigate", to)

    def terms(self, t: float) -> tuple:
        """d, rem and the costs of the decision rule, from the leading plan (as BipV2.choose)."""
        cfg, lead = self.cfg, self.B.lead()
        gaps = np.diff([0.0] + sorted(self.B.done_t[lead]))
        d = float(np.mean(gaps)) if len(gaps) else 20.0
        rem = max(1.0, d * float(np.interp((t - self.read_t) / d, np.arange(len(cfg.rem_curve)) * cfg.rem_step, cfg.rem_curve)))
        costs = Costs(cfg.beta_rel * d, cfg.beta_prec_rel * d, cfg.kappa, 0.05, cfg.robot_slowdown, cfg.notice_s)
        unknown = lambda k: not any(self.B.T[lead].scene.known(o) for o in self.spec_for(k)["objects"])
        return d, rem, costs, unknown

    def wonder(self, t: float) -> bool:
        """Ask about the most valuable need, if it beats the interruption and patience lasts."""
        if self.B is None or self.questions >= self.cfg.max_questions:
            return False
        if self.cfg.talk_policy == "vlm":  # baseline: S2 decides alone whether and what to ask
            q = vlm_question(self.cfg, self.dialogue, self.home, self.B)
            if q is None:
                return False
            self.question, self.why = q, {"need": "vlm"}
            self.decisions.append({"t": round(t, 2), "kind": "ask", "x": "vlm", "value": 0.0})
            return True
        d, rem, costs, unknown = self.terms(t)
        skip = {a for a, p in self.asked.items() if p == self.progress}  # the same again only after progress
        more = self.more(t, d, rem, costs, unknown, skip)
        pk = self.p_know()
        ranked = sorted([(v * pk, a) for v, a in needs(self.B, d, rem, costs, self.cfg, unknown, skip) + more], key=lambda x: -x[0])
        if self.cfg.talk_policy == "vlm_num":  # S2 decides, from the human model's values
            key = (self.progress, tuple(a[:2] for _, a in ranked[:4]))
            if not ranked or ranked[0][0] < self.cfg.ask_floor or t - self.consulted < self.cfg.ask_cooldown or key == self.declined:
                return False
            self.consulted = t
            r = vlm_decide(self.cfg, self.dialogue, self.home, self.B, ranked, self.situation())
            if r is None:
                self.declined = key
                return False
            value, aspect = next((v, a) for v, a in ranked if a is r[1])
            self.asked[aspect[:2]] = self.progress
            self.question = r[0]
            self.why = {"need": aspect[0], "class": aspect[1], "value": round(float(value), 2), "by": "vlm"}
            self.decisions.append({"t": round(t, 2), "kind": "ask", "x": f"{aspect[0]} {aspect[1]}", "value": round(float(value), 2)})
            return True
        if not ranked or ranked[0][0] < self.cfg.ask_cost:
            return False
        value, aspect = ranked[0]
        self.asked[aspect[:2]] = self.progress
        self.question = compose(self.cfg, self.dialogue, self.home, self.B, aspect)
        self.why = {"need": aspect[0], "class": aspect[1], "value": round(float(value), 2)}
        self.decisions.append({"t": round(t, 2), "kind": "ask", "x": f"{aspect[0]} {aspect[1]}", "value": round(float(value), 2)})
        return True

    def more(self, t: float, d: float, rem: float, costs, unknown, skip) -> list:
        """When no plan has anything left for the robot while the person is still at work (an action in the last
        30 s), learning what else the task needs is worth about one subgoal of the person's time, d."""
        acts = [a["t"] for a in self.history["acts"] if a["actor"] == "H"]
        if ("more", "") in skip or not acts or t - acts[-1] > 30 or self.B.values(d, rem, costs, self.cfg.beta_wrong_rel * d, unknown):
            return []
        return [(d, ("more", "", {}))]

    def p_know(self) -> float:
        """The chance the person can answer a question now: Beta(1, 1) on how many answers settled what was asked."""
        return (1 + self.settled) / (2 + self.answered) if self.cfg.know_model else 1.0

    def situation(self, carrying: bool = False) -> str:
        """What S2 is told about the moment it decides in."""
        return (f"The person has completed {self.progress} steps so far."
                + (f" Their answers settled {self.settled} of your {self.answered} questions." if self.answered else "")
                + (" You are about to carry an object to where you think it goes." if carrying else ""))

    def hear(self, t: float, text: str) -> None:
        asked = self.dialogue[-1][3] if self.dialogue and self.dialogue[-1][1] == "Robot" and len(self.dialogue[-1]) > 3 else {}
        self.dialogue.append([round(t, 2), "Person", text])
        self.replan(t, follow=False)
        if self.B is not None and asked.get("need") in ("where", "exists", "count", "next to", "more"):
            self.answered += 1
            if asked["need"] == "more":  # settled if the robot now has something to do
                d, rem, costs, unknown = self.terms(t)
                self.settled += bool(self.B.values(d, rem, costs, self.cfg.beta_wrong_rel * d, unknown))
            else:
                self.settled += (asked["need"], asked["class"]) not in {a[:2] for a in aspects(self.B)}
        if self.B is not None and self.cur is None:
            self.wonder(t)  # a follow-up right away while the answer left a need worth asking about

    def replan(self, t: float, follow: bool = True) -> None:
        """S2 turns the conversation and the placements seen into plans; S1 replays them on everything observed."""
        S, prior = propose(self.cfg, self.dialogue, self.home, list(self.seen.items()))
        if not S:
            self.log.append(f"{t:.1f} no plan from the conversation")
            return
        self.S, self.prior = S, prior
        self.B = StructureBelief(self.cfg, S, prior, self.cfg.seed)
        self.B.observe(self.history["views"], self.history["acts"], self.history["fixations"] if self.cfg.gaze else [])
        self.B.claim(set(self.claims))
        show = lambda sp: "|".join(sp["tos"]) + (f"~{sp['refs'][0].rsplit('_', 1)[0]}" if sp["refs"] else "")
        self.log.append(f"{t:.1f} plans " + "; ".join(f"{n} {p:.2f}: " + ", ".join(f"{v}>{show(S[n]['spec'][v])}" for v in S[n]["nodes"])
                                                    + f" edges {len(S[n]['edges'])}" for n, p in prior.items()))
        self.retarget()
        if follow and self.cur is None:
            self.wonder(t)

    def observe(self, t: float, seen: dict) -> None:
        new = {k: seen[k][self.ptr[k]:] for k in self.ptr}
        for k in self.ptr:
            self.ptr[k] = len(seen[k])
            self.history[k] += new[k]
        if self.B is None or not any(new.values()):
            return
        super().observe_new(t, new)
        placed = [(a["action"][1], a["action"][3]) for a in new["acts"] if a["actor"] == "H" and a["action"][0] == "Place"]
        self.seen.update(placed)
        if any(not self.explains(o, f) for o, f in placed):  # a placement no plan predicts: S2 re-plans
            self.log.append(f"{t:.1f} unexplained " + ", ".join(f"{o}>{f}" for o, f in placed))
            self.replan(t)

    def explains(self, obj: str, f: str) -> bool:
        """Whether some plan sends an object of this class to this furniture."""
        return any(re.sub(r"_\d+$", "", obj) == re.sub(r"_\d+$", "", sp["objects"][0]) and f in sp["tos"]
                   for n in self.B.names for sp in self.B.st[n]["spec"].values())

    def choose(self, t: float, seen: dict) -> str | None:
        self.progress = len(seen.get("completions", []))
        if self.B is None:
            return None
        if t - self.last_ask >= self.cfg.ask_cooldown and self.wonder(t):
            return None
        return super().choose(t, seen)


def make_robot(cfg: Config, sim) -> Robot:
    kinds = {"idle": Robot, "bip": Bip, "graph": Graph, "oracle": Oracle, "bipv2": BipV2, "talker": Talker}
    if cfg.robot not in kinds:
        raise SystemExit(f"unknown robot: {cfg.robot}")
    return kinds[cfg.robot](cfg, cfg.seed + 17)
