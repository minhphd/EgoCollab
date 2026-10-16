"""The robot's policies on SkillEnv: each step it takes is a pick of its object then a place on its target.

    oracle   knows the task's orderings and the person's step (info, privileged): an open, ready step the person is not
             on, the farthest from theirs
    bip      BIP (helper/collab/bip.py) on the observation events only; speech: none, announce (delegation), ask
             (clarification: declined if the person is on it), tell (correction by value, when it picks an object up
             and when it puts it in place)

    agent = make("bip+tell", env, seed); obs, info = env.reset(seed=seed); agent.reset(obs, info)
    while ...: obs, r, term, trunc, info = env.step(agent.act(obs, info))
"""

from __future__ import annotations

import numpy as np

from . import bip


class Executor:
    """Claim a step, pick its object, [open the container,] place it, [close it,] release the claim; a failed step waits
    15 s before another try; an object it cannot place goes back onto the furniture it came off. An object that moves
    10 cm before the robot has it was taken: the robot stops and leaves the step 15 s."""

    def __init__(self, env, seed: int):
        self.env, self.task, self.rng = env, env.unwrapped.task, np.random.default_rng([seed, 4])
        self.cur, self.phase, self.log, self.segments, self.since, self.source = None, None, [], [], 0.0, None
        self.blocked = {v: np.inf for v, ok in enumerate(env.unwrapped.robot_can) if not ok}  # too big for the gripper

    def reset(self, obs: dict, info: dict) -> None:
        pass

    def start(self, t: float, v: int) -> dict:
        s, w = self.task["steps"][v], self.env.unwrapped.w
        self.cur, self.phase, self.since, self.at = v, "pick", t, w.pos(s["obj"])
        self.source = w.support(self.at, skip=(s["obj"],))
        self.log.append(f"{t:.1f} start {v}")
        return {"skill": "pick", "object": s["obj"], "step": v}

    def follow(self, t: float, info: dict) -> dict:
        """The next skill of the current step from the last one's status."""
        status, s, w = info["skill"], self.task["steps"][self.cur], self.env.unwrapped.w
        if self.phase == "pick" and w.held["robot"] != s["obj"] and float(np.linalg.norm(np.subtract(w.pos(s["obj"]), self.at))) > 0.1:
            self.blocked[self.cur] = t + 15.0  # someone took it
            self.end(t, "taken")
            return {"skill": "stop", "step": None}
        if status == "running":
            return {"skill": "noop", "step": self.cur}
        if status.startswith("failed"):
            self.log.append(f"{t:.1f} {self.phase} {status}")
            held = self.env.unwrapped.w.held["robot"] == s["obj"]
            if self.phase == "pick" and held and not s.get("closed"):  # it has it, though drawing in failed: on with it
                self.phase = "place"
                return {"skill": "place", "object": s["obj"], "rel": s["rel"], "target": s["target"], "step": self.cur}
            self.blocked[self.cur] = t + 15.0
            if self.phase == "place" and self.source is not None:  # in hand: back where it came from
                self.phase = "return"
                return {"skill": "place", "object": s["obj"], "rel": "ontop", "target": self.source, "step": self.cur}
            return self.end(t, "failed")
        nxt = {"pick": "open" if s.get("closed") else "place", "open": "place",
               "place": "close" if s.get("closed") else None, "close": None, "return": None}[self.phase]
        if nxt is None:
            return self.end(t, "returned" if self.phase == "return" else "done")
        self.phase = nxt
        if nxt in ("open", "close"):  # a container with a door
            return {"skill": nxt, "object": s["target"], "step": self.cur}
        return {"skill": "place", "object": s["obj"], "rel": s["rel"], "target": s["target"], "step": self.cur}

    def end(self, t: float, why: str) -> dict:
        self.log.append(f"{t:.1f} {why} {self.cur}")
        self.segments.append([round(self.since, 2), round(t, 2), self.cur])
        self.cur, self.phase = None, None
        return {"skill": "noop", "step": None}


class Oracle(Executor):
    def act(self, obs: dict, info: dict) -> dict:
        t = info["t"]
        if self.cur is not None:
            return self.follow(t, info)
        done, theirs, T = info["done"], info["person_step"], self.task
        if theirs is None and t < 1.0:  # the person has not chosen yet
            return {"skill": "noop", "step": None}
        ready = [v for v in range(len(T["steps"])) if v not in done and v != theirs and self.blocked.get(v, -1) < t
                 and all(u in done for u, x in T["orderings"] if x == v)]
        if not ready:
            return {"skill": "noop", "step": None}
        if theirs is None:
            return self.start(t, ready[self.rng.integers(len(ready))])
        here = self.env.unwrapped.w.pos(T["steps"][theirs]["obj"])[:2]
        return self.start(t, max(ready, key=lambda v: float(np.linalg.norm(self.env.unwrapped.w.pos(T["steps"][v]["obj"])[:2] - here))))


class Bip(Executor):
    def __init__(self, env, seed: int, speech: str | None = None, P: bip.Params | None = None):
        super().__init__(env, seed)
        self.speech, self.P = speech, P or bip.Params(seed=seed)
        self.next_decide, self.dwell, self.told = 1.0, {}, set()  # first after a second of watching, as the oracle
        self.claimed = set()  # steps the robot told the person it is on

    def reset(self, obs: dict, info: dict) -> None:
        T = self.task
        self.bel = bip.Belief({"title": T["title"], "nodes": T["nodes"], "labels": T["labels"], "objects": T["step_objects"],
                               "closed": [bool(s.get("closed")) for s in T["steps"]]}, self.P)

    def observe(self, t: float, events: list) -> None:
        for e in events:
            if e["kind"] == "clip":
                self.bel.clip(e["t"], e["verbs"], e["object"], e["receptacle"], e["visible"])
            elif e["kind"] == "fixation":
                self.bel.fixation(e["t"], e["object"], e["visible"])
                for o in self.dwell:  # gaze on a moved object (or on the robot carrying it): it may have been seen
                    if e["object"] == o or (e["object"] == "robot" and self.phase == "place"):
                        self.dwell[o] += 0.6
            elif e["kind"] == "robot_done":
                self.bel.robot_done(e["t"], e["step"])
                self.tell(t, e["step"], "put")
            elif e["kind"] == "speech" and e["text"].startswith("No") and e["step"] == self.cur:
                self.blocked[self.cur] = t + 15.0  # they are on it: leave it
                self.decline = True

    def tell(self, t: float, v: int, kind: str):
        """Correction by value: tell when P(missed) P(they look for it soon) 20 s > 3 s, P(missed) = exp(-dwell / 0.4)."""
        if self.speech != "tell" or (v, kind) in self.told:
            return
        o = self.task["steps"][v]["obj"]
        p_missed = float(np.exp(-self.dwell.get(o, 0.0) / 0.4))
        p_seek = 1.0 if kind == "put" else float(min(1.0, self.bel.current()[v] + self.bel.next()[v]))
        if bip.tell_value(p_missed, p_seek, self.P) > 0:
            self.told.add((v, kind))
            self.say = (kind, v)

    def act(self, obs: dict, info: dict) -> dict:
        t = info["t"]
        self.say, self.decline = None, False
        self.observe(t, obs["events"])
        if self.cur is not None and self.speech and self.cur not in self.claimed and self.bel.current()[self.cur] > 0.5:
            self.claimed.add(self.cur)  # the person is going for the robot's step: say the robot is on it
            self.say = ("claim", self.cur)
        if self.cur is not None:
            if self.decline:
                a = self.end(t, "declined")
            else:
                phase = self.phase
                a = self.follow(t, info)
                if phase == "pick" and self.phase in ("open", "place"):  # just picked it up
                    self.dwell[self.task["steps"][self.cur]["obj"]] = 0.0
                    self.tell(t, self.cur, "have")
        elif t >= self.next_decide:
            self.next_decide = t + 1.0
            self.bel.claim(t, set())
            v = bip.decide(self.bel, t, blocked=[u for u, until in self.blocked.items() if until >= t])
            a = {"skill": "noop", "step": None}
            if v is not None:
                a = self.start(t, v)
                self.bel.claim(t, {v})
                if self.speech in ("announce", "ask"):
                    self.say = (self.speech, v)
        else:
            a = {"skill": "noop", "step": None}
        if self.say:
            a["say"] = self.say
        return a


def make(policy: str, env, seed: int):
    base, _, speech = policy.partition("+")
    if base == "alone":
        return None
    return Oracle(env, seed) if base == "oracle" else Bip(env, seed, speech or None)
