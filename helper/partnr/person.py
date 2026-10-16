"""The scripted person (humanoid agent 1): one subgoal at a time, through its skill script.

A subgoal is allowed when it is not done, every subgoal it requires (true orderings) is done, and the person
has not yielded it to the robot. Which allowed subgoal comes next, and how the person behaves, depends on
cfg.person:
    uniform  a random allowed subgoal; detours (p_detour) and pauses (p_pause); always yields to the robot
    hitl     calibrated on PARTNR's human traces (hitl.py): P(v) prop. to exp(-choice_rank * instruction rank
             - choice_dist * distance), so mostly the nearest object; an idle pause before each subgoal of
             idle_ratio times the last one's work; a first placement goes to the wrong furniture with
             p_misplace and is fixed later; yields a subgoal the robot has taken with p_yield
A failed placement is retried once, plainly; a subgoal is given up after max_fails failures.
"""

from __future__ import annotations

import numpy as np

from helper.bip.config import Config

from .perceive import relation, sim_position


def recover(failed: tuple, queue: list, tries: dict) -> bool:
    """After a failed Place, walk to the receptacle again and place plainly (once per object); True if queued."""
    if failed[0] != "Place":
        return False
    obj, prep, rec = [x.strip() for x in failed[1].split(",")[:3]]
    if tries.get(obj, 0) >= 1:
        return False
    tries[obj] = tries.get(obj, 0) + 1
    queue[:0] = [("Navigate", rec), ("Place", f"{obj}, {prep}, {rec}, none, none")]
    return True


def from_hand(script: list, held: str | None) -> list:
    """The script from after the Pick of the held object, when restarting with it in hand."""
    picks = [k for k, (tool, arg) in enumerate(script) if tool == "Pick" and arg == held]
    return script[picks[0] + 1:] if picks else list(script)


def openable(sim, f: str) -> bool:
    """Whether furniture f has doors or drawers to open (a cabinet can be open shelving)."""
    try:
        return bool(sim.env.full_world_graph.get_node_from_name(f).is_articulated())
    except Exception:
        return False


def instance_script(sim, spec: dict, here: np.ndarray, held: str | None, avoid: set) -> tuple[str | None, list]:
    """For a category subgoal: the instance to use (the held candidate, else the nearest free one that nobody
    else is going for) and its script, opening its container first. (None, []) when no candidate is free."""
    tos = spec.get("tos") or [spec["to"]]
    if held in spec["objects"]:
        o, steps = held, []
    else:
        free = [(float(np.linalg.norm((p - here)[[0, 2]])), o) for o in spec["objects"] if o not in avoid
                and (r := relation(sim, o)) is not None and r[1] not in tos and (p := sim_position(sim, o)) is not None]
        if not free:
            return None, []
        o = min(free)[1]
        rel, f = relation(sim, o)
        steps = ([("Navigate", f), ("Open", f)] if rel == "in" and openable(sim, f) else []) + [("Navigate", o), ("Pick", o)]
    ref = next((r for r in spec.get("refs") or [] if (x := relation(sim, r)) and x[1] in tos), None)
    near = f"next_to, {ref}" if ref else "none, none"
    return o, steps + [("Navigate", tos[0]), ("Place", f"{o}, on, {tos[0]}, {near}")]


class Person:
    def __init__(self, cfg: Config, task: dict, furniture: list[str], seed: int):
        self.cfg, self.task = cfg, task
        self.furniture = sorted(f for f in furniture if not f.startswith("floor_"))
        self.rng = np.random.default_rng(seed)
        self.req = {v: {u for u, w in task["edges"] if w == v} for v in task["nodes"]}
        self.cur, self.queue, self.skill, self.wait_until = None, [], None, -1.0
        self.fails = {v: 0 for v in task["nodes"]}
        self.achieved = {}  # subgoal -> time its script ended with a successful placement
        self.tries = {}  # Place retries per object
        self.skills_done = []  # (t, tool, arg) of every successful skill
        self.segments, self.actions, self.log = [], [], []  # [start, end, node], skill reads, text
        self.seg_start, self.skill_start, self.skill_end, self.last_read = None, None, -1.0, -1.0
        self.misplaced, self.mistake, self.yields, self.last_work = set(), False, {}, 15.0
        self.target, self.taken = None, set()  # the instance the person goes for; instances the robot goes for
        self.order = self.discovery(np.random.default_rng([seed, 11]))

    def discovery(self, rng) -> list[str]:
        """The order the person works the steps out in: a random order that respects the task's orderings."""
        order, left = [], list(self.task["nodes"])
        while left:
            ready = [v for v in left if self.req[v] <= set(order)] or left
            order.append(ready[rng.integers(len(ready))])
            left.remove(order[-1])
        return order

    def known(self, done: set, busy: set = frozenset()) -> list[str]:
        """The open steps the person has worked out: the next person_horizon of the discovery order, or all. Steps
        the robot is seen doing (busy) are left to it, and the person works out the next one instead."""
        rest = [v for v in self.order if v not in done and v not in busy and self.fails[v] < self.cfg.max_fails]
        h = self.cfg.person_horizon
        k = int(np.ceil(h * len(self.task["nodes"]))) if 0 < h < 1 else int(h)
        return rest[:k] if k else rest

    def allowed(self, t: float, done: set, robot: dict) -> list[str]:
        for v, when in list(self.achieved.items()):  # placed, yet the placement does not hold: a failure
            if v not in done and v != self.cur and t - when > 1.0:
                self.fails[v] += 1
                del self.achieved[v]
        busy = {v for v, since in robot.items() if self.yielded(t, v, since)}
        known = set(self.known(done, busy))
        return [v for v in self.task["nodes"] if v not in done and v not in busy and self.req[v] <= done
                and self.fails[v] < self.cfg.max_fails and v in known]

    def yielded(self, t: float, v: str, since: float) -> bool:
        """Whether the person has noticed the robot on v and gives it up (once decided per claim)."""
        if t - since < self.cfg.notice_s:
            return False
        if (v, since) not in self.yields:
            self.yields[(v, since)] = self.cfg.person != "hitl" or self.rng.random() < self.cfg.p_yield
        return self.yields[(v, since)]

    def choose(self, options: list[str], sim) -> str:
        """The next subgoal: random (uniform), or mostly the nearest object, a little by instruction order (hitl).
        Of interchangeable category subgoals (plate1, plate2), only the lowest open rank is a choice."""
        low = {}
        for v in options:
            sp = self.task["spec"].get(v) or {}
            if "rank" in sp:
                c = tuple(sp["objects"])
                low[c] = v if c not in low or sp["rank"] < self.task["spec"][low[c]]["rank"] else low[c]
        options = [v for v in options if "rank" not in (self.task["spec"].get(v) or {}) or v in low.values()]
        if self.cfg.person != "hitl" or len(options) == 1:
            return options[self.rng.integers(len(options))]
        here = np.array(sim.sim.agents_mgr[1].articulated_agent.base_pos)
        score = []
        for v in options:
            sp = self.task["spec"].get(v) or {}
            o = self.task["object"][v] if "rank" not in sp else instance_script(sim, sp, here, None, self.taken)[0]
            there = sim_position(sim, o) if o is not None else None
            d = float(np.linalg.norm((there - here)[[0, 2]])) if there is not None else 5.0
            score.append(-self.cfg.choice_rank * self.task["nodes"].index(v) - self.cfg.choice_dist * d)
        p = np.exp(np.array(score) - max(score))
        return options[self.rng.choice(len(options), p=p / p.sum())]

    def nearest(self, sim) -> str | None:
        """The furniture nearest the person."""
        here = np.array(sim.sim.agents_mgr[1].articulated_agent.base_pos)
        near = [(float(np.linalg.norm((p - here)[[0, 2]])), f) for f in self.furniture if (p := sim_position(sim, f)) is not None]
        return min(near)[1] if near else None

    def misplace(self, sim) -> None:
        """Send this attempt's placement to the furniture nearest the right one (a mistake fixed later)."""
        place = [k for k, (tool, arg) in enumerate(self.queue) if tool == "Place"]
        if not place:
            return
        obj, prep, rec = [x.strip() for x in self.queue[place[-1]][1].split(",")[:3]]
        target = sim_position(sim, rec)
        near = [(float(np.linalg.norm((sim_position(sim, f) - target)[[0, 2]])), f) for f in self.furniture
                if f != rec and target is not None and sim_position(sim, f) is not None]
        if not near:
            return
        wrong = min(near)[1]
        self.queue = [(t_, x.replace(rec, wrong)) if k >= place[-1] - 1 else (t_, x) for k, (t_, x) in enumerate(self.queue)]
        self.mistake = True

    def _start(self, t: float, v: str, held: str | None, sim) -> None:
        script, self.target = self.task["scripts"][v], self.task["object"][v]
        if script is None:  # a category subgoal: choose the instance now
            here = np.array(sim.sim.agents_mgr[1].articulated_agent.base_pos)
            self.target, script = instance_script(sim, self.task["spec"][v], here, held, self.taken)
            if self.target is None:
                self.fails[v] += 1
                self.log.append(f"{t:.1f} no free instance for {v}")
                return
        self.cur, self.queue, self.seg_start, self.mistake = v, from_hand(script, held), t, False
        if held is not None and held == self.target:  # restarting with the object already in hand
            return
        if held is not None:  # holding another subgoal's object: put it down nearby first
            f = self.nearest(sim)
            if f is not None:
                self.queue = [("Navigate", f), ("Place", f"{held}, on, {f}, none, none")] + self.queue
        r = self.rng.random()
        if self.cfg.person == "hitl":  # idle before the subgoal, as people do in the traces
            self.wait_until = t + self.cfg.idle_ratio * self.last_work * float(np.exp(0.5 * self.rng.standard_normal()))
            if v not in self.misplaced and self.rng.random() < self.cfg.p_misplace:
                self.misplace(sim)
                self.misplaced.add(v)
        elif r < self.cfg.p_detour and self.furniture:
            self.queue.insert(0, ("Navigate", self.furniture[self.rng.integers(len(self.furniture))]))
        elif r < self.cfg.p_detour + self.cfg.p_pause:
            self.wait_until = t + self.cfg.pause_s * (0.5 + self.rng.random())
        self.log.append(f"{t:.1f} start {v}" + (" (will misplace)" if self.mistake else ""))

    def _end(self, t: float, why: str) -> None:
        if self.cur is not None:
            self.segments.append([round(self.seg_start, 2), round(t, 2), self.cur, self.target])
            self.last_work = max(3.0, t - max(self.seg_start, self.wait_until))
            self.log.append(f"{t:.1f} {why} {self.cur}")
        self.cur, self.queue, self.skill = None, [], None

    def act(self, t: float, sim, done: set, robot: dict, visible: list[str]):
        """Low-level action vector for this step (or None to stand still)."""
        taken = self.cur is not None and self.cur in robot and self.yielded(t, self.cur, robot[self.cur])
        busy = self.skill is not None and self.skill[0] != "Navigate"  # a Pick or Place finishes first
        if self.cur is not None and (self.cur in done or taken) and sim.held(1) is None and not busy:
            self._end(t, "done" if self.cur in done and not taken else "dropped")
        if self.cur is None:
            options = self.allowed(t, done, robot)
            if not options:
                return None
            self._start(t, self.choose(options, sim), sim.held(1), sim)
        if t < self.wait_until:
            return None
        if self.skill is None:
            if not self.queue:
                if self.mistake:  # placed on the wrong furniture: the subgoal stays open for later
                    self._end(t, "misplaced")
                    return None
                self.achieved[self.cur] = t  # the script ended with a successful placement
                self._end(t, "finished")
                return None
            self.skill, self.skill_start, self.last_read = self.queue.pop(0), t, -1.0
        tool, arg = self.skill
        if t - self.last_read >= self.cfg.action_every:  # what an observer of this skill could read now
            ents = [x.strip() for x in arg.split(",") if x.strip() and x.strip() not in ("on", "within", "none", "next_to")]
            self.actions.append({"t": round(t, 2), "verb": tool, "entities": ents, "visible": visible, "node": self.cur})
            self.last_read = t
        vec, resp = sim.skill(1, tool, arg)
        if resp:
            ok = resp.startswith("Successful")
            if ok:
                self.skills_done.append((t, self.skill[0], self.skill[1]))
            self.log.append(f"{t:.1f} {tool}[{arg}] {'ok' if ok else resp[:60]}")
            failed, self.skill, self.skill_end = self.skill, None, t
            if not ok and not recover(failed, self.queue, self.tries):
                self.fails[self.cur] += 1
                self._end(t, "failed")
        return vec

    def close(self, t: float) -> None:
        self._end(t, "stopped")
