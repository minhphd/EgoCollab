"""The robot's decision each second: execute a subgoal, ask the person, or wait, from S1's belief.

    V(execute v) = P_exec(v) [(1 - Y_v) D(v) - beta Y_v] - beta_prec (1 - P_exec(v))
    V(ask)       = E_answer[best V after the answer] - kappa                     V(wait) = 0
    V(tell o)    = P(false belief about o) P(the person looks for o soon) search - talk     (tell_value)
P_exec(v)  probability that every predecessor of v is done, over graphs and particles (exec_prob)
Y_v        probability the person works on v while the robot would: v is their current subgoal, or their
           planned next one and they switch to it within `notice` s (after that they see the robot's claim)
D(v)       the person's time saved: v's duration, less any wait for a slower robot once the person runs out of
           work that does not depend on v (saved)
beta, beta_prec, kappa: the costs of a conflict, an ordering violation and a question (Costs). The robot
executes the best v when its value is positive and waits otherwise; blocked subgoals are never executed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .s1 import Filter


@dataclass
class Costs:
    beta: float = 60.0  # s lost per conflict
    beta_prec: float = 120.0  # s lost per precedence violation
    kappa: float = 5.0  # s-equivalent cost of interrupting with a question
    eps_answer: float = 0.05
    robot_slowdown: float = 1.5  # robot time per unit of person time
    notice: float = float("inf")  # s until the person notices the robot's claim and works around it
    search: float = 20.0  # s a person loses looking for an object where it no longer is
    talk: float = 3.0  # s of the person's time to hear where it is


def exec_prob(F: Filter) -> np.ndarray:
    """P(every required predecessor of v is complete), over graphs and each particle's completed set."""
    Dp = F.D | F.C
    ready = (F.anc[:, None, :] & ~Dp[..., None]) == 0  # [K, M, N]
    return np.einsum("km,kmn->n", F.joint(), ready)


def overlap(F: Filter, J: np.ndarray, remaining: np.ndarray, d_robot: np.ndarray) -> np.ndarray:
    """Y[v] under particle weights J: the person is on v, or moves to v before the robot would finish it."""
    g, n = F.g, F.n
    soon = np.where(g >= 0, remaining[np.maximum(g, 0)], 0.0)
    Y = np.zeros(F.N)
    for v in range(F.N):
        Y[v] = (J * ((g == v) | ((n == v) & (soon < d_robot[v])))).sum()
    return Y


def saved(F: Filter, d: np.ndarray, remaining: np.ndarray, d_robot: np.ndarray) -> np.ndarray:
    E = F.edge_probs()  # E[v, u] = P(u needs v)
    claimed = ((F.claimed >> np.arange(F.N)) & 1).astype(bool)
    on = F.marginal(F.g)  # the person's current subgoal has `remaining` left; any other one needs all of d
    work = (on * np.minimum(d, remaining) + (1 - on) * d) * (1 - F.done_probs()) * ~claimed
    free = np.array([((1 - E[v]) * work).sum() - work[v] for v in range(F.N)])
    return d - np.maximum(0.0, d_robot - free)


def act_values(F: Filter, J: np.ndarray, P_exec: np.ndarray, d: np.ndarray, remaining: np.ndarray,
               costs: Costs, anticipate: bool = True) -> np.ndarray:
    d_robot = d * costs.robot_slowdown
    D = saved(F, d, remaining, d_robot)
    # a person who sees the robot's claim within `notice` s picks another subgoal: only switches before then collide
    Y = overlap(F, J, remaining, np.minimum(d_robot, costs.notice)) if anticipate else np.array([(J * (F.g == v)).sum() for v in range(F.N)])
    started = np.array([(J * ((F.S >> v) & 1)).sum() for v in range(F.N)])
    V = P_exec * ((1 - Y) * D - costs.beta * Y) - costs.beta_prec * (1 - P_exec)
    blocked = (((F.C | F.claimed) >> np.arange(F.N)) & 1).astype(bool) | (F.done_probs() > 0.5)
    return np.where(blocked | (started > 0.5), -np.inf, V)


def decide(F: Filter, d: np.ndarray, remaining: np.ndarray, costs: Costs, ask: str = "voi",
           anticipate: bool = True, entropy: float = 1.0, blocked: np.ndarray | None = None) -> tuple[str, int | list[int] | None, dict]:
    """("execute", v) | ("ask", choices) | ("wait", None), with the values behind the choice; blocked subgoals
    (e.g. whose object the robot has never located) are never executed."""
    J, P_exec = F.joint(), exec_prob(F)
    V = act_values(F, J, P_exec, d, remaining, costs, anticipate)
    if blocked is not None:
        V = np.where(blocked, -np.inf, V)
    best = float(np.max(V, initial=0.0))
    info = {"value": best}
    if ask != "never":
        nm = F.marginal(F.n)
        choices = [int(v) for v in np.argsort(-nm)[:3] if nm[v] > 0]
        if choices and ask == "voi":
            gain = 0.0
            for a in choices + [None]:
                hit = (F.n == a) if a is not None else ~np.isin(F.n, choices)
                L = np.where(hit, 1 - costs.eps_answer, costs.eps_answer / len(choices))
                p = (J * L).sum()
                Ja = J * L / max(p, 1e-12)
                gain += p * max(0.0, float(np.max(act_values(F, Ja, P_exec, d, remaining, costs, anticipate), initial=0.0)))
            info["voi"] = gain - best
            if gain - costs.kappa > best:
                return "ask", choices, info
        elif choices and ask == "entropy":
            p = nm[nm > 0] / nm.sum()
            info["entropy"] = float(-(p * np.log(p)).sum())
            if info["entropy"] > entropy:
                return "ask", choices, info
    if best > 0:
        return "execute", int(np.argmax(V)), info
    return "wait", None, info


def tell_value(p_false: float, p_seek: float, costs: Costs) -> float:
    """V(tell the person where o is): the search a false belief about o would waste if they look for o soon, less
    the time to hear it. Tell when positive."""
    return p_false * p_seek * costs.search - costs.talk
