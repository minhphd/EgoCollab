"""The evidence log P(o_1:t | H) of one structure, as a reward for training a proposer (BIP v2).

A structure is a stream's subgoals (nodes, spec, objects) with orderings (truth.edges, by node name). S1 replays
the stream's events with cfg.evidence_particles particles over cfg.evidence_seeds seeds. log Z-hat is biased
low by about half its variance across seeds, so the estimate is mean + var / 2.
    log_evidence(cfg, st)  ->  float
"""

from __future__ import annotations

import dataclasses

import numpy as np

from . import observe
from .config import Config
from .s1 import Filter


def replay(cfg: Config, st: dict, events: list, seed: int) -> float:
    idx = {v: i for i, v in enumerate(st["nodes"])}
    graph = frozenset((idx[u], idx[v]) for u, v in st["truth"]["edges"])
    F = Filter(dataclasses.replace(cfg, particles=cfg.evidence_particles), len(idx), [graph], [0.0], seed)
    step = {"action": F.action, "fixation": lambda x: F.fixation(*x), "completion": F.completion,
            "detected": F.detected, "robot_done": F.robot_done}
    for t, _, kind, x, info in events:
        step[kind](x)
    return float(F.logZ[0])


def log_evidence(cfg: Config, st: dict) -> float:
    events = observe.events(cfg, st)
    z = np.array([replay(cfg, st, events, seed) for seed in range(cfg.evidence_seeds)])
    return float(z.mean() + z.var() / 2)
