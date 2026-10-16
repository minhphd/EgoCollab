"""Synthesized eye gaze for the person's head camera, calibrated on HD-EPIC and independent of S1's model.

Settings are the gen_* and gaze fields of helper.bip.config.Config. Each frame comes with the visible entities as panoptic masks {name: (cx, cy, area)} in [0, 1] image
coordinates. Gaze dwells on one target for a log-normal time (median dwell_s), then picks the next:
    - priming: before a pick (place), with probability p_prime_pick (p_prime_place) gaze goes to the object
      (receptacle) a log-normal lead ahead (HD-EPIC: 65% of pick-ups primed, median lead 3.1 s; 32% of
      put-downs, 1.8 s)
    - otherwise the current subgoal's entities with w_cur, the next subgoal's with w_next, else any visible one
The point is the mask centre plus jitter. A fixation is reported once gaze has held one entity for fix_min s.
"""

from __future__ import annotations

import numpy as np

from helper.bip.config import Config


class Gaze:
    def __init__(self, cfg: Config, seed: int):
        self.cfg, self.rng = cfg, np.random.default_rng(seed)
        self.target, self.until, self.held_since, self.reported = None, -1.0, 0.0, False
        self.primes: list[tuple[float, float, str]] = []  # (from, to, entity) windows of primed gaze

    def plan(self, t_event: float, entity: str, kind: str) -> None:
        """Register an upcoming pick or place so gaze may prime it ahead of time."""
        c = self.cfg
        p, lead = (c.p_prime_pick, c.lead_pick) if kind == "pick" else (c.p_prime_place, c.lead_place)
        if self.rng.random() < p:
            self.primes.append((t_event - lead * np.exp(c.lead_sigma * self.rng.standard_normal()), t_event, entity))

    def _choose(self, t: float, masks: dict, cur: set, nxt: set) -> str | None:
        primed = [e for a, b, e in self.primes if a <= t < b and e in masks]
        if primed:
            return primed[0]
        r = self.rng.random()
        for pool, w in ((cur, self.cfg.gen_w_cur), (nxt, self.cfg.gen_w_next)):
            vis = sorted(pool & masks.keys())
            if r < w and vis:
                return vis[self.rng.integers(len(vis))]
            r -= w
        names = sorted(masks)
        return names[self.rng.integers(len(names))] if names else None

    def step(self, t: float, masks: dict, cur: set, nxt: set) -> tuple[list[float], dict | None]:
        """Gaze point [x, y] at t and a fixation {t, object, visible} when one is completed now."""
        c = self.cfg
        primed_now = any(a <= t < b and e in masks and e != self.target for a, b, e in self.primes)
        if t >= self.until or self.target not in masks or primed_now:
            new = self._choose(t, masks, cur, nxt)
            if new != self.target:
                self.target, self.held_since, self.reported = new, t, False
            self.until = t + c.gen_dwell_s * np.exp(c.gen_dwell_sigma * self.rng.standard_normal())
        if self.target in masks:
            cx, cy, _ = masks[self.target]
            xy = [cx + c.gaze_jitter * self.rng.standard_normal(), cy + c.gaze_jitter * self.rng.standard_normal()]
        else:
            xy = [0.5 + c.center_bias * self.rng.standard_normal(), 0.45 + c.center_bias * self.rng.standard_normal()]
        xy = [float(np.clip(v, 0, 1)) for v in xy]
        fix = None
        if self.target is not None and not self.reported and t - self.held_since >= c.fix_min:
            fix, self.reported = {"t": round(t, 2), "object": self.target, "visible": sorted(masks)}, True
        return xy, fix
