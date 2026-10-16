"""What the robot's recognizers report of the person, in simulation: every `clip` s, a clip event with the verb
recognizer's probabilities, drawn from the real recognizer's calibrated outputs on held-out HD-EPIC clips of the same
true verb (helper/action_cls/stream.py's bank.npz, copied to BANK), and the detector's reads of the object and the
receptacle, each right with probability acc, else another object in view.

    rec = Recognizer(BANK, acc=0.7, rng=np.random.default_rng(0))
    event = rec.clip(t, clip, person.completed, held, visible)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

VERBS = ("pick", "place", "open", "close", "other")
BANK = Path(__file__).resolve().parents[2] / "data/collab/verb_bank.npz"


class Recognizer:
    def __init__(self, bank: str | Path | None, acc: float, rng, verb_acc: float = 0.6):
        """Without a bank, the verb read is right with probability verb_acc and gets 0.6, the rest 0.1 each."""
        self.acc, self.rng, self.verb_acc, self.rows = acc, rng, verb_acc, None
        if bank and Path(bank).exists():
            b = np.load(bank)
            self.p, self.rows = b["p"], [np.flatnonzero(b["y"] == c) for c in range(len(VERBS))]

    def verbs(self, label: str) -> np.ndarray:
        c = VERBS.index(label)
        if self.rows is not None and len(self.rows[c]):
            return self.p[self.rows[c][self.rng.integers(len(self.rows[c]))]]
        read = c if self.rng.random() < self.verb_acc else [x for x in range(len(VERBS)) if x != c][self.rng.integers(4)]
        q = np.full(len(VERBS), 0.1)
        q[read] = 0.6
        return q

    def read(self, x: str | None, visible) -> str | None:
        others = sorted(set(visible) - {x})
        return x if x is None or not others or self.rng.random() < self.acc else others[self.rng.integers(len(others))]

    def clip(self, t: float, clip: float, completed: list[dict], held: str | None, visible) -> dict:
        """The clip (t - clip, t]: its true verb is the last step completed in it (else other); the object is that
        step's (a container for open and close), else what the person holds; a place also has its receptacle."""
        steps = [c for c in completed if t - clip < c["t"] <= t]
        last = steps[-1] if steps else None
        label = last["verb"] if last else "other"
        obj, rec = (last["obj"], last["target"]) if last else (held, None)
        return {"t": round(t, 2), "kind": "clip", "verbs": self.verbs(label).tolist(), "object": self.read(obj, visible),
                "receptacle": self.read(rec, visible), "visible": sorted(visible), "truth": label}
