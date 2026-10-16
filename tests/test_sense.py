"""helper/collab/sense.py: clip events from the person's completed steps (no simulator).

    python -m pytest tests/test_sense.py -q
"""

import numpy as np

from helper.collab.sense import VERBS, Recognizer

DONE = [{"t": 1.2, "verb": "pick", "obj": "cup", "target": None},
        {"t": 2.5, "verb": "place", "obj": "cup", "target": "cabinet"},
        {"t": 7.0, "verb": "open", "obj": "fridge", "target": None}]
SEEN = {"cup", "cabinet", "fridge", "milk", "table"}


def test_a_clip_is_labelled_by_its_last_step():
    rec = Recognizer(None, acc=1.0, rng=np.random.default_rng(0))
    a, b, c = (rec.clip(t, 3.0, DONE, "milk", SEEN) for t in (3.0, 6.0, 9.0))
    assert (a["truth"], a["object"], a["receptacle"]) == ("place", "cup", "cabinet")  # pick and place, read as the place
    assert (b["truth"], b["object"], b["receptacle"]) == ("other", "milk", None)  # nothing done: what the person holds
    assert (c["truth"], c["object"]) == ("open", "fridge")


def test_reads_are_wrong_at_the_detectors_rate_and_verbs_come_from_the_bank(tmp_path):
    p = np.eye(len(VERBS))[[0, 0, 1, 4]] * 0.9 + 0.02
    np.savez(tmp_path / "bank.npz", p=p, y=np.array([0, 0, 1, 4]))
    rec = Recognizer(tmp_path / "bank.npz", acc=0.7, rng=np.random.default_rng(1))
    reads = [rec.clip(3.0, 3.0, DONE, None, SEEN) for _ in range(2000)]
    assert abs(np.mean([r["object"] == "cup" for r in reads]) - 0.7) < 0.04
    assert all(np.argmax(r["verbs"]) == VERBS.index("place") for r in reads)  # drawn from the bank's place rows
