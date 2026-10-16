"""helper/behavior/floor.py on a synthetic map: a box stood on the floor is walked around.

    python -m pytest tests/test_floor.py -q
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from helper.behavior.floor import Floor  # noqa: E402


def test_a_blocked_box_is_walked_around(tmp_path):
    cv2.imwrite(str(tmp_path / "floor.png"), np.full((400, 400), 255, np.uint8))  # 4 m of open floor, 1 cm pixels
    fl = Floor(tmp_path / "floor.png")
    assert fl.clearance((0.0, 0.0)) > 1.0
    fl.block((-0.3, -0.3), (0.3, 0.3))
    assert fl.clearance((0.0, 0.0)) == 0.0
    way = fl.plan((-1.2, 0.0), (1.2, 0.0))
    assert way is not None
    along = [a + t * (b - a) for a, b in zip(way, way[1:]) for t in np.linspace(0, 1, 20)]
    assert min(fl.clearance(p) for p in along) > 0.15  # the way keeps off the box
