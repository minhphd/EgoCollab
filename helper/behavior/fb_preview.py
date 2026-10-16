"""Stills of false-belief designs in BEHAVIOR-1K (OmniGibson): the object X on the surface A, then inside the open
container B, then B closed, seen from a person's eye height facing A and B.

    OMNI_KIT_ACCEPT_EULA=YES python -m helper.behavior.fb_preview <designs.json> <out_dir> b000,b002   # behavior env
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

EYE = 1.6  # m, the person's eye height
BACK = 2.6  # m from the middle of A and B


def look_at(pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Quaternion (x, y, z, w) of a camera at pos looking at target, z up (the camera looks along its -z)."""
    from omnigibson.utils.transform_utils import mat2quat
    import torch as th
    f = (target - pos) / np.linalg.norm(target - pos)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return mat2quat(th.tensor(np.stack([r, u, -f], 1), dtype=th.float32)).numpy()


def eye_points(a, b) -> tuple[list, np.ndarray]:
    """Two points BACK m from the middle of A and B, one on each side of the line A-B, at eye height."""
    pa, pb = (np.array(o.get_position_orientation()[0]) for o in (a, b))
    mid = (pa + pb) / 2
    side = np.array([-(pb - pa)[1], (pb - pa)[0], 0.0])
    side /= np.linalg.norm(side) + 1e-9
    return [np.array([*(mid + s * BACK * side)[:2], EYE]) for s in (1, -1)], mid


def main() -> None:
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    designs = {d["design"]: d for d in json.load(open(sys.argv[1]))}
    out, keep = Path(sys.argv[2]), sys.argv[3].split(",")
    out.mkdir(parents=True, exist_ok=True)
    import omnigibson as og
    from omnigibson.macros import gm
    from omnigibson.object_states import Inside, OnTop, Open
    from PIL import Image
    gm.HEADLESS, gm.ENABLE_OBJECT_STATES = True, True
    env = None
    for k in keep:
        d = designs[k]
        lib = Path(os.environ.get("B1K_DATASETS", Path.home() / "BEHAVIOR-1K/datasets")) / "behavior-1k-assets/objects"
        model = sorted(p.name for p in (lib / d["X_kind"]).iterdir())[0]
        cfg = {"scene": {"type": "InteractiveTraversableScene", "scene_model": d["scene"]}, "robots": [],
               "objects": [{"type": "DatasetObject", "name": "X", "category": d["X_kind"], "model": model,
                            "position": [100.0, 100.0, 100.0]}]}  # loaded with the scene, out of sight until sampled
        if env is not None:
            og.clear()
        env = og.Environment(configs=cfg)
        scene = env.scene
        a, b, x = (scene.object_registry("name", n) for n in (d["A"], d["B"], "X"))
        import torch as th
        for _ in range(5):  # env.step starts the physics views that state sampling reads
            env.step(th.empty(0))
        cam = og.sim.viewer_camera
        cam.image_width, cam.image_height = 1280, 720
        eyes, mid = eye_points(a, b)
        shots = []

        def shot(tag):
            for i, eye in enumerate(eyes):
                cam.set_position_orientation(eye, look_at(eye, mid + [0, 0, 0.5]))
                for _ in range(30):
                    og.sim.render()
                rgb = cam.get_obs()[0]["rgb"][..., :3].cpu().numpy()
                Image.fromarray(rgb).save(out / f"{k}_{tag}_side{i}.png")
            shots.append(tag)

        on = x.states[OnTop].set_value(a, True)
        shot("1_on_table")
        b.states[Open].set_value(True)
        inside = x.states[Inside].set_value(b, True)
        for _ in range(10):
            env.step(th.empty(0))
        shot("2_inside_open")
        b.states[Open].set_value(False)
        for _ in range(10):
            env.step(th.empty(0))
        shot("3_closed")
        print(json.dumps({"design": k, "scene": d["scene"], "on_top": bool(on), "inside": bool(inside), "shots": shots}), flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
