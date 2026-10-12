"""Gaze fixations on objects in HD-EPIC, in 3D: the Aria gaze ray against where each object rests.

An object rests where it was picked up before its first move and where it was put down after each
move (HD-EPIC scene-and-object-movements); while carried it has no resting place and is skipped.
A fixation is gaze held on one object for `min_s` s; it is reported when that time has passed.

    from helper.datasets.hd_epic.gaze3d import fixations, resting
    fixations("P01-20240202-110250")   # [{"t": 12.3, "object": <id>, "visible": [<ids>]}, ...]
"""

import json
import re
from functools import cache

import numpy as np

from .data import DATA

ANN = DATA / "annotations" / "scene-and-object-movements"
FPS = 30


@cache
def annotations() -> tuple[dict, dict]:
    return json.loads((ANN / "assoc_info.json").read_text()), json.loads((ANN / "mask_info.json").read_text())


def resting(vid: str) -> list[dict]:
    """Per object: its name and the intervals [a, b) in which it rests at a 3D point."""
    assoc, masks = annotations()
    out = []
    for oid, obj in assoc.get(vid, {}).items():
        spans, before = [], None
        for tr in sorted(obj["tracks"], key=lambda tr: tr["time_segment"][0]):
            a, b = tr["time_segment"]
            ms = sorted((masks[vid][m] for m in tr["masks"] if masks[vid][m].get("3d_location")),
                        key=lambda m: m["frame_number"])
            start = next((m for m in ms if abs(m["frame_number"] / FPS - a) < abs(m["frame_number"] / FPS - b)), None)
            end = next((m for m in reversed(ms) if abs(m["frame_number"] / FPS - b) <= abs(m["frame_number"] / FPS - a)), None)
            if not spans and start:
                spans.append((-np.inf, a, start["3d_location"]))
            elif spans and spans[-1][1] == np.inf:
                spans[-1] = (spans[-1][0], a, spans[-1][2])
            if end:
                spans.append((b, np.inf, end["3d_location"]))
        if spans and not re.fullmatch(r"Track \d+ \(skipped\)", obj["name"].strip()):
            out.append({"id": oid, "name": obj["name"].strip(), "spans": spans})
    return out


def fixations(vid: str, hz: float = 10.0, angle: float = 6.0, min_s: float = 0.3, fov: float = 50.0) -> list[dict]:
    """Fixations in time order: {"t", "object" (instance id), "visible" (instance ids in view)}.

    Gaze is sampled at `hz`; an object within `angle` degrees of the gaze ray is fixated, and objects
    within `fov` degrees of the camera axis are visible.
    """
    folder = DATA / "framewise" / vid[:3] / vid
    if not (folder / "framewise_info.jsonl").exists():
        return []
    cam = np.array(json.loads((folder / "device_calibration.json").read_text())["cameras"]["camera-rgb"]["T_device_camera"])
    axis_dev = cam[:3, 2]
    objs = resting(vid)
    step = round(FPS / hz)
    samples = []  # (t, fixated id or None, visible ids)
    with open(folder / "framewise_info.jsonl") as f:
        for i, line in enumerate(f):
            if i % step:
                continue
            row = json.loads(line)
            T, d = row["T_world_device"], row["gaze_direction_in_world"]
            if not T or not d:
                samples.append((i / FPS, None, []))
                continue
            T, d, t = np.array(T), np.array(d) / np.linalg.norm(d), i / FPS
            here = [(o["id"], p) for o in objs for a, b, p in o["spans"] if a <= t < b]
            if not here:
                samples.append((t, None, []))
                continue
            ids, P = zip(*here)
            rays = np.array(P) - T[:, 3]
            rays /= np.linalg.norm(rays, axis=1, keepdims=True)
            gaze_angle = np.degrees(np.arccos(np.clip(rays @ d, -1, 1)))
            view_angle = np.degrees(np.arccos(np.clip(rays @ (T[:, :3] @ axis_dev), -1, 1)))
            visible = [ids[k] for k in np.flatnonzero(view_angle < fov)]
            k = int(np.argmin(gaze_angle))
            samples.append((t, ids[k] if gaze_angle[k] < angle else None, visible))
    out, run = [], None
    for t, oid, visible in samples + [(np.inf, None, [])]:
        if run and oid == run[1]:
            if not run[2] and t - run[0] >= min_s:
                out.append({"t": round(t, 2), "object": oid, "visible": sorted(set(visible) | {oid})})
                run[2] = True
            continue
        run = [t, oid, False] if oid else None
    return out
