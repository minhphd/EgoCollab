"""Viewer episodes from recorded PARTNR streams: the person's head camera with gaze, the third-person views
in sync, the subgoals as steps, the skill reads as actions, and tracks for what the person holds, gaze
fixations, completions and the robot's subgoals.

    python -m helper.partnr.export <record_dir> [<episodes_root>]    # default root: <record_dir>/episodes
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from helper.datasets.episode import Action, Episode, Node, Segment, Track, write_episode


def intervals(rows: list[dict], key: str, t1: float) -> list[dict]:
    """Runs of equal non-empty values of rows[key] as track items."""
    out, cur, start = [], None, 0.0
    for r in rows + [{"t": t1, key: None}]:
        if r[key] != cur:
            if cur:
                out.append({"start": start, "end": r["t"], "label": cur, "detail": {}})
            cur, start = r[key], r["t"]
    return out


def episode(st: dict) -> Episode:
    label = dict(zip(st["nodes"], st["labels"]))
    reads = st["actions"]
    actions = [Action(a["t"], (reads[i + 1]["t"] if i + 1 < len(reads) else st["t1"]),
                      f"{a['verb']} {', '.join(a['entities'])}", {"subgoal": label.get(a["node"], "")})
               for i, a in enumerate(reads)]
    tracks = [Track("Holding", intervals(st["held"], "object", st["t1"]), "simulator grasp state"),
              Track("Gaze fixations", [{"start": f["t"], "end": f["t"], "label": f["object"],
                                        "detail": {"visible": len(f["visible"])}} for f in st["fixations"]], "synthesized gaze"),
              Track("Completions", [{"start": c["t"], "end": c["t"], "label": label[c["node"]], "detail": {}}
                                    for c in st["completions"]], "PARTNR proposition tracker"),
              Track("Robot", [{"start": a, "end": b, "label": label.get(v, v), "detail": {}} for a, b, v, *_ in st.get("robot_segments", [])],
                    f"robot policy: {st.get('robot', 'idle')}")]
    info = {"Task": [st["title"], f"{len(st['nodes'])} subgoals",
                     "required orderings: " + (", ".join(f"{label[u]} > {label[v]}" for u, v in st["truth"]["edges"]) or "none")],
            "Result": [f"{k}: {v}" for k, v in st["result"].items()],
            "Person log": st["person_log"][:80]}
    if st.get("robot_log"):
        info["Robot log"] = st["robot_log"][:80]
    return Episode(dataset="partnr", id=st["id"], title=st["title"], nodes=[Node(v, label[v]) for v in st["nodes"]],
                   edges=[tuple(e) for e in st["truth"]["edges"]], segments=[Segment(v, a, b) for a, b, v, *_ in st["truth"]["segments"]],
                   actions=actions, graph_source="PARTNR temporal constraints", video_id=st["id"], task=st["title"],
                   tracks=[t for t in tracks if t.items], info=info)


def main() -> None:
    rec = Path(sys.argv[1])
    root = Path(sys.argv[2]) if len(sys.argv) > 2 else rec / "episodes"
    for p in sorted((rec / "streams").glob("*.json")):
        st = json.loads(p.read_text())
        vids = rec / "videos"
        write_episode(root, episode(st), vids / f"{st['id']}_head.mp4", st["gaze"]["points"],
                      views={"third": vids / f"{st['id']}_third.mp4"})
    print(f"episodes in {root}")


if __name__ == "__main__":
    main()
