"""Ego-Exo4D: download keystep takes with eye gaze, and convert them into viewer episodes.

Requires a signed licence (https://ego4d.dev/request/ego-exo4d) and the emailed AWS keys
stored with `aws configure`.

    python -m helper.datasets.ego_exo4d download --num-takes 4
    python -m helper.datasets.ego_exo4d build
"""

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from .episode import Episode, Node, Segment, precedence_dag, write_episode

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "ego_exo4d"
EPISODES = REPO / "data" / "episodes"
RAW_GAZE_SIZE = 1408  # 2D gaze pixels refer to the full-resolution Aria RGB frame
GAZE_SUBSAMPLE = 3  # 2D gaze rows are at one third of the 30 fps video rate


def egoexo(*args: str) -> None:
    egoexo_bin = Path(sys.executable).with_name("egoexo")
    subprocess.run([str(egoexo_bin), "-o", str(DATA), "-y", *args], check=True)


def keystep_annotations() -> dict:
    """Merged {take_uid: annotation} over the train and val keystep files."""
    merged = {}
    for split in ("train", "val"):
        path = DATA / "annotations" / f"keystep_{split}.json"
        if path.exists():
            merged |= json.loads(path.read_text())["annotations"]
    return merged


def has_gaze(take: dict) -> bool:
    return bool(take.get("has_trimmed_gaze") or take.get("has_trimmed_eye_gaze"))


def download(num_takes: int, uids: list[str] | None) -> None:
    egoexo("--parts", "metadata")
    egoexo("--parts", "annotations", "--benchmarks", "keystep")
    if not uids:
        takes = {t["take_uid"]: t for t in json.loads((DATA / "takes.json").read_text())}
        ks = keystep_annotations()
        pool = [takes[u] for u in ks if u in takes and has_gaze(takes[u]) and len(ks[u]["segments"]) >= 6]
        pool.sort(key=lambda t: t["duration_sec"])
        uids = [t["take_uid"] for t in pool[:num_takes]]
    print("takes:", uids)
    # --uids drops metadata and annotations, so the take parts are fetched separately.
    egoexo("--parts", "downscaled_takes/448", "--uids", *uids)
    egoexo("--parts", "take_eye_gaze", "--uids", *uids)


def ego_video(take: dict) -> Path:
    cam = next(c for c in take["frame_aligned_videos"] if c.lower().startswith("aria"))
    rel = Path(take["frame_aligned_videos"][cam]["rgb"]["relative_path"])
    return DATA / take["root_dir"] / rel.parent / "downscaled" / "448" / rel.name


def load_gaze(take: dict) -> list:
    folder = DATA / take["root_dir"] / "eye_gaze"
    path = next((p for p in (folder / "personalized_eye_gaze_2d.csv", folder / "general_eye_gaze_2d.csv") if p.exists()))
    with open(path) as f:
        return [[round(float(r["x"]) / RAW_GAZE_SIZE, 4), round(float(r["y"]) / RAW_GAZE_SIZE, 4)]
                if r["x"] and r["y"] else None for r in csv.DictReader(f)]


def build() -> None:
    takes = {t["take_uid"]: t for t in json.loads((DATA / "takes.json").read_text())}
    ks = keystep_annotations()
    orders = defaultdict(list)  # scenario -> first-occurrence keystep order of every annotated take
    for a in ks.values():
        order = [str(s["step_id"]) for s in sorted(a["segments"], key=lambda s: s["start_time"])]
        orders[a["scenario"]].append(list(dict.fromkeys(order)))
    for uid, a in ks.items():
        take = takes.get(uid)
        if take is None or not ego_video(take).exists():
            continue
        segs = sorted(a["segments"], key=lambda s: s["start_time"])
        names = {str(s["step_id"]): s["step_name"] for s in segs}
        ids = list(dict.fromkeys(names))
        ep = Episode(
            "ego_exo4d", take["take_name"], take.get("task_name") or a["scenario"],
            nodes=[Node(i, names[i], a["scenario"]) for i in ids],
            edges=precedence_dag(orders[a["scenario"]], ids, min_support=2),
            segments=[Segment(str(s["step_id"]), s["start_time"], s["end_time"]) for s in segs],
            graph_source=f"Keystep precedence shared by all {len(orders[a['scenario']])} annotated {a['scenario']} takes",
            task=a["scenario"],
        )
        print("built", write_episode(EPISODES, ep, ego_video(take), load_gaze(take), GAZE_SUBSAMPLE))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["download", "build"])
    ap.add_argument("--num-takes", type=int, default=4, help="shortest keystep takes with gaze to fetch")
    ap.add_argument("--uids", nargs="*", default=None, help="explicit take uids instead of automatic picks")
    args = ap.parse_args()
    download(args.num_takes, args.uids) if args.command == "download" else build()


if __name__ == "__main__":
    main()
