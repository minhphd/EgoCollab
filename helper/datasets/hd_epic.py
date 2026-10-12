"""HD-EPIC: download a subset of videos with gaze, and convert them into viewer episodes.

    python -m helper.datasets.hd_epic download --videos P01-20240202-110250 ...
    python -m helper.datasets.hd_epic build
    python -m helper.datasets.hd_epic export-actions   # numbered narrations for graph annotation
    python -m helper.datasets.hd_epic check            # validate post-hoc graphs
"""

import argparse
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from . import posthoc_graph
from .episode import Action, Episode, Node, Segment, probe_video, write_episode
from .sharepoint import SharedFolder

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "hd_epic"
EPISODES = REPO / "data" / "episodes"
ANNOTATIONS_GIT = "https://github.com/hd-epic/hd-epic-annotations"
# OneDrive mirrors listed on https://hd-epic.github.io/site (faster than data.bris.ac.uk).
VIDEOS_URL = "https://uob-my.sharepoint.com/:f:/g/personal/wq23021_bristol_ac_uk/IgA0PLwURLmfQZOl_J256iZtAY_uzFr0gTXVMWCWY4MSGd0"
INTERMEDIATE_URL = "https://uob-my.sharepoint.com/:f:/g/personal/jc17360_bristol_ac_uk/IgCCGb5qDbiOR7cmj1R9OyUWAXQFYL7FP_d0eMzB4ENPVQk"
MIN_ACTIONS = 3  # smaller subgoals give trivial graphs
DEFAULT_VIDEOS = [
    "P01-20240202-110250",  # coffee, 5 steps
    "P03-20240217-131219",  # red lentil squash curry, 11 steps
    "P07-20240529-131737",  # chickpea salad, part 1 of 2
    "P07-20240529-134410",  # chickpea salad, part 2 of 2
]


def all_video_ids(videos: SharedFolder) -> list[str]:
    return sorted(name[:-4] for p in videos.folders("Videos") for name in videos.files(f"Videos/{p}")
                  if name.endswith(".mp4"))


def download(video_ids: list[str] | None, workers: int = 4) -> None:
    """Downloads videos and per-frame gaze; `None` means every video. Reruns skip finished files."""
    ann = DATA / "annotations"
    if not ann.exists():
        subprocess.run(["git", "clone", "--depth", "1", ANNOTATIONS_GIT, str(ann)], check=True)
    videos, inter = SharedFolder(VIDEOS_URL), SharedFolder(INTERMEDIATE_URL)
    video_ids = video_ids or all_video_ids(videos)
    sizes = {}

    def fetch(vid: str) -> None:
        p = vid[:3]
        if p not in sizes:
            sizes[p] = videos.files(f"Videos/{p}")
        videos.download(f"Videos/{p}/{vid}.mp4", DATA / "videos" / p / f"{vid}.mp4", sizes[p].get(f"{vid}.mp4"))
        for name, size in inter.files(f"{p}/{vid}").items():
            inter.download(f"{p}/{vid}/{name}", DATA / "framewise" / p / vid / name, size)

    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(fetch, vid): vid for vid in video_ids}
        for i, fut in enumerate(as_completed(futures), 1):
            err = fut.exception()
            print(f"[{i}/{len(video_ids)}] {futures[fut]} {'FAILED: ' + str(err) if err else 'ok'}", flush=True)


def load_framewise(vid: str) -> tuple[list, np.ndarray]:
    """Per-frame gaze centre (normalised by the RGB image size) and horizontal head position (NaN if unknown)."""
    folder = DATA / "framewise" / vid[:3] / vid
    if not (folder / "framewise_info.jsonl").exists():  # a few videos have no released gaze or pose
        return [], np.empty((0, 2))
    w, h = json.loads((folder / "device_calibration.json").read_text())["cameras"]["camera-rgb"]["image_size"]
    points, xy = [], []
    with open(folder / "framewise_info.jsonl") as f:
        for line in f:
            row = json.loads(line)
            g, T = row["gaze_centre_in_pixels"], row["T_world_device"]
            points.append([round(g[0] / w, 4), round(g[1] / h, 4)] if g else None)
            xy.append(T[0][3:] + T[1][3:] if T else [np.nan, np.nan])  # world z is vertical
    return points, np.array(xy, dtype=float)


def head_displacement(q: np.ndarray) -> tuple[float | None, float]:
    """Largest horizontal distance (m) of the head from its median position, and pose coverage."""
    ok = ~np.isnan(q[:, 0])
    if not ok.any():
        return None, 0.0
    q = q[ok]
    return float(np.linalg.norm(q - np.median(q, axis=0), axis=1).max()), float(ok.mean())


def recipe_graph(vid: str, recipes: dict) -> tuple[list, list, list, list]:
    """Nodes, edges, segments and already-done nodes for every recipe capture that touches `vid`."""
    nodes, edges, segments, done_before = [], [], [], []
    for rid, r in recipes.items():
        for cap in r["captures"]:
            if vid not in cap["videos"]:
                continue
            earlier = set(cap["videos"][: cap["videos"].index(vid)])
            steps = list(r["steps"])
            nodes += [Node(s, r["steps"][s], r["name"]) for s in steps]
            edges += list(zip(steps, steps[1:]))
            for kind, table in (("step", cap["step_times"]), ("prep", cap["prep_times"])):
                for s, times in table.items():
                    segments += [Segment(s, t["start"], t["end"], kind) for t in times if t["video"] == vid]
            done_before += [s for s, t in cap["step_times"].items() if t and all(x["video"] in earlier for x in t)]
    return nodes, edges, segments, done_before


def load_narrations() -> pd.DataFrame:
    narr = pd.read_pickle(DATA / "annotations" / "narrations-and-action-segments" / "HD_EPIC_Narrations.pkl")
    narr = narr.sort_values(["video_id", "start_timestamp", "unique_narration_id"])
    suffix = narr.unique_narration_id.str.slice(len("P01-20240202-110250-"))
    narr["index"] = pd.to_numeric(suffix, errors="coerce")
    # A few videos have nested ids ("...-3-2"); those are numbered 1..n by start time instead.
    nested = narr.video_id.isin(narr.video_id[narr["index"].isna()])
    narr.loc[nested, "index"] = narr[nested].groupby("video_id").cumcount() + 1
    narr["index"] = narr["index"].astype(int)
    return narr


def action_times(narr: pd.DataFrame, vid: str) -> dict[int, tuple[float, float]]:
    rows = narr[narr.video_id == vid]
    return dict(zip(rows["index"], zip(rows.start_timestamp, rows.end_timestamp)))


def local_videos() -> list[str]:
    return sorted(p.stem for p in (DATA / "videos").glob("P*/*.mp4"))


def build(video_ids: list[str] | None = None) -> None:
    """One episode per subgoal per video where a post-hoc graph exists, else one recipe episode per video.

    Subgoals with fewer than MIN_ACTIONS actions are skipped.
    """
    recipes = json.loads((DATA / "annotations" / "high-level" / "complete_recipes.json").read_text())
    narr = load_narrations()
    for vid in video_ids or local_videos():
        rows = narr[narr.video_id == vid]
        actions = [Action(r.start_timestamp, r.end_timestamp, r.narration) for r in rows.itertuples()]
        video, (gaze, xy) = DATA / "videos" / vid[:3] / f"{vid}.mp4", load_framewise(vid)
        for old in EPISODES.glob(f"hd_epic/{vid}*"):
            shutil.rmtree(old)
        g = posthoc_graph.find(vid)
        if not g:
            nodes, edges, segments, done_before = recipe_graph(vid, recipes)
            title = " + ".join(dict.fromkeys(n.group for n in nodes)) or "no recipe"
            ep = Episode("hd_epic", vid, title, nodes, edges, segments, actions, done_before,
                         "Recipe steps in the participant's written order", video_id=vid, task=title)
            print("built", write_episode(EPISODES, ep, video, gaze))
            continue
        labels = {r.index: "; ".join(" ".join(a) for a in r.main_actions) for r in rows.itertuples()}
        episodes = [e for e in posthoc_graph.subgoal_episodes(g, vid, action_times(narr, vid), labels)
                    if len(e["nodes"]) >= MIN_ACTIONS]
        try:
            fps = len(gaze) / probe_video(video)["duration"]
        except subprocess.CalledProcessError:
            print(f"skipped {vid}: unreadable video, download it again")
            continue
        for e in episodes:
            lo, hi = e["clip"]
            disp, coverage = head_displacement(xy[int(lo * fps): int(hi * fps) + 1])
            ep = Episode("hd_epic", f"{vid}__{e['subgoal']['id']}", e["subgoal"]["label"], e["nodes"], e["edges"],
                         e["segments"], [a for a in actions if a.end > lo and a.start < hi],
                         graph_source="Primitive actions of this subgoal; edges are annotated preconditions",
                         clip=e["clip"], video_id=vid, task=f"{g['task']} · {e['subgoal'].get('group', '')}",
                         node_kind="action", stats={"head_displacement_m": disp and round(disp, 3),
                                                    "pose_coverage": round(coverage, 3)})
            write_episode(EPISODES, ep, video, gaze)
        print(f"built {vid}: {len(episodes)} subgoal episodes")


def export_actions(video_ids: list[str] | None = None) -> None:
    """Writes numbered narrations per video, with subgoal and verb-noun pairs, for graph annotation."""
    narr, out = load_narrations(), DATA / "actions"
    out.mkdir(exist_ok=True)
    for vid in video_ids or local_videos():
        g = posthoc_graph.find(vid)
        owner = {i: n["id"] for n in (g or {}).get("nodes", [])
                 for i in posthoc_graph.parse_ranges(n["actions"].get(vid, ""))}
        rows = narr[narr.video_id == vid]
        lines = [f"{r.index:4d} [{r.start_timestamp:7.1f}-{r.end_timestamp:7.1f}] <{owner.get(r.index, '-')}> "
                 f"{r.narration} || {'; '.join(' '.join(p) for p in r.main_actions)}" for r in rows.itertuples()]
        (out / f"{vid}.txt").write_text("\n".join(lines) + "\n")
        print(out / f"{vid}.txt", len(lines), "actions")


def check(video_ids: list[str] | None = None) -> bool:
    """Validates every post-hoc graph, or only those covering `video_ids`."""
    narr, ok = load_narrations(), True
    for path in sorted(posthoc_graph.ANNOTATIONS.glob("*.json")):
        g = json.loads(path.read_text())
        if video_ids and not set(video_ids) & set(g["videos"]):
            continue
        errors, warnings = posthoc_graph.validate(g, {v: action_times(narr, v) for v in g["videos"]})
        n_act = sum(len(posthoc_graph.parse_ranges(s)) for n in g["nodes"] for s in n["actions"].values())
        total = sum(len(action_times(narr, v)) for v in g["videos"])
        n_prim = sum(len(t) for t in g.get("primitives", {}).values())
        print(f"{path.name}: {len(g['nodes'])} subgoals, {len(g['edges'])} subgoal edges, {n_act}/{total} actions "
              f"assigned, {n_prim} primitives, {len(posthoc_graph.primitive_edges(g))} primitive edges")
        for e in errors:
            print("   error:", e)
        for w in warnings:
            print("   warning:", w)
        ok &= not errors
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["download", "build", "export-actions", "check"])
    ap.add_argument("--videos", nargs="*", default=None, help="HD-EPIC video ids")
    ap.add_argument("--all", action="store_true", help="download every video")
    ap.add_argument("--workers", type=int, default=4, help="parallel downloads")
    args = ap.parse_args()
    if args.command == "download":
        download(None if args.all else args.videos or DEFAULT_VIDEOS, args.workers)
    elif args.command == "build":
        build(args.videos)
    elif args.command == "export-actions":
        export_actions(args.videos)
    else:
        raise SystemExit(0 if check(args.videos) else 1)


if __name__ == "__main__":
    main()
