"""HD-EPIC: download a subset of videos with gaze, and convert them into viewer episodes.

    python -m helper.datasets.hd_epic download --videos P01-20240202-110250 ...
    python -m helper.datasets.hd_epic build
    python -m helper.datasets.hd_epic export-actions   # numbered narrations for graph annotation
    python -m helper.datasets.hd_epic check            # validate post-hoc graphs
"""

import argparse
import glob
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import posthoc_graph
from ..episode import Action, Box, Episode, Node, Segment, Track, probe_video, write_episode
from ..sharepoint import SharedFolder

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data" / "hd_epic"
EPISODES = REPO / "data" / "episodes"
ANN = DATA / "annotations"
FPS, FRAME_PX = 30, 1408  # every HD-EPIC video; frame numbers and boxes refer to these
ANNOTATIONS_GIT = "https://github.com/hd-epic/hd-epic-annotations"
# OneDrive mirrors listed on https://hd-epic.github.io/site (faster than data.bris.ac.uk).
VIDEOS_URL = "https://uob-my.sharepoint.com/:f:/g/personal/wq23021_bristol_ac_uk/IgA0PLwURLmfQZOl_J256iZtAY_uzFr0gTXVMWCWY4MSGd0"
INTERMEDIATE_URL = "https://uob-my.sharepoint.com/:f:/g/personal/jc17360_bristol_ac_uk/IgCCGb5qDbiOR7cmj1R9OyUWAXQFYL7FP_d0eMzB4ENPVQk"
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


def recipe_steps(vid: str, recipes: dict) -> tuple[list, list, list]:
    """Steps, their step and prep segments in `vid`, and steps already done in earlier videos, for every
    recipe capture that touches `vid` (high-level/complete_recipes.json)."""
    nodes, segments, done_before = [], [], []
    for r in recipes.values():
        for cap in r["captures"]:
            if vid not in cap["videos"]:
                continue
            earlier = set(cap["videos"][: cap["videos"].index(vid)])
            nodes += [Node(s, label, r["name"]) for s, label in r["steps"].items()]
            for kind, table in (("step", cap["step_times"]), ("prep", cap["prep_times"])):
                for s, times in table.items():
                    segments += [Segment(s, t["start"], t["end"], kind) for t in times if t["video"] == vid]
            done_before += [s for s, t in cap["step_times"].items() if t and all(x["video"] in earlier for x in t)]
    return nodes, segments, done_before


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


@cache
def _json(rel: str) -> dict:
    return json.loads((ANN / rel).read_text())


@cache
def _csv(pattern: str) -> pd.DataFrame:
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(ANN / pattern)))])


def _seconds(hms: str) -> float:
    h, m, s = str(hms).split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _clock(t: float) -> str:
    return f"{int(t) // 60}:{t % 60:04.1f}"


def activities(vid: str, duration: float) -> Track:
    """High-level activity segments; an end time of "end" means the end of the video."""
    rows = _csv("high-level/activities/*_recipe_timestamps.csv")
    end = lambda x: duration if str(x).strip() == "end" else float(x)
    items = [{"start": float(r.start_time), "end": end(r.end_time), "label": r.high_level_activity_label,
              "detail": {"recipe": r.recipe_id} if isinstance(r.recipe_id, str) else {}}
             for r in rows[rows.video_id == vid].itertuples()]
    return Track("Activities", items, "high-level/activities/*_recipe_timestamps.csv")


def movements(vid: str) -> tuple[Track, Track, list[Box]]:
    """Object movements (pick-up to put-down) with their boxes, and the gaze priming before each."""
    assoc = _json("scene-and-object-movements/assoc_info.json").get(vid, {})
    masks = _json("scene-and-object-movements/mask_info.json").get(vid, {})
    items, boxes, name_at = [], [], {}
    for obj in assoc.values():
        name = obj["name"].strip()
        for tr in obj["tracks"]:
            ms = sorted((masks[m] for m in tr["masks"] if m in masks), key=lambda m: m["frame_number"])
            where = lambda m: m.get("fixture") if m.get("fixture") not in (None, "Null") else "no fixture"
            a, b = tr["time_segment"]
            items.append({"start": a, "end": b, "label": name,
                          "detail": {"from": where(ms[0]), "to": where(ms[-1])} if ms else {}})
            for m in ms:
                name_at[m["frame_number"]] = name
                boxes.append(Box(round(m["frame_number"] / FPS, 3), [round(c / FRAME_PX, 4) for c in m["bbox"]], name))
    primed = []
    for x in _json("eye-gaze-priming/priming_info.json").get(vid, {}).values():
        for key, word in (("start", "pick-up"), ("end", "put-down")):
            y = x.get(key)
            stats = y.get("prime_stats", {}) if isinstance(y, dict) else {}
            if stats.get("frame_primed", -1) > 0:
                t = stats["frame_primed"] / FPS
                primed.append({"start": t, "end": t, "label": f"gaze on the {word} spot of {name_at.get(y['frame'], 'an object')}",
                               "detail": {"ahead by": f"{stats.get('prime_gap', 0):.1f} s", "hand at": _clock(y["frame"] / FPS)}})
    return (Track("Object movements", sorted(items, key=lambda i: i["start"]), "scene-and-object-movements/assoc_info.json"),
            Track("Gaze priming", sorted(primed, key=lambda i: i["start"]), "eye-gaze-priming/priming_info.json"), boxes)


def sounds(vid: str) -> Track:
    rows = _csv("audio-annotations/HD_EPIC_Sounds.csv")
    items = [{"start": _seconds(a), "end": _seconds(b), "label": c}
             for a, b, c in rows.loc[rows.video_id == vid, ["start_timestamp", "stop_timestamp", "class"]].values]
    return Track("Sounds", items, "audio-annotations/HD_EPIC_Sounds.csv")


def ingredients(vid: str, recipes: dict) -> tuple[Track, dict]:
    """Weighing and adding of each ingredient in this video, and the recipe facts for the info panel."""
    items, info = [], {}
    for rid, r in recipes.items():
        for cap in r["captures"]:
            if vid not in cap["videos"]:
                continue
            info[f"Recipe {rid}: {r['name']}"] = [f"type: {r['type']}", f"source: {r['source'] or 'none'}",
                                                  f"videos: {', '.join(cap['videos'])}"]
            lines = []
            for ing in cap["ingredients"].values():
                facts = [f"{k} {ing[k]}" for k in ("calories", "carbs", "fat", "protein") if ing.get(k) not in (None, "N/A")]
                amount = f"{ing['amount']} {ing['amount_unit']}" if ing.get("amount") not in (None, "N/A") else ""
                lines.append(" · ".join(x for x in [ing["name"], amount, ", ".join(facts)] if x))
                for kind in ("weigh", "add"):
                    items += [{"start": e["start"], "end": e["end"], "label": f"{kind} {ing['name']}"}
                              for e in ing.get(kind, []) if e.get("video") == vid]
            info[f"Ingredients of {r['name']}"] = lines
    return Track("Ingredients", sorted(items, key=lambda i: i["start"]), "high-level/complete_recipes.json"), info


def narration_detail(r) -> dict:
    out = {"verbs": ", ".join(r.verbs), "nouns": ", ".join(r.nouns),
           "main action": "; ".join(" ".join(p) for p in r.main_actions), "hands": ", ".join(r.hands)}
    return {k: v for k, v in out.items() if v}


def build(video_ids: list[str] | None = None) -> None:
    """One viewer episode per video with every HD-EPIC label: the narrated actions (with verbs, nouns and
    hands), the steps of every recipe captured in the video with their step and prep times, high-level
    activities, object movements with their boxes, gaze priming, sounds, and ingredient weighing and adding.
    HD-EPIC gives no dependencies between steps, so episodes have no edges; the post-hoc graphs in
    annotations/ are not used here."""
    recipes = _json("high-level/complete_recipes.json")
    narr = load_narrations()
    for vid in video_ids or local_videos():
        rows = narr[narr.video_id == vid]
        actions = [Action(r.start_timestamp, r.end_timestamp, r.narration.strip(), narration_detail(r)) for r in rows.itertuples()]
        for old in EPISODES.glob(f"hd_epic/{vid}*"):
            shutil.rmtree(old)
        nodes, segments, done_before = recipe_steps(vid, recipes)
        moved, primed, boxes = movements(vid)
        ing, info = ingredients(vid, recipes)
        video = DATA / "videos" / vid[:3] / f"{vid}.mp4"
        tracks = [t for t in (activities(vid, probe_video(video)["duration"]), ing, moved, primed, sounds(vid)) if t.items]
        info = {"Video": [vid, f"participant {vid[:3]}", f"{len(actions)} narrated actions, {len(moved.items)} object movements, "
                               f"{len(sounds(vid).items)} sound events"]} | info
        title = " + ".join(dict.fromkeys(n.group for n in nodes)) or "no recipe"
        ep = Episode("hd_epic", vid, title, nodes, [], segments, actions, done_before,
                     "HD-EPIC recipe steps with their step and prep times (no dependencies are provided)",
                     video_id=vid, task=title, tracks=tracks, boxes=boxes, info=info)
        print("built", write_episode(EPISODES, ep, video, load_framewise(vid)[0]), flush=True)


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
