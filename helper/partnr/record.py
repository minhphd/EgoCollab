"""Runs PARTNR episodes with the scripted person, synthesized gaze and a robot policy, and records them.

Per episode, under cfg.record_dir (or the run folder for closed-loop runs):
    streams/<id>.json      the BIP stream (helper/bip/replay.py documents its fields)
    frames/<id>/           the person's view with gaze at 1 fps, for S2
    videos/<id>_head.mp4   the person's head camera; videos/<id>_third.mp4 a third-person view
    episodes/partnr/<id>/  a viewer episode (python -m helper.viewer.server --episodes <record_dir>/episodes)
Run on a GPU node from the repo root:
    python -m helper.partnr.record episodes=40 workers=4
"""

from __future__ import annotations

import json
import re
import zlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from helper.bip.actions import OBSERVED
from helper.bip.config import Config, parse

from .gaze import Gaze
from .perceive import parse as parse_action
from .perceive import recognize, relation, sim_position, survey, view
from .person import Person
from .voice import answer as voice_answer
from .voice import step
from .tasks import drop, subgoals


def encode(cfg: Config, frames: Path, out: Path, fps: float) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([cfg.ffmpeg, "-v", "error", "-y", "-framerate", f"{fps:g}", "-i", str(frames / "%06d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out)], check=True)


def dot(img: np.ndarray, xy: list[float], r: int = 6) -> np.ndarray:
    import cv2
    h, w = img.shape[:2]
    out = img.copy()
    cv2.circle(out, (int(xy[0] * w), int(xy[1] * h)), r, (255, 0, 0), 2)
    return out


def position(sim, name: str) -> np.ndarray | None:
    try:
        return np.array(sim.env.full_world_graph.get_node_from_name(name).properties["translation"])
    except Exception:
        return None


def done_now(sim, task: dict, achieved: set) -> set:
    """Subgoals whose placement holds now with nobody carrying the object, and whose next-to holds too (or
    whose placement fell back to a plain one after PARTNR found no next-to spot). A category subgoal of rank j
    is done once j of its candidates rest on its furniture."""
    now, carried = sim.holding(), {sim.held(0), sim.held(1)}
    out, placed = set(), {}
    for v in task["nodes"]:
        sp = task["spec"].get(v) or {}
        if "rank" in sp:
            key = tuple(sp["objects"])
            if key not in placed:
                placed[key] = sum(1 for o in sp["objects"] if o not in carried and (r := relation(sim, o)) and r[1] in sp["tos"])
            if placed[key] >= sp["rank"]:
                out.add(v)
            continue
        i = task["index"][v]
        if now[i] and task["object"][v] not in carried and (all(now[j] for j in task["props"][v]) or v in achieved):
            out.add(v)
    return out


def robot_credit(robot, completions: list, task: dict | None = None) -> list[str]:
    """Subgoals the robot completed: its own script placed the object, and the subgoal first became done
    during the robot's work on it (not before, by the person). A category subgoal: it became done within a
    second of the robot's own Place of one of its candidates."""
    first = {c["node"]: c["t"] for c in completions}
    if task is not None:  # any naming: the robot's own Place of the subgoal's object, within a second before
        places = [(t, arg.split(",")[0].strip()) for t, tool, arg in robot.skills_done if tool == "Place"]
        objs = lambda v: (task["spec"].get(v) or {}).get("objects") or [task["object"][v]]
        return sorted(v for v, tc in first.items() if any(o in objs(v) and 0 <= tc - tp <= 1.0 for tp, o in places))
    return sorted(v for v in robot.done if v in first and any(w == v and a <= first[v] <= b + 0.5 for a, b, w, *_ in robot.segments))


def rooms(sim) -> dict:
    """Each furniture's room, for the robot's premapped home."""
    wg, out = sim.env.full_world_graph, {}
    for f in wg.get_all_furnitures():
        try:
            out[f.name] = wg.get_room_for_entity(f).name
        except Exception:
            pass
    return out


class Delayed:
    """The streams the robot receives, with the person's egocentric items held back `delay` s and stamped with the
    time they arrive; append-only, since robots read them by position."""
    def __init__(self, delay: float):
        self.delay, self.seen, self.pending, self.out = delay, {}, {}, {}

    def __call__(self, t: float, name: str, src: list, mine) -> list:
        pend, out = self.pending.setdefault(name, []), self.out.setdefault(name, [])
        pend += [(x["t"] + (0.0 if mine(x) else self.delay), x) for x in src[self.seen.get(name, 0):]]
        self.seen[name] = len(src)
        out += [x if mine(x) else {**x, "t": round(r, 2)} for r, x in pend if r <= t]  # seen when it arrives
        self.pending[name] = [(r, x) for r, x in pend if r > t]
        return out


def neighbours(sim, furniture: list, k: int = 2) -> dict:
    """Each furniture's k nearest pieces (floor-plan distance), for describing it as a person would."""
    pos = {f: p for f in furniture if not f.startswith(("floor", "wall")) and (p := sim_position(sim, f)) is not None}
    out = {}
    for f, p in pos.items():
        d = sorted((float(np.linalg.norm((q - p)[[0, 2]])), g) for g, q in pos.items() if g != f)
        out[f] = [g for _, g in d[:k]]
    return out


def converse(cfg: Config, t: float, task: dict, person, robot, done: set) -> None:
    """The robot's pending question goes to the person's voice; the person stops to answer."""
    q, robot.question = robot.question, None
    robot.say(t, q)
    lab = {v: step(task["spec"][v], task.get("rooms", {}), task.get("near")) if task["spec"].get(v) else lab_
           for v, lab_ in zip(task["nodes"], task["labels"])}
    state = {"instruction": task["say"], "steps": [lab[v] for v in person.known(done)], "partial": bool(cfg.person_horizon),
             "done": [lab[v] for v in task["nodes"] if v in done], "current": lab.get(person.cur)}
    try:
        a = voice_answer(q, state, cfg.voice_port)
    except Exception as e:
        a = "(no answer)"
        robot.log.append(f"{t:.1f} voice failed: {str(e)[:80]}")
    talk = cfg.talk_overhead + (len(q.split()) + len(a.split())) / cfg.talk_wps  # hearing the question, answering
    person.wait_until = max(person.wait_until, t + talk)
    robot.talk_s = getattr(robot, "talk_s", 0.0) + talk
    robot.hear(t, a)


def wrong_places(robot, task: dict) -> int:
    """The robot's placements that no subgoal of the true task asks for: an object outside the task, or a task
    object on furniture where no subgoal sends its kind."""
    kind = lambda x: re.sub(r"_\d+$", "", x)
    ok = set()
    for v in task["nodes"]:
        sp = task["spec"].get(v) or {}
        for o in sp.get("objects") or [task["object"][v]]:
            ok |= {(o, f) for f in sp.get("tos") or [sp.get("to")]}
    ok |= {(kind(o), f) for o, f in list(ok)}
    place = [[x.strip() for x in arg.split(",")] for t, tool, arg in robot.skills_done if tool == "Place"]
    return sum(1 for p in place if (p[0], p[2]) not in ok and (kind(p[0]), p[2]) not in ok)


def look_target(sim, skill, furniture: set) -> np.ndarray | None:
    """Where the person looks: the skill's target while handling it or within 3 m of it, else ahead.
    Furniture positions sit at the floor, so the gaze goes 0.6 m higher, about a tabletop."""
    if skill is None:
        return None
    tool, arg = skill
    name = arg.split(",")[2].strip() if tool == "Place" else arg.split(",")[0].strip()
    there = position(sim, name)
    if there is None:
        return None
    there = there + [0.0, 0.6 if name in furniture else 0.0, 0.0]
    here = np.array(sim.sim.agents_mgr[1].articulated_agent.base_pos)
    return there if tool != "Navigate" or np.linalg.norm((here - there)[[0, 2]]) < 3.0 else None


def run(cfg: Config, sim, ep_id: str, robot, out: Path) -> dict:
    import cv2
    sim.reset(ep_id)
    task = subgoals(sim, cfg.seed)
    before = sorted(done_now(sim, task, set()))
    task = drop(task, set(before))
    if len(task["nodes"]) < 2:
        raise ValueError(f"only {len(task['nodes'])} subgoal left to do at the start")
    seed = cfg.seed * 7919 + (int(ep_id) if str(ep_id).isdigit() else zlib.crc32(str(ep_id).encode()) % 100000)
    furniture = [f.name for f in sim.env.full_world_graph.get_all_furnitures()]
    person, gaze = Person(cfg, task, furniture, seed), Gaze(cfg, seed + 1)
    sid = f"{cfg.split}_{ep_id}"
    task["furniture"] = sorted(furniture)
    task["rooms"], task["near"] = rooms(sim), neighbours(sim, furniture)
    task["map"] = survey(cfg, sim) if cfg.premap else []  # the robot's map from before the task
    task["say"] = json.loads(Path(cfg.say_file).read_text()).get(str(ep_id), task["title"]) if cfg.say_file else task["title"]
    robot.start(task, sid, person)
    idx, nodes = task["index"], task["nodes"]
    tmp = Path(out) / "tmp" / sid
    for d in ("head", "third"):
        shutil.rmtree(tmp / d, ignore_errors=True)
        (tmp / d).mkdir(parents=True)
    fdir = Path(out) / "frames" / sid
    shutil.rmtree(fdir, ignore_errors=True)
    fdir.mkdir(parents=True)
    done, ever, completions, fixations, held, points, gaze_at = set(), set(), [], [], [], [], None
    views, acts, n_person, n_robot, prng = [], [], 0, 0, np.random.default_rng([seed, 5])
    late = Delayed(cfg.ego_delay)
    last_skill, idle_since, step, t0 = None, None, 0, time.time()
    while step < cfg.max_steps:
        t = step / cfg.fps
        masks = sim.masks()
        done = done_now(sim, task, set(person.achieved) | robot.done)
        for v in sorted(done - ever):
            ever.add(v)
            completions.append({"t": round(t, 2), "node": v})
        if len(done) == len(nodes):
            break
        if getattr(robot, "question", None):
            converse(cfg, t, task, person, robot, done)
        person.taken = {robot.target} if robot.cur is not None and getattr(robot, "target", None) else set()
        robot.avoid = {person.target} if person.cur is not None and person.target else set()
        vec1 = person.act(t, sim, done, robot.claims, sorted(masks))
        if person.skill is not None and person.skill != last_skill:  # gaze may prime the next pick or place
            tool, arg = person.skill
            nxt = person.queue[0] if person.queue else None
            if tool == "Navigate" and nxt and nxt[0] in ("Pick", "Place"):
                here, there = sim.sim.agents_mgr[1].articulated_agent.base_pos, position(sim, arg)
                eta = np.linalg.norm(np.array(here) - there) / 2.5 if there is not None else 2.0
                gaze.plan(t + eta, arg, "pick" if nxt[0] == "Pick" else "place")
        last_skill = person.skill
        goal = look_target(sim, person.skill, set(furniture))  # people look at what they are about to handle
        if goal is not None:
            gaze_at = goal if gaze_at is None else gaze_at + 0.2 * (goal - gaze_at)
        else:
            gaze_at = None
        sim.head_look(gaze_at)
        cur = set(task["objects"][person.cur]) if person.cur else set()
        xy, fix = gaze.step(t, masks, cur, set())
        if fix and cfg.partner_view == "ego":
            fixations.append(fix)
        if step % int(cfg.fps) == 0 and cfg.partner_view == "ego":
            held.append({"t": round(t, 2), "object": sim.held(1), "visible": sorted(masks)})
            img = sim.frame("agent_1_head_rgb")
            cv2.imwrite(str(fdir / f"{int(round(t)):06d}.jpg"), cv2.cvtColor(dot(cv2.resize(img, (256, 256)), xy), cv2.COLOR_RGB2BGR))
        if step % max(1, int(cfg.fps * cfg.view_every)) == 0:  # detections in both cameras for the scene graph
            fur = set(furniture)
            if cfg.partner_view == "ego":
                views.append({"t": round(t, 2), "src": "ego", "triples": view(cfg, sim, "agent_1_head_panoptic", 1, "H", fur, prng)})
            views.append({"t": round(t, 2), "src": "robot",
                          "triples": view(cfg, sim, "agent_0_head_panoptic", 0, "R", fur, prng)})
        new = person.skills_done[n_person:]
        seen_by = sorted(masks) if cfg.partner_view in ("ego", "acts") or not new else sorted(sim.masks("agent_0_head_panoptic"))
        for when, tool, arg in new:  # the person's actions, as recognized
            a = recognize(cfg, tool, arg, seen_by, set(furniture), prng)
            if a and cfg.partner_view == "robot" and a[1] not in seen_by:  # the robot's camera misses it
                a = None
            if a:
                acts.append({"t": round(when, 2), "actor": "H", "action": a})
        n_person = len(person.skills_done)
        if cfg.ego_delay:  # the person's view reaches the robot late
            obs = {"held": late(t, "held", held, lambda x: False), "fixations": late(t, "fixations", fixations, lambda x: False),
                   "views": late(t, "views", views, lambda x: x["src"] == "robot"),
                   "acts": late(t, "acts", acts, lambda x: x["actor"] == "R")}
        else:
            obs = {"held": held, "fixations": fixations, "views": views, "acts": acts}
        vec0 = robot.act(t, sim, {"actions": person.actions, **obs, "completions": completions, "frames": str(fdir), "done": done})
        for when, tool, arg in robot.skills_done[n_robot:]:  # the robot knows its own actions
            if tool in OBSERVED:
                acts.append({"t": round(when, 2), "actor": "R", "action": list(parse_action(tool, arg))})
        n_robot = len(robot.skills_done)
        if vec0 is None and vec1 is None and person.cur is None and t >= person.wait_until:  # not pausing or talking
            idle_since = t if idle_since is None else idle_since
            if t - idle_since > 10:  # nothing left the person can do
                break
        else:
            idle_since = None
        if step % cfg.video_every == 0:
            k = step // cfg.video_every
            points.append([round(xy[0], 4), round(xy[1], 4)])
            cv2.imwrite(str(tmp / "head" / f"{k:06d}.jpg"), cv2.cvtColor(sim.frame("agent_1_head_rgb"), cv2.COLOR_RGB2BGR))
            third = np.concatenate([cv2.resize(sim.frame("agent_0_third_rgb"), (256, 256)),
                                    cv2.resize(sim.frame("agent_1_third_rgb"), (256, 256))], 1)
            cv2.imwrite(str(tmp / "third" / f"{k:06d}.jpg"), cv2.cvtColor(third, cv2.COLOR_RGB2BGR))
        sim.step({0: vec0, 1: vec1})
        step += 1
    t_end = step / cfg.fps
    person.close(t_end)
    vfps = cfg.fps / cfg.video_every
    for d in ("head", "third"):
        encode(cfg, tmp / d, Path(out) / "videos" / f"{sid}_{d}.mp4", vfps)
    shutil.rmtree(tmp, ignore_errors=True)
    stream = {"id": sid, "episode": ep_id, "title": task["title"], "nodes": nodes, "labels": task["labels"],
              "objects": task["objects"], "t0": 0.0, "t1": round(t_end, 2),
              "inventory": task.get("inventory", {}), "recipe": task.get("recipe"),
              "truth": {"edges": task["edges"], "segments": person.segments, "completions": completions}, "done_before": before,
              "actions": person.actions, "held": held, "fixations": fixations, "completions": completions,
              "views": views, "acts": acts, "furniture": sorted(furniture), "spec": task["spec"], "map": task["map"],
              "frames": str(fdir), "frame_fps": 1.0, "gaze": {"fps": vfps, "points": points},
              "scripts": task["scripts"], "person_log": person.log, "robot": robot.name, "robot_log": robot.log,
              "robot_segments": robot.segments, "robot_belief": getattr(robot, "belief", []),
              "robot_s2": getattr(getattr(robot, "B", None), "calls", []), "robot_decisions": getattr(robot, "decisions", [])[-400:],
              "dialogue": getattr(robot, "dialogue", []),
              "result": {"steps": step, "seconds": round(t_end, 2), "percent_complete": sim.percent_complete(),
                         "success": len(done) == len(nodes), "done_by_robot": robot_credit(robot, completions, task), "robot_wrong": wrong_places(robot, task), "questions": getattr(robot, "questions", 0), "talk_s": round(getattr(robot, "talk_s", 0.0), 1), "robot_failed": robot.failed,
                         "wall_s": round(time.time() - t0, 1)}}
    (Path(out) / "streams").mkdir(parents=True, exist_ok=True)
    (Path(out) / "streams" / f"{sid}.json").write_text(json.dumps(stream))
    return stream


def worker(args) -> list[dict]:
    cfg, ids, gpu, out = args
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from .env import Sim
    from .robot import make_robot
    sim, results = Sim(cfg, ids, Path(out) / "partnr_logs"), []
    for ep_id in ids:
        try:
            st = run(cfg, sim, ep_id, make_robot(cfg, sim), Path(out))
            results.append({"id": st["id"], **st["result"]})
            print(f"{st['id']}: {len(st['nodes'])} subgoals, {st['result']}", flush=True)
        except Exception as e:  # one bad episode must not stop the worker
            import traceback
            print(f"{ep_id}: FAILED {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}", flush=True)
    return results


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    from .env import episodes
    cfg = parse(sys.argv[1:])
    out = Path(cfg.record_dir) if cfg.name.startswith("rec") else cfg.run_dir  # recordings vs experiment runs
    eps = episodes(cfg)
    eps = eps[: cfg.episodes] if cfg.episodes else eps
    ids = [e["id"] for e in eps]
    out.mkdir(parents=True, exist_ok=True)
    (out / f"config_{cfg.name}.json").write_text(json.dumps(cfg.to_dict(), indent=1))
    chunks = [ids[i::cfg.workers] for i in range(cfg.workers)]
    with ProcessPoolExecutor(cfg.workers, mp_context=mp.get_context("spawn")) as pool:
        rows = [r for rs in pool.map(worker, [(cfg, c, i % 4, str(out)) for i, c in enumerate(chunks) if c]) for r in rs]
    (out / f"results_{cfg.name}.json").write_text(json.dumps(rows, indent=1))
    ok = [r for r in rows if r["success"]]
    print(f"{cfg.name}: {len(rows)} episodes, {len(ok)} solved, mean {np.mean([r['seconds'] for r in rows]):.1f} s", flush=True)


if __name__ == "__main__":
    main()
