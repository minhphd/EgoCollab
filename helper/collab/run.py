"""Run collaboration episodes and score them against the person alone.

    python -m helper.collab.run <out_dir> task=lunch policy=alone,oracle,bip,bip+announce,bip+ask,bip+tell seed=0,1,2 [clip=1] [t_max=300]

One process per task (OmniGibson loads one home per process). Each (policy, seed) writes <out>/streams/<task>_<policy>_<seed>.json
(and <out>/clips/... .mp4 with clip=1); an existing stream is skipped, and an episode the process died in is recorded as
crashed when it restarts; then each episode's metrics (helper/collab/score.py).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .agents import make
from .env import CollabEnv, SkillEnv
from .score import score


def episode(env: SkillEnv, policy: str, seed: int, clip: Path | None) -> dict:
    env.unwrapped.robot_on = policy != "alone"
    obs, info = env.reset(seed=seed)
    agent = make(policy, env, seed)
    if agent is not None:
        agent.reset(obs, info)
    frames, wall, k, crashed, said = [], time.time(), 0, None, 0
    while True:
        a = agent.act(obs, info) if agent is not None else {"skill": "noop"}
        for line in (agent.log[said:] if agent is not None else []):  # the robot's log as it happens
            print(f"  {policy} robot {line}", flush=True)
        said = len(agent.log) if agent is not None else 0
        try:
            obs, _, done, out_of_time, info = env.step(a)
        except Exception as err:  # the simulator blew up (a NaN pose): the episode ends; the next reset restores it
            crashed, done, out_of_time = f"{type(err).__name__}: {err}"[:300], False, True
            print(f"  {policy} crashed at t {info['t']:.1f} s: {crashed}", flush=True)
            break
        if clip is not None and k % 2 == 0:
            frames.append(env.render())
        if k % 450 == 0:  # every 30 s of task time
            last = (info["person_log"] or [""])[-1]
            print(f"  {policy} t {info['t']:.0f} s: {len(info['done'])} done, person on {info['person_step']} ({last}), "
                  f"robot {info.get('skill')}", flush=True)
        k += 1
        if done or out_of_time:
            break
    if clip is not None:
        write_video(frames, clip)
    log = info["log"]
    return {"task": env.unwrapped.task["title"], "policy": policy, "seed": seed, "seconds": info["t"], "finished": bool(done),
            "done": log["done"], "by": {str(v): w for v, w in info["by"].items()}, "utterances": log["utterances"],
            "person_segments": info["person_segments"], "robot_segments": agent.segments if agent is not None else [],
            "person_log": info["person_log"], "robot_log": agent.log if agent is not None else [], "frames": log["frames"],
            "fixations": log["fixations"], "wall_s": round(time.time() - wall, 1), "crashed": crashed}


def write_video(frames: list, path: Path) -> None:
    import cv2
    tmp = path.with_suffix("")
    tmp.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(str(tmp / f"{i:06d}.jpg"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    ff = str(Path.home() / ".conda/envs/egocollab/bin/ffmpeg")
    subprocess.run([ff, "-v", "error", "-y", "-framerate", "7.5", "-i", str(tmp / "%06d.jpg"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "23", str(path)], check=True)
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()


def main() -> None:
    out = Path(sys.argv[1])
    args = dict(a.split("=", 1) for a in sys.argv[2:])
    task, policies = args["task"], args.get("policy", "alone,bip").split(",")
    seeds = [int(s) for s in args.get("seed", "0").split(",")]
    clip = args.get("clip") == "1"
    (out / "streams").mkdir(parents=True, exist_ok=True)
    env = SkillEnv(CollabEnv(task, t_max=float(args.get("t_max", 300)), render_mode="rgb_array" if clip else None))
    for seed in seeds:
        for policy in policies:
            path = out / "streams" / f"{task}_{policy}_{seed}.json"
            mark = path.with_suffix(".started")
            if path.exists():
                continue
            if mark.exists():  # the process died in this episode (the simulator aborted): recorded, not rerun
                st = {"task": env.unwrapped.task["title"], "policy": policy, "seed": seed, "seconds": None, "finished": False,
                      "done": [], "by": {}, "utterances": [], "person_segments": [], "robot_segments": [], "person_log": [],
                      "robot_log": [], "frames": [], "fixations": [], "wall_s": None, "crashed": "the simulator process died"}
            else:
                mark.touch()
                st = episode(env, policy, seed, out / "clips" / f"{task}_{policy}_{seed}.mp4" if clip else None)
            path.write_text(json.dumps(st))
            mark.unlink()
            print(json.dumps({k: st[k] for k in ("policy", "seed", "seconds", "finished", "by", "wall_s")}), flush=True)
    streams = {(s["policy"], s["seed"]): s for s in (json.loads(p.read_text()) for p in sorted((out / "streams").glob(f"{task}_*.json")))}
    rows = [score(st, streams.get(("alone", sd))) for (pol, sd), st in sorted(streams.items()) if pol != "alone"]
    for r in rows:
        print("SCORE", json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # all written: skip the simulator's teardown, which can crash
