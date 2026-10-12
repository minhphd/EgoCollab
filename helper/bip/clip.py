"""A clip for inspecting one BIP replay and one simulated episode, written to runs/bip/clips/<video>.mp4.

Part 1 replays a test video at 8x: the egocentric frame with gaze, what S1 receives, its belief over
the current and next subgoal against the truth, the inferred graph against the annotated one, edge F1
over time and every Gemini call. Part 2 animates the same task in the closed-loop simulation: the
person alone, the person with the robot, and the robot's decisions.

    python -m helper.bip.clip video=P04-20240414-175337 [sim_video=P01-20240204-145458 sim_seed=0]
"""

import json
import shutil
import subprocess
import sys
import textwrap
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import stream
from .config import Config, parse
from .decide import Costs
from .run import Replay
from .score import closure
from .sim import episode
from .tasks import clock, load
from .view import View

FPS, SPEED, W, H = 10, 8.0, 16, 9
_DATA: dict = {}  # shared with forked frame workers
INK, MUTED, GOOD, BAD = "#14213D", "#6B7585", "#2E8B57", "#C0392B"


def colors(n: int) -> list:
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    return [cmap(i % cmap.N) for i in range(n)]


def short(label: str, k: int = 30) -> str:
    return label if len(label) <= k else label[: k - 1] + "…"


def layers(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """Node positions: x by longest path from a source in the true graph, y spread within a layer."""
    depth = [0] * n
    for _ in range(n):
        for u, v in edges:
            depth[v] = max(depth[v], depth[u] + 1)
    pos = np.zeros((n, 2))
    for d in set(depth):
        members = [v for v in range(n) if depth[v] == d]
        for i, v in enumerate(members):
            pos[v] = (d, (i - (len(members) - 1) / 2) * 1.0)
    return pos


def reduction(n: int, pairs: set) -> set:
    return {(u, v) for u, v in pairs if not any((u, w) in pairs and (w, v) in pairs for w in range(n) if w not in (u, v))}


# -- part 1: the replay ----------------------------------------------------------------------------------
def replay_frame(args) -> None:
    k, i_frame, T = args
    data, out = _DATA, _DATA["out"]
    st, tr, calls = data["stream"], data["trace"], data["calls"]
    N, labels, cols, pos = len(st["nodes"]), st["labels"], data["colors"], np.array(data["pos"])
    truth, t0, t1 = {tuple(e) for e in data["truth"]}, st["t0"], st["t1"]
    i = max(0, np.searchsorted(data["times"], T, side="right") - 1)
    snap = tr[i]
    cur, nxt, E = np.array(snap["cur"]), np.array(snap["next"]), np.array(snap["edges"])
    true_cur, true_next, done = data["true_cur"][i_frame], data["true_next"][i_frame], data["done"][i_frame]

    fig = plt.figure(figsize=(W, H), dpi=100, facecolor="white")
    fig.text(0.01, 0.965, f"Fast-slow BIP replay · {st['title']} · HD-EPIC {st['video']}", fontsize=17, color=INK, weight="bold")
    fig.text(0.01, 0.935, "Perception: narrated actions, completions and gaze on tracked objects (oracle) · S1 particle filter per graph · "
             "S2 gemini-3.5-flash on surprise", fontsize=11, color=MUTED)
    fig.text(0.99, 0.965, f"{clock(T - t0)} / {clock(t1 - t0)}   ·   {SPEED:g}×", fontsize=15, color=INK, ha="right")

    ax = fig.add_axes([0.01, 0.29, 0.355, 0.63])
    ax.imshow(data["view"].image(T))
    ax.axis("off")

    # what S1 receives
    tx = fig.add_axes([0.01, 0.12, 0.355, 0.16])
    tx.axis("off")
    act = next((e for e in reversed(data["actions"]) if e[0] <= T), None)
    fix = next((e for e in reversed(data["fixations"]) if T - 2.5 <= e[0] <= T), None)
    fin = next((e for e in reversed(data["completions"]) if T - 4 <= e[0] <= T), None)
    lines = [("S1 input", INK, "bold")]
    if act:
        lines.append(("Action: " + textwrap.shorten(act[1], 70), INK, "normal"))
    lines.append(("Gaze: " + (f"{fix[1]}" + (f"  (objects of {', '.join(map(str, fix[2]))})" if fix[2] else "  (no subgoal's objects)") if fix else "no fixation"), INK, "normal"))
    if fin:
        lines.append((f"Completed: {fin[1] + 1} {short(labels[fin[1]], 40)}", GOOD, "normal"))
    for j, (s, c, w) in enumerate(lines):
        tx.text(0, 1 - j * 0.24, s, fontsize=11.5, color=c, weight=w, va="top", transform=tx.transAxes)

    # belief bars
    y = np.arange(N)[::-1]
    for x0, p, title, true in ((0.535, cur, "P(current subgoal)", true_cur), (0.775, nxt, "P(next subgoal)", true_next)):
        b = fig.add_axes([x0, 0.55, 0.2, 0.33])
        b.barh(y, p, color=[cols[v] for v in range(N)], height=0.7)
        b.set_xlim(0, 1)
        b.set_ylim(-0.6, N - 0.4)
        b.set_title(title, fontsize=12, color=INK, loc="left")
        b.tick_params(labelsize=9, colors=MUTED)
        b.set_yticks(y)
        b.set_yticklabels([f"{v + 1} {short(labels[v])}" for v in range(N)] if x0 < 0.6 else [], fontsize=9.5, color=INK)
        for s in ("top", "right"):
            b.spines[s].set_visible(False)
        if true is not None:
            b.plot(1.0, y[true], marker="<", color=GOOD, markersize=12, clip_on=False)
            b.text(1.02, y[true], "truth", color=GOOD, fontsize=9, va="center", transform=b.get_yaxis_transform())
        for v in range(N):
            if done[v]:
                b.text(0.98, y[v], "done", fontsize=8, color=MUTED, ha="right", va="center")

    # graph: inferred against annotated
    g = fig.add_axes([0.385, 0.13, 0.36, 0.36])
    g.axis("off")
    g.set_title("Task graph: inferred (p ≥ 0.5) against annotated", fontsize=12, color=INK, loc="left")
    stated = {(u, v) for u in range(N) for v in range(N) if E[u, v] >= 0.5}
    tc = closure(list(range(N)), truth)
    draw = [(u, v, GOOD if (u, v) in tc else BAD, "-") for u, v in reduction(N, stated)]
    draw += [(u, v, MUTED, "--") for u, v in truth if (u, v) not in closure(list(range(N)), stated)]
    for u, v, c, ls in draw:
        g.annotate("", xy=pos[v], xytext=pos[u], arrowprops=dict(arrowstyle="-|>", color=c, lw=2, ls=ls,
                   shrinkA=14, shrinkB=14, connectionstyle="arc3,rad=0.12"))
    bip = int(np.argmax(cur)) if cur.sum() > 0 else -1
    for v in range(N):
        face = "#C8CCD2" if done[v] else cols[v]
        g.scatter(*pos[v], s=900, color=face, alpha=1.0 if (v == bip or done[v]) else 0.35, zorder=3,
                  edgecolors=INK if v == true_cur else "white", linewidths=3 if v == true_cur else 1)
        g.text(*pos[v], str(v + 1), ha="center", va="center", fontsize=12, weight="bold", color=INK, zorder=4)
    g.margins(0.12, 0.2)
    fig.text(0.385, 0.105, "solid fill = BIP's current · dark ring = true current · grey = done · "
           "green = right ordering · red = wrong · dashed = missed", fontsize=8.5, color=MUTED, transform=g.transAxes)

    # edge F1 over time and the S2 log
    f = fig.add_axes([0.79, 0.33, 0.19, 0.15])
    ts = np.array(data["f1_t"])
    f.plot((ts - t0) / 60, data["f1"], color=INK, lw=1.5)
    f.axhline(data["f1"][0], color=MUTED, ls="--", lw=1)
    f.axvline((T - t0) / 60, color=BAD, lw=1)
    f.set_ylim(0, 1)
    f.set_title("Edge F1 (dashed: Gemini prior)", fontsize=10.5, color=INK, loc="left")
    f.tick_params(labelsize=8, colors=MUTED)
    f.set_xlabel("min", fontsize=8, color=MUTED)
    log = fig.add_axes([0.765, 0.10, 0.23, 0.17])
    log.axis("off")
    log.text(0, 1, "S2 calls (Gemini)", fontsize=11, weight="bold", color=INK, va="top", transform=log.transAxes)
    shown = [c for c in calls if c["t"] <= T]
    for j, c in enumerate(shown[-3:]):
        ready = T >= c["t_ready"]
        s = (f"{clock(c['t'] - t0)} surprise {c['surprise']:.1f} → {c['latency']:.1f} s, "
             + (f"{c['graphs']} graphs, {c.get('added', 0)} new" if ready else "thinking…"))
        log.text(0, 0.75 - j * 0.25, s, fontsize=10, color=INK if ready else BAD, va="top", transform=log.transAxes)
    if not shown:
        log.text(0, 0.75, "none yet: S1 alone", fontsize=10, color=MUTED, va="top", transform=log.transAxes)

    # timeline of the current subgoal, truth against BIP
    tl = fig.add_axes([0.075, 0.025, 0.905, 0.055])
    for row, seq in ((1, data["band_true"]), (0, data["band_bip"])):
        for a, b, v in seq:
            if v is not None and v >= 0:
                tl.barh(row, (b - a) / 60, left=(a - t0) / 60, color=cols[v], height=0.8)
    for c in calls:
        tl.plot((c["t"] - t0) / 60, 1.75, marker="v", color=BAD, markersize=8, clip_on=False)
    tl.axvline((T - t0) / 60, color=INK, lw=2)
    tl.set_yticks([0, 1])
    tl.set_yticklabels(["BIP current", "true current"], fontsize=9, color=INK)
    tl.set_xlim(0, (t1 - t0) / 60)
    tl.tick_params(labelsize=8, colors=MUTED)
    for s in ("top", "right"):
        tl.spines[s].set_visible(False)
    fig.text(0.985, 0.002, "red ▼ = S2 call", fontsize=8, color=MUTED, ha="right")
    fig.savefig(out / f"{k:05d}.png")
    plt.close(fig)


def replay_data(cfg: Config, vid: str) -> dict:
    st = stream.read(cfg, vid)
    out = Replay(replace(cfg, trace=True, tier="action", s2="llm"), st).run()
    N, index, t0, t1 = len(st["nodes"]), {v: i for i, v in enumerate(st["nodes"])}, st["t0"], st["t1"]
    task = load(cfg.min_nodes)[vid]
    tr = out["trace"]
    truth = [(index[u], index[v]) for u, v in st["edges"]]
    tc = closure(list(range(N)), truth)
    f1 = []
    for s in tr:
        E = np.array(s["edges"])
        guess = closure(list(range(N)), {(u, v) for u in range(N) for v in range(N) if E[u, v] >= 0.5})
        tp = len(guess & tc)
        f1.append(2 * tp / max(1, len(guess) + len(tc)))
    times = np.arange(t0, t1, SPEED / FPS)
    owner = lambda t: index.get(task.current(t)) if task.current(t) else None
    true_cur = [owner(T) for T in times]
    true_next = [index.get(task.next(T)) if task.next(T) else None for T in times]
    done = [[task.last[v] <= T for v in st["nodes"]] for T in times]
    obj = {i: set(p) for i, p in enumerate(out_objects(cfg, st))}
    fixes = [(f["t"], st["objects"][f["object"]], sorted(v + 1 for v in range(N) if st["objects"][f["object"]] in obj[v]))
             for f in st["fixations"]]
    bands = lambda seq: [(a, b, v) for a, b, v in zip(times[:-1], times[1:], seq[:-1])]
    bip_cur = []
    tt = np.array([s["t"] for s in tr])
    for T in times:
        j = max(0, np.searchsorted(tt, T, side="right") - 1)
        c = np.array(tr[j]["cur"])
        bip_cur.append(int(np.argmax(c)) if c.sum() > 0 else None)
    return {"stream": st, "trace": tr, "times": tt, "calls": out["s2_calls"], "colors": colors(N),
            "pos": layers(N, truth).tolist(), "truth": truth, "true_cur": true_cur, "true_next": true_next,
            "done": done, "f1": f1, "f1_t": tt.tolist(), "frames": times,
            "actions": [(s["t"], s["text"]) for s in st["steps"]], "fixations": fixes,
            "completions": [(c["t"], index[c["node"]]) for c in st["completions"]],
            "band_true": bands(true_cur), "band_bip": bands(bip_cur)}


def out_objects(cfg: Config, st: dict) -> list[list[str]]:
    from . import llm
    p = llm.prior(cfg, st, llm.ids_for(cfg, st["nodes"], st["video"]))
    return [p["objects"].get(v, []) for v in st["nodes"]]


# -- part 2: the simulation --------------------------------------------------------------------------
def segments(trace: list, key: str) -> list[tuple]:
    """Runs of one value in a simulation lane: (start s, end s, subgoal or -1, "pause" | "fail" | "")."""
    out = []
    for tick in trace:
        state = "pause" if key == "g" and tick["pause"] > 0 else "fail" if key == "robot" and tick["fail"] else ""
        cell = (tick[key] if tick[key] is not None else -1, state)
        if out and out[-1][2:] == cell and out[-1][1] == tick["t"] - 1:
            out[-1] = (out[-1][0], tick["t"], *cell)
        else:
            out.append((tick["t"] - 1, tick["t"], *cell))
    return out


def sim_frame(args) -> None:
    k, T = args
    data, out = _DATA, _DATA["out"]
    st, cols, alone, robot = data["stream"], data["colors"], data["alone"], data["robot"]
    N, labels, Tmax = len(st["nodes"]), st["labels"], max(alone["time"], robot["time"])
    fig = plt.figure(figsize=(W, H), dpi=100, facecolor="white")
    fig.text(0.01, 0.955, f"Closed-loop simulation · {st['title']} · synthetic person, robot "
             f"{data['slowdown']:g}× slower", fontsize=17, color=INK, weight="bold")
    fig.text(0.01, 0.92, "The person follows the annotated graph and the recorded durations; the robot acts on the BIP belief: "
             "execute a subgoal, ask which subgoal is next, or wait, each second.", fontsize=11, color=MUTED)
    ax = fig.add_axes([0.12, 0.50, 0.85, 0.36])
    for row, tr, key in ((2, alone["trace"], "g"), (1, robot["trace"], "g"), (0, robot["trace"], "robot")):
        for a, b, v, state in data["lanes"][(row, key)]:
            if a >= T or (v < 0 and state != "pause"):
                continue
            c = "#DDDDDD" if state == "pause" else cols[v]
            ax.barh(row, (min(b, T) - a) / 60, left=a / 60, color=c, height=0.7,
                    hatch="///" if state == "fail" else None, edgecolor=BAD if state == "fail" else c, lw=0.5)
        prev = {"conflicts": 0, "violations": 0, "questions": 0}
        for tick in tr:
            if tick["t"] > T:
                break
            for what, mark, c in (("conflicts", "X", BAD), ("violations", "^", "#E67E22"), ("questions", "$?$", "#7D3C98")):
                if tick[what] > prev[what] and (row == 1 or what == "violations"):
                    ax.plot(tick["t"] / 60, row + 0.5, marker=mark, color=c, markersize=13, clip_on=False)
                prev[what] = tick[what]
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(["Person, alone", "Person, with robot", "Robot"], fontsize=12, color=INK)
    ax.set_xlim(0, Tmax / 60 * 1.02)
    ax.set_ylim(-0.6, 2.9)
    ax.axvline(T / 60, color=INK, lw=1.5)
    ax.set_xlabel("minutes", fontsize=10, color=MUTED)
    ax.tick_params(labelsize=9, colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    leg = fig.add_axes([0.12, 0.395, 0.85, 0.06])
    leg.axis("off")
    for v in range(N):
        leg.text((v % 5) / 5, 0.8 - (v // 5) * 0.55, f"\u25a0 {v + 1} {short(labels[v], 28)}", color=cols[v], fontsize=9.5, va="center")
    fig.text(0.12, 0.37, "grey = paused (question or conflict) · hatched = robot attempt that breaks an ordering · "
             "X conflict · ▲ violation · ? question", fontsize=9.5, color=MUTED)
    # decision log
    lg = fig.add_axes([0.03, 0.04, 0.6, 0.33])
    lg.axis("off")
    lg.text(0, 1, "Robot decisions", fontsize=12, weight="bold", color=INK, va="top", transform=lg.transAxes)
    ds = [tick for tick in robot["trace"] if tick["t"] <= T and tick["decision"] and tick["decision"][0] != "wait"]
    for j, tick in enumerate(ds[-7:]):
        kind, x, value, voi = tick["decision"][:4]
        if kind == "execute":
            s = f"{clock(tick['t'])}  execute {x + 1} {short(labels[x], 34)}   (value {value:+.0f} s)"
        else:
            ans = tick["decision"][4] if len(tick["decision"]) > 4 else None
            s = (f"{clock(tick['t'])}  ask \"which next: {', '.join(str(c + 1) for c in x)} or something else?\" → "
                 f"{ans + 1 if ans is not None else 'something else'}   (value of the answer {voi:+.0f} s)")
        lg.text(0, 0.85 - j * 0.12, s, fontsize=10.5, color=INK, va="top", transform=lg.transAxes)
    # counters
    ct = fig.add_axes([0.66, 0.04, 0.32, 0.33])
    ct.axis("off")
    last = next((tick for tick in reversed(robot["trace"]) if tick["t"] <= T), robot["trace"][0])
    rows = [("Conflicts", last["conflicts"]), ("Violations", last["violations"]), ("Questions", last["questions"])]
    for j, (name, val) in enumerate(rows):
        ct.text(0, 0.85 - j * 0.16, f"{name}: {val}", fontsize=13, color=INK, va="top", transform=ct.transAxes)
    if T >= Tmax:
        ct.text(0, 0.30, f"Alone {clock(alone['time'])}, with robot {clock(robot['time'])}", fontsize=14, weight="bold", color=INK, transform=ct.transAxes)
        ct.text(0, 0.16, f"{robot['time'] / alone['time']:.2f} of the time alone; test-set mean "
                f"{data['mean_ratio']:.2f}", fontsize=12, color=MUTED, transform=ct.transAxes)
    fig.savefig(out / f"{k:05d}.png")
    plt.close(fig)


def card(k: int, out: Path, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(W, H), dpi=100, facecolor="#F6F5F0")
    fig.text(0.07, 0.72, title, fontsize=30, weight="bold", color=INK)
    for j, s in enumerate(lines):
        fig.text(0.07, 0.6 - j * 0.07, s, fontsize=17, color=INK)
    fig.savefig(out / f"{k:05d}.png")
    plt.close(fig)


def main() -> None:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if a.startswith(("video=", "sim_video=", "sim_seed=", "slowdown=")))
    args = [a for a in sys.argv[1:] if a.split("=", 1)[0] not in opt]
    vid, sim_vid = opt["video"], opt.get("sim_video", opt["video"])
    slowdown = float(opt.get("slowdown", 1.5))
    cfg = parse(args + ["split=all"])
    out_dir = Path(cfg.runs) / "clips"
    frames = out_dir / f"{vid}_frames"
    shutil.rmtree(frames, ignore_errors=True)
    frames.mkdir(parents=True)

    data = replay_data(cfg, vid)
    data["view"], data["out"] = View(cfg, vid), frames
    (out_dir / f"{vid}_replay.json").write_text(json.dumps({"calls": data["calls"], "f1": data["f1"]}))
    sims = json.loads((Path(cfg.runs) / "sim_test" / "episodes.json").read_text())["episodes"]
    alone_t = {(r["video"], r["seed"]): r["time"] for r in sims if r["condition"] == "none"}
    mean_ratio = float(np.mean([r["time"] / alone_t[r["video"], r["seed"]] for r in sims if r["condition"] == "ours"]))
    costs = Costs(robot_slowdown=slowdown)
    cfg_sim = replace(cfg, pool=64, particles=32)
    seeds = [int(opt["sim_seed"])] if "sim_seed" in opt else range(3)
    runs = [(episode(cfg_sim, sim_vid, "none", s, costs, True), episode(cfg_sim, sim_vid, "ours", s, costs, True)) for s in seeds]
    alone, robot = min(runs, key=lambda ab: abs(ab[1]["time"] / ab[0]["time"] - mean_ratio))  # the most typical seed

    st, N = data["stream"], len(data["stream"]["nodes"])
    sim_st = stream.read(cfg, sim_vid)
    k = 0
    intro = ["One HD-EPIC test video, replayed at 8× speed as if live.",
             "S1, a particle filter per candidate task graph, updates on every narrated action, gaze fixation and completion.",
             "S2, Gemini, proposes graphs at the start and again only when S1 is surprised.",
             "Then a typical task in the closed-loop simulation, with a robot that acts on the belief."]
    for _ in range(4 * FPS):
        card(k, frames, "Fast-slow BIP over task graphs", intro)
        k += 1
    _DATA.clear()
    _DATA.update(data)
    jobs = [(k + i, i, T) for i, T in enumerate(data["frames"])]
    with ProcessPoolExecutor(32) as pool:
        list(pool.map(replay_frame, jobs, chunksize=4))
    k += len(jobs)
    Tmax = max(alone["time"], robot["time"])
    lanes = {(2, "g"): segments(alone["trace"], "g"), (1, "g"): segments(robot["trace"], "g"),
             (0, "robot"): segments(robot["trace"], "robot")}
    _DATA.clear()
    _DATA.update({"stream": sim_st, "colors": colors(len(sim_st["nodes"])), "alone": alone, "robot": robot, "slowdown": slowdown,
                  "mean_ratio": mean_ratio, "lanes": lanes, "out": frames})
    times = list(np.linspace(0, Tmax, 20 * FPS)) + [Tmax] * (4 * FPS)
    jobs = [(k + i, T) for i, T in enumerate(times)]
    with ProcessPoolExecutor(32) as pool:
        list(pool.map(sim_frame, jobs, chunksize=4))
    k += len(jobs)
    outro = ["Perception here is oracle: narrations, completions and object positions come from HD-EPIC's annotations.",
             "On raw video, the prompted VLM of RQ1 reads the current subgoal right 23% of the time; BIP reads 67% from narrations.",
             "The simulated person follows the model BIP assumes, so the robot numbers are an upper bound.",
             "Deployment needs the perception layer: action and completion recognition from the Aria stream."]
    for _ in range(5 * FPS):
        card(k, frames, "What this does not show yet", outro)
        k += 1
    mp4 = out_dir / f"{vid}.mp4"
    subprocess.run([str(Path(sys.executable).with_name("ffmpeg")), "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", str(frames / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", str(mp4)], check=True)
    shutil.rmtree(frames)
    print(mp4, f"{k} frames, {k / FPS:.0f} s; replay S2 calls {len(data['calls'])}; sim seed {robot['seed']}: "
          f"alone {alone['time']:.0f} s, with robot {robot['time']:.0f} s")


if __name__ == "__main__":
    main()
