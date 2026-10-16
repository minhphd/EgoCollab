"""How much recognizer error BIP tolerates. Synthetic episodes in which the step lists and the graph prior are right,
so the only errors are the simulated recognizers': the verb read right with probability `verb` on clips that complete
a step, a step read on clips that complete none with probability `false_step`, and the object read right with
probability `obj` (None: no object reads). The read verb gets 0.4 and each other 0.15, a ratio of 2.7 (1 nat); `bank` instead draws the real
recognizer's calibrated outputs (stream3, held-out clips of the same true verb), as helper/collab/sense.py does.

A task: 9 subgoals (HD-EPIC's median), each pick(o_v) then place(o_v, f_v), with 8 random required orderings; the
person takes them in a random order the orderings allow: idle clips before the pick, the pick and the place (in one
clip 35% of the time, as in HD-EPIC), idle clips after. The object read is the step's object, or the held one on a
clip with no step. Two fixations a clip: on the current subgoal's objects 30% of the time, the next one's 10%.

    python -m helper.collab.tolerance
    python -m helper.collab.tolerance show=1 seed=0   # one episode with the real recognizer's outputs, drawn
"""

from __future__ import annotations

import json
import multiprocessing
import sys
import tempfile
from pathlib import Path

import numpy as np

from . import bip
from .sense import BANK, Recognizer

ROOT = Path(__file__).resolve().parents[2]
SUB, EDGES, EPISODES, CONF = 9, 8, 100, 0.4
GOOD = {"verb": 0.9, "false_step": 0.05, "obj": 0.9}
REAL = {"verb": 0.5, "false_step": 0.45, "obj": 0.7}  # stream3 on test clips; obj is BIP's assumed detector


def episode(rng) -> tuple[set, list, list]:
    """(required orderings, the person's order, clips as (true current subgoal, verb, object, receptacle))."""
    order = list(rng.permutation(SUB))
    pairs = [(order[i], order[j]) for i in range(SUB) for j in range(i + 1, SUB)]
    true = {pairs[i] for i in rng.choice(len(pairs), EDGES, replace=False)}
    seq, done = [], set()
    while len(seq) < SUB:
        ready = [v for v in range(SUB) if v not in done and all(u in done for u, w in true if w == v)]
        v = int(rng.choice(ready))
        seq.append(v)
        done.add(v)
    clips = []
    for v in seq:
        clips += [(v, "other", None, None)] * rng.poisson(1)
        if rng.random() < 0.35:
            clips.append((v, "place", f"o{v}", f"f{v}"))
        else:
            clips += [(v, "pick", f"o{v}", None)] + [(v, "other", f"o{v}", None)] * rng.poisson(1) + [(v, "place", f"o{v}", f"f{v}")]
        clips += [(v, "other", None, None)] * rng.poisson(2)
    return true, seq, clips


def run(job: tuple[dict, int], trace: dict | None = None) -> dict:
    """One episode's accuracy; `trace`, if given, receives the episode and, per clip, the reads and the belief."""
    rates, seed = job
    rng = np.random.default_rng(seed)
    true, seq, clips = episode(rng)
    every = [x for v in range(SUB) for x in (f"o{v}", f"f{v}")]
    task = {"title": f"synthetic task {seed}", "nodes": [f"v{i}" for i in range(SUB)],
            "labels": [f"put o{i} on f{i}" for i in range(SUB)], "objects": {f"v{i}": [f"o{i}", f"f{i}"] for i in range(SUB)}}

    def gemini(text, schema, P):  # the prior: true orderings 0.8, two false ones 0.3; pick -> place step lists
        s2 = bip.S2(task, P)
        if "edges" in schema["properties"]:
            false = {(int(u), int(v)) for u, v in rng.choice(SUB, (2, 2))} - true - {(v, u) for u, v in true}
            return {"edges": [{"before": s2.id[u], "after": s2.id[v], "p": q} for (u, v), q in
                              [(e, 0.8) for e in true] + [(e, 0.3) for e in false if e[0] != e[1]]]}, 0.0
        if "graphs" not in schema["properties"]:
            return {"subgoals": [{"id": s2.id[v], "steps": [{"verb": "pick", "object": f"o{v}", "optional": False},
                                                            {"verb": "place", "object": f"o{v}", "target": f"f{v}", "optional": False}],
                                  "undo": {"verb": "pick", "object": f"o{v}"}} for v in range(SUB)]}, 0.0
        return {"graphs": []}, 0.0
    bip.gemini = gemini
    with tempfile.TemporaryDirectory() as tmp:
        bel = bip.Belief(task, bip.Params(seed=seed, cache=tmp))
    bank = Recognizer(BANK, rates["obj"], rng) if rates.get("bank") else None
    read = lambda x: None if rates["obj"] is None else x if x is None or rng.random() < rates["obj"] else every[rng.integers(len(every))]
    hit, done_ok, none_ok, fin, t = [], [], [], set(), 0.0
    for v, verb, o, f in clips:
        nxt = seq[seq.index(v) + 1] if seq.index(v) + 1 < SUB else None
        fx = []
        for _ in range(2):
            r = rng.random()
            pool = [f"o{v}", f"f{v}"] if r < 0.3 else [f"o{nxt}", f"f{nxt}"] if r < 0.4 and nxt is not None else every
            fx.append(pool[rng.integers(len(pool))])
            bel.fixation(t, fx[-1], every)
            t += 1.0
        if bank:
            q = bank.verbs(verb)
        else:
            if verb == "other":
                seen = ["pick", "place"][rng.integers(2)] if rng.random() < rates["false_step"] else "other"
            else:
                seen = verb if rng.random() < rates["verb"] else [c for c in bip.VERBS if c != verb][rng.integers(4)]
            q = np.full(len(bip.VERBS), (1 - CONF) / 4)
            q[bip.VERBS.index(seen)] = CONF
        ro, rf = read(o), read(f)
        bel.clip(t, q, ro, rf, every)
        t += 1.0
        fin |= {v} if verb == "place" else set()
        if trace is not None:
            trace.update(true=sorted(true), seq=seq)
            trace.setdefault("clips", []).append({"t": t, "v": v, "verb": verb, "read": bip.VERBS[int(np.argmax(q))], "obj": o,
                                                  "obj_read": ro, "rec_read": rf, "gaze": fx, "fin": sorted(fin),
                                                  "current": bel.current().tolist(), "done": bel.done().tolist()})
        hit.append(int(np.argmax(bel.current())) == v)
        done_ok.append(np.mean([(p > 0.5) == (u in fin) for u, p in enumerate(bel.done())]))
        none_ok.append(1 - len(fin) / SUB)
    return {"current": float(np.mean(hit)), "done": float(np.mean(done_ok)), "done_none": float(np.mean(none_ok))}


def summary(rates: dict, rows: list[dict]) -> dict:
    out = {**rates, **{k: float(np.mean([r[k] for r in rows])) for k in rows[0]}}
    out["current_se"] = float(np.std([r["current"] for r in rows]) / np.sqrt(len(rows)))
    print(f"{json.dumps(rates)}: current {out['current']:.3f} +- {out['current_se']:.3f}, done {out['done']:.3f} "
          f"(none done {out['done_none']:.3f})", flush=True)
    return out


def main():
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    if args.get("show") == "1":
        seed = int(args.get("seed", 0))
        episode_figure(dict(REAL, bank=True), seed, ROOT / f"runs/collab/tolerance_episode{seed}.png")
        return
    grid = {"verb": (0.3, 0.5, 0.7, 0.9, 1.0), "false_step": (0.0, 0.15, 0.3, 0.45, 0.6), "obj": (0.2, 0.4, 0.6, 0.9, 1.0)}
    sweeps = {k: [dict(GOOD, **{k: x}) for x in v] for k, v in grid.items()}
    sweeps |= {f"{k}, no object reads": [dict(GOOD, obj=None, **{k: x}) for x in grid[k]] for k in ("verb", "false_step")}
    real = {"real-like": REAL, "real recognizer (bank)": dict(REAL, bank=True)}
    real |= {f"fix {k}": dict(REAL, **{k: GOOD[k]}) for k in ("verb", "false_step", "obj")}
    conds = [r for rows in sweeps.values() for r in rows] + list(real.values())
    with multiprocessing.get_context("fork").Pool(16) as pool:
        res = pool.map(run, [(c, s) for c in conds for s in range(EPISODES)])
    by = {i: summary(c, res[i * EPISODES:(i + 1) * EPISODES]) for i, c in enumerate(conds)}
    it = iter(range(len(conds)))
    out = {"from good": {k: [by[next(it)] for _ in rows] for k, rows in sweeps.items()},
           "from real": {name: by[next(it)] for name in real}}
    path = ROOT / "runs/collab/tolerance.json"
    path.write_text(json.dumps(out, indent=1))
    plot(out, path.with_suffix(".png"))


def plot(res: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = {"verb": "verb read right on clips with a step", "false_step": "step read on clips with none",
             "obj": "object read right"}
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.9), gridspec_kw={"width_ratios": [1, 1, 1, 1.3]})
    good = res["from good"]
    for ax, k in zip(axes, names):
        for rows, ls, tag in ((good[k], "-", ""), (good.get(f"{k}, no object reads"), "--", ", no object reads")):
            if rows:
                x = [r[k] for r in rows]
                ax.errorbar(x, [r["current"] for r in rows], yerr=[r["current_se"] for r in rows], marker="o", ls=ls,
                            color="#14213D", label="current subgoal" + tag)
                ax.plot(x, [r["done"] for r in rows], marker="s", ls=ls, color="#E07B39", label="done set" + tag)
        ax.axhline(good[k][0]["done_none"], ls=":", color="#E07B39", lw=1, label="done set: nothing done")
        ax.set_xlabel(names[k])
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("BIP accuracy, from a good recognizer")
    axes[0].legend(loc="lower right", fontsize=7)
    real = res["from real"]
    x = np.arange(len(real))
    axes[3].bar(x - 0.2, [r["current"] for r in real.values()], 0.4, color="#14213D", label="current subgoal")
    axes[3].bar(x + 0.2, [r["done"] for r in real.values()], 0.4, color="#E07B39", label="done set")
    plain = {"fix verb": "verb right 90%", "fix false_step": "false steps 5%", "fix obj": "object right 90%"}
    axes[3].set_xticks(x, [plain.get(n, n).replace(" (", "\n(") for n in real], fontsize=8)
    axes[3].set_ylim(0, 1.15)
    axes[3].set_title("from a real-like recognizer, one error fixed")
    axes[3].legend(loc="upper center", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def episode_figure(rates: dict, seed: int, path: Path) -> None:
    """One episode: BIP's P(current subgoal) and P(done) at each clip, subgoals in the person's order, over what the
    person did and what the sensors read; printed as a table too."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr = {}
    res = run((rates, seed), tr)
    seq, C = tr["seq"], tr["clips"]
    row, x = {v: k for k, v in enumerate(seq)}, np.arange(len(C))
    names = [f"{k + 1}. v{v}" for k, v in enumerate(seq)]
    color = {"pick": "#3B6FB6", "place": "#E07B39", "open": "#6A7179", "close": "#6A7179", "other": "#E3E6E4"}
    fig, ax = plt.subplots(3, 1, figsize=(16, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1.1, 3]})
    for a, key, title in ((ax[0], "current", "P(current subgoal)"), (ax[2], "done", "P(done)")):
        a.imshow(np.array([[c[key][v] for v in seq] for c in C]).T, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        a.set_yticks(range(len(seq)), names)
        a.set_ylabel(title)
    ax[0].plot(x, [row[c["v"]] for c in C], "o", ms=5, mfc="none", mec="#E07B39", mew=1.5, label="the person's current subgoal")
    ax[0].plot(x, [row[int(np.argmax(c["current"]))] for c in C], ".", ms=4, color="#14213D", label="BIP's top subgoal")
    ax[0].legend(loc="lower left", fontsize=9, framealpha=0.9)
    first = {v: next(i for i, c in enumerate(C) if v in c["fin"]) for v in seq}
    ax[2].plot([first[v] for v in seq], range(len(seq)), "|", ms=14, mew=2.5, color="#E07B39", label="the subgoal's place: done from here")
    ax[2].legend(loc="lower left", fontsize=9, framealpha=0.9)
    for i, c in enumerate(C):
        ax[1].scatter(i, 2, marker="s", s=60, color=color[c["verb"]])
        ax[1].scatter(i, 1, marker="s", s=60, color=color[c["read"]])
        if c["read"] != c["verb"]:
            ax[1].scatter(i, 1, marker="x", s=30, color="#14213D")
        if c["obj"] is not None:
            ax[1].scatter(i, 0, marker="o", s=40, color="#14213D" if c["obj_read"] == c["obj"] else "none", edgecolors="#14213D")
    ax[1].set_yticks([0, 1, 2], ["object read (hollow: wrong)", "verb read (x: wrong)", "what happened"])
    ax[1].set_ylim(-0.6, 2.6)
    for key, lab in (("pick", "pick"), ("place", "place (with its pick if\nboth fall in one clip)"), ("open", "open or close"), ("other", "no step")):
        ax[1].scatter([], [], marker="s", color=color[key], label=lab)
    ax[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9, frameon=False)
    ax[2].set_xlabel("clip (3 s each; two gaze fixations before each)")
    order = ", ".join(f"v{u} before v{v}" for u, v in tr["true"])
    fig.suptitle(f"Synthetic episode {seed}: 9 subgoals, each pick o_v then place it on f_v; the real recognizer's verb outputs, "
                 f"objects read right 70%\nRequired orderings: {order}\nBIP's top subgoal right on {res['current']:.0%} "
                 f"of clips; done set right on {res['done']:.0%} of subgoal-clips", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"{'clip':>4s} {'person on':>9s} {'happened':>12s} {'verb read':>9s} {'object read':>11s} {'gaze':>9s}   BIP top (P)   done (P > 0.5)")
    for i, c in enumerate(C):
        top = int(np.argmax(c["current"]))
        step = c["verb"] if c["verb"] != "other" else "nothing"
        step += f" {c['obj']}" if c["verb"] != "other" else ""
        print(f"{i:4d} {'v' + str(c['v']):>9s} {step:>12s} {c['read']:>9s} {str(c['obj_read'] or '-'):>11s} {' '.join(c['gaze']):>9s}   "
              f"v{top} ({c['current'][top]:.2f})   {' '.join(f'v{v}' for v in range(SUB) if c['done'][v] > 0.5) or '-'}")


if __name__ == "__main__":
    main()
