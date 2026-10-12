# EgoCollab: fast-slow Bayesian inverse planning over task graphs

A robot watches a person through Aria glasses and infers, online, the person's task graph (which
subgoals must come before which), the subgoal they work on now and the one they will work on next.
It then decides each second whether to take over a subgoal, ask a question or wait.

Inference is fast-slow Bayesian inverse planning (BIP):

- **S1 (fast):** a particle filter per candidate task graph, updated on every narrated action, gaze
  fixation and subgoal completion, in about 1.6 ms.
- **S2 (slow):** Gemini, which proposes candidate graphs from the task description at the start and
  revises them only when S1 is surprised, about 1.5 times per video.

The code replays HD-EPIC kitchen videos through this inference, scores it against annotated
subgoal graphs, and runs the robot's decision rule in a closed-loop simulation with synthetic people.

- Design: the BIP and RSS tabs of [Task-Graph Inference and Graph-Grounded Collaboration](https://claude.ai/artifact/BuvrnPpnCXtEXsXAhKPfUA)
- Results: sections G2 and G3 of [EgoCollab Results So Far](https://claude.ai/artifact/ApwJ5t52z1y5YSwtTuzngF), and the
  [Subgoal Model Progress](https://claude.ai/artifact/FKycbHwrsGyGByUGiC9ew6) deck

## Layout

```
helper/
  bip/
    config.py     every setting, overridable on the command line as key=value
    tasks.py      ground truth per video: subgoals, required orderings, which subgoal runs when
    gaze.py       gaze fixations on objects in 3D: the Aria gaze ray against tracked object positions
    stream.py     builds what S1 observes per video from annotations (oracle tiers), data/bip/stream/
    detect.py     YOLOE objects (prompted with Gemini's vocabulary) and YOLO hands on the frames
    perceive.py   turns detections and 2D gaze into causal observations (real tier), data/bip/real/
    view.py       2 fps JPEG frame cache, and frames with the gaze painted as a red glow
    gemini.py     Gemini on Vertex AI, with the service account read from a .env file
    llm.py        S2: the prior graph and object sets per video, and graph revisions on surprise
    model.py      S1: particle filter per graph over current subgoal, planned next subgoal, started set
    run.py        streaming replay: events through S1, S2 on surprise, an estimate every 10 s
    score.py      edge F1, current and next subgoal accuracy (BIP runs and RQ1 VLM runs alike)
    fit.py        gaze mixture weights by EM on the dev videos, and gaze look-ahead by time to a switch
    decide.py     the robot's rule: execute a subgoal, ask which subgoal is next, or wait
    sim.py        closed-loop simulation: a synthetic person and the robot on each test task
    clip.py       inspection video of one replay and one simulated episode
  datasets/
    episode.py    shared viewer episode format and graph utilities
    hd_epic.py    HD-EPIC download, build, and loaders for narrations and per-frame gaze
    posthoc_graph.py  post-hoc subgoal graphs: format, validation, action ownership
    sharepoint.py anonymous OneDrive client used by hd_epic.py
    ego_exo4d.py  Ego-Exo4D download and build (not used by BIP yet)
    industreal.py IndustReal build (not used by BIP yet)
  annotations/hd_epic/  one post-hoc task graph per HD-EPIC task, and GUIDELINES.md
  viewer/       local web viewer: video, gaze, true graph and BIP's inferred graph over time
  slurm/        bip_sweep.sh (parallel replays), cpu.sbatch, dataset downloads
  environment.yml
data/           hd_epic/, episodes/, frames448/, bip/ (streams, Gemini priors), ego_exo4d/, industreal/
runs/           bip/ (replays, simulations, clips), rq1/ (the earlier prompted-VLM baseline, still scorable)
models/hf/      Hugging Face cache (all-MiniLM-L6-v2 for narration embeddings)
models/yolo/    yoloe-26l-seg.pt, hand_yolov9c.pt (Bingsu/adetailer) and the YOLOE text encoder
```

## Setup

```bash
conda env create -f helper/environment.yml     # in home, never on scratch
conda activate egocollab
```

CHPC scratch deletes files by their dates after about 60 days, and conda unpacks files with their
package build dates, so an environment on scratch breaks at the next sweep.

**Gemini.** `config.env` points at `../MiniWeb-src/MiniWeb/.env`, which holds a Vertex AI service account:
`GOOGLE_CREDENTIALS_JSON` (multi-line JSON), `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`.
`gemini.py` reads it directly. Never print the file: the private key spans several lines. Models
that answer on this project: `gemini-3.5-flash` (default), `gemini-3.6-flash` and `gemini-3.1-pro-preview`.

**CHPC.** Run everything on a compute node, never on a login node. With an allocation held, a
command runs on it with

```bash
srun --jobid=<allocation> --overlap bash -c "cd $PWD; python -m helper.bip.run name=..."
```

`stream build` loads MiniLM from the local cache: set `HF_HOME=$PWD/models/hf HF_HUB_OFFLINE=1`.
A file edited on the login node can read as its old version on a compute node for tens of seconds;
check the node sees the edit (for example with `grep`) before launching, and check each run's
`config.json`.

## Data

### HD-EPIC

Public (CC BY-NC 4.0). The download fetches the annotations, the MP4 videos and the per-frame gaze
and head pose (HD-EPIC Intermediate Data); it skips the 1.9 TB of VRS files.

```bash
python -m helper.datasets.hd_epic download --all --workers 4    # every video, about 125 GB
python -m helper.datasets.hd_epic check                         # validate the post-hoc graphs
python -m helper.bip.tasks export                               # viewer episodes, dataset hd_epic_task
python -m helper.bip.view --shard 0/1                           # 2 fps frame cache, data/frames448
python -m helper.bip.stream build workers=16                    # observation streams, data/bip/stream
```

On a transfer node, `helper/slurm/download_hd_epic.sbatch` runs the download. The frame cache
splits over Slurm array tasks: `sbatch --array=0-11 helper/slurm/cpu.sbatch python -m helper.bip.view --shard ARRAY/12`.

**Subgoal graphs** (`helper/annotations/hd_epic/`, one JSON per task, rules in `GUIDELINES.md`) have
two levels. Every narrated action is a node with direct precondition edges; subgoals ("Pan hot and
oiled") group actions and carry their own required orderings, each with a one-line reason. `check`
verifies that every cross-subgoal action edge follows a subgoal path and every subgoal edge is
backed by an action edge. Graphs were drawn by LLM subagents; spot-check before trusting fine details.

**Task set.** `tasks.load(3)` keeps the 138 videos with at least 3 subgoals that have actions, in a
fixed shuffled order. The first 30 are the dev split (fitting parameters), the other 108 the test split.

### Ego-Exo4D and IndustReal

Both build viewer episodes in the same format; BIP does not use them yet.

- Ego-Exo4D needs a licence (https://ego4d.dev/request/ego-exo4d) and AWS keys from the approval
  email, stored with `aws configure`. `python -m helper.datasets.ego_exo4d download --num-takes 4`,
  then `build`. Its graphs are keystep orderings shared by every annotated take of a scenario.
  `helper/slurm/egoexo_cooking.sbatch` downloads the cooking takes with gaze.
- IndustReal (tabletop assembly, HoloLens 2 gaze) downloads with `helper/slurm/industreal.sbatch`
  from 4TU.ResearchData, then `python -m helper.datasets.industreal build --shard 0/8` encodes the
  frames and writes episodes.

## How BIP works

### Ground truth (`tasks.py`)

For each video, a `Task` holds the subgoals with actions in the video, their labels, the annotated
required orderings (u, v), and the first and last action time of each subgoal. A subgoal is done once
its last action ends. The current subgoal at t is the subgoal of the action running at t (a pause
inside one subgoal keeps it), and the next subgoal is the next different one to start.

### Observations: oracle tiers (`gaze.py`, `stream.py`)

`tier=step` and `tier=action` take perception from HD-EPIC's annotations, not from detectors:

- **Actions:** every narration inside the task, with its text. `tier=step` also gives S1 the
  subgoal each action belongs to; `tier=action` gives only the text, scored against each subgoal
  label by MiniLM cosine similarity: `L[v] = exp(beta_action * (sim[v] - max sim))`.
- **Gaze fixations:** HD-EPIC records where each moved object is picked up and put down in 3D,
  so every object has a resting place between moves. At 10 Hz, the fixated object is the one
  closest to the gaze ray, if within `fix_angle` (6 degrees). Gaze held `fix_min` (0.3 s) on one
  object is a fixation, reported once that time has passed. Objects within `fov` (50 degrees) of
  the camera axis are the visible set X.

`data/bip/stream/<video>.json` holds `steps` (t, end, text, owner), `completions`, `fixations`
(t, object, visible), `objects` (id to name), `sim` (narration to subgoal similarity), the subgoal
`nodes`, `labels`, annotated `edges`, and the task extent `t0`, `t1`.

### Observations: real tier (`llm.vocab`, `detect.py`, `perceive.py`)

`tier=real` uses only what a deployed system has: the egocentric video, the Aria 2D gaze point and
the task text.

1. **Vocabulary:** Gemini lists each subgoal's objects as short nouns a detector recognizes, from the
   task text alone (`llm.vocab`, about 21 names per video, cached in `data/bip/vocab/`). These names
   are the detector prompts and the object sets O(v).
2. **Detection** (`detect.py`): YOLOE-26L (open-vocabulary YOLO) prompted with the vocabulary finds
   objects, and a YOLOv9c hand detector finds hands, at `det_fps` (2) on 448 px frames. Weights live in
   `models/yolo/`. Both detectors take 21.7 ms per frame together on one RTX PRO 4000 (53 ms at worst),
   46 frames per second, so a live system keeps up with 23 times headroom. `det_fps=10` decodes the
   video instead of reading the frame cache.
3. **Observations** (`perceive.py`), causal as a live system: a frame counts only `det_latency`
   (0.05 s) after it is taken, and only frames already available are used.
   - Every `window` (3 s), an action event lists the objects a hand touched: the hand box grown by
     `hand_margin` overlaps the object box; objects larger than `max_area` (counters, stoves) never
     count. Its likelihood per subgoal is the product over handled objects x of
     `w_hand U_O(v)(x) + (1 - w_hand) U_X(x)`.
   - Gaze held `fix_min` on one object box (the smallest box holding the gaze point) is a fixation.

### Completions (`completion=`)

- `oracle`: each subgoal's completion is observed when its last action ends, standing in for a
  perfect scene-graph predicate.
- `latent`: nothing is observed. Each particle keeps its own completed set D; leaving a subgoal marks
  it complete with probability `p_done` (0.27 on dev: most switches in HD-EPIC are pauses), and a
  complete subgoal cannot be resumed. The estimate's done set is P(complete) >= 0.5.
- `noisy`: the latent model plus a detector that reports `det_rate` (70%) of completions, on average
  `det_delay` (10 s) late. A report marks the subgoal complete in every particle.

### S2 prior (`llm.prior`) and the graph pool (`run.sample_pool`)

One text-only Gemini call per video, cached in `data/bip/prior/<model>/<video>.json`. Gemini sees
the task title, the subgoals as shuffled IDs S1 to Sn, and the scene's object names. A JSON
schema restricts its reply to valid IDs and names. It returns:

- required orderings `{before, after, p}`, where A > B means A must be complete before B starts
  because of how the task works, not habit;
- the objects of each subgoal, O(v), used by the gaze model.

The pool is the most probable acyclic graph (orderings with p >= 0.5) plus up to `pool` - 1
distinct samples that keep each ordering with probability p and skip cycles. Any graph's prior is
`log_prior`: independent orderings with Gemini's probabilities, 0.02 for orderings Gemini never named.

### S1 filter (`model.py`)

Each graph k has `particles` particles, each a current subgoal g, a planned next subgoal n and a
started set S (bitmask). The completed set C is observed and shared. Graphs are stored as ancestor
bitmasks, so "is v executable" is one bitwise test.

**Choice weights** (`options`), the policy pi over subgoals to work on:

| Subgoal v | Weight |
| --- | --- |
| complete, or taken by the robot | 0 |
| started, not complete | `resume` (1) |
| not started, every ancestor complete | 1 |
| not started, some ancestor incomplete | `delta` (0.05) |

`rule=started` relaxes "ancestor complete" to "ancestor started".

**Events**, each multiplying particle weights and adding the graph's predictive probability of the
event to its log marginal likelihood `logZ[k]`:

- `action(L)`: the subgoal transition. For each particle, over every possible next subgoal g':

  ```
  switch = plan * 1[g' = n] + (1 - plan) * pi(g' | not g)
  T      = stay * 1[g' = g] + (1 - stay) * switch     if g is incomplete, else switch
  q      = T * L
  ```

  The particle is weighted by sum(q) and g' is sampled from q, so the action informs the move. If g
  changes, a new plan n is drawn from pi with g' counted as done. Graph evidence enters only
  through pi: a graph that forbids an observed start pays `delta`.
- `fixation(u, noise)`: `L = w_cur U_O(g)(x) + w_next U_O(n)(x) + w_noise U_X(x)`, where
  `U_S(x) = 1[x in S and X] / |S and X|`. EM on dev sets the weights to 0.17, 0 and 0.83 with
  oracle gaze, and 0.105, 0 and 0.895 with YOLO gaze.
- `completion(v)`: weight `1 - eps_done` if g = v, else `eps_done`, then C gains v.
- `detected(v)`: a completion detector's report, possibly late: C gains v, no evidence about g.
- `answer(a, choices, eps)` and `robot_done(v)`: used by the simulation.

Each graph resamples when its effective sample size falls below half. **Readouts:** graph weights
are softmax(log prior + logZ); the current and next subgoal are particle marginals of g and n;
`edge_probs()[u, v]` is the weighted share of graphs whose closure has u before v; `done_probs()`
gives P(complete) per subgoal over each particle's completed set.

### Surprise and S2 revisions (`run.py`, `llm.revise`)

A reference filter with only the empty graph runs alongside the pool. On every action,

```
surprise = log P(event | empty graph) - log sum_k w_k P(event | graph k)
```

When surprise exceeds `surprise` (1.4), no call is pending and `cooldown` (30 s) has passed, S2 is
called. With `s2=llm`, Gemini sees the three leading graphs with weights, S1's log of reads
("1:20 started S3; 2:40 S3 complete"), the surprising event and the last 9 s of frames with gaze,
and returns up to `revisions` (3) corrected graphs. The graphs join the pool only once the call's
latency has passed, and each new graph is scored on the whole stored history before it joins. S2
only proposes; S1 decides every graph's weight. `s2=repair` replaces Gemini with a rule that drops
the orderings into each observed start whose source was not done; `s2=never` keeps the prior pool.

### Estimates and scoring (`run.estimate`, `score.py`)

Every `grid` (10 s) the replay writes an estimate: done subgoals, the current subgoal, the top 3
next subgoals and every ordering with p >= 0.05. `score.py` pools over updates:

- **Edge F1:** orderings with p >= 0.5 against the truth, both on transitive closure, at the prior,
  online and in the last tenth of the task. `truth=annotated` (default) uses the annotated
  orderings, `consistent` only those the person never broke, `followed` every ordering the person
  followed (u complete before v started).
- **Current**, **next top-1**, **next top-3**, **progress** (done subgoals).
- Baselines: the prior held fixed, and the true order of first starts as a chain.

### Decision rule (`decide.py`)

On the belief, for each subgoal v:

```
V(execute v) = P_exec(v) [ (1 - Y[v]) D(v) - beta Y[v] ] - beta_prec (1 - P_exec(v))
V(ask)       = E_answer[ best V after the answer ] - kappa
V(wait)      = 0
```

- `P_exec(v)`: probability that every required predecessor of v is done, over graph weights.
- `Y[v]`: probability the person works on v while the robot would: v is their current subgoal,
  or their plan and their current subgoal ends before the robot would finish v.
- `D(v)`: the person's time saved, v's duration less any wait for a slower robot once the person
  runs out of work that does not depend on v.
- The question is "which subgoal next: the top 3 or something else?". Its value comes from
  reweighting the particles by each possible answer.
- Costs (`Costs`): `beta` 60 s per conflict, `beta_prec` 120 s per violation, `kappa` 5 s per
  question, `robot_slowdown` 1.5.

### Simulation (`sim.py`)

Per test task and seed, at 1 s ticks:

- **Person:** follows the annotated graph. Each subgoal takes its recorded action time (10 to 600 s).
  The next subgoal is drawn uniformly among the executable ones not taken by the robot, and the robot
  is noticed 2 s after it starts. A question costs a 3 s pause; answers are true 95% of the time.
- **Robot sensing:** a step label every 3 s, correct `acc` of the time (0.9); fixations from the
  fitted mixture with a 2 s look-ahead; completions per `completion` (with `p_done=1.0` for `latent`,
  since the simulated person never pauses a subgoal). A conflict shows the robot what the person is doing.
- **Robot thinking:** S2 is the repair rule with Gemini's measured 5.6 s latency, so the simulation
  makes no API calls.
- **Robot acting:** a robot attempt that breaks a true ordering fails and wastes its time.
- **Conditions:** `none` (person alone), `ours`, `ours_nogaze`, `never_ask`, `entropy_ask`,
  `no_anticipation`. The person's random draws are shared across conditions.

## Running

```bash
# replays on the 108 test videos (step oracle or narration only; S2 llm, repair or never)
python -m helper.bip.run name=t_act_llm split=test tier=action s2=llm
python -m helper.bip.score name=t_act_llm split=test truth=annotated
python -m helper.bip.score runs=runs/rq1 name=all_27b split=test       # the RQ1 VLM baseline

# several variants in parallel on one node, each scored
helper/slurm/bip_sweep.sh "t_step_llm split=test tier=step" "t_act_nogaze split=test gaze=False"

# real perception: vocabulary, detection (one GPU per shard), observations, fitting
CUDA_VISIBLE_DEVICES=0 python -m helper.bip.detect --shard 0/4          # also calls llm.vocab
python -m helper.bip.perceive build workers=32
python -m helper.bip.fit split=dev tier=real        # w_hand, YOLO gaze weights, p_done
python -m helper.bip.run name=r_latent split=test tier=real completion=latent \
    w_hand=0.082 w_cur=0.105 w_next=0.0 w_noise=0.895 p_done=0.274

# fitting and diagnostics on the 30 dev videos
python -m helper.bip.fit split=dev                  # oracle gaze EM weights and look-ahead
python -m helper.bip.llm gemini-3.5-flash gemini-3.6-flash   # prior edge F1 by model, test videos

# closed-loop simulation and the inspection clip
python -m helper.bip.sim name=sim_test split=test seeds=3 slowdown=1.5
python -m helper.bip.sim name=sim_a0.5_latent split=test seeds=3 acc=0.5 completion=latent p_done=1.0
python -m helper.bip.clip video=P04-20240414-175337 sim_video=P01-20240204-145458 sim_seed=0

# viewer: pick "HD-EPIC tasks (subgoal graphs)", a video, then an estimate such as bip_t_act_llm
python -m helper.viewer.server --port 8766          # http://127.0.0.1:8766
```

On a remote machine, forward the viewer port: `ssh -L 8766:127.0.0.1:8766 <host>`.

## Outputs

- `runs/bip/<name>/config.json`: the settings of the run.
- `runs/bip/<name>/<video>.json`: `records` (one estimate per grid time: `t`, `estimate` with
  `done`, `current`, `next`, `edges`), `s2_calls` (`t`, `surprise`, `latency`, `t_ready`, `graphs`,
  `added`, tokens), `prior_call`, `pool` (final number of graphs), `s1_ms`, and `trace` (the belief
  after every event) when `trace=True`.
- `runs/bip/<name>/summary_<split>_<truth>.json`: the scores, with curves by tenth of the task.
- `data/episodes/hd_epic_task/<video>/pred_bip_<name>.json`: the same estimates, for the viewer.
- `runs/bip/<sim name>/episodes.json`: one row per task, condition and seed: `time`, `conflicts`,
  `violations`, `questions`, `s2_calls`, `robot_subgoals`.
- `runs/bip/clips/<video>.mp4`: the inspection clip.

## Settings (`helper/bip/config.py`)

| Group | Setting | Default | Meaning |
| --- | --- | --- | --- |
| Data | `split`, `dev` | test, 30 | dev = first 30 videos of the fixed order, test = the other 108 |
| Data | `videos`, `only_videos` | 0, () | cap or pick videos |
| Data | `grid` | 10 | s between written estimates |
| Perception | `tier` | action | step: subgoal of each action given; action: narration text only; real: detections |
| Perception | `completion` | oracle | oracle, latent (inferred by S1) or noisy (a lagging detector) |
| Perception | `p_done`, `det_rate`, `det_delay` | 0.36, 0.7, 10 | latent and noisy completion |
| Real tier | `det_model`, `hand_model`, `det_fps`, `det_latency` | yoloe-26l-seg.pt, hand_yolov9c.pt, 2, 0.05 | detectors and timing |
| Real tier | `window`, `hand_margin`, `max_area`, `w_hand` | 3, 0.15, 0.2, 0.5 | handled-object events (fit `w_hand` on dev) |
| Model | `rule` | complete | an ordering lets its target start once the source is complete (or started) |
| Model | `stay`, `plan`, `resume` | 0.95, 0.5, 1.0 | keep the subgoal per action; a switch goes to the plan; weight of resuming |
| Model | `delta` | 0.05 | weight of a start that breaks an ordering |
| Model | `beta_action`, `eps_step`, `eps_done` | 20, 0.001, 0.05 | narration sharpness; step oracle error; stray completion |
| Gaze | `gaze`, `w_cur`, `w_next`, `w_noise` | True, 0.17, 0, 0.83 | mixture weights (EM on dev) |
| Gaze | `fix_angle`, `fix_min`, `fov` | 6, 0.3, 50 | fixation and visibility thresholds |
| S1 | `pool`, `particles` | 128, 64 | graphs from the prior; particles per graph |
| S2 | `s2` | llm | llm, repair or never |
| S2 | `surprise`, `cooldown`, `revisions` | 1.4, 30, 3 | trigger threshold; s between calls; graphs per call |
| S2 | `model`, `thinking`, `s2_frames`, `s2_frame_gap` | gemini-3.5-flash, low, 4, 3 | Gemini model and its images |
| Scoring | `truth` | annotated | annotated, consistent or followed |

`gaze=False` removes fixations from S1 and the gaze glow from S2's images.

## Viewer and episode format

The viewer (`helper/viewer/`, Python standard-library server, plain JavaScript) plays any episode with
its gaze, subgoal and action timelines and the task graph coloured by status. For HD-EPIC tasks it
also shows a chosen estimate file: the inferred graph at the time each estimate would have been
available, with edges green when right, red when wrong and dashed when missed, and an F1 curve.
Click a node or a timeline to seek. Keys: space, arrows (5 s), `,` and `.` (one frame), `n` and `p`
(next and previous episode).

An episode lives in `data/episodes/<dataset>/<id>/`:

- `video.mp4`: a symlink to the ego RGB video.
- `episode.json`: `nodes` (id, label, group), `edges` ([u, v]: v requires u), `segments` (node,
  start, end, kind), `actions`, `done_before`, `graph_source`, `edge_notes`, `clip`, `video_id`,
  `task`, `node_kind`, `stats`, `video` (size, fps, duration).
- `gaze.json`: `fps`, `t0` and `points`, one `[x, y]` in [0, 1] or `null` per sample.
- `pred_<name>.json`: estimates from a replay, in the format of `runs/bip/<name>/<video>.json`.

A new dataset needs only a build script that writes this format with `episode.write_episode`.

## Current results (108 HD-EPIC test videos)

| Perception | Edge F1 | Progress | Current | Next top-1 | Next top-3 | Gemini calls per video |
| --- | --- | --- | --- | --- | --- | --- |
| Gemini prior alone | 0.76 | | | | | 1 |
| Step oracle, oracle completions | 0.74 | 1.00 | 1.00 | 0.30 | 0.68 | 1.5 |
| Narrations, oracle completions | 0.69 | 1.00 | 0.67 | 0.25 | 0.59 | 1.6 |
| Narrations, completions inferred | 0.65 | 0.67 | 0.46 | 0.12 | 0.34 | 9.3 |
| YOLO, oracle completions | 0.73 | 1.00 | 0.43 | 0.22 | 0.58 | 0 |
| YOLO, completion detector (70%, 10 s late) | 0.72 | 0.82 | 0.30 | 0.15 | 0.44 | 0 |
| **YOLO, completions inferred (fully real)** | 0.71 | 0.68 | 0.24 | 0.10 | 0.30 | 0 |
| Same at 10 fps detection | 0.69 | 0.69 | 0.25 | 0.12 | 0.34 | 0 |
| RQ1 prompted VLM (Qwen3.5-27B) on raw video | 0.67 | 0.64 | 0.23 | 0.08 | 0.17 | 113 |
| Chance | | 0.58 | 0.10 | 0.10 | 0.30 | |

In simulation, the robot saves 16% of the person's time when 1.5 times slower than the person and
23% when as fast; anticipating the next subgoal cuts conflicts from 0.82 to 0.54 per episode. With
completions inferred, it still saves 12 to 13% even at near-chance step labels, by acting on the
graph and the subgoal durations.

**Limits of this version:**

- With real perception, BIP reads the current subgoal no better than the raw-video VLM (0.24 against
  0.23), and its next-subgoal prediction is at chance. Completion detection decides the next subgoal;
  action recognition decides the current one. YOLO hand contacts are weak evidence: the handled
  object is only 1.5 times likelier to belong to the current subgoal than to another, and detecting
  at 10 fps instead of 2 adds only 0.01 to 0.03.
- Watching lowers the graph below Gemini's prior. HD-EPIC subgoals are done in bouts, and 29% of
  starts break an annotated ordering. S1 can drop orderings but has no move that adds one. With
  real perception, Gemini revisions never fire.
- Gaze adds little: it points to the next subgoal only in the last 2 s before a switch, and helps the
  current subgoal only when actions are weak.
- The simulated person follows the model S1 assumes and never pauses a subgoal, so the simulation is
  an upper bound: its belief accuracy stays at 0.49 or more, against 0.24 on real video. The robot
  can act before any observation and break an ordering the prior missed.
- Questions cover only the next subgoal; ordering questions ("does B need A first?") would target
  the violations.

Code removed in the streamlining (earlier pipelines, the RQ1 VLM runner, notes and reports) is in
`~/archive/EgoCollab-before-streamline-2026-10-11.tar.gz`.
