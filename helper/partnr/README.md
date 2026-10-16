# PARTNR

A benchmark for planning and reasoning in human-robot collaboration: a Spot robot and a humanoid partner
rearrange objects in furnished HSSD apartments from a free-form instruction ("Place the candle and candle
holder on the white table in the living room. Move the plant to the same table."). From Chang et al.,
*PARTNR: A Benchmark for Planning and Reasoning in Embodied Multi-Agent Tasks*
([arXiv 2411.00081](https://arxiv.org/abs/2411.00081)); code: https://github.com/facebookresearch/partnr-planner.
It runs in Habitat 3.0 (habitat-sim 0.3.3, habitat-lab 0.3.3).

## Layout

| Path | What |
| --- | --- |
| `third_party/partnr/partnr-planner/` | the code (`habitat_llm`), with habitat-lab and transformers-CFG as submodules (git-ignored) |
| `third_party/partnr/partnr-planner/data` | a symlink to `data/partnr/`, described in [data/partnr/README.md](../../data/partnr/README.md) |
| `~/.conda/envs/partnr/` | Python 3.9, habitat-sim 0.3.3 (headless, bullet), torch 2.7.1 with CUDA 12.8, git-lfs |

## Setup

```bash
helper/partnr/setup.sh       # from the repo root: code, submodules, environment; reruns skip finished parts
helper/partnr/download.sh    # on a compute node: 26 GB of scenes, objects, episodes and skill checkpoints
```

- torch is 2.7.1 with CUDA 12.8 instead of the upstream 2.4.1 with CUDA 12.4, which has no Blackwell
  kernels; it runs on the RTX PRO 4000 (Blackwell), A800 and H200 nodes.
- `requirements.txt` pins protobuf 3.20.1, so the setup pins tensorboard 2.19 (2.20 and later need
  protobuf 6 and break `import habitat_baselines`) and pillow 10.4 (habitat-sim's pin).
- None of the Hugging Face datasets are gated; no token is needed. The task data is CC BY-NC.

## Running

On a GPU node (habitat-sim renders through EGL):

```bash
helper/partnr/smoke_test.sh 2 runs/partnr/smoke   # heuristic planner, no LLM, first 2 val_mini episodes
```

The heuristic planner (`baselines/heuristic_full_obs.yaml`) splits the task between both agents with full
observability and no LLM. It solves both smoke episodes (100% complete, about 575 simulator steps, 25 s
each on one RTX PRO 4000) and writes results, traces and side-by-side third-person videos to
`runs/partnr/smoke/results/`. LLM planners take `llm@...=openai_chat` or `inference_mode=hf` overrides
(see the upstream README).

## BIP testbed

PARTNR stands in for the EgoCollab setting: the humanoid is the person wearing the glasses, its head
camera is the ego video, gaze is synthesized, and the Spot robot helps. BIP itself lives in `helper/bip`
and knows nothing about PARTNR; it reads streams.

| File | What |
| --- | --- |
| `env.py` | the environment for one worker: our controllers drive both agents through PARTNR's oracle skills; only the person's head RGB and object-ID cameras and two third-person cameras render |
| `tasks.py` | subgoals: one per success condition, with the Navigate, (Open,) Pick, Navigate, Place script PARTNR's resolver picks; true orderings from the temporal constraints |
| `person.py` | the scripted person: a random allowed subgoal at a time, detours (25%) and pauses (20%), never drops a held object, a failed "next to" placement falls back to a plain one |
| `gaze.py` | gaze from the head camera's object masks, calibrated on HD-EPIC (priming before pick-ups and put-downs, dwell times, a mixture over current, next and other objects) |
| `robot.py` | robot policies: `idle`, `bip` (S1 belief and the act-or-wait rule), `graph` (belief-free control: a random ready subgoal), `oracle` (true state) |
| `record.py` | runs episodes and writes streams, 1 fps ego frames with gaze for S2, and head and third-person videos |
| `export.py` | viewer episodes from streams |
| `clip.py` | clips: ego view with gaze, both third-person views, and the belief per subgoal against the truth |
| `belief_grid.sh` | offline replays over perception tiers and ablations |
| `loop_score.py` | closed-loop scores against the person alone |
| `loop_grid.sh` to `loop_grid5.sh` | the closed-loop rounds: robots and perception (1), prior model, true graph and conflict cost on ordered tasks (2), looking first (3), a person who never notices the robot (4), the violation cost against speed (5) |

```bash
# on a GPU node, from the repo root
python -m helper.partnr.record name=rec_val_mini workers=8                  # person alone; rec_* names -> data/bip_partnr
helper/partnr/belief_grid.sh data/bip_partnr/streams runs/partnr/belief     # offline belief, CPU
python -m helper.partnr.record name=loop_bip robot=bip runs=runs/partnr     # closed loop -> runs/partnr/loop_bip
python -m helper.partnr.loop_score data/bip_partnr/streams runs/partnr/loop_bip runs/partnr/loop_graph
python -m helper.partnr.export data/bip_partnr                              # viewer episodes
python -m helper.partnr.clip data/bip_partnr/streams/val_mini_3.json clip.mp4 runs/partnr/belief/<run>/val_mini_3.json
```

Results and clips: the "BIP in PARTNR" section of the progress deck; runs in `runs/partnr/` (`belief2`,
`loop` to `loop5`), clips in `runs/partnr/clips/`.

PARTNR details the testbed handles: an is_next_to proposition is not its own task step (it joins the
placement of the same object, on the reference's receptacle); `proposition_satisfied_at` only records the
first time a proposition held, so done-ness is read from the current state; a Pick or Place in progress
finishes before a subgoal ends; "next to" placements can still find no valid spot, and then fall back to a
plain one; the humanoid's head camera looks straight ahead, so the recorder turns it toward the target
within 3 m of it.
