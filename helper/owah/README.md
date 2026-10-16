# Online Watch-And-Help (NOPA)

A household assistance benchmark in the VirtualHome simulator: a helper agent (Bob) watches a main agent
(Alice) act, infers her goal online and helps her finish faster. From Puig, Shu, Tenenbaum and Torralba,
*NOPA: Neurally-guided Online Probabilistic Assistance for Building Socially Intelligent Home Assistants*
([arXiv 2301.05223](https://arxiv.org/abs/2301.05223)); code: https://github.com/xavierpuigf/online_watch_and_help.
It builds on *Watch-And-Help* (Puig et al., ICLR 2021; https://github.com/xavierpuigf/watch_and_help).

## Layout

Everything is installed in `third_party/owah/` (git-ignored) and `~/.conda/envs/owah/`:

| Path | What |
| --- | --- |
| `online_watch_and_help/` | the NOPA code: agents (MCTS, goal inference), arenas, test scripts in `testing_agents/` |
| `online_watch_and_help/dataset/` | task files copied from Watch-And-Help: `test_env_set_help.pik`, `train_env_set_help.pik`, `env_to_pred.json`, `match_demo_test.json`, `object_info.json`, `watch_scenes_split.json` (the NOPA repo's own dataset is "coming soon") |
| `virtualhome/` | the VirtualHome Python API on its `wah` branch; the code expects it next to `online_watch_and_help/` |
| `executable/` | the VirtualHome v2.3.0 Linux simulator (`linux_exec.v2.3.0.x86_64`) and `virtualhome.sif`, an Apptainer image (Ubuntu 22.04, Mesa) that runs it headless |
| `~/.conda/envs/owah/` | Python 3.7 with the repo's pinned packages; torch 1.6 is the CPU build |

## Status

The simulator runs headless and the API works (reset, scene graphs). The Watch-And-Help task scenes
do not load: they were made for VirtualHome v2.2.0, and v2.3.0 lacks several of their prefabs
(`Cupcake_1`, `Juice_1`, `Pudding_1`, `Apple_1`), so `expand_scene` fails with a NullReferenceException.
Without fixed prefab names, v2.3.0 places juice but leaves cupcakes, pudding and apples unplaced, and the
full task scene still fails.

## Setup

```bash
helper/owah/setup.sh          # from the repo root, on a login node: clones, downloads, environment; reruns skip finished parts
```

- virtual-home.org no longer serves any simulator (every link returns 404), and Watch-And-Help asks for
  v2.2.0, which is no longer published. The setup takes VirtualHome v2.3.0 with two rendering patches from
  the TieJean/virtualhome fork (release `sim-v2.3.0-p1p2`, sha256 checked), plus its Mesa container.
  The official v2.0.0 build is still on GitHub (1.5 GB) if v2.3.0 turns out to differ.
- The environment is made with micromamba (`~/.local/bin/micromamba`), because the conda install under
  `/scratch/.../miniconda` is broken: scratch purged part of it (`ImportError: frozendict`).
- torch 1.6 has no build for Blackwell or Ampere GPUs that fits this stack, so the agents run on CPU.

## Running

On a compute node (CHPC: never on a login node):

```bash
helper/owah/start_sim.sh 8080 &                                  # the simulator, headless, on port 8080
~/.conda/envs/owah/bin/python helper/owah/smoke_test.py 0 30      # API check, then episode 0 for 30 steps
```

`start_sim.sh` starts Xvfb and the simulator inside the container. Python code connects with
`UnityCommunication(port="8080")`, or through `UnityEnvironment(..., use_editor=True)`, which attaches to the
simulator on port 8080 instead of launching one. One simulator serves one environment; start more on
other ports (`start_sim.sh 8081 100`) for parallel episodes.

The repo's own test scripts (`testing_agents/test_hp.py` and others) run every test episode five
times and launch the simulator themselves from `--executable_file`; run them from
`third_party/owah/online_watch_and_help/`.
