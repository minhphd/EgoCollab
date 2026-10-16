# BIP architecture (locked)

A robot helper that models its partner: what the person is doing and will do next, and what the person believes
about where things are. One runtime, `helper/bip/live.py`, serves the simulator replays and the real robot. Its
settings are frozen in `helper/bip/locked.json` (`python -m helper.bip.lock` writes it).

## Data flow

```
 robot camera ─► SceneObs ─┐                               ┌─► execute subgoal ─► robot skills ─► Act (R)
 robot skills ─► Act (R) ──┤                               │
 glasses ──────► EgoFrame ─┼─► scene graph ─► task belief ─┼─► ask ─────────────► speech out
 recognizer ───► Act (H) ──┤        │         (S1, S2)      │
 microphone ───► Speech ───┘        └───────► partner's ────┴─► tell where X is ─► speech out
                                              beliefs
```

## Components

| Component | File | Rate | What it does |
|---|---|---|---|
| Inputs and outputs | `io.py` | | `SceneObs`, `EgoFrame`, `Act`, `Speech` in; `Command` out; the gaze-to-entity rule |
| Scene graph | `scene.py`, `track.py` | per input | where every object is, from detections and actions; subgoal completions |
| Task belief, fast (S1) | `s1.py`, `belief.py` | per event | particle filter over task graphs and the partner's progress |
| Task belief, slow (S2) | `s2.py` | on surprise | Gemini proposes task graphs; S1 scores them on the whole history |
| Partner's beliefs, fast | `mind.py` | per frame | for each object, where the person thinks it is |
| Partner's beliefs, slow | `s2.saw` | when unsure | Gemini judges from the person's view whether they saw the object move |
| Decision | `decide.py` | 1 Hz | execute, ask, tell or wait, each valued in seconds of the person's time |
| Runtime | `live.py` | | fast path inline, slow path in background threads |

## The partner's beliefs

- **Gaze as entities.** Each frame's gaze point names one entity (`io.fixated`): the nearest object within aoi of it,
  else the furniture under it. Set aoi to the eye tracker's error plus a small object's radius, in image units.
- **Fast.** For each object o, b(o) is a distribution over the places the person may think o is. Their own moves they
  always register. Another agent's move they register only by attending o while it is in their view: dwell of the gaze
  on o registers it with 1 - exp(-dwell / tau_reg), and o in view with the gaze elsewhere with a small hazard q_vis per
  second. Attending o in the robot's hand shortly before it reaches its new place shows where it went.
- **First look.** When the person starts looking for o, the dwell on o's candidate places over the first look_s of
  looking at them is shared out; the believed place draws the share w_seek beyond chance. This is evidence about b(o),
  applied once per search.
- **Slow.** When P(false belief) is between 0.25 and 0.75 as a search starts, Gemini reads the person's view during
  the move and returns P(they saw it), which joins b(o) as a likelihood.
- **Tell.** V(tell o) = P(false belief about o) × P(the person looks for o soon) × 20 s − 3 s. The person looks for
  o soon when their current or planned next subgoal (S1) uses o, or when a hint says so.

## Rates and latency

- EgoFrame: 10 to 30 Hz. The partner's beliefs and the gaze term of S1 update on every frame; each update is
  microseconds.
- SceneObs: 2 Hz. Act: on completion of each skill.
- Decisions: 1 Hz. Gemini calls: 1 to 2 s, in background threads; the loop never waits for them.

## What the robot side provides

| Input | Source | Notes |
|---|---|---|
| `EgoFrame.visible` | an open-vocabulary detector on the glasses' RGB | entity names must match the scene graph's IDs (3D association with the robot's map) |
| `EgoFrame.gaze` | the glasses' eye gaze, projected into the RGB image with the device calibration | Aria Gen2: on-device eye gaze (the bridge's `eye_gaze` callback) |
| `EgoFrame.image` | the glasses' RGB with the gaze drawn | kept 60 s for the slow path |
| `SceneObs` | the robot's detections registered to its map | (object, on or in, furniture), (agent, holding, object) |
| `Act (R)` | the robot's skill executor, on completion | Pick, Place, Open, Close |
| `Act (H)` | an action recognizer on the glasses (optional) | hand-object contact plus the detector |
| Clock | one clock for every input | the glasses' timestamps anchored to the robot's |
| `Command` | to the skill executor and text to speech | execute a subgoal, ask, tell |

## Frozen settings

`helper/bip/locked.json` holds every setting of `Config` (task belief), `MindConfig` (partner's beliefs) and `Costs`
(decision). A change to the architecture or to a frozen setting needs a new lock file and a rerun of the replays below.

## Checks before a deployment

- `python -m helper.bip.fb_live runs/fb/main2/streams <out>`: the live loop on the recorded false-belief tasks.
- `python -m helper.bip.fb_eval runs/fb/main2/streams <out> sweep=1`: the partner's beliefs, with and without gaze,
  and away from the simulator's settings.
- `python -m helper.bip.replay ...`: the task belief on the PARTNR recordings.
- `python -m helper.behavior.fb_analysis runs/fb/b1k_silent/streams <out>`: detection, anticipation and the ways of
  talking on the BEHAVIOR-1K recordings.
- `python -m helper.behavior.fb_noise runs/fb/b1k_silent/streams <out>`: the same under eye tracker errors (calibration
  offset, jitter, glitches) and readout radii.
