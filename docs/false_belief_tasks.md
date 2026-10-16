# False-belief tasks

A Sally-Anne task in a home. The person puts an object X on a surface A. Later the robot moves X into a closed
container B in the same room (open, place inside, close). The person comes back for X and first looks where they
think it is. The query is at the end of that first look: where does the person think X is?

## Conditions

| Condition | During the move | Belief |
|---|---|---|
| left | in another room | false |
| turned | in the room, facing away | false |
| watched | in view, gaze follows X | true |
| missed | in view, gaze elsewhere | false |
| back_watched | away for the pick, back in view for the placement, gaze follows X | true |
| back_missed | as back_watched, gaze elsewhere | false |

watched and missed (and the two back_ conditions) share one video: the scripts and the simulator's random draws
are the same, and only the gaze differs.

## The person's belief

The person learns where X is only by attending X while it is in view. Coming back, the first look goes to the
believed place with probability 0.8, else to the other place first (anticipatory looking).

## What every method may read

- The ground truth of the world: every object's place at every moment, from the start state and its changes.
- Each agent's actions, from a fixed set: navigate, pick, place, open, close.
- The person's view: frames every 0.5 s, the entities in view, and the gaze point with the entity it falls on.

## Checks

A task is kept only if its recording shows what its condition claims: X in view for at least 1 s of the move
(watched, missed, back_), gaze on X at the placement (watched) or never on X (missed), X out of view during the
move (left, turned), X hidden once the container is closed, and both A and B in view at the query.

## Design

A design fixes X, A, B, the watching spot, a furniture in another room and the robot's parking spot. Each design is
validated by a dry run of every scripted move and a survey of watching spots, then runs in all six conditions.

## PARTNR

- 28 designs over 10 scenes, validated by a dry run and a survey of standing points; 168 tasks recorded, 113 pass
  their checks (runs/fb/main2).
- Code: helper/partnr/fb_layout.py, fb_design.py, fb_record.py, fb_clip.py.

## Results (runs/fb/eval)

| Method | Balanced accuracy | AUROC |
|---|---|---|
| Partner's beliefs (helper/bip/mind.py), with gaze | 1.00 | 1.00 |
| Partner's beliefs, first look only | 0.62 | 0.76 |
| Partner's beliefs, no gaze | 0.60 | 0.76 |
| Gemini 3.5 Flash, frames with gaze drawn and the world's true state | 0.68 | 0.68 |
| Gemini 3.5 Flash, plus the ordered list of what the person looked at | 0.92 | 0.92 |

- The simulated person's attention and the partner model share one rule, so the model's perfect score is close to
  circular; with its settings away from the simulator's it scores 0.78 to 1.00 (median 0.96).
- Without gaze, watched and missed (one video) cannot be told apart.
- The first look alone is weak: free glances often rest on things near the believed place.
- The live loop (helper/bip/live.py) tells the person where the object is in all 76 false-belief tasks and in none
  of the 37 true-belief tasks.

## BEHAVIOR-1K

The same six conditions in OmniGibson (Isaac Sim 5.1), with photorealistic homes, on the lab's RTX 5090
(helper/behavior: sim.py, fb_design.py, fb_record.py, fb_analysis.py; runs/fb_b1k on that machine).

- The person: a head camera with a standing mannequin body just behind it, turned with the head, walking the scene's
  traversable map; their view is RGB.
- What they see and where their gaze lands come from physics raycasts from the eye: an object is in view when a ray
  reaches a point on its box first; the gaze label is the small object within 0.04 of the gaze point, else whatever the
  ray through the gaze point hits. No gaze pixels reach any method. (Isaac Sim's segmentation annotators crash on
  some views.)
- The robot: a visual-only Stretch moved along the map (it never pushes furniture); it carries the object at 1.1 m and
  works from spots where the person can see what it holds. Objects go on surfaces with OmniGibson's OnTop; in a
  container they rest at the middle of its closed box, held there every frame; containers open and close with Open.
- BIP's live loop runs during each task, reading the person's view, the gaze labels and both agents' actions. When it
  tells the person where the object is, the person goes straight to the container (closed loop). A second pass records
  a silent robot.
- After the query the person goes for the object: to the believed place, and, finding it empty, looks around and goes
  to the container.
- Designs: 43 (a surface and an openable container 1 to 3.5 m apart in one room of 11 home scenes; the object spawned
  from the BEHAVIOR library by room type), validated by a survey of standing points by raycast.
- Each scene runs in its own process, retried after a crash; tasks already saved are skipped.

## Communication policies (helper/behavior/fb_analysis.py)

| Policy | What the robot says | When |
|---|---|---|
| silent | nothing | |
| announce (delegation) | "I'll put the mug in the cabinet." | before it moves the object |
| ask (clarification) | "Shall I put the mug in the cabinet?", the person answers | before it moves the object |
| tell by value (correction) | "The mug is in the cabinet." | when P(false belief) × P(looks for it soon) × 20 s > 3 s |
| tell by value, no gaze | the same | the same rule with P(false belief) from the model without gaze |
| always tell | "The mug is in the cabinet." | after every move |

Each is scored by the person's time to get the object, wrong trips, the person's talking time and needless utterances,
from the scene's walking distances; the closed-loop recordings check the model.

## BEHAVIOR-1K results (runs/fb/eval/b1k_*.json)

Passes main_v2 (the person hears BIP) and silent_v2 (the robot silent): 444 tasks recorded, 350 pass their checks
(181 and 169; 37 and 35 designs; 11 homes). Clips: runs/fb/b1k_main_v2/clips, runs/fb/b1k_silent_v2/clips.

| Inputs (all with the world's true state), silent pass | Balanced accuracy | AUROC | Destination right at the query |
|---|---|---|---|
| Position only, no head camera | 0.50 | 0.64 | 72% |
| Head camera view, no gaze | 0.50 | 0.77 | 70% |
| Head camera and gaze, first look only | 0.76 | 0.90 | 83% |
| Head camera and gaze, dwell on entities (BIP) | 1.00 | 1.00 | 100% |
| Gemini 3.5 Flash, frames | 0.53 | 0.58 | 71% |
| Gemini 3.5 Flash, plus the entities gazed at, in order | 0.92 | 0.94 | 95% |
| Walking heading | | | 43% |

- BIP names the destination 8.2 s before arrival. Gemini with the gaze list is less calibrated (Brier 0.046 against
  0.001).
- Live, BIP tells after all 128 missed moves and none of the 53 seen ones, 0.5 s after the move. The person then has
  the object 4.7 s after the query; with the robot silent, 11.6 s.

| Policy (main pass, 181 tasks) | Person's time (s) | Wrong trips | Needless utterances |
|---|---|---|---|
| silent | 7.5 | 71% | 0% |
| announce | 6.1 | 0% | 29% |
| ask | 8.1 | 0% | 29% |
| tell by value | 5.3 | 0% | 0% |
| tell by value, no gaze | 6.1 | 0% | 29% |
| always tell | 6.1 | 0% | 29% |

The walking model predicts the live retrieval (3.57 s against 3.61 s measured).

## Gaze readout under tracker error (helper/behavior/fb_noise.py)

Calibration offset, per-frame jitter and glitches (a random entity for a 0.2 s window) on the silent pass, 3 seeds.
With a 6° readout radius, balanced accuracy stays at 0.98 or better under every error tested; with 2.5°, a 4° offset
drops it to 0.84. A 0.2 s fixation filter needs the wide radius (0.50 to 0.74 under offset or jitter with 2.5°).
For eye-tracking glasses: radius = tracker error plus 2°, filter optional. A gaze that ignores the object stays 9.5°
from it here; a real person's may pass closer, which caps the radius.

## Known limits

- The simulated attention and the partner model share one rule, so BIP's perfect score is close to circular.
- The head video can keep a fading image of a moved object at its old place for several seconds; the geometry
  (gaze labels, checks, BIP) is unaffected, the Gemini frames are.
- About 3% of head frames are near-uniform (the camera against or inside geometry while walking).
