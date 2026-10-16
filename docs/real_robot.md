# Real-robot system test

The same partner model, decision rule and VLM as in simulation, fed by real perception: the robot's own camera
and the partner's Aria glasses. A lab member plays a scripted partner, as the simulated person does. Every
measure is about the robot, which matches the robot-centric style of RSS collaboration papers
(docs/rss_survey.md).

## Hypotheses

1. With the partner's glasses, the robot reads the partner's current subgoal more accurately and sooner than with
   its own camera, including a camera that watches the partner.
2. The glasses cut clashes and wrong placements; task time improves less.
3. Gaze adds little beyond the partner's view.

Prior user studies find that gaze rarely shortens task time and shows in errors and early intent instead, so
reading accuracy and wrong placements are the primary measures and task time is secondary.

## Task

- A shared tabletop with 6 to 10 objects, on the bimanual setup: set a table, or sort items into bins.
- The partner states the full instruction. Where each object goes and the order stay open to the robot, as in
  the PARTNR full-instruction tasks, where the VLM's plans match being told the subgoals.
- 4 task variants (object sets and targets).
- The Stretch runs one room-scale tidying task as a breadth demonstration, closer to PARTNR.

## The scripted partner

- Each trial has a script drawn from the task's subgoal graph: a valid order of subgoals, shown on a card.
- Scripted events test the robot: a change of order midway, a pause, reaching for the object the robot is about
  to take.
- When the robot asks, the partner answers from the card. The card covers part of the task.

## Conditions on the robot

The robot, the decision rule and the task stay fixed; only what the robot sees of its partner changes.

| Condition | What the robot sees |
|---|---|
| Robot camera | Its own camera |
| Watching | Its own camera, turned toward the partner whenever it is idle |
| Glasses | Its own camera, plus the partner's view: their actions and the object in hand |
| Glasses and gaze | As above, plus gaze |

The partner alone sets the baseline time. The simulator carries the wider ladder: every action from outside, a
late view, and the asking policies (never, always, a fixed rule, the VLM alone, the VLM with the partner model's
values, and conformal asking as in KnowNo).

## Design

- 3 partners × 4 tasks × 4 conditions × 2 repeats: 96 trials, 24 per condition, about 3 minutes each, conditions in
  counterbalanced order.
- Ground truth: the script, with subgoal start and end times from the glasses video, labelled offline.
- Analysis: mixed-effects models with partner and task as random effects; bootstrap confidence intervals over
  trials.

## Measures

From logs only:
- reading accuracy: the robot's belief about the partner's current subgoal against ground truth, every 0.5 s;
- subgoals named within 5 s;
- wrong placements;
- clashes: both reaching for the same object within 2 s;
- task time over the alone time, and robot idle time;
- questions asked.

No questionnaires and no analysis of the partner's behavior beyond ground-truth timing.

## Failures

Every failed or slow trial is assigned to one stage: perception, the partner model's belief, the VLM's plans, the
decision to ask, or execution. The paper reports the breakdown and a Limitations section.

## Perception

- Glasses: RGB, eye-tracking images and head pose, streamed over USB to the robot's computer. On first-generation
  Aria, gaze comes from Meta's gaze model run on that computer.
- Events: a hand-object contact detector and an open-vocabulary detector turn the partner's view into picks,
  places and the object in hand.
- The robot's camera runs through the same detectors.
- A marker on the table registers both cameras in the robot's frame; an object detected in both views confirms the
  match.
- The VLM runs in the background, so its 1 to 2 s calls never stall the robot.

## Offline check before the robot moves

- Record 10 sessions of a partner doing the tasks alone, with the glasses and the robot's camera.
- Run perception and the partner model on the recordings, and measure reading accuracy per condition.
- These recordings also give real per-view error rates for the simulator.
- Go on to the robot only if the glasses read the partner clearly better than the robot's camera.

## Optional: people's experience

If the Extension Charter asks for it: 6 to 10 participants who are not authors, unscripted, on robot camera
against glasses and gaze, with NASA-TLX. This needs ethics review.

## Data

Recordings show faces and contain gaze. They stay on Redwood.

## Timeline (RSS 2027)

| Dates | Step |
|---|---|
| Oct 13 to Oct 24 | Simulator: completion fix, viewpoint controls, asking baselines, more seeds |
| Oct 20 to Nov 7 | Glasses streaming, registration, detectors |
| Nov 3 to Nov 21 | Offline recordings and the offline check |
| Nov 17 to Dec 4 | Stage 1: hypotheses, protocol, simulator and offline results |
| Dec to Feb | Robot skills, the live loop, pilot trials |
| Feb 26 to Apr 16 | Trials as the Extension Charter requires, analysis, the final paper |

## Open

- Aria generation: on-device gaze or gaze computed on the robot's computer.
- Whether the bimanual setup already picks and places reliably.
