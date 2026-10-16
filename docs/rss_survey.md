# Evaluation depth at RSS for a collaboration system

What RSS asks of systems papers, and the evaluation depth of comparable accepted papers (RSS 2022 to 2026,
plus closely related CoRL, IROS and arXiv work). Counts come from the papers' text; "about" marks arithmetic
from stated per-condition counts.

## RSS rules

- RSS 2027 runs in two stages. Stage 1, due December 4, 2026: a 6-page extended abstract in which "the problem
  statement, hypotheses, and methodology must be fully fleshed out", while "experiments or empirical validations
  will be missing or preliminary". The review becomes an Extension Charter (February 26, 2027) listing the
  experiments the final paper must add; the 8-page paper is due April 16, 2027.
- "Submissions will not be rejected for yielding negative experimental results."
- Systems contributions must be "properly evaluated against alternatives"; papers need a Limitations section
  (2025 author information). Results "need to be quantitative" (2026 review guidelines).
- No rule on human subjects or on real-robot experiments.

Sources: [RSS 2027 call](https://roboticsconference.org/information/cfp/),
[RSS 2025 author information](https://roboticsconference.org/2025/information/authorinfo/),
[RSS 2026 call](https://roboticsconference.org/2026/information/cfp/).

## Two evaluation styles

| Style | Who plays the person | Real trials | Baselines | Statistics | Ethics review |
|---|---|---|---|---|---|
| Robot-centric claims | the experimenter, an operator or a few lab members | 20 to 60 per condition | 3 to 6, often never-ask or always-ask bounds | rarely | not stated |
| Claims about people's experience | 8 to 14 participants who are not authors | 1 per condition per person, 32 to 140 in all | 2 to 4 conditions | Friedman, Mann-Whitney or mixed-effects; NASA-TLX | stated in every 2025 and 2026 paper |

Robot-centric examples: [RCIP](https://arxiv.org/abs/2403.15959) (RSS 2024),
[UPS: when to act, ask or learn](https://arxiv.org/abs/2602.22474) (RSS 2026),
[KnowNo](https://arxiv.org/abs/2307.01928) (CoRL 2023), [OK-Robot](https://arxiv.org/abs/2401.12202) (RSS 2024, Stretch,
10 homes, 171 trials), [Constraint-aware intent](https://arxiv.org/abs/2409.00215) (RSS 2024, 3 subjects).

Experience examples: [Knowing when not to help](https://www.roboticsproceedings.org/rss22/p119.pdf) (RSS 2026, 10
students), [ASQ](https://arxiv.org/abs/2605.22986) (RSS 2026, 12 people, mixed-effects, 2 of 3 hypotheses
unsupported), [FEAST](https://arxiv.org/abs/2506.14968) (RSS 2025 award, a 5-day in-home study with 2 care
recipients, a lessons-learned section).

Simulation appears in 7 of 11 RSS collaboration papers surveyed. Code release is explicit in 3 of 11.

## Robots that use the partner's view or gaze

| Paper | Comparison | Result |
|---|---|---|
| [EDITH](https://arxiv.org/abs/2606.10276) (2026), bimanual arms, Aria | first-person view and gaze together against robot-view frames | success drops 33.8% without them; view and gaze never separated |
| [Chan et al.](https://arxiv.org/abs/2208.11856), 37 people | AR and gaze, crossed | time unchanged (178 to 181 s); placing errors 0.19 against 1.27 only with both |
| [Aronson and Admoni](https://roboticsproceedings.org/rss18/p025.html) (RSS 2022), 12 people | joystick, gaze, merged | gaze alone slowest; 35% of gaze trials never fixate the goal |
| [FAM-HRI](https://arxiv.org/abs/2503.16492), Aria | speech, gaze, both | success 0.90, 0.65, 0.94 |
| [GEAR](https://arxiv.org/abs/2507.18947) (IROS 2025), 30 people | gaze against a touchscreen | time not significant; workload lower only in the complex task |

No paper found compares the robot's camera, the partner's view, and the view with gaze in a real
collaboration. Gaze rarely shortens task time; its effects show in errors, workload and how early intent is read.

## Aria glasses in live systems

- First-generation Aria computes no gaze on the device: EDITH and FAM-HRI stream the eye-tracking images to a PC
  and run Meta's gaze model there. The second generation advertises on-device eye tracking.
- Personal gaze calibration exists only in Meta's offline processing; live systems use the factory calibration.
- No paper reports glasses-to-robot latency. The VLM is the bottleneck: 1.3 to 2.2 s in FAM-HRI.
- USB streaming is more stable than Wi-Fi. Streams are synchronized by timestamp and resampled to the control rate.
- Mapping the first-person view to the robot: markers break when the person moves away from the robot camera's
  viewpoint; detecting the object in both views catches errors.
