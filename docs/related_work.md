# Related work map

Working notes for the paper: model-based collaboration with a model of the partner, and whether seeing through the
partner's eyes makes collaboration better and more natural. Entries come from a web search on 2026-10-12; check
each citation against the paper before it goes into a draft.

## Positioning in one paragraph

Each part exists alone: LLM proposals filtered by Bayesian inverse planning (NOPA, AutoToM, CLIPS), cost-aware
asking (REVOIR, BALI, GOMA), egocentric goal inference (EgoToM, WAGIBench), gaze for intent (Huang and Mutlu,
Aronson and Admoni). No found work combines VLM-proposed task structures with orderings, inference over the
partner's progress (done, doing, next), the partner's egocentric actions and gaze as evidence, and one decision
over executing, waiting and asking, at household scale. The measured value of the partner's view is the
centerpiece; the rest supports it.

## Cognitive science anchors

| Work | Point we use |
|---|---|
| Tomasello et al., BBS 2005 | Shared intentionality: joint goals and joint attention set human collaboration apart |
| Sebanz, Bekkering, Knoblich, TiCS 2006; Vesper et al., Neural Networks 2010 | Joint action rests on shared task representations, prediction and monitoring of the partner |
| Baker, Jara-Ettinger, Saxe, Tenenbaum, Nat. Hum. Behav. 2017 | Bayesian theory of mind infers beliefs, desires and percepts; the partner's view turns percepts into observations |
| Land et al., Perception 1999; Flanagan and Johansson, Nature 2003 | Gaze reaches the next target before the hand; observers' eyes anticipate an actor's goal |
| Samson et al., JEP:HPP 2010; Keysar et al., Cognition 2003 | People compute what others see, yet use it unreliably |
| Brennan et al., Cognition 2008 | Shared gaze halved joint search time and beat shared gaze plus voice: seeing replaces talking |
| Clark and Krych, JML 2004; Gergle et al. 2004 | A shared visual space lets action replace language |
| Fussell et al., CHI 2003; Mueller et al., QJEP 2013 | Counterpoints: a head camera or gaze alone does not always help |

## Bayesian theory of mind and assistance

| Work | What it does | Difference |
|---|---|---|
| Baker et al., Cognition 2009 | Goal inference by inverse planning | Our core inference |
| SIPS, NeurIPS 2020 | Sequential Monte Carlo goal inference for boundedly rational agents | Fixed goal sets; no help |
| CLIPS, AAMAS 2024 | Goals from actions and instructions, LLM likelihoods, assistance | Flat goals, no asking, no partner view |
| NOPA, ICRA 2023 | Neural goal proposals, particle filter, helper in Watch-And-Help | Closest architecture; flat goals, third-person, no asking |
| Watch-And-Help, ICLR 2021 | Helping benchmark; success and speedup over a solo agent | Metrics to reuse |
| AutoToM, NeurIPS 2025 | LLM proposes the agent model, then Bayesian inverse planning | Observer only |
| Bayesian Delegation, CogSci 2020 | Inverse planning over which subtask the partner does | No language, no egocentric input |
| CIRL, NeurIPS 2016; Fern et al., JAIR 2014 | Assistance as a game or POMDP | Formal frame to state the problem in |
| AssistanceZero, ICML 2025 | Planning with a learned human model in Minecraft | ICML precedent for model-based assistance |
| Carroll et al., NeurIPS 2019; Strouse et al. (FCP), NeurIPS 2021 | Human models beat self-play; model-free population training | The model-based versus model-free debate |

## Cooperative LLM agents and communication

| Work | What it does | Difference |
|---|---|---|
| PARTNR, ICLR 2025 | Habitat 3.0 human-robot benchmark, LLM planners, no communication | Our testbed; their robots slow real people (4268 steps with a ReAct robot, 3047 alone) |
| Dongre and Hakkani-Tuer, SIGDIAL 2026 | Free dialogue on PARTNR | Conflicts drop, success drops (70.6% to 52.9%): the LLM should not choose what to say alone |
| LLawCo, ICML 2026 | Learned rules such as "talk when necessary" on PARTNR | Learned rules, no partner model |
| CoELA, ICLR 2024 | LLM agents that decide when to send messages | Our "VLM decides alone" baseline |
| GOMA, ICRA 2025 | Communication to align beliefs (KL proxy) in VirtualHome and Overcooked | Value proxy, not value of information; no partner view |
| BALI, HRI 2026 | LLM goals; asks when goal entropy beats a decaying cost | Entropy threshold, flat goals, turn-taking |
| REVOIR, arXiv 2026 | LLM-scored intent particles and one-step value of information | Closest on asking; one-step, no acting while asking, the user knows the whole task |
| Dong et al., ACL 2026 | One-step value of information minus a per-turn cost | Not embodied |
| Grand et al., ICLR 2026 | Asking versus acting in collaborative Battleship | ML-venue precedent |
| KnowNo, CoRL 2023; Introspective Planning, NeurIPS 2024 | Ask when a conformal set has several options | Ambiguity of the robot's own plan, no partner, no cost of asking |
| Ask-to-Act, arXiv 2025 | RL-trained MLLM asks minimal questions in Habitat | Robot works alone; ambiguity fixed at the start |
| UoT, NeurIPS 2024; BED-LLM, ICLR 2026 | Questions by expected information gain | Ignore what the robot does with the answer |
| Tellex et al., RSS 2014; Nikolaidis et al., THRI 2018; CommPlan, HRI 2020 | Robots that ask; choosing between acting and speaking | Classics to cite |
| Talker-Reasoner 2024; DPT-Agent, ACL 2025; Hi Robot, ICML 2025 | Fast and slow systems | Our split: the human model decides, the VLM talks |

## Egocentric perception and gaze

| Work | What it does | Use for us |
|---|---|---|
| Huang et al., Front. Psych. 2015; Huang and Mutlu, HRI 2016 | Gaze predicts the requested item 1.8 s before speech; anticipatory robot control | Gaze gives timeliness |
| Admoni and Srinivasa 2016; Aronson and Admoni, RSS 2022 | Gaze as POMDP observations for shared autonomy | Gaze channel precedent |
| Shen et al., Gaze Meets ML 2022 | Probabilistic intention in hierarchical tasks with gaze | Closest method on gaze |
| HoloAssist, ICCV 2023 | Instructor watches the performer's ego video with gaze and intervenes | Human version of our robot; data for when to intervene |
| EgoToM, 2025; WAGIBench, NeurIPS 2025 | Goal inference from egocentric video | Observer only |
| Ego-Exo4D, CVPR 2024 | Aria ego video with gaze, exo cameras, keysteps, task graphs | Real-data ladder: exo only, plus ego actions, plus gaze |
| HD-EPIC, CVPR 2025 | Aria gaze, fine-grained actions, recipes, digital twin | Closest to our scene-graph representation; loaders in the repo |
| Habitat 3.0, ICLR 2024 | Robot sees only its own sensors | Defines the robot-camera-only baseline |

## Experiments the literature implies

1. Information ladder with one planner: the robot's camera only; plus the person's actions and view; plus gaze;
   with the person alone and an oracle as bounds. Report time, conflicts, questions, and inference accuracy and
   lead time.
2. Controls for "their perspective, not more pixels": an extra robot viewpoint of matched coverage; time-shifted
   or shuffled ego streams.
3. Stratify by co-visibility and by how far candidate structures diverge.
4. Gaze realism: calibrated lead time and priming, noise, dropout, and an uninformative-gaze condition.
5. Asking: a Pareto front of task time against questions and talking time, against never ask, ask everything,
   entropy or information-gain rules, one-step value of information, and an LLM that decides alone.
6. Offline real data: the same ladder on Ego-Exo4D or HD-EPIC.
7. Report PARTNR's own metrics with a full-knowledge partner, for comparability.
