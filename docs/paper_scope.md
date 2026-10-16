# Paper scope (draft)

Working title: **Seeing Through Your Partner's Eyes: Model-Based Collaboration with a Model of the Human**

Target: ICML 2027 (RSS as the fallback). Related work: [related_work.md](related_work.md).

## Question

People collaborate by tracking what a partner intends, and they do it better when they share the partner's
perspective: shared gaze halves joint search time and makes words less necessary (Brennan et al. 2008). A robot
helper can share the partner's perspective literally, through the person's smart glasses. The paper asks whether
a robot that sees through its partner's eyes collaborates better and more naturally, and why.

## Abstract (draft)

> A robot that helps a person must know what the person is doing and what remains to be done, without being
> told and without interrupting them. We cast helping as model-based collaboration: the robot keeps a model of
> its partner, Bayesian inverse planning over task structures (which subgoals exist, their order, and which the
> person has done, is doing and will do next), and chooses to act, wait or ask by the value of each in the
> person's time. A large vision-language model proposes the candidate task structures from what the person
> says and does, and words the robot's questions; the partner model decides what is worth asking. The model's
> evidence includes the partner's own perspective: the view, actions and gaze from the person's glasses.
> On 26 long household tasks in Habitat 3.0 with a person simulated from human data, the robot's own camera
> sees almost none of the person's actions. Given those actions, the task takes 0.43 of the person's time alone
> against 0.53 (the oracle: 0.38); the person's view adds the object in hand, which cuts clashes from 0.09 to
> 0.03 per task, and gaze makes the reading timely without shortening the task; the same view 10 s late loses
> most of the gain. From a casual request, seeing through the person's eyes takes the task from 0.94 to 0.81 of
> the time alone without a word, more than asking does; asking with the partner model's values raises the share of
> the robot's placements that are right from 0.42 to 0.58, while a VLM deciding alone asks nearly three times as often
> and is slower than never asking. On real cooking video (Ego-Exo4D, 68 takes),
> the same kind of partner model reads the person's current step from their view (0.19, and 0.30 zoomed to
> where people work) while the best-placed outside camera barely beats the step order alone (0.15 against
> 0.14); moment-to-moment gaze adds little beyond where people usually look.

## Formal setting

- **Task.** A structure $G = (V, E)$: subgoals $V$ (an object kind to a furniture, possibly next to another
  kind) and orderings $E$. The person knows $G$ (or part of it, and works out the rest); the robot does not.
- **Partner state.** $z_t = (D_t, g_t, n_t)$: subgoals done, the one in progress, the next. The person acts by
  a boundedly rational policy $\pi_H(a \mid z, s, G)$; this policy is the robot's model of its partner.
- **Robot belief.** $b_t(G, z) \propto P(G \mid \text{language}) \, P(o_{1:t} \mid G, z)$, with candidate $G$
  proposed by the VLM and $z$ tracked by a particle filter per candidate.
- **Observations.** $o_t$ from the robot's camera, and optionally from the partner's view: the person's actions
  $a^H_t$ as recognized from their glasses, their view of the scene, and gaze fixations. The ladder of
  observation models is the experimental variable.
- **Decision.** Execute subgoal $v$ with value $V(v)$ (time saved against conflicts, ordering violations and a
  wrong placement, under $b_t$), wait (0), or ask $q$ with value $\mathrm{VOI}(q) - c(q)$, where $\mathrm{VOI}$ is
  the expected gain in the robot's decision values over the rest of the task and $c$ is the person's talking time.
- **Claim to make precise.** Without the partner's view, the action likelihood $P(a^H_t \mid g_t)$ drops out
  almost always (the robot's camera saw none of the person's actions in 47 of 67 episodes) and $b_t(z)$ falls
  back to the prior predictive of $\pi_H$, corrected only by objects found moved. The gain in time is larger
  where the two work close enough for the robot to catch some actions (0.14 against 0.02 of the time alone),
  where reading the partner decides clashes.

- **What theory fixes, and what the experiment measures.** The ladder's observation channels are nested (the
  robot's camera; plus every action of the person; plus their view; plus gaze), and a delayed view is a garbling of
  the timely one. By Blackwell (1953), the value of a Bayes-optimal robot can only rise along the ladder and fall
  with delay. The experiment measures how much of that value an approximate model realizes, and where it
  saturates: gaze raises how well the robot reads the person without changing what it does, so its value in task
  time is near zero, while seeing the person's actions changes decisions (clashes) and carries the gain.

## Contributions

1. A partner model for household collaboration: Bayesian inverse planning over VLM-proposed task structures
   with orderings, tracking the partner's progress, with one decision over acting, waiting and asking.
2. The measured value of the partner's perspective: a ladder from the robot's own camera to the person's view,
   actions and gaze, with controls for timing (a delayed view) and for gaze leaking the target (uninformative
   gaze).
3. Communication driven by the partner model: the VLM decides whether and what to ask from the model's values;
   compared with a fixed rule, a VLM alone, asking for everything and never asking.
4. A simulated partner calibrated on human data (PARTNR human episodes, HD-EPIC gaze), who may know only part of
   the task and answers with a small LM; and offline validation on real egocentric video.

## Figures

1. The loop: perception, the partner model (fast), the VLM (slow), the decision.
2. The ladder: reading accuracy and task time per observation model (runs/analysis/ladder_report.py).
3. Help against interruption: task time against the person's talking time per asking policy
   (runs/analysis/casual_figure.py, casual_2x2.py).
4. Belief over time on one episode, with and without the partner's view (clip frames).
5. Real data: subgoal reading on HD-EPIC by evidence (hand contacts, narrations, gaze).

## Status (2026-10-13)

| Piece | Status |
|---|---|
| Full-instruction tasks: the plans match being told the subgoals; asking adds nothing | done, one seed |
| Partner-view ladder with controls (late view, gaze not drawn to targets), 26 tasks, 3 seeds | done (runs/partnr/final) |
| Casual request: seeing crossed with asking, who decides to ask, with a wrong placement valued as saving nothing and a model of what the person knows | done (runs/partnr/casual2): seeing 0.12 of the time alone, asking 0.02 to 0.05; asking raises right placements 0.42 to 0.58 |
| The remaining gap from a casual request (0.77 at best against 0.47 told the subgoals): the plans, not the partner's state | open |
| Real video ladder on Ego-Exo4D: outside cameras, the person's view, a gaze crop and a fixed crop; zero-shot and probes | done (runs/egoexo); probes: fixed crop 0.23, gaze crop 0.22 |
| Actions only, from an ideal outside recognizer (sim control) | done: carries the time gain |
| Extra robot viewpoint control | next |
| Real data: earlier HD-EPIC results (narrations 0.67, hand contacts 0.24 to 0.43, gaze adds little) | superseded by Ego-Exo4D for the view question |
| PARTNR's own metrics with a full-knowledge partner and its LLM planners | next |
| Formal statement and proof sketch | next |
