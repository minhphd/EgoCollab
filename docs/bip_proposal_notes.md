# BIP with learned terms: notes on the proposal "BIP, explained"

The proposal is implemented as written in `helper/collab/bip.py`, documented in `helper/collab/README.md`. The
code before it is in `runs/collab_snapshots/2026-10-15_before_proposal`. This note lists what the data say about
it and the changes that need approval.

## Built

| Part of the proposal | Where | State |
| --- | --- | --- |
| step lists, vocabulary check, fallback | `S2.steps`, `checked`, `fallback` | done; 22 of 22 lists for the 3 simulation tasks pass the check |
| progress vector, S and D from it | `Filter.k`, `Filter._masks` | done |
| transition: stay or switch, one or two steps, optional skip, undo | `Filter.clip`, `Filter._outcomes` | done |
| verb term with the training prior, object term | `Filter._reads`, `Belief.clip` | done |
| gaze: saliency x boosts, normalized over the view | `Filter.fixation` | done; saliency uniform |
| repair with step lists and P(done), replay scoring | `S2.revise`, `Belief._add` | done |
| fitting: transition counts, gaze boosts | `helper/collab/fit.py` | done on HD-EPIC |
| fitting: temperature, recognizer bank | `helper/action_cls/stream.py` | stream3: temperature 1.82; bank in data/collab/verb_bank.npz |
| simulator clips drawn from the real recognizer | `helper/collab/sense.py`, `env.py` | wired; not run (the simulator is on the 5090) |
| BIP on real HD-EPIC video | `helper/collab/hdepic.py` | 22 test videos, real and oracle inputs, and the old BIP |
| tests | `tests/test_bip.py` (12), `tests/test_sense.py` (2) | all 35 repo tests pass |

## What the data say

- **Two steps per clip are common.** Step actions (pick, place, open, close) last a median 1.0 s. Of the 3 s clips
  that complete a step, 35% complete two (2 s: 21%; 1 s: 6%). The proposal's own mitigation, a second advance, is
  built; its probability is fitted (0.35).
- **Real people interleave.** Fitted on 153 HD-EPIC videos: leave once done 0.28 per clip, leave before 0.05,
  a step per clip 0.41, resume weight 2.7, and a subgoal whose annotated predecessors are not done is started at
  14% of switches (delta 0.35, against the hand-set 0.05). The graph constrains real people less than assumed.
- **Gaze favours the current subgoal, not the next.** Fitted boosts: current 1.30, next 0.14, noise 0.18;
  0.08 nats per fixation over a uniform choice.
- **On cooking video, steps alone lose the subgoal.** On the 22 test videos (8,371 clips), the current subgoal
  is right 13.5% of the time with every input real, 14.6% with oracle verbs and step objects, and 8.7% without
  gaze, which is chance: all tracking comes from gaze. The pre-proposal BIP, reading every narrated action's
  objects, gets 45.9% (done set 0.71 against 0.55). With a second seed (fresh Gemini priors and step lists) the
  two give 15.4% and 44.2%: seeds move the score by about 2 points, the gap is about 30. In 55% of clips no step
  completes, and the proposal scores their object reads as uniform.
- **The verb recognizer is weak per clip** (`stream3`, test accuracy 0.52, macro F1 0.47): one clip's evidence is a
  likelihood ratio of 1.6 to 2.4 for its verb against other, and 45% of clips with no step read as pick or place.
  With oracle verbs the current subgoal moves only from 13.5% to 14.9%: the recognizer is not the limit.
- **Gemini's step lists fit cooking poorly.** On the 22 test videos (227 subgoals, 5,713 narrated pick, place,
  open and close actions): 24% of the narrated steps match a step of their subgoal's list by verb and object, 76% of
  consecutive matched steps follow the list's order, and 46% of the lists' required steps are ever seen. Matching
  narration nouns to the detector's names by string undercounts matches, but most real steps are not in the
  lists, and half the listed ones never happen as written, so "done once the last required step is" rarely holds.
- **The detected object rarely touches a hand.** The cached detections hold about 3 objects per frame, mostly
  large ones; the median overlap of a hand with any of them is 3%.

## Changes for approval

1. **Object evidence in clips with no step.** Rule: with no step on subgoal $u$,
   $P(r \mid u) = w\,[r \in O(u)] / |O(u) \cap X| + (1 - w) / |X|$, instead of $1/|X|$. $w$ fitted on HD-EPIC as the
   share of no-step clips whose handled object belongs to the current subgoal. Why: it restores the evidence the
   pre-proposal BIP used. Measured on the training videos: in 16,005 clips with no step, the narrated object
   (matched in 9,621) belongs to the current subgoal 80% of the time against 18% by chance, a likelihood ratio of
   4.4; $w$ would be 0.76.
2. **Steps in any order within a subgoal.** Rule: progress is the set of steps done; a step outcome may complete
   any step not done whose required predecessors in the list are done, with optional steps free. Why: Gemini puts
   open before pick for the fridge while people (and the simulated person) often pick first. On HD-EPIC 24% of
   consecutive matched steps break the list's order; coverage (24% of real steps in the lists) is the larger gap.
3. **The recognizer BIP reads, with or without gaze.** Gaze enters both the recognizer and the fixation term.
   Without gaze the recognizer is weaker (macro F1 0.44 against 0.47; 0.1 to 0.2 nats less evidence per clip), and
   BIP on the 22 test videos is the same either way (current 0.135 both; done 0.55 both). Keep the recognizer with
   gaze; compare again once changes 1 and 2 give the clips more weight.
4. **The clip length.** Keep the proposal's 3 s. With 1 s clips (recognizer `stream1`, macro F1 0.39 against 0.47,
   and the person model fitted on 1 s clips), BIP on the 22 test videos tracks the current subgoal no better
   (0.140 against 0.135) and reads the done set worse (0.48 against 0.55): three times the reads bring three times
   the spurious steps.

## Corrections to the proposal text

- One symbol per thing: s is both the leave probability and a step index; c both the verb class and the conflict
  cost; T the horizon, the temperature and a move's probability; a both the advance probability and a subgoal in
  the marginals example. The undo probability (upsilon) reads like u in u -> v.
- The verb term divides by the class prior the recognizer trained under. Training is class-balanced, so that
  prior is uniform, not HD-EPIC's class frequencies.
- The recognizer is not "already trained" for clips of the stream: `stream.py` trains it, with "other" meaning
  no step completed.
- `log_Z` is `Filter.logZ`. In the decision, min(d_R, 2 s) is always 2 s, since d is at least 3 s.
- "Still hand-set" leaves out the telling constants (0.4 s, 0.6 s per fixation) and the duration floor and
  default (3 s, 20 s).
- Three tables use em dashes as empty cells.
- The graph weights rest on each filter's estimate of its evidence; graphs that differ little can be separated by
  sampling noise.
