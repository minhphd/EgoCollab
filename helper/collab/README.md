# Fast-slow Bayesian inverse planning (BIP)

The robot infers which subgoal the person is on, which comes next, which are done, and which orderings the
task requires. Then it decides each second whether to take a subgoal over, and when to tell the person about a
move they may have missed. All of BIP is one file, `bip.py`. BIP knows no particular simulator: the env
(`env.py`, with `sense.py` for what the recognizers report) supplies the observations, and the agent (`agents.py`)
feeds them in and asks for decisions.

Parameter names in `code font` are fields of `Params`, with their defaults. All logarithms are natural.

## Overview: model, inference, decision

- **Generative model** (section 2): how the observations arise if everything hidden were known. A task
  graph $G$ is drawn from Gemini's prior. The person moves through the subgoals as $G$ allows, completing each
  subgoal's steps, which Gemini writes once per task. What the person does reaches the robot as clips, each
  scored by a verb recognizer and an object detector, and as gaze fixations. The model gives a probability to
  every history and runs forward, hidden to observed.
- **Inference** (section 3): the same model run backward. Bayes' rule turns $P(\text{observed} \mid
  \text{hidden})\,P(\text{hidden})$ into $P(\text{hidden} \mid \text{observed})$: which graph, which subgoal the
  person is on, which comes next, how far each subgoal is. S1 and S2 only approximate the sums.
- **Decision** (section 5): the posterior, carried one step forward through the generative model, predicts
  what happens if the robot takes subgoal $v$:
  - whether its predecessors are done ($P_{\mathrm{exec}}$);
  - whether the person is on it or about to start it ($Y_v$);
  - how much of the person's time it saves ($\Delta$).

  Telling (section 6) weighs the person's search time against the time spent listening.

The whole algorithm. Inference runs on every event; the decision runs every second while the robot is free:

```python
# Once per task
p_uv = S2.prior(instruction, subgoals)          # Gemini: P(ordering u -> v is required)          (2.1)
R_v = S2.steps(instruction, subgoals, vocab)    # Gemini: each subgoal's steps, checked           (2.2)
pool = sample_pool(p_uv)                        # up to 64 graphs G, each with log P(G | l)       (2.1, 3.1)
for G in pool:
    log_Z[G] = 0                                # evidence log P(o_1:t | G)                       (3.2)
    particles[G] = M x (g ~ pi, n ~ pi, k = 0, weight = 1/M)                                      # (2.10)
C, Q = {}, {}                                   # the robot's own subgoals done; claimed by the robot

# Every simulator step
for t in steps:
    person.act()
    for event in env.events():                  # clips every 3 s, gaze fixations, the robot's own completions
        infer(event)                            # S1, on every event                              (section 3)
        if event.kind == "clip" and surprise() > tau:        # the pool's prediction vs. no orderings
            S2.revise(...)                      # graphs or step lists, scored before they join   (section 4)
    if robot.free and t >= next_decide:
        decide(t)                               # every 1 s                                       (section 5)
    robot.step()

def infer(event):                               # what each event contributes
    for G in pool:
        for p in particles[G]:
            if event.kind == "clip":            # the verb recognizer's P(c | clip), the object reads
                moves = stay or switch x (no step, one step, two steps) on the subgoal, or an undo   # (2.4)
                p.factor = sum(T(m) * L(m) for m in moves)                                          # (2.7)
                m = sample(T * L); update p.g, p.k, then p.n                                        # (2.5, 2.6)
            elif event.kind == "fixation":
                p.factor = P(x | p.g, p.n, X)   # gaze                                            (2.8)
        log_Z[G] += log(sum(p.weight * p.factor))       # evidence factor P(o_k | o_<k, G)        (3.2)
        for p in particles[G]:
            p.weight = normalize(p.weight * p.factor)   # Bayes update of the person's state      (3.3)
    P(G | o) = normalize(P(G | l) * exp(log_Z))         # graph posterior                         (3.4)

def decide(t):                                  # how the posterior becomes an action
    J[G, p] = P(G | o) * p.weight               # joint weight of each (graph, particle)          (3.4)
    for v in subgoals:
        P_exec = P(every predecessor of v done)            # sum of J over particles             (5.1)
        Y = P(person on v, or switching to v first)        # sum of J over particles             (5.2)
        Delta = person's time saved by the robot doing v   # from P(u done), P(v before u)       (5.3)
        U[v] = P_exec * ((1 - Y) * Delta - c * Y) - c_prec * (1 - P_exec)
    claim argmax(U) if max(U) > 0 else wait     # a claim enters the belief as Q
```

## 1. Variables and inputs

| Symbol | Meaning |
| --- | --- |
| $\ell$ | the instruction (text) |
| $V$ | the subgoals, given with the task (not inferred) |
| $G = (V, E)$ | the task graph; $(u, v) \in E$ means $v$ cannot start before $u$ is done |
| $R_v,\ K_v$ | subgoal $v$'s step list, written by Gemini, and its length; each step is a verb with an object, and a place also names its target |
| $g_t$ | the subgoal the person works on now; $-1$ only before the first draw |
| $n_t$ | the subgoal the person plans to do next; $-1$ when none is possible |
| $k_t = (k_v)$ | the progress vector: for each subgoal $v$, the steps of $R_v$ done, from 0 to $K_v$ |
| $S_t,\ D_t$ | the subgoals started ($k_v > 0$) and done (every required step of $R_v$ done), from $k_t$ |
| $C_t,\ Q_t$ | the robot's own subgoals done, and the subgoal it has claimed; shared by all particles, never inferred |
| $q_t$ | a clip's verb probabilities over pick, place, open, close, other, from the verb recognizer |
| $r_t$ | the clip's object read and receptacle read, from the detector (either may be missing) |
| $x_t,\ X_t$ | the fixated object and the objects visible at that moment ($x_t \in X_t$) |
| $o_t$ | one observation: a clip $(q_t, r_t)$, or a fixation $(x_t, X_t)$ |

The objects of $v$, $O(v)$, are every object and target its step list names. The person's completions are never
observed: BIP infers them from the clips (2.7).

In pseudocode, the state S1 keeps:

```python
pool      = K graphs G, each with log P(G | l) and log Z(G)
particles = M per graph: (g, n, k) and a weight; S and D follow from k
C, Q      = the robot's own subgoals done, claimed by the robot (shared by all particles)
```

## 2. Generative model (joint factorization)

$$
\begin{aligned}
& P(G, z_{1:T}, o_{1:T} \mid \ell) \\[4pt]
={} & P(G \mid \ell) && \text{prior over orderings (2.1)} \\
& \times \prod_{t=1}^{T} \Big[\, P(z_t \mid z_{t-1}, G, C, Q) && \text{transition (2.4, 2.5)} \\
& \qquad\quad \times P(o_t \mid z_t, z_{t-1}) \,\Big] && \text{clip (2.7) or gaze (2.8)}
\end{aligned}
$$

with $z_t = (g_t, n_t, k_t)$. Time advances by events, not by clock ticks: each event carries one observation. A
clip is scored against what the person's state says happened in it, so its likelihood depends on the move from
$z_{t-1}$ to $z_t$; at a fixation the state does not move. $C$ and $Q$ come from the robot itself and carry no
likelihood (2.9). The step lists $R_v$ are fixed per task (2.2), and repair can replace them (section 4).

In pseudocode, the model run forward (it simulates a person; inference only evaluates it):

```python
G = sample_prior(p_uv)                          # 2.1
g, n, k = draw from pi, draw from pi without g, zeros     # 2.10
every 3 s, a clip:
    if rand() < s(g):                           # 2.4: leave, s_done once g is done, else s_mid
        g = n w.p. lambda if n is valid, else draw from pi
    k[g] += 0, 1 or 2 steps (prob 1 - a, a (1 - a2), a a2), skipping an optional step
    n = draw from pi with g counted done, if g changed or n is used up       # 2.5
    q = the recognizer's probabilities for the verb of the last step done, or "other"   # 2.7
    r = that step's object and target, each misread w.p. 1 - obj_acc
x = a visible object, favouring the objects of g, then of n                  # 2.8
```

### 2.1 Prior over orderings $P(G \mid \ell)$

$$
P(G \mid \ell) = \prod_{(u,v) \in E,\ \text{listed}} p_{uv} \prod_{(u,v) \notin E,\ \text{listed}} \big(1 - p_{uv}\big) \prod_{(u,v) \in E,\ \text{not listed}} 0.02
$$

**Computed by** `S2.prior` and `log_prior`.

1. One Gemini call per task (`model`, Gemini 3.5 Flash Lite).
2. The prompt holds:
   - a sentence framing the task ("a robot assistant watches a person through their smart glasses");
   - $\ell$;
   - the subgoals in words, under IDs S1 to Sn shuffled with the seed, so their order leaks nothing;
   - a definition: an ordering A > B is required when the instruction asks for that order (for example
     "first ... then ...") or when the task cannot be done otherwise; a habit or a convenient order is not
     required.
3. The reply follows a JSON schema: a list of orderings (before, after, p).
4. Each $p_{uv}$ is clipped to $[0.02, 0.98]$; orderings of a subgoal with itself are dropped.
5. An ordering Gemini did not list costs a factor 0.02 when $E$ contains it, and nothing otherwise.
6. The reply is cached per task, model and seed (`cache`).

### 2.2 Step lists $R_v$

A second Gemini call at the task's start writes each subgoal as the short list of steps a recognizer can see.
A subgoal is done once its last required step is.

**Computed by** `S2.steps`, `checked` and `fallback`.

1. The prompt holds the header of 2.1, the rules, and the object names the detector knows (the task's objects
   and targets, and `vocab` when the task gives one). The rules: steps in the order a person does them; each
   step is one verb (pick, place, open, close) with one object, and a place also names its target; a step that
   happens only sometimes is optional; the undo is the step that would reverse the subgoal once it is done.
2. The reply follows a JSON schema whose verbs and objects are enumerations of the vocabulary.
3. Each list is checked: every verb and object in the vocabulary, a target on every place, at least one step
   that is not optional. A subgoal whose list fails, or that the reply leaves out, falls back to
   pick($o_v$) then place($o_v$, $f_v$), with an optional open and close of $f_v$ when it has a door.
4. The reply is cached per task, model and seed, next to the prior.

| Subgoal | Steps | Undo |
| --- | --- | --- |
| put the bowl on the table | pick(bowl) → place(bowl, table) | pick(bowl) |
| put the milk in the fridge | open(fridge) → pick(milk) → place(milk, fridge) → close(fridge) | pick(milk) |

The rows are Gemini's lists for the lunch and groceries tasks: all 22 lists of the three tasks pass the check.
For the fridge, Gemini puts the open before the pick, while the simulated person picks first; with a fixed step
order (A8), that pick can only be explained as part of a later clip's two steps.

### 2.3 Policy $\pi$ over the next subgoal

$$
\pi(v \mid D, C, S, G, Q) \propto
\begin{cases}
0 & v \in D \cup C \\
\rho & v \in S \quad\text{(started, not done)} \\
1 & \text{every ancestor of } v \text{ in } G \text{ is in } D \cup C \quad\text{(ready)} \\
\delta & \text{otherwise}
\end{cases}
\;\times\;
\begin{cases}
\kappa & v \in Q \\
1 & v \notin Q
\end{cases}
$$

- The ancestors of $v$ are all of $v$'s transitive predecessors in $G$.
- $\rho$ = `resume` (2.7), $\delta$ = `delta` (0.35), fitted on HD-EPIC (2.11): people go back to a subgoal they
  started more than they open a new one, and 14% of their switches start a subgoal whose annotated
  predecessors are not done.
- $\kappa$ = `claimed` (0.3, hand-set): the person may not know the robot is on $v$, so they may still go for it.

**Computed by** `Filter.options`, for every (graph, particle) pair. Normalized over $v$ after the current
subgoal $g$ is excluded.

### 2.4 Transition $P(g_t, k_t \mid g_{t-1}, n_{t-1}, k_{t-1}, \dots)$

At each clip the person either **stays** on their subgoal or **switches**; then the clip completes no step, one
step, or two steps of the subgoal they are now on. With small probability it instead undoes a subgoal they did.

```
was the person on a subgoal g?
├─ yes, its required steps not done ─┬─ stay, prob 1 − s_mid ─────────────────► on g
│                                    └─ leave, prob s_mid ──┐
├─ yes, done ────────────────────────┬─ stay, prob 1 − s_done ────────────────► on g
│                                    └─ leave, prob s_done ─┤
└─ no (only before the first draw) ──── leave, prob 1 ──────┤
                                                            ├─ plan n valid ─┬─ prob λ ─────► on n
                                                            │                └─ prob 1 − λ ─► on a draw from π
                                                            └─ no valid plan ───────────────► on a draw from π
then, on the subgoal u the person is now on, with progress k_u:
├─ no step ─────── prob 1 − a          the clip shows "other"
├─ one step ────── prob a (1 − a2)     the next step (or the one after it, if the next is optional)
└─ two steps ───── prob a a2           the next two, the clip showing the second
```

As one formula, for the moves that do not undo:

$$
P(g_t = u,\ \text{outcome } o) = \Big[ (1 - s)\,[u = g_{t-1}] + s\,\mathrm{sw}(u) \Big]\; P(o \mid u, k_u)\; (1 - \upsilon\,[D^{\text{person}} \neq \emptyset])
$$

$$
s = \begin{cases} s_{\mathrm{done}} & g_{t-1} \text{ done} \\ s_{\mathrm{mid}} & g_{t-1} \text{ not done} \\ 1 & g_{t-1} = -1 \end{cases}
\qquad
\mathrm{sw}(u) = \begin{cases}
\lambda\,[u = n_{t-1}] + (1 - \lambda)\,\pi(u) & n_{t-1} \text{ valid} \\
\pi(u) & \text{otherwise}
\end{cases}
$$

- $s_{\mathrm{done}}$ = `s_done` (0.28), $s_{\mathrm{mid}}$ = `s_mid` (0.05), $a$ = `adv` (0.41), $a_2$ = `adv2` (0.35):
  fitted on HD-EPIC 3 s clips (2.11). People linger after a subgoal's last step: they leave within a clip only
  28% of the time.
- $\lambda$ = `plan` (0.5, hand-set): HD-EPIC does not annotate the plan.
- **Optional steps:** when the next step is optional, a step outcome may complete it or skip it and complete the
  one after; the two split the outcome's probability evenly.
- **Two steps per clip:** in HD-EPIC, 35% of the 3 s clips that complete a step complete two (a pick and its
  place within 3 s), so a clip may complete the next two steps; its likelihood is that of the second.
- **Undo** ($\upsilon$ = `undo`, 0.01, hand-set): with probability $\upsilon$ the clip is the undo of a subgoal the
  person did (a uniform choice among them), which puts that subgoal back to one required step short of done.
  Subgoals the robot did ($C$) are not undone.
- **No possible move** (no subgoal and none left): the particle keeps its state and explains the clip as
  "other" (2.7). A graph that thinks the task is finished still pays for every later clip.

**Example.** Subgoals $v_0, \dots, v_3$ with pick then place each; the ordering $v_0 \to v_1$; the robot has
claimed $v_3$. A particle is on $v_0$ with its pick done ($k = 1$), and plans $v_2$. The policy over the others:
$v_1$ waits for $v_0$ ($\delta = 0.35$), $v_2$ is ready (1), $v_3$ is ready but claimed ($\kappa = 0.3$), so
$\pi = (0, 0.212, 0.606, 0.182)$ and the switch goes to $v_2$ with $0.5 + 0.5 \times 0.606 = 0.803$. Then:

| Move | $T(m)$ | The clip should show |
| --- | --- | --- |
| stay on $v_0$, no step | $0.95 \times 0.59 = 0.561$ | other |
| stay on $v_0$, its place | $0.95 \times 0.41 = 0.390$ | place($o_0$, $f_0$) |
| switch to $v_2$, no step | $0.05 \times 0.803 \times 0.59 = 0.024$ | other |
| switch to $v_2$, its pick | $0.05 \times 0.803 \times 0.41 \times 0.65 = 0.011$ | pick($o_2$) |
| switch to $v_2$, pick and place | $0.05 \times 0.803 \times 0.41 \times 0.35 = 0.006$ | place($o_2$, $f_2$) |
| switch to $v_1$ or $v_3$ | 0.009 and 0.008 in all | |

**Computed by** `Filter.clip` with `Filter._outcomes` (the step outcomes per subgoal and progress), as an array
over every particle's moves.

### 2.5 Planned next $P(n_t \mid g_t, \dots)$

$n_t$ is redrawn when $g$ changed, $n_{t-1} = g_t$, $n_{t-1}$ is done or claimed by the robot, or $n_{t-1} = -1$;
otherwise it is kept. A redraw samples $n_t \sim \pi$, computed with $g_t$ treated as done for readiness (the
person plans what comes after finishing $g_t$) and with $g_t$ excluded. If no subgoal is possible, $n_t = -1$.

**Computed by** `Filter._replan`, after the move.

### 2.6 Progress, started and done

- **Progress** $k_v$: the steps of $R_v$ done, set by the clip's move (2.4). A switch leaves the old subgoal's
  progress as it is, so a half-done subgoal keeps its steps if the person comes back.
- **Started** $S = \{v : k_v > 0\}$; **done** $D = \{v : k_v \geq K^{\text{req}}_v\}$, $K^{\text{req}}_v$ the
  steps up to the last one that is not optional. A trailing optional step (close the fridge) may still follow.
- **The robot's done** $C$: set by the robot's own completions (2.9); a subgoal in $C$ counts as done everywhere.

**Computed by** `Filter._masks`, which keeps $S$ and $D$ as 64-bit masks.

### 2.7 Clip likelihood $P(o_t \mid z_t, z_{t-1})$

A clip is the video of the last `clip` (3) seconds. The verb recognizer (`helper/action_cls/stream.py`) gives
$q(c \mid \text{clip})$ over pick, place, open, close, other, where "other" means the clip completed no step; it
is trained class-balanced, so its prior is uniform and $q$ divided by that prior (`verb_prior`) is a likelihood up
to a constant. The detector reports the object read $r_o$ (what the person handles) and, at a place, the
receptacle read $r_f$; either may be missing. If the move completed step $s$ (verb $c_s$, object $b_s$, target
$f_s$):

$$
P(\text{clip}, r \mid s) \propto \frac{q(c_s \mid \text{clip})}{p_{\text{train}}(c_s)}\;
P(r_o \mid b_s)\; P(r_f \mid f_s),
\qquad
P(r \mid b) = \begin{cases} \text{acc} & r = b \\ \frac{1 - \text{acc}}{|X| - 1} & r \neq b \end{cases}
$$

- acc = `obj_acc` (0.7), the detector's rate of right reads; $X$ is what is in view.
- A read the step does not name (a receptacle at a pick), and both reads of a move that completed no step, are
  uniform, $1/|X|$; a missing read is 1.
- A move that completed no step has likelihood $q(\text{other}) / p_{\text{train}}(\text{other})$ times the
  uniform reads.
- Two steps in one clip are scored as the second, the last the clip shows; an undo as its step.
- The verb says what happened; the object says which subgoal it happened to. Neither alone is enough: every
  subgoal has a pick, and one object can appear in two subgoals.

**Computed by** `Filter._reads`, once per clip for every step of every subgoal; each move then costs a table
lookup. `Belief.clip` takes $q$ (as a sequence or a dict), floors it at $10^{-6}$ and divides by `verb_prior`.

### 2.8 Gaze likelihood $P(x \mid g, n, X)$

The person's glasses report a fixation on one object $x$ among the visible objects $X$:

$$
P(x \mid g, n, X) = (1 - \varepsilon)\, \frac{s(x \mid X)\, e^{\gamma_1 [x \in O(g)] + \gamma_2 [x \in O(n)]}}{\sum_{x' \in X} s(x' \mid X)\, e^{\gamma_1 [x' \in O(g)] + \gamma_2 [x' \in O(n)]}} + \frac{\varepsilon}{|X|}
$$

- $s(x \mid X)$ is a saliency model's weight of each visible object, regardless of the task; uniform until one
  exists (`saliency` in `Belief.fixation`).
- $\gamma_1$ = `gamma_cur` (1.3), $\gamma_2$ = `gamma_next` (0.14), $\varepsilon$ = `eps` (0.18): maximum likelihood on
  36,562 fixations of HD-EPIC's training participants (`fit.py gaze=1`): real gaze on the detector's boxes at
  2 fps, each subgoal's objects from Gemini's vocabulary, the annotated current and next subgoals, uniform
  saliency. Gaze lands on the current subgoal's objects 53% of the time, though they are 38% of what is in view;
  it barely favours the next subgoal's. The fit gains 0.08 nats per fixation over a uniform choice.
- The likelihood is normalized over $X$. A fixation reweighs the particles and never moves them.

**Computed by** `Filter.fixation`, per particle, from the counts of each subgoal's objects in view.

### 2.9 The robot's own subgoals: $C$ and $Q$

When the robot's own action completes $v$, $v$ joins $C$ and leaves $Q$, with no likelihood: it says nothing about
the person. Before each decision the agent clears $Q$, and it sets $Q = \{v\}$ when it takes $v$, so the policy
(2.3) knows what the robot is doing.

**Computed by** `Filter.robot_done` and `Belief.claim`.

### 2.10 The start

Each particle draws its first subgoal from $\pi$ and its plan from $\pi$ without it, with $k = 0$. No weight
changes and no surprise is computed.

### 2.11 Fitted parameters (`fit.py`)

On HD-EPIC's 153 videos with annotated subgoals (`helper/datasets/hd_epic/tasks.py`), cut into 3 s clips (48,684
clips). At each clip's end the person is on the subgoal of the latest action (pauses inside a subgoal kept); a
subgoal's steps are its actions whose main verb is pick, place, open or close.

| Parameter | Value | Count | From |
| --- | --- | --- | --- |
| $s_{\mathrm{done}}$ | 0.28 | 2,590 clips | the next clip is on another subgoal, once the current one's steps are done |
| $s_{\mathrm{mid}}$ | 0.05 | 46,041 clips | ... before they are |
| $a$ | 0.41 | 43,794 clips | the clip completes a step of the subgoal it stays on, given one remains |
| $a_2$ | 0.35 | 17,542 clips | it completes two, given one and two remain |
| $\rho$, $\delta$ | 2.7, 0.35 | 2,967 switches | maximum likelihood of the subgoal chosen at each switch, the annotated orderings as $G$ (66% started, 20% ready, 14% not ready) |

With 1 s clips the same counts give $s_{\mathrm{done}}$ 0.12, $s_{\mathrm{mid}}$ 0.017, $a$ 0.19, $a_2$ 0.065
(`python -m helper.collab.fit clip=1.0`). $\lambda$, $\upsilon$ and $\kappa$ are not annotated and stay hand-set.

## 3. Inference

Write $z_t = (g_t, n_t, k_t)$ for the person's hidden state. The chain rule, then Bayes' rule on the graph, gives
three terms:

$$
\begin{aligned}
P(G, z_t \mid o_{1:t}, \ell)
&= P(G \mid o_{1:t}, \ell)\; P(z_t \mid G, o_{1:t}) \\
&\propto \underbrace{P(G \mid \ell)}_{\text{(1) prior}}\;
\underbrace{P(o_{1:t} \mid G)}_{\text{(2) evidence}}\;
\underbrace{P(z_t \mid G, o_{1:t})}_{\text{(3) person state}}
\end{aligned}
$$

In pseudocode, `infer(event)` of the overview in full:

```python
def infer(event):
    for G in pool:
        for p in particles[G]:                                  # p = (g, n, k, weight)
            if event.kind == "clip":
                pi = policy(p.D | C, p.S, G, Q), without p.g    # (2.3)
                s = 1 if p.g == -1 else s_done if p.g in p.D | C else s_mid
                sw = plan * onehot(p.n) + (1 - plan) * pi if p.n is valid else pi
                on = (1 - s) * onehot(p.g) + s * sw             # P(on u) for every subgoal u    (2.4)
                moves = [(on[u] * P(o | u, p.k[u]), L(o on u)) for u, o]       # no, one or two steps
                moves += [(undo / |done by the person|, L(undo of v)) for v done by the person]
                p.factor = sum(T * L for T, L in moves)          # the predictive score           (2.7)
                m = sample(moves, prob T * L)                    # the locally optimal proposal
                p.g, p.k = the move's subgoal and progress       # (2.6)
                if p.g changed or p.n in (-1, p.g) or p.n in p.D | C | Q:
                    p.n = sample(policy(p.D | C | {p.g}, p.S, G, Q), without p.g)   # (2.5)
            elif event.kind == "fixation":                      # x among the visible X
                p.factor = P(x | p.g, p.n, X)                   # (2.8)
        log_Z[G] += log(sum(p.weight * p.factor for p in particles[G]))   # (3.2), weights normalized
        for p in particles[G]:
            p.weight = normalize(p.weight * p.factor)                       # (3.3)
        if 1 / sum(p.weight ** 2 for p in particles[G]) < M / 2:
            systematic_resample(particles[G])
    if event.kind == "robot_done":
        C.add(event.v); Q.discard(event.v)                                  # (2.9)
```

### 3.1 Term (1): the prior

The prior is 2.1, evaluated exactly for each graph of a finite **pool** $\mathcal{G}$, built by `sample_pool`:

1. **The most probable graph:** add the listed orderings with $p_{uv} \ge 0.5$ in decreasing $p_{uv}$,
   skipping any that would close a cycle.
2. **Samples:** shuffle the listed orderings, keep each with probability $p_{uv}$, and skip cycles. Repeat
   until the pool holds `pool` (64) distinct graphs, or after $20 \times 64$ tries.
3. **S2's revisions** join later (section 4).

The pool holds distinct graphs, so weighting each by its prior and its evidence (3.4) is the posterior restricted
to the pool.

### 3.2 Term (2): the evidence

$$
P(o_{1:t} \mid G) = \prod_{k=1}^{t} P(o_k \mid o_{<k}, G) = Z_t(G),
\qquad
P(o_k \mid o_{<k}, G) \approx \sum_m w^{(m)} \ell^{(m)}_k
$$

with normalized weights $w^{(m)}$ and the particle's predictive $\ell^{(m)}_k$:

- for a clip, $\ell^{(m)}_k = \sum_{\text{moves}} T(m)\, L(m)$, with the transition 2.4 and the likelihood 2.7,
  both from the particle's state before the move;
- for a fixation, $\ell^{(m)}_k$ is the likelihood 2.8 at the particle's state.

**Computed by** `Filter._weigh`, which keeps $\log Z$ (`Filter.logZ`).

### 3.3 Term (3): the person's state, a particle filter per graph

The Bayes filter: predict with the transition, then update with the observation and normalize by 3.2's factor.

**Computed by** `Filter`, with `particles` ($M = 64$) weighted particles per graph.

- **Clip** (`Filter.clip`), for each particle:
  1. List its moves (2.4): stay or switch to each subgoal, with no step, one step or two steps; or undo a subgoal
     it did. At most $7|V| + |V|$ moves.
  2. Multiply the weight by $\sum_m T(m) L(m)$; this also feeds 3.2. The score uses the state before the move,
     so a particle is not rewarded for a jump the clip told it to make.
  3. Draw a move with probability $T(m) L(m)$ over that sum: the locally optimal proposal, under which the weight
     update above is exact. The clip pulls particles toward the step it shows.
  4. Apply the move to $g$ and $k$, then redraw $n$ (2.5).
- **Fixation** (`Filter.fixation`): multiply each weight by the likelihood at the particle's state.
- **Resampling** (`Filter._resample`): when a graph's effective sample size $1/\sum_m (w^{(m)})^2$ falls below
  $M/2$, resample its particles systematically and reset the weights to $1/M$.

The recognizer and the detector run once per clip, not once per particle. A clip update takes 6 ms for 32 graphs
of 64 particles, a fixation 0.4 ms.

### 3.4 Graph weights and read-outs

$$
P(G \mid o_{1:t}, \ell) = \frac{P(G \mid \ell)\, Z_t(G)}{\sum_{G' \in \mathcal{G}} P(G' \mid \ell)\, Z_t(G')}
$$

Every read-out averages over graphs, and within each graph over particles, with the joint weight
$J_{G,m} = P(G \mid o_{1:t}, \ell)\, w^{(m)}_G$ (`Filter.joint`):

$$
\begin{aligned}
P(g_t = v) &= \textstyle\sum_{G,m} J_{G,m}\, [g^{(m)} = v] \\
P(n_t = v) &= \textstyle\sum_{G,m} J_{G,m}\, [n^{(m)} = v] \\
P(v \text{ done}) &= \textstyle\sum_{G,m} J_{G,m}\, [v \in D^{(m)} \cup C] \\
P(v \text{ started}) &= \textstyle\sum_{G,m} J_{G,m}\, [v \in S^{(m)}] \\
P(u \prec v) &= \textstyle\sum_{G} P(G \mid o_{1:t}, \ell)\, [u \text{ is an ancestor of } v \text{ in } G]
\end{aligned}
$$

`Belief.current`, `Belief.next` and `Belief.done` return the first three for every subgoal;
`Filter.edge_probs` returns the last. The joint weights keep the correlation between the graph and the person's
state, which separate marginals would lose.

### 3.5 Approximations

| | Approximation | Where |
| --- | --- | --- |
| A1 | graphs come from a finite pool, not all DAGs on $V$ | `sample_pool`, S2 |
| A2 | the person's state given a graph is a particle approximation | `Filter` |
| A3 | the planned next subgoal is proposed from $\pi$, without the coming evidence | `Filter._replan` |
| A4 | a clip completes at most two steps, both of the subgoal the person is on | `Filter._outcomes` |
| A5 | the saliency model is uniform until one is trained | `Filter.fixation` |
| A6 | no positions are used: walking carries no evidence | `Belief` |
| A7 | at most 63 subgoals: $S$, $D$, $C$ and $Q$ are 64-bit masks | `bit`, `has` |
| A8 | a subgoal's steps come in the order its list gives, apart from skipped optional steps | `Filter._outcomes` |
| A9 | consecutive clips are scored as independent given the state, though their recognizer windows overlap | `Filter.clip` |

## 4. S2: revisions on surprise

- **Surprise** (`Belief._feed`, clips only):
  - Let $p_{\mathcal{G}} = \sum_G P(G \mid o_{<k}, \ell)\, P(o_k \mid o_{<k}, G)$, the pool's prediction of the
    clip.
  - Let $p_\emptyset$ be the prediction of a reference filter whose only graph has no orderings.
  - The surprise is $\ln p_\emptyset - \ln p_{\mathcal{G}}$: how much better "no orderings" explains the clip than
    the pool does. Either the pool asserts an ordering the person breaks, or a step list does not fit.
- **Trigger:** S2 is called when all four hold:
  - the surprise exceeds $\tau$ = `surprise` (1.4);
  - no call is pending;
  - at least `cooldown` (20 s) has passed since the last call;
  - fewer than `max_calls` (4) calls have been made.
- **Prompt** (`S2.revise`):
  - the prior's header;
  - each subgoal's step list and its current $P(\text{done})$;
  - the three leading graphs with their probabilities;
  - S1's last 30 reads ("12 s working on S3", "15 s read place mug into cabinet");
  - one sentence on the surprise ("the person appears to work on S4 while S2 is not complete; the
    hypotheses explain this poorly");
  - the rules: if the person started B while A was not complete, A > B is not required; and the step rules of 2.2.
- **Reply:** up to `revisions` (3) corrected graphs, corrected step lists for some subgoals, or both. Model:
  `model` (Gemini 3.5 Flash Lite). Step lists are checked as in 2.2.
- **Joining** (`Belief._add`), after the call's measured latency:
  - corrected step lists replace the old ones only if the pool, run through the whole history with them in a
    fresh filter, explains it better ($\log \sum_G P(G)\, Z(G)$, `Filter.evidence`);
  - each new acyclic graph is run through the whole history in a fresh filter, then joins. Its weight is
    therefore its own prior times its own evidence.
- Gemini never sets the done set: a missed completion is the filter's to recover from later clips.

In pseudocode (`Belief._feed`, `Belief._revise`, `Belief._add`):

```python
if clip and surprise > tau and S2 is free:
    graphs, steps = gemini(instruction, step lists and P(done), top graphs, recent reads, the surprise)
    after its latency: keep steps if the history is likelier with them; score each graph on the history, add it
```

## 5. Decision (`decide`)

Only while the robot has no subgoal, every second (`agents.py`):

$$
\begin{aligned}
U(v)
=\; & P_{\mathrm{exec}}(v)\, \big[ (1 - Y_v)\, \Delta(v) - c\, Y_v \big] \\
-\; & c_{\mathrm{prec}}\, \big( 1 - P_{\mathrm{exec}}(v) \big),
\qquad U(\text{wait}) = 0
\end{aligned}
$$

The robot executes $\arg\max_v U(v)$ when it is positive (ties go to the subgoal listed first), and waits
otherwise.

In pseudocode, `decide(t)` of the overview in full:

```python
def decide(t, blocked):
    d = mean gap between changes of the leading subgoal, at least 3 s; 20 s before two gaps    # (5.4)
    rem = max(1, d * r((t - t_read) / d))                                # time left on the current subgoal
    d_R = slowdown * d                                                   # the robot's time for a subgoal
    on, done, started = P(g_t = u), P(u done), P(u started) for every u  # read-outs (3.4)
    for v in subgoals:
        P_exec = sum(J[G, p] for all G, p if every ancestor of v in G is in p.D | C)          # (5.1)
        Y = sum(J[G, p] for all G, p if p.g == v or (p.n == v and rem < min(d_R, notice)))  # (5.2)
        work = [(on[u] * min(d, rem) + (1 - on[u]) * d) * (1 - done[u]) * (u not in Q) for u in subgoals]
        free = sum((1 - P(v before u)) * work[u] for u in subgoals) - work[v]
        Delta = d - max(0, d_R - free)                                                       # (5.3)
        U[v] = P_exec * ((1 - Y) * Delta - c * Y) - c_prec * (1 - P_exec)
        if v in C | Q or done[v] > 0.5 or started[v] > 0.5 or v in blocked:
            U[v] = -inf                                                                      # (5.5)
    return argmax(U) if max(U) > 0 else None
```

Each term, with $J$ the joint weight of 3.4:

### 5.1 $P_{\mathrm{exec}}(v)$: the predecessors are done

$$
P_{\mathrm{exec}}(v) = \sum_{G,m} J_{G,m}\, \big[\text{every ancestor of } v \text{ in } G \text{ is in } D^{(m)} \cup C\big]
$$

### 5.2 $Y_v$: the person works on $v$ before seeing the robot

$$
Y_v = \sum_{G,m} J_{G,m}\, \Big[ g^{(m)} = v \;\text{ or }\; \big( n^{(m)} = v \text{ and } \mathrm{rem}(g^{(m)}) < \min(d_R, \text{notice}) \big) \Big]
$$

- notice = `notice` (2 s): the time the person needs to see the robot on $v$ and leave it.
- $d_R = d \cdot$ `slowdown` (3.0): the robot's time for a subgoal (the Stretch drives at 0.3 m/s, people walk
  at 1.2). With $d \geq 3$ s, $d_R \geq 9$ s, so the minimum is the notice time.
- $\mathrm{rem}(g)$: the person's time left on their current subgoal (5.4); $\mathrm{rem}(-1) = 0$.

$P_{\mathrm{exec}}$ and $Y_v$ are each summed over the pairs, then multiplied: the product does not keep their
correlation.

### 5.3 $\Delta(v)$: the person's time saved

$$
\begin{aligned}
\mathrm{work}(u) &= \Big[ P(g_t = u)\, \min\big(d, \mathrm{rem}(u)\big) + \big(1 - P(g_t = u)\big)\, d \Big] \big(1 - P(u \text{ done})\big)\, [u \notin Q] \\
\mathrm{free}(v) &= \textstyle\sum_{u \in V} \big(1 - P(v \prec u)\big)\, \mathrm{work}(u) - \mathrm{work}(v) \\
\Delta(v) &= d - \max\big(0,\ d_R - \mathrm{free}(v)\big)
\end{aligned}
$$

$\mathrm{work}(u)$ is the person's work left on $u$: the remaining time if they are on it, all of $d$ if not.
$\mathrm{free}(v)$ is the work the person can do meanwhile that does not depend on $v$. The robot saves the full
$d$ unless the person runs out of such work before the slower robot finishes.

### 5.4 Durations and costs

- $d$: the mean gap between changes of the leading current subgoal ($\arg\max_v P(g_t = v)$, read at each clip),
  at least 3 s, or 20 s before two gaps exist. It is the same for every subgoal.
- $\mathrm{rem}(u) = \max\big(1,\ d \cdot r\big((t - t_{\text{read}}) / d\big)\big)$: the person's expected time left
  on their subgoal, given the time $t - t_{\text{read}}$ already spent on it. $t_{\text{read}}$ is when the leading
  subgoal last changed. $r(x) = E[D - x \mid D > x]$ is the mean residual time of a person's subgoal, measured on
  the PARTNR human traces (`rem_curve`, at $x = 0, 0.25, \dots, 3$).
- $c$ = `beta_rel` $\times\, d$ (1.0), the cost of a conflict.
- $c_{\mathrm{prec}}$ = `beta_prec_rel` $\times\, d$ (2.0), the cost of an ordering violation.

### 5.5 Never executed

$U(v) = -\infty$ when any of these holds:

- $v \in C \cup Q$;
- $P(v \text{ done}) > 0.5$;
- $P(v \text{ started}) > 0.5$;
- $v$ is in `blocked`, the agent's list: subgoals the robot cannot do, or has left for a while.

## 6. Telling (`tell_value`)

When the robot moves an object the person may look for, it weighs telling them where it is:

$$
V(\text{tell } o) = P(\text{missed})\; P(\text{seek})\; \text{search} - \text{talk},
\qquad
P(\text{missed}) = e^{-\text{dwell} / 0.4\,\text{s}}
$$

- dwell grows 0.6 s per fixation on $o$, or on the robot carrying it, after the robot picks $o$ up (`agents.py`).
- $P(\text{seek})$: 1 after the robot's subgoal is done ("I put ..."); $P(g_t = v) + P(n_t = v)$ when it picks the
  object up ("I have the ...").
- search = `search` (20 s), the time a person loses looking where the object no longer is; talk = `talk` (3 s).
  The 0.4 s and 0.6 s are hand-set.
- The robot speaks when the value is positive.

**Example:** the robot puts a figure away; the person looked at it once while it was carried (dwell 0.6 s), so
$P(\text{missed}) = e^{-1.5} = 0.22$ and $V = 0.22 \times 1 \times 20 - 3 = +1.5$ s: it tells.

## 7. Main loop

- **Online** (`run.py`, `agents.Bip.act`), every simulator step:
  1. The person acts (`person.py`), recording each pick, place, open and close it completes.
  2. The env produces events (`env.py`): every 3 s a clip (`sense.py`: the verb recognizer's probabilities drawn
     from the real recognizer's calibrated outputs on held-out clips of the same true verb, `data/collab/verb_bank.npz`,
     and the detector's reads), gaze fixations, the robot's own completions.
  3. `Belief` runs S1 on each event (section 3), and S2 on surprise (section 4).
  4. Every second while the robot is free, the agent clears the claim, runs `decide` and claims the chosen
     subgoal (section 5).
  5. At each pick and place of its own, the agent weighs telling (section 6).
- **On HD-EPIC video** (`hdepic.py`): the same `Belief` with every input real: the verb recognizer's clips, the
  detected object touching a hand, real gaze on the detections; scored against the annotated subgoals.

## 8. Files

| File | Role |
| --- | --- |
| `bip.py` | all of BIP: `Params`; task graphs (`ancestors`, `log_prior`, `sample_pool`); step lists (`checked`, `fallback`); S1 (`Filter`); S2 (`S2`, `gemini`); `Belief`; `decide`; `tell_value` |
| `fit.py` | the person model's parameters as counts on HD-EPIC (2.11) |
| `sense.py` | the clips the recognizers report in simulation |
| `hdepic.py` | BIP on HD-EPIC video with real inputs, scored against the annotations |
| `agents.py` | the robot's policies: when BIP is fed and consulted, claims and speech |
| `env.py`, `person.py`, `world.py` | the domain: the events BIP observes (`docs/collab_env.md`) |
| `../../tests/test_bip.py`, `test_sense.py` | BIP on synthetic episodes (Gemini replaced by canned replies); the clip events |

```bash
# on a compute node, from the repo root
python -m pytest tests/test_bip.py tests/test_sense.py -q
python -m helper.collab.fit clip=3.0
python -m helper.collab.hdepic name=real bank=runs/action_cls/stream3/bank.npz
# on the lab 5090 (the simulator); copy the recognizer's bank to data/collab/verb_bank.npz first
python -m helper.collab.run runs/collab/x task=tidy policy=bip+tell seed=1
```
