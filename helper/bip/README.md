> This README documents the earlier BIP package in this folder. The BIP the robot runs is `helper/collab/bip.py`, described in `helper/collab/README.md`.

# Fast-slow Bayesian inverse planning (BIP)

The robot infers which subgoal the person is on, which comes next, which are done, and which orderings the
task requires; then it decides each second whether to take a subgoal over. BIP knows no particular simulator:
a domain supplies the observations (`helper/partnr` for PARTNR). Parameter names in `code font` are fields of
`config.py`, with their defaults. This README describes the scene tier (`action_tier=scene`, `completion=scene`),
used in every PARTNR experiment; the code defaults to the oracle tiers of section 4. All logarithms are natural.

## Overview: model, inference, decision

- **Generative model** (section 2): how the observations arise if everything hidden were known. A task
  graph $G$ is drawn from Gemini's prior; the person moves through the subgoals as $G$ allows; their actions,
  gaze and completions change the scene and reach the cameras. It gives a probability to every history, and
  runs forward: hidden to observed.
- **Inference** (section 3): the same model run backward. Bayes' rule turns $P(\text{observed} \mid
  \text{hidden})\,P(\text{hidden})$ into $P(\text{hidden} \mid \text{observed})$: which graph, which subgoal the
  person is on, which comes next, which are done. No second model exists; S1 and S2 only approximate the sums.
- **Decision** (section 6): the posterior, carried one step forward through the generative model, predicts
  what happens if the robot takes subgoal $v$: whether its predecessors are done ($P_{\mathrm{exec}}$), whether
  the person is on it or about to start it ($Y_v$), and how much of the person's time it saves ($\Delta$).

The whole algorithm. Inference runs on every event; the decision runs every second while the robot is free:

```python
# Once per task
p_uv = S2.prior(instruction, subgoals)          # Gemini: P(ordering u -> v is required)          (2.1)
pool = sample_graphs(p_uv)                      # 64 graphs G, each with log P(G | l)             (2.1, 3.1)
for G in pool:
    log_Z[G] = 0                                # evidence log P(o_1:t | G)                       (3.2)
    particles[G] = M x (g=-1, n=-1, S={}, D={}, weight=1/M)
C, Q = {}, {}                                   # observed done; claimed by the robot

# Every simulator step
for t in steps:
    person.act()
    obs = detections(both cameras, every 0.5 s) + recognized person actions + robots own actions
    for event in tracker.update(obs):           # updates the scene graph s_hat, emits events     (section 4)
        infer(event)                            # S1, on every event                              (section 3)
        if event.kind == "action" and surprise() > tau:      # the pool's evidence factor vs. no orderings
            S2.revise(...)                      # its graphs join the pool after the call         (section 5)
    if robot.free and t >= next_decide and t >= last_person_completion + look_s:
        decide(t)                               # every decide_every (1 s)                        (section 6)
    robot.step()                                # its skill script

def infer(event):                               # what each event contributes
    for G in pool:
        for p in particles[G]:
            if event.kind == "action":          # L[v] = P(a | g=v, s_hat), from the tracker       (2.6)
                T = transition(p, policy(p, G)) # P(g_t = v | g, n)                               (2.2, 2.3)
                p.factor = sum(T * L)           # the particle's predictive (L[N] if nothing is possible)
                p.g = sample(T * L); update p.D, p.S, p.n                                         # (2.4, 2.5)
            elif event.kind == "fixation":
                p.factor = P(x | p.g, p.n, X)   # gaze                                            (2.7)
            elif event.kind == "completion":
                p.factor = 1                    # v joins C; evidence only with completion_lik    (2.8)
        log_Z[G] += log(sum(p.weight * p.factor))       # evidence factor P(o_k | o_<k, G)        (3.2)
        for p in particles[G]:
            p.weight = normalize(p.weight * p.factor)   # Bayes update of the person's state      (3.3)
    P(G | o) = normalize(P(G | l) * exp(log_Z))         # graph posterior                         (3.4)

def decide(t):                                  # how the posterior becomes an action
    J[G, p] = P(G | o) * p.weight               # joint weight of each (graph, particle)          (3.4)
    for v in subgoals:
        P_exec = P(every predecessor of v done)            # sum of J over particles             (6.1)
        Y = P(person on v, or switching to v first)        # sum of J over particles             (6.2)
        Delta = person's time saved by the robot doing v   # from P(u done), P(v before u)       (6.3)
        V[v] = P_exec * ((1 - Y) * Delta - beta * Y) - beta_prec * (1 - P_exec)
    claim argmax(V) if max(V) > 0 else wait     # a claim enters the belief as Q
```

## 1. Variables and inputs

| Symbol | Meaning |
| --- | --- |
| $\ell$ | the instruction (text) |
| $V$ | the subgoals, given with the task (not inferred) |
| $G = (V, E)$ | the task graph; $(u, v) \in E$ means $v$ cannot start before $u$ is done |
| $s_t$ | the scene graph (section 4) |
| $g_t$ | the subgoal the person works on now; $-1$ before any evidence |
| $n_t$ | the subgoal the person plans to do next; $-1$ when none is possible |
| $S_t$ | the subgoals the person has started |
| $D_t$ | the subgoals done, as one particle believes it |
| $C_t$ | the subgoals observed done, shared by all particles |
| $Q_t$ | the robot's claimed subgoals (the person works around them) |
| $a_t$ | the person's action |
| $x_t,\ X_t$ | the fixated object and the objects visible to the person at that moment ($x_t \in X_t$) |
| $c_t$ | a subgoal seen to become done at step $t$ (a completion) |
| $o_t$ | one observation: a recognized action, a gaze fixation, or a completion read off $\hat s_t$ |

Each subgoal $v$ has a spec: an object $o_v$, a relation (on or in), a target furniture $f_v$, and optionally a
reference object $r_v$ it must end next to. Its **objects** are $O(v) = \{o_v, f_v\}$, plus $r_v$ when given. The
relation only shapes the plan's Place (2.6); the predicate (section 4) and the action match ignore it.

A **category subgoal** names a kind of object instead of one: candidates $K_v$ (every instance of the class) and a
rank $j$, read "the $j$-th fork on the table". Any candidate serves it, so two forks need two subgoals, ranks 1
and 2, which are interchangeable. Its target can be a set of furniture (any dining table). $O(v) = K_v \cup
\{f_v\}$. Instruction-level tasks name kinds, not instances, so structures are compared at this level.

In pseudocode, the state S1 keeps:

```python
pool      = K graphs G, each with log P(G | l) and log Z(G)
particles = M per graph: (g, n, S, D) and a weight
C, Q      = observed done, claimed by the robot (shared by all particles)
```

## 2. Generative model (joint factorization)

$$
\begin{aligned}
& P(G, g_{1:T}, n_{1:T}, a_{1:T}, x_{1:T}, s_{1:T}, o_{1:T} \mid \ell) \\[4pt]
={} & P(G \mid \ell) && \text{prior over orderings (2.1)} \\
& \times \prod_{t=1}^{T} \Big[\, P(g_t \mid g_{t-1}, n_{t-1}, D_{t-1}, G, Q_t) && \text{transition (2.3)} \\
& \qquad\quad \times P(n_t \mid g_t, D_{t-1}, G, Q_t) && \text{planned next (2.4)} \\
& \qquad\quad \times P(a_t \mid g_t, s_{t-1}) && \text{action (2.6)} \\
& \qquad\quad \times P(x_t \mid g_t, n_t, X_t) && \text{gaze (2.7)} \\
& \qquad\quad \times P(c_t \mid g_t) && \text{completion (2.8)} \\
& \qquad\quad \times P(s_t \mid s_{t-1}, a_t, a^R_t) && \text{effects (section 4)} \\
& \qquad\quad \times P(o_t \mid s_t, a_t) \,\Big] && \text{perception (section 4)}
\end{aligned}
$$

Time advances by events, not by clock ticks: a step $t$ carries one observation, so only the term of its kind
(action, gaze or completion) is present at that step. $g$ and $n$ change only at action steps; at the other
steps the transition is the identity.

In pseudocode, the model run forward (it simulates a person; inference only evaluates it):

```python
G = sample_prior(p_uv)                      # 2.1: keep each listed ordering with prob p_uv
g, n, S = -1, -1, {}
while subgoals remain:
    g = stay_or_switch(g, n)                # 2.3: stay w.p. sigma; else n w.p. lambda; else pi (2.2)
    if g changed or n is used up:
        n = sample(pi with g counted done)  # 2.4
    S.add(g)                                # 2.5
    a = plan(g, scene) w.p. 1 - eps, else any observable action   # 2.6
    x = an object of g w.p. w_cur, of n w.p. w_next, else anything visible   # 2.7
    scene = effects(scene, a)               # section 4; newly done subgoals are completions (2.8)
```

### 2.1 Prior over orderings $P(G \mid \ell)$

$$
P(G \mid \ell) = \prod_{(u,v) \in E,\ \text{listed}} p_{uv} \prod_{(u,v) \notin E,\ \text{listed}} \big(1 - p_{uv}\big) \prod_{(u,v) \in E,\ \text{not listed}} 0.02
$$

**Computed by** `s2.prior` and `graph.log_prior`.

1. One Gemini call per task (`prior_model`, else `model`).
2. The prompt holds:
   - a sentence framing the task ("a robot watches a person through their smart glasses");
   - $\ell$;
   - the subgoals in words, under IDs S1 to Sn shuffled with the seed, so their order leaks nothing;
   - a definition: an ordering A > B is required when the instruction asks for that order (for example
     "first ... then ...") or when the task cannot be done otherwise; a habit or a convenient order is not
     required.
3. The reply follows a JSON schema: a list of orderings (before, after, p).
4. Each $p_{uv}$ is clipped to $[0.02, 0.98]$; orderings of a subgoal with itself are dropped.
5. An ordering Gemini did not list costs a factor 0.02 when $E$ contains it, and nothing otherwise.
   $E$ is the graph's own set of orderings, not its transitive closure.
6. The reply is cached per model and task.

Two alternatives:

- `prior=flat`: every ordering gets $p_{uv}$ = `p_flat` (0.2).
- `prior=truth`: every true ordering gets 0.98, and nothing else is listed (an oracle).

### 2.2 Policy $\pi$ over the next subgoal

$$
\pi(v \mid D, C, S, G, Q) \propto
\begin{cases}
0 & v \in D \cup C \cup Q \\
\rho & v \in S \quad\text{(started, not done)} \\
1 & \text{every ancestor of } v \text{ in } G \text{ is in } D \cup C \quad\text{(ready)} \\
\delta & \text{otherwise}
\end{cases}
$$

The ancestors of $v$ are all of $v$'s transitive predecessors in $G$. The default rule `rule=complete` asks
them to be done; `rule=started` accepts started ones too. $\rho$ = `resume` (1.0), $\delta$ = `delta` (0.05).

**Computed by** `Filter.options`, for every (graph, particle) pair. Normalized over $v$ after the current
subgoal $g$ is excluded. In the transition (2.3), readiness uses the particle's $D \cup C$ as it is; only the
planned next (2.4) also counts $g_t$ as done.

### 2.3 Transition $P(g_t \mid g_{t-1}, n_{t-1}, \dots)$

At each action event the person either **stays** on the subgoal they were on, or **switches**. A switch goes
to the subgoal they had planned next, or to a fresh pick from the policy $\pi$ (2.2):

```
was the person on a subgoal g_{t-1}?
├─ yes, not done ─┬─ stay, prob σ ───────────────────────────────► g_t = g_{t-1}
│                 └─ switch, prob 1 − σ ──┐
├─ yes, done ─────── switch, prob 1 ──────┤
└─ no (the start) ── switch, prob 1 ──────┤
                                          ├─ planned n_{t-1} valid ─┬─ prob λ ─────► g_t = n_{t-1}
                                          │                         └─ prob 1 − λ ─► g_t drawn from π
                                          └─ no valid plan ──────────────────────► g_t drawn from π
```

As one formula, with $s$ the probability of staying:

$$
P(g_t = v) = s\,[v = g_{t-1}] + (1 - s)\,\mathrm{sw}(v),
\qquad
s = \begin{cases} \sigma & g_{t-1} \text{ exists and is not done} \\ 0 & \text{otherwise} \end{cases}
$$

$$
\mathrm{sw}(v) = \begin{cases}
\lambda\,[v = n_{t-1}] + (1 - \lambda)\,\pi(v) & n_{t-1} \text{ valid} \\
\pi(v) & \text{otherwise}
\end{cases}
$$

- $\sigma$ = `stay` (0.9): consecutive actions mostly serve one subgoal (Pick, then Place of the same object).
- $\lambda$ = `plan` (0.5): how often a switch follows the person's own plan. The plan is what gaze reveals
  ahead of time (2.7), so this is where gaze helps predict the next subgoal.
- $\pi$ excludes $g_{t-1}$; its readiness uses the particle's $D \cup C$ as it is.
- "Done" means in the particle's $D \cup C$. $n_{t-1}$ is valid when it exists, is not done, differs from
  $g_{t-1}$ and is not claimed by the robot.

**Example.** Subgoals $v_0, \dots, v_3$, the ordering $v_0 \to v_1$, the robot has claimed $v_3$. A particle has
$g_{t-1} = v_0$ (not done) and $n_{t-1} = v_2$. The policy over the others: $v_1$ waits for $v_0$ ($\delta = 0.05$),
$v_2$ is ready (1), $v_3$ is claimed (0), so $\pi = (0, 0.048, 0.952, 0)$. Then:

| $v$ | stay | switch to the plan | switch by $\pi$ | $P(g_t = v)$ |
| --- | --- | --- | --- | --- |
| $v_0$ | 0.9 | | | 0.9 |
| $v_1$ | | | $0.05 \times 0.048$ | 0.0024 |
| $v_2$ | | $0.1 \times 0.5 = 0.05$ | $0.05 \times 0.952$ | 0.0976 |
| $v_3$ | | | 0 | 0 |

**Computed by** `Filter.action`, as a vector over $v$ for every particle. It is applied only at action events
(section 3.3).

- **No possible subgoal** (the task looks finished to this particle, or everything left is claimed): the vector
  is zero, so the particle explains the action with the null model, $P(a \mid \text{no subgoal}) = 1 / |A_{\mathrm{obs}}(s)|$
  (the last entry of $L$, 2.6), and keeps $g$. Actions are never free: a structure that thinks the task is done
  still pays for every later action.

### 2.4 Planned next $P(n_t \mid g_t, \dots)$

$n_t$ is redrawn when $g$ changed, $n_{t-1} = g_t$, $n_{t-1}$ is done or claimed by the robot, or $n_{t-1} = -1$;
otherwise it is kept.
A redraw samples $n_t \sim \pi$, computed with $g_t$ treated as done for readiness (the person plans what
comes after finishing $g_t$) and with $g_t$ excluded. If no subgoal is possible, $n_t = -1$.

**Computed by** `Filter.action`, after $g_t$ is sampled.

### 2.5 Started and done sets

- **Started:** $S_t = S_{t-1} \cup \{g_t\}$.
- **Observed done** $C_t$: set by completion events (2.8) and by the robot's own completions. It is shared by
  all particles and only grows: a subgoal undone later stays in $C$.
- **Order at an action event:** $g_t$ is sampled, then $D$ is updated, then $S$, then $n_t$ is redrawn on the
  updated $D$.
- **Inferred done** $D_t$, per particle: when a particle's $g$ changes away from a subgoal not yet in $D \cup C$,
  that subgoal joins $D$ with probability $p_{\mathrm{done}}$ = `p_done` (0.5). This covers completions the
  scene graph missed. It is off with `completion=oracle`.

### 2.6 Action likelihood $P(a \mid g, s)$

Only manipulations are observed: Pick($o$), Place($o$, rel, $f$, $r$), Open($f$), Close($f$). Walking has no
evidence without the person's position.

**The plan** $\mathrm{plan}(g, s)$, the next manipulation(s) toward subgoal $g$ in scene $s$:

1. If $\mathrm{done}_g(s)$ holds, the plan is empty.
2. Else, if the person holds $o_g$ (for a category subgoal, any candidate): $[\mathrm{Place}(o_g, \text{rel}_g, f, r_g)]$
   for each target furniture $f$.
3. Else: $[\mathrm{Pick}(o_g)]$, preceded by $\mathrm{Open}(f)$ when $o_g$ is *in* furniture $f$ that is not open in
   $\hat s$ (never opened, or closed again). For a category subgoal, this for every candidate not in a hand and
   not already in place, so $1 / |\mathrm{plan}|$ spreads the probability over which instance the person takes.

**The likelihood:**

$$
P(a \mid g, s) = (1 - \epsilon)\, \frac{[a \text{ matches an action of } \mathrm{plan}(g, s)]}{|\mathrm{plan}(g, s)|} + \frac{\epsilon}{|A_{\mathrm{obs}}(s)|}
$$

- Two actions **match** when they have the same verb and the same object; for Place, also the same furniture.
  The relation and the reference object are not compared. The object of Open and Close is the furniture.
- When the plan is empty, only the $\epsilon$ term remains.
- $L$ has one more entry, $L[N] = 1 / |A_{\mathrm{obs}}(s)|$: the null model, an action of a person with no subgoal left
  (2.3). For the uninformative events of 2.9 every entry is 1.
- $\epsilon$ = `eps_act` (0.1).
- $|A_{\mathrm{obs}}(s)|$, the number of observable actions in $s$, is $N_{\text{free}} + 2\,N_{\text{held}}\,|F| + 2\,|F|$:
  - $N_{\text{free}}$: known objects that neither agent holds (known: every $o_v$ and $r_v$, not the furniture;
    every object of the premap; every object ever detected);
  - $N_{\text{held}}$: objects held by the person or the robot;
  - $|F|$: the number of furniture pieces in the map.

**Computed by** `actions.likelihood` on the scene before the action, giving a vector $L[v] = P(a \mid g = v, s)$
over all subgoals.

**Example:** the person is seen to Pick(cup), and subgoal $v$ is "cup on table" with the cup not yet held, so
its plan is [Pick(cup)]. With 40 observable actions, $L[v] = 0.9 + 0.1/40 = 0.9025$. Every subgoal whose plan
does not match gets $0.1/40 = 0.0025$.

### 2.7 Gaze likelihood $P(x \mid g, n, X)$

A fixation is gaze held for at least `fix_min` (0.3 s) on one object $x$, among the visible objects $X$.

$$
U_v(x) = \frac{[x \in O(v)]}{\max\big(1, |O(v) \cap X|\big)},
\qquad
P(x \mid g, n, X) = \Big( w_{\mathrm{cur}}\, U_g(x) + w_{\mathrm{next}}\, U_n(x) + \frac{w_{\mathrm{noise}}}{|X|} \Big)^{\gamma}
$$

- $U_{-1} = 0$.
- $w_{\mathrm{cur}}, w_{\mathrm{next}}, w_{\mathrm{noise}}$ = `w_cur` (0.17), `w_next` (0.05), `w_noise` (0.78), fitted by EM
  on HD-EPIC gaze.
- $\gamma$ = `gaze_temper` (1.0).
- When $O(g) \cap X$ is empty or $n = -1$, the terms do not sum to 1 over $X$; the likelihood is not renormalized
  (A7).

**Computed by** `Filter.fixation`, per particle. The domain supplies the object vectors
$u[v] = U_v(x)$ and $1/|X|$ (`observe.fixation`).

### 2.8 Completion likelihood

When subgoal $v$ is seen to become done by the person (section 4), $v$ joins $C$. By default that is all: the
completion is not evidence. With `completion_lik` on, it is also evidence of who did it:

$$
P(\text{completion of } v \mid g) = \begin{cases} 1 - \epsilon_d & g = v \\ \epsilon_d & g \ne v \end{cases}
$$

$\epsilon_d$ = `eps_done` (0.05). It is off because a structure without $v$ sees no completion event for $v$ and
skips its cost: on the 20 dev episodes, a structure missing a real subgoal then outscored the true one 55% of
the time. The Place action already carries the evidence of who did it.

When the robot's own action completes $v$, $v$ joins $C$ and leaves $Q$, with no likelihood: it says
nothing about the person.

**Computed by** `Filter.completion` and `Filter.robot_done`.

### 2.9 Uninformative actions

Two events carry no information about which subgoal, $L[v] = 1$ for all $v$, but let the transition run:

- **Start:** at $t = 0$, so every particle draws a first subgoal from $\pi$.
- **Moves on:** right after each person completion, so the particles that were on the finished subgoal move
  to another.

Both are action events in every respect: they can trigger resampling, they update S1's reads (section 5), and
their surprise is always 0.

## 3. Inference

Write $z_t = (g_t, n_t, S_t, D_t)$ for the person's hidden state. The scene is tracked, not marginalized: it is
replaced by the point estimate $\hat s_t$ of section 4, so every term conditions on $\hat s_{1:t}$. The chain
rule, then Bayes' rule on the graph, gives three terms:

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
        for p in particles[G]:                                  # p = (g, n, S, D, weight)
            if event.kind == "action":                          # L[v] = P(a | g=v, s_hat)       (2.6)
                pi = policy(p.D | C, p.S, G, Q), without p.g    # (2.2)
                s = stay if p.g exists and is not done else 0
                sw = plan * onehot(p.n) + (1 - plan) * pi if p.n is valid else pi
                T = s * onehot(p.g) + (1 - s) * sw              # (2.3)
                if sum(T) == 0:                                 # nothing possible: the null model
                    p.factor = L[N]; continue                   # (2.3, 2.6)
                p.factor = sum(T * L)
                g = sample(T * L)                               # locally optimal proposal
                if g != p.g and p.g not in p.D | C and rand() < p_done:
                    p.D.add(p.g)                                # a missed completion           (2.5)
                p.S.add(g)
                if g != p.g or p.n in (-1, g) or p.n in p.D | C | Q:
                    p.n = sample(policy(p.D | C | {g}, p.S, G, Q), without g)   # (2.4)
                p.g = g
            elif event.kind == "fixation":                      # x among the visible X
                p.factor = w_cur * U(p.g, x) + w_next * U(p.n, x) + w_noise / len(X)   # (2.7)
            elif event.kind == "completion":
                p.factor = (1 - eps_d if p.g == event.v else eps_d) if completion_lik else 1   # (2.8)
        log_Z[G] += log(sum(p.weight * p.factor for p in particles[G]))   # (3.2), weights normalized
        for p in particles[G]:
            p.weight = normalize(p.weight * p.factor)                       # (3.3)
        if 1 / sum(p.weight ** 2 for p in particles[G]) < M / 2:
            systematic_resample(particles[G])
    if event.kind == "completion":
        C.add(event.v)
```

### 3.1 Term (1): the prior

The prior is 2.1, evaluated exactly for each graph of a finite **pool** $\mathcal{G}$, built by `graph.sample_pool`:

1. **The most probable graph:** add the listed orderings with $p_{uv} \ge 0.5$ in decreasing $p_{uv}$,
   skipping any that would close a cycle.
2. **Samples:** shuffle the listed orderings, keep each with probability $p_{uv}$, and skip cycles. Repeat
   until the pool holds `pool` (64) distinct graphs, or after $20 \times 64$ tries.
3. **S2's revisions** join later (section 5).

### 3.2 Term (2): the evidence

$$
P(o_{1:t} \mid G) = \prod_{k=1}^{t} P(o_k \mid o_{<k}, G) = Z_t(G),
\qquad
P(o_k \mid o_{<k}, G) = \sum_{z_k} P(o_k \mid z_k, \hat s)\; P(z_k \mid G, o_{<k})
$$

Each factor is how well graph $G$ predicted observation $o_k$. It is the normalizer of term (3)'s update.
The pool's graphs are distinct as sets of orderings; two graphs with the same closure can both be in it.

**Computed by** `Filter._weigh`. With the particles of 3.3, at each event:

$$
P(o_k \mid o_{<k}, G) \approx \sum_m w^{(m)} \ell^{(m)}_k,
\qquad
\log Z(G) \mathrel{+}= \log \sum_m w^{(m)} \ell^{(m)}_k
$$

with normalized weights $w^{(m)}$ and the particle's predictive $\ell^{(m)}_k$:

- for an action, $\ell^{(m)}_k = \sum_{v} P(g_k = v \mid z^{(m)}_{k-1}) \, L[v]$, with the transition 2.3 and $L$ from 2.6;
- for a fixation or a completion, $\ell^{(m)}_k$ is the likelihood 2.7 or 2.8 at the particle's current state (1 for a
  completion by default).

**As a reward** for a structure proposer (`evidence.log_evidence`): one graph, `evidence_particles` (256)
particles, `evidence_seeds` (4) seeds. $\log \hat Z$ is biased low by about half its variance across seeds (0.26 nats
at 64 particles, more for larger structures; 0.06 at 256), so the estimate is the seeds' mean plus half their
variance.

### 3.3 Term (3): the person's state, a particle filter per graph

The Bayes filter: predict with the transition, then update with the observation and normalize by 3.2's
factor.

$$
P(z_k \mid G, o_{1:k}) = \frac{P(o_k \mid z_k, \hat s) \sum_{z_{k-1}} P(z_k \mid z_{k-1}, G)\, P(z_{k-1} \mid G, o_{<k})}{P(o_k \mid o_{<k}, G)}
$$

- **The assumption:** the observation depends only on the current state and the tracked scene,
  $P(o_k \mid z_k, G, o_{<k}) = P(o_k \mid z_k, \hat s)$.
- **Which scene:** $\hat s$ is $\hat s_{k-1}$ for an action (scored before its effects) and $\hat s_k$ for a
  completion.
- **When the state moves:** the transition $P(z_k \mid z_{k-1}, G)$ is 2.3 for $g$, 2.4 for $n$ and 2.5 for $S$
  and $D$, applied only at action events. At a fixation or completion event, $z_k = z_{k-1}$.

**Computed by** `s1.Filter`, with `particles` ($M = 64$) weighted particles per graph. A particle is
$(g, n, S, D)$, with $S$ and $D$ stored as bitmasks.

- **Action event** (`Filter.action`), for each particle $m$:
  1. Compute the transition vector $T^{(m)}[v] = P(g_k = v \mid z^{(m)}_{k-1})$ (2.3).
  2. Multiply the weight by $\ell^{(m)}_k = \sum_v T^{(m)}[v] L[v]$; this also feeds 3.2.
  3. Sample $g_k \propto T^{(m)}[v] L[v]$, the locally optimal proposal.
  4. Update $D$ (2.5), then $S$, then redraw $n$ (2.4) on the updated $D$.
- **Fixation or completion event** (`Filter.fixation`, `Filter.completion`): multiply each weight by the
  likelihood at the particle's state.
- **Resampling** (`Filter._resample`): when a graph's effective sample size $1/\sum_m (w^{(m)})^2$ falls below
  $M/2$, resample its particles systematically and reset the weights to $1/M$.

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

`Belief.estimate` reports:

- the current subgoal: $\arg\max_v P(g_t = v)$;
- the next three: subgoals with $P(v \text{ done}) < 0.5$ other than the current, ranked by $P(n_t = v)$, then by
  $P_{\mathrm{exec}}(v)$ (6.1);
- the done subgoals: those with $P(v \text{ done}) \ge 0.5$;
- every ordering with $P(u \prec v) \ge 0.05$; scoring counts $\ge 0.5$ as stated.

### 3.5 Approximations

| | Approximation | Where |
| --- | --- | --- |
| A1 | the scene graph is a point estimate, not a distribution | `track.Tracker` |
| A2 | detections and recognized actions are taken at face value; their noise is not modelled | `track.py` |
| A3 | graphs come from a finite pool, not all DAGs on $V$ | `graph.sample_pool`, S2 |
| A4 | the person's state given a graph is a particle approximation | `s1.Filter` |
| A5 | the planned next subgoal is proposed from $\pi$, without the coming evidence | `Filter.action` |
| A6 | a missed completion is inferred only when the person leaves a subgoal, with $p_{\mathrm{done}}$ | `Filter.action` |
| A7 | the gaze likelihood is not renormalized when the subgoals' objects are out of view | `Filter.fixation` |

## 4. The scene graph $\hat s_t$

**Nodes:** furniture (a known static map), objects, the person H and the robot R. **Edges:** on($o, f$), in($o, f$),
next_to($o, o'$), holding(agent, $o$).

**Start** (`premap`, on): the robot mapped the home before the task. $\hat s_0$ holds every object's furniture and
next-to edges at that time; the map goes stale as objects move.

**Updates** (`scene.Scene`, `track.Tracker`), in time order:

- **Detections** from the person's head camera and the robot's camera, every `view_every` (0.5 s): triples
  ($o$, on/in, $f$), ($o$, next_to, $o'$), (agent, holding, $o$). An object seen on furniture is set there; if
  it moved, its old next-to edges are removed. An object seen in a hand is lifted from its furniture.
- **The person's recognized actions** and **the robot's own actions:**
  - Pick($o$) lifts $o$ and sets holding;
  - Place($o$, rel, $f$, $r$) sets $o$ rel $f$ and, if $r$ is given, next_to($o$, $r$);
  - Open and Close mark furniture open or closed.

**Predicate** of subgoal $v$ (`Scene.holds`): $o_v$ rests on or in $f_v$ (the two are not told apart), nobody holds
it, and, if $r_v$ is given, next_to($o_v$, $r_v$). A category subgoal of rank $j$ holds once $j$ of its candidates
rest on or in its target furniture.

**Events** (`track.Tracker`): after every update, each subgoal whose predicate starts to hold emits:

- a robot completion (2.8), when the robot's own action made it hold;
- otherwise a person completion (2.8) followed by a "moves on" event (2.9).

A recognized person action emits, in this order: an action event with $L$ from 2.6, scored on the scene before
the action; then the completions it causes; then, after each person completion, the "moves on" event.

**Perception** $P(o_t \mid s_t, a_t)$ belongs to the domain. In PARTNR (`helper/partnr/perceive.py`):

- a visible object is missed with `view_miss` (0.2), and put on the wrong furniture with `view_rel_err` (0.1);
- a person action is missed with `act_miss` (0.2), and names a wrong object of the same kind with
  `act_arg_err` (0.15).

BIP does not model these errors (A2).

In pseudocode (`track.Tracker`):

```python
on detections:           scene.see(triples)
on a recognized action:  emit an action event, L scored on the scene; then scene.apply(action)
after either:            for each subgoal that just became done, emit its completion (and "moves on")
```

## 5. S2: revisions on surprise

- **Surprise** (`Belief.feed`, action events only):
  - Let $p_{\mathcal{G}} = \sum_G P(G \mid o_{<k}, \ell)\, P(o_k \mid o_{<k}, G)$, the pool's prediction of the
    action.
  - Let $p_\emptyset$ be the prediction of a reference filter whose only graph has no orderings.
  - The surprise is $\ln p_\emptyset - \ln p_{\mathcal{G}}$ (natural logarithm): how much better "no orderings"
    explains the action than the pool does.
- **Trigger:** S2 is called when all four hold:
  - the surprise exceeds $\tau$ = `surprise` (1.4);
  - no call is pending;
  - at least `cooldown` (20 s) has passed since the last call;
  - fewer than `max_calls` (4) calls have been made.
- **Prompt** (`s2.revise`):
  - the prior's header;
  - the three leading graphs with their probabilities;
  - S1's last 30 reads ("12 s started S3", "40 s S2 complete");
  - one sentence on the surprise ("the person appears to work on S4 while S2 is not complete; the
    hypotheses explain this poorly");
  - the last `s2_frames` (3) ego frames, `s2_frame_gap` (2 s) apart, with the gaze drawn on them.
- **Reply:** up to `revisions` (3) corrected graphs. Model: `model` (Gemini 3.5 Flash Lite).
- **Joining the pool** (`Belief.add`): after the call's measured latency, each new acyclic graph is run through
  the whole event history in a fresh filter, then joins. Its weight is therefore its own prior times its own
  evidence.
- **Without a model** (`s2=repair`): drop every ordering into a subgoal that was observed starting before the
  ordering's source was done (`graph.repair`).

In pseudocode (`Belief.feed`, `Belief.fire`, `Belief.add`):

```python
if action and surprise > tau and S2 is free:
    graphs = gemini(instruction, top graphs, recent reads, ego frames)
    after its latency: score each graph on the whole history, then add it to the pool
```

## 6. Decision (`decide.py`)

Only while the robot has no subgoal: every `decide_every` (1 s), and at least `look_s` after the start and
after each observed completion by the person, when the person picks their next subgoal (`look_s` is 0 by
default and 3 s in the experiments):

$$
\begin{aligned}
V(\text{execute } v)
=\; & P_{\mathrm{exec}}(v)\, \big[ (1 - Y_v)\, \Delta(v) - \beta\, Y_v \big] \\
-\; & \beta_{\mathrm{prec}}\, \big( 1 - P_{\mathrm{exec}}(v) \big) \\[4pt]
V(\text{wait}) =\; & 0
\end{aligned}
$$

The robot executes $\arg\max_v V$ when it is positive (ties go to the subgoal listed first), and waits
otherwise. `look_s` is a minimum delay: the first decision comes once both conditions hold.

In pseudocode, `decide(t)` of the overview in full:

```python
def decide(t):
    d = mean gap between completions, or 20 s before the first           # (6.4)
    rem = max(1, d * r((t - t_read) / d))                                # time left on the current subgoal
    on, done, started = P(g_t = u), P(u done), P(u started) for every u  # read-outs (3.4)
    for v in subgoals:
        P_exec = sum(J[G, p] for all G, p if every ancestor of v in G is in p.D | C)          # (6.1)
        Y = sum(J[G, p] for all G, p if p.g == v or (p.n == v and rem < min(d_R, notice)))  # (6.2)
        work = [(on[u] * min(d, rem) + (1 - on[u]) * d) * (1 - done[u]) * (u not in Q) for u in subgoals]
        free = sum((1 - P(v before u)) * work[u] for u in subgoals) - work[v]
        Delta = d - max(0, d_R - free)                                                       # (6.3)
        V[v] = P_exec * ((1 - Y) * Delta - beta * Y) - beta_prec * (1 - P_exec)
        if v in C | Q or done[v] > 0.5 or started[v] > 0.5 or its object was never located:
            V[v] = -inf                                                                      # (6.5)
    v = argmax(V)
    claim(v) if V[v] > 0 else wait
```

Each term, with $J$ the joint weight of 3.4:

### 6.1 $P_{\mathrm{exec}}(v)$: the predecessors are done

$$
P_{\mathrm{exec}}(v) = \sum_{G,m} J_{G,m}\, \big[\text{every ancestor of } v \text{ in } G \text{ is in } D^{(m)} \cup C\big]
$$

**Computed by** `exec_prob`.

### 6.2 $Y_v$: the person works on $v$ before seeing the robot

$$
Y_v = \sum_{G,m} J_{G,m}\, \Big[ g^{(m)} = v \;\text{ or }\; \big( n^{(m)} = v \text{ and } \mathrm{rem}(g^{(m)}) < \min(d_R, \text{notice}) \big) \Big]
$$

- $P(v \text{ started})$, $P(v \text{ done})$: the read-outs of 3.4.
- notice = `notice_s` (2 s): the time the person needs to see the robot's claim.
- $d_R = d \cdot$ `robot_slowdown` (1.0): the robot's time for a subgoal.
- $\mathrm{rem}(g)$: the person's time left on their current subgoal (6.4); $\mathrm{rem}(-1) = 0$.

**Computed by** `overlap`.

### 6.3 $\Delta(v)$: the person's time saved

$$
\begin{aligned}
\mathrm{work}(u) &= \Big[ P(g_t = u)\, \min\big(d, \mathrm{rem}(u)\big) + \big(1 - P(g_t = u)\big)\, d \Big] \big(1 - P(u \text{ done})\big)\, [u \notin Q] \\
\mathrm{free}(v) &= \textstyle\sum_{u \in V} \big(1 - P(v \prec u)\big)\, \mathrm{work}(u) - \mathrm{work}(v) \\
\Delta(v) &= d - \max\big(0,\ d_R - \mathrm{free}(v)\big)
\end{aligned}
$$

$\mathrm{work}(u)$ is the person's work left on $u$: the remaining time if they are on it, all of $d$ if not.
$\mathrm{free}(v)$ is the work the person can do meanwhile that does not depend on $v$ (the sum includes $u = v$,
with $P(v \prec v) = 0$, and the last term removes it). The robot saves the full
$d$ unless the person runs out of such work before the robot finishes.

**Computed by** `saved`.

### 6.4 Durations and costs

- $d$: the mean gap between successive observed completions, the robot's included, counted from $t = 0$
  (completions at 30 s and 50 s give gaps 30 and 20, so $d = 25$), or 20 s before the first. It is the same for
  every subgoal.
- $\mathrm{rem}(u) = \max\big(1,\ d \cdot r\big((t - t_{\text{read}}) / d\big)\big)$: the person's expected time left on their
  subgoal, given the time $t - t_{\text{read}}$ already spent on it. $t_{\text{read}}$ is when the argmax of $P(g_t)$
  last changed ($t_{\text{read}} = 0$ until then). $r(x) = E[D - x \mid D > x]$ is the mean residual time of the
  gaps $D$ between a person's successful placements, in units of the episode's mean gap, measured on the
  PARTNR human traces (`rem_curve`, at $x = 0, 0.25, \dots, 3$; `hitl.residual`). It is the same for every
  subgoal $u$.
- $\beta = $ `beta_rel` $\times\, d$ (1.0), the cost of a conflict.
- $\beta_{\mathrm{prec}} = $ `beta_prec_rel` $\times\, d$ (2.0), the cost of an ordering violation.

### 6.5 Never executed

$V(\text{execute } v) = -\infty$ when any of these holds:

- $v \in C \cup Q$;
- $P(v \text{ done}) > 0.5$;
- $P(v \text{ started}) > 0.5$;
- the robot has never located $o_v$ in $\hat s$.

Asking the person is implemented (`decide.py`) but switched off.

### 6.6 Subgoals not given (`structures.py`)

When the subgoals themselves are uncertain, BIP keeps a few candidate structures $H$ (a task family's recipes),
each with its own tracker and S1 filter on the same observations. Subgoals match across structures by name
("plate1").

$$
w(H) \propto P(H \mid \ell)\, Z_t(H), \qquad P_{\mathrm{exist}}(v) = \sum_{H \ni v} w(H)
$$

$$
V(\text{execute } v) = \sum_{H \ni v,\ f_H(v) = f^*} w(H)\, V_H(v) - \beta_{\mathrm{wrong}} \big(1 - P_{\mathrm{right}}(v)\big)
$$

- $V_H(v)$ is 6.1 to 6.5 inside structure $H$; $\beta_{\mathrm{wrong}}$ = `beta_wrong_rel` $\times\, d$ (1.0), the cost of
  a wrong placement.
- $f^*$ is the furniture most of the weight sends $v$ to (the robot goes there), and $P_{\mathrm{right}}(v)$ the weight of
  the structures that contain $v$ and send it to $f^*$. A placement is wrong, and saves nothing, when the subgoal
  does not exist or goes elsewhere.
- $v$ is never executed when $P_{\mathrm{exist}}(v) < 0.5$, or when a structure of weight above 0.2 rules it out.
- $P(H \mid \ell)$: `recipe_prior=llm` (Gemini's probability per object class, multiplied over the family's
  classes, `families.recipe_prior`) or `uniform`.

### 6.7 What to ask (`dialogue.py`)

When the robot is not told the task, the human model sets what to ask and the slow VLM (S2) asks it.

1. **Plans from the conversation.** S2 reads the conversation, the premapped home (furniture by room and
   neighbours, objects by furniture) and where the person has put things, and gives the 3 most likely plans with
   their probabilities: per object class, how many, where they go (one furniture per entry), what they go next to,
   and orderings between classes. Each plan is a structure of 6.6, covering the whole task including what is done.
   A subgoal is named by what it is ("spoon1"), so plans can disagree on where it goes. Where the conversation
   leaves the furniture open, the plans differ on it. "Next to" is symmetric: the first object of a group needs no
   neighbour, the later ones go next to it.
2. **Needs.** Where the plans disagree on a class (whether it is in the task, how many, where it goes, what it
   goes next to), the answer splits the plans into groups. Its value is

$$
\mathrm{VOI} = \sum_{a} P(a) \sum_{v} \max\big(0, V_a(v)\big) - \sum_{v} \max\big(0, V(v)\big),
$$

   with $V_a$ the decision values under the plans answer $a$ leaves: what the robot could do over the rest of the
   task, not only next.
3. **Asking.** S2 writes one question for the most valuable need, describing furniture as a person would.
   The robot asks while the best need beats `ask_cost` (8 s), up to `max_questions`; after every answer it
   re-plans and asks again if a need remains. Before carrying an object to an uncertain destination, it asks
   where it goes when $(d + \beta_{\mathrm{wrong}}) \big(1 - P(\text{most likely destination})\big)$ beats `ask_cost`.
   It asks about the same need again only after the person has completed another subgoal, since the person may
   not have worked it out yet. When no plan has anything left for the robot while the person is still at work, a
   need "what else does the task need?" is worth about one subgoal of the person's time, $d$: a step no plan
   contains cannot show up as a disagreement between plans.
4. **Surprise.** When the person places an object where no plan sends its class, S2 re-plans with that placement.
5. **Cost.** Each exchange costs the person `talk_overhead` (1.5 s) plus the words at `talk_wps` (2.5 per s).
   With `know_model`, a question's value is multiplied by the chance the person can answer it, $(1 + s)/(2 + n)$
   after $n$ answers of which $s$ settled what was asked (the plans then agree on it): the robot learns that a
   partner who works the task out as they go often cannot answer yet.
6. **Who decides to ask** (`talk_policy`): `model` asks when the best need beats `ask_cost`; `vlm_num` gives S2
   the open needs with their values in seconds and the cost of asking, and S2 decides whether and what to ask;
   `vlm` lets S2 decide alone, without the values.
7. **One decision.** Executing a subgoal, waiting and asking are scored by the same human model: $V(v)$ for
   executing $v$, 0 for waiting, $\mathrm{VOI} - c$ for asking. The robot asks when $\mathrm{VOI} > c$ and keeps
   acting by $V$; S2 only words the question (or, with `vlm_num`, also judges it).
8. **Starts** (`talk_start`): an open question, "tell me every step", or what the person says first
   (`say_file`: a casual request per task; else PARTNR's instruction). Baseline (`talk_policy=vlm`): S2 decides
   alone whether and what to ask.

The simulated person (`helper/partnr/person.py`) may work the task out as they go (`person_horizon`): they know a
share of the steps, in an order that respects the task's orderings, and work out the next one when one is done.
Their voice (`helper/partnr/voice.py`, a small LM) answers only from the steps worked out so far and says when it
is not sure yet.

What the robot sees of its partner (`partner_view`): `ego` adds the person's glasses to the robot's own camera
(their view of the scene, their recognized actions, the object in hand, and gaze when `gaze`); `robot` keeps only
the robot's camera, which recognizes a person's action only when it sees the action's target; `acts` keeps the
robot's camera and adds every action of the person, as an ideal outside action recognizer would give them.

## 6.8 The partner's beliefs (`mind.py`)

BIP also models what the person believes about where things are. For each object o, b(o) is a distribution over the
places the person may think o is; the scene graph gives where o is. P(false belief) = 1 - b(o)[actual(o)].

- Gaze enters as the entity it falls on, never as pixels, and its dwell accumulates as evidence.
- The person registers their own moves always, and another agent's move of o only by attending o while it is in
  their view: dwell on o registers it with 1 - exp(-dwell / `tau_reg`) (0.15 s; 0.3 s of dwell gives 86%); o in view
  with the gaze elsewhere registers with hazard `q_vis` (0.02 per s). Attending o in the robot's hand within
  `follow_s` (2 s) of its arrival shows where it went (`q_fix`, 0.9).
- When the person starts looking for o (S1's current or next subgoal uses o), their dwell on o's candidate places is
  accumulated over the first `look_s` (0.5 s) of looking at any of them; each place's share of it, weighted by
  `w_seek` (0.6) against chance, is evidence about b(o), once per search. `destination(o)` reads where they will go.
- After the robot tells the person where o is, they know (`inform`).
- The slow path: when P(false belief) is in `s2_band` (0.25 to 0.75) as a search starts, Gemini reads the person's view
  during the move (`s2.saw`) and returns P(they saw it), which joins b(o) as a likelihood.
- Telling: V(tell o) = P(false belief) P(looks for o soon) `search` (20 s) - `talk` (3 s) (`decide.tell_value`).

## 7. Main loops

- **Online** (`helper/partnr/robot.py`), every simulator step:
  1. The person acts.
  2. The cameras and the action recognizer produce observations.
  3. `Tracker` updates $\hat s$ and emits events (section 4).
  4. `Belief.feed` runs S1 (section 3), and S2 on surprise (section 5).
  5. Every second, `decide` chooses to execute or wait (section 6).
- **Offline** (`replay.py`):
  1. The same events, from a recorded stream.
  2. The estimate (3.4) is written every `grid` (2 s).
  3. `score.py` compares it with the truth.

The overview's pseudocode is this loop.

## 8. Files

| File | Role |
| --- | --- |
| `config.py` | every setting, grouped by the term it belongs to |
| `graph.py` | task graphs: ancestors, closure, $\log P(G \mid \ell)$, the pool, the model-free repair |
| `scene.py` | the scene graph and the subgoal predicates |
| `actions.py` | the plan and $P(a \mid g, s)$ |
| `track.py` | observations to events through $\hat s$ |
| `observe.py` | events for each perception tier |
| `s1.py` | the particle filter: policy, transition and every likelihood |
| `s2.py` | Gemini: the prior and the revisions |
| `belief.py` | S1 and S2 together, one event at a time |
| `structures.py` | subgoals not given: candidate structures, their weights and the mixed decision (6.6) |
| `dialogue.py` | plans from the conversation, the needs worth asking about and the questions (6.7) |
| `evidence.py` | $\log P(o \mid H)$ of one structure, debiased, as a reward (3.2) |
| `decide.py` | execute, ask or wait |
| `replay.py`, `score.py` | the offline loop and its metrics |
| `mind.py` | the partner's beliefs about where things are (6.8) |
| `io.py`, `live.py` | the inputs and commands of a robot, and the runtime that serves the robot and the replays |
| `lock.py`, `locked.json` | the frozen settings the live loop loads (docs/architecture.md) |
| `fb_eval.py`, `fb_vlm.py`, `fb_live.py` | false-belief detection: the partner's beliefs, VLM baselines, the live loop |

```bash
# on a compute node, from the repo root
python -m helper.bip.replay name=scene action_tier=scene completion=scene streams_dir=data/bip_partnr_dev/streams
python -m helper.bip.score name=scene streams_dir=data/bip_partnr_dev/streams
```
