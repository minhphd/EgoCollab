"""Every setting of fast-slow Bayesian inverse planning (BIP) and of the PARTNR testbed, in one place.

Override fields on the command line as key=value, e.g.
    python -m helper.bip.replay name=oracle action_tier=oracle completion=oracle
"""

import ast
from dataclasses import asdict, dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    name: str = "debug"
    runs: str = str(ROOT / "runs/bip")
    streams_dir: str = str(ROOT / "data/bip_partnr/streams")  # recorded episodes as BIP streams
    cache: str = str(ROOT / "data/bip_partnr")  # S2 priors per model and episode
    episodes: int = 0  # cap on the number of streams (0 = all)
    grid: float = 2.0  # s between estimates written out
    trace: bool = False  # keep the belief after every event (for clips)

    # Observations o_t: which perception tier feeds S1 (observe.py). "scene" builds the scene graph s_t from both
    # views and both agents' actions (track.py) and scores actions with P(a | g, s) (actions.py)
    action_tier: str = "oracle"  # scene, oracle (skill and target), noisy (target confused), ego (held object only)
    completion: str = "oracle"  # scene (from s_t), oracle (exact), noisy (missed and late), latent (inferred by S1)
    view_every: float = 0.5  # scene: s between detections in each view
    view_miss: float = 0.2  # scene: P(a visible object goes undetected)
    view_rel_err: float = 0.1  # scene: P(a detected object is placed on the wrong furniture)
    act_miss: float = 0.2  # scene: P(a person's Pick, Place, Open or Close goes unrecognized)
    act_arg_err: float = 0.15  # scene: P(a recognized action names a wrong object)
    next_to_m: float = 1.0  # scene: objects on the same furniture this close are next to each other (PARTNR's)
    premap: bool = True  # scene: the robot mapped the home before the task; s_0 is that map
    ego_delay: float = 0.0  # s the person's egocentric items (view, actions, held object, gaze) reach the robot late
    partner_view: str = "ego"  # ego: the robot also gets the person's glasses (their view, actions, held object,
    # gaze if cfg.gaze); robot: only its own camera, which sees a person's action when it sees the action's target;
    # acts: its own camera and every action of the person, as an ideal outside action recognizer would give them
    action_acc: float = 0.7  # noisy: P(the read target is the true one)
    hand_miss: float = 0.0  # ego: P(the hand-object detector misses a held object)
    hand_acc: float = 1.0  # ego: P(a read names the held object); otherwise a random visible object
    fix_acc: float = 1.0  # P(a fixation's object label is right); otherwise a random visible object
    det_rate: float = 0.7  # noisy completions: share detected
    det_delay: float = 5.0  # noisy completions: mean delay (s)

    # Action likelihood P(a_t | g_t, s_t)
    eps_act: float = 0.1  # scene: P(an action off the subgoal's plan), spread over the observable actions
    beta_action: float = 4.0  # other tiers: L[v] = exp(beta * share of the action's entities in O(v))

    # Person model: transition P(g_t | g_t-1, n_t-1), policy pi over the next subgoal, completions
    rule: str = "complete"  # an ordering u -> v lets v start once u is "complete" or "started"
    stay: float = 0.9  # P(the next action keeps the current subgoal | it is incomplete)
    plan: float = 0.5  # P(a switch goes to the planned next subgoal)
    resume: float = 1.0  # pi: weight of returning to a started subgoal, against 1 for an executable new one
    delta: float = 0.05  # pi: weight of starting a subgoal whose predecessors are not done
    evidence_particles: int = 256  # log P(o | H) as a reward (evidence.py): bias -0.06 nats at 256, -0.26 at 64
    evidence_seeds: int = 4
    completion_lik: bool = False  # completions as evidence of who did them; off keeps structures comparable (README 2.8)
    eps_done: float = 0.05  # P(a subgoal completes while another one is being worked on)
    p_done: float = 0.5  # P(leaving an unfinished-looking subgoal means it is done), when completions can be missed

    # Gaze likelihood P(x_t | g_t, n_t, X_t): a mixture over the current subgoal's, next subgoal's or any visible object
    gaze: bool = True
    w_cur: float = 0.17
    w_next: float = 0.05
    w_noise: float = 0.78
    gaze_temper: float = 1.0

    # S1: one particle filter per graph of the pool
    particles: int = 64  # per graph
    pool: int = 64  # graphs sampled from the prior P(G | l)
    seed: int = 0

    # S2 (Gemini): the prior P(G | l) once per task, revisions on surprise
    prior: str = "llm"  # llm (Gemini reads the instruction), flat (every ordering at p_flat), truth (oracle graph)
    p_flat: float = 0.2
    s2: str = "llm"  # revisions: llm (Gemini), repair (drop falsified orderings, no model), never
    surprise: float = 1.4  # revise when log P(event | no orderings) - log P(event | pool) exceeds this
    cooldown: float = 20.0  # s between S2 calls
    max_calls: int = 4  # S2 calls per episode, to keep the cost down
    revisions: int = 3  # graphs per S2 revision
    model: str = "gemini-3.5-flash-lite"  # the cheapest current Gemini on Vertex (3.8 Flash Lite is not served)
    prior_model: str = ""  # model for the one prior call per task ("" = model)
    thinking: str = ""  # the model default (the Python 3.9 SDK has no thinking levels; Lite thinks least)
    s2_frames: int = 3  # ego frames with gaze per revision
    s2_frame_gap: float = 2.0

    # PARTNR testbed (helper/partnr): episodes, simulation and recording
    split: str = "val_mini"
    only: tuple = ()  # episode ids to run (empty: every eligible one, up to `episodes`)
    kinds: tuple = ("is_on_top", "is_inside", "is_next_to", "is_on_floor")  # rearrangement propositions only
    min_subgoals: int = 3
    ordered_only: bool = False  # keep only tasks with required orderings
    record_dir: str = str(ROOT / "data/bip_partnr")  # streams/, frames/, videos/, episodes/
    fps: float = 30.0  # env steps per simulated second (agents walk about 2.5 m/s)
    video_every: int = 2  # keep every k-th step in the videos
    head_px: int = 512
    max_steps: int = 9000
    ffmpeg: str = str(Path.home() / ".conda/envs/egocollab/bin/ffmpeg")
    workers: int = 4  # simulator processes, one GPU each

    # The person: scripted humanoid. "hitl" is calibrated on 5,041 PARTNR human-in-the-loop episodes (hitl.py);
    # "uniform" picks a random allowed subgoal and always yields to the robot
    person: str = "hitl"
    choice_rank: float = 0.4  # hitl: P(v) prop. to exp(-choice_rank * instruction rank - choice_dist * distance in m)
    choice_dist: float = 4.4  # hitl: people take the nearest remaining object 80% of the time
    p_misplace: float = 0.28  # hitl: P(a first placement goes to the wrong furniture; fixed later)
    idle_ratio: float = 0.72  # hitl: pause before a subgoal, times the last subgoal's work time (42% of time idle)
    p_yield: float = 0.9  # hitl: P(the person gives up a subgoal the robot has taken)
    p_detour: float = 0.25  # before a subgoal, walk to a random piece of furniture first
    p_pause: float = 0.2  # before a subgoal, stand still for pause_s
    pause_s: float = 3.0
    max_fails: int = 2  # failed attempts before the person gives a subgoal up
    action_every: float = 3.0  # s between repeated reads of a running skill
    notice_s: float = 2.0  # the person avoids the robot's subgoal this long after the robot starts it

    # Synthesized gaze (helper/partnr/gaze.py), calibrated on HD-EPIC, independent of S1's gaze model
    gen_dwell_s: float = 0.6  # median dwell on one target
    gen_dwell_sigma: float = 0.5
    fix_min: float = 0.3  # s on one entity before it counts as a fixation (as in HD-EPIC gaze3d)
    gen_w_cur: float = 0.3  # share of free fixations on the current subgoal's entities
    gen_w_next: float = 0.1
    p_prime_pick: float = 0.65  # HD-EPIC: 65% of pick-ups primed by gaze, median lead 3.1 s
    lead_pick: float = 3.1
    p_prime_place: float = 0.32  # HD-EPIC: 32% of put-downs, median lead 1.8 s
    lead_place: float = 1.8
    lead_sigma: float = 0.8
    gaze_jitter: float = 0.02
    center_bias: float = 0.15

    # The robot (closed loop, decide.py): idle, bip (belief + decision rule), graph (belief-free control), oracle
    robot: str = "idle"
    decide_every: float = 1.0  # s between robot decisions
    look_s: float = 0.0  # s the robot watches after the start and after each completion before choosing
    explore: bool = False  # with nothing worth doing, the robot walks to unvisited furniture to map objects
    beta_rel: float = 1.0  # cost of a conflict, in mean subgoal durations
    say_file: str = ""  # JSON {episode id: what the person says at the start}; empty: the PARTNR instruction
    person_horizon: float = 0  # steps the person has worked out ahead; below 1, a share of the task (0: all of it)
    talk_start: str = "ask"  # talker: ask (an open question), ask_all ("tell me every step"), instruction (is told it)
    s2_thinking: str = "1024"  # talker: Gemini thinking for plans and questions, a token budget ("" = model default)
    talk_policy: str = "model"  # talker: model (ask when the value beats ask_cost), vlm_num (S2 decides from the
    # human model's values) or vlm (S2 decides alone)
    ask_floor: float = 2.0  # vlm_num: S2 is consulted only when a question is worth at least this many seconds
    know_model: bool = False  # talker: a question's value times the chance the person can answer it, from how
    # often their answers so far settled what was asked
    talk_overhead: float = 1.5  # talker: s the person loses to an exchange, plus their words at talk_wps
    talk_wps: float = 2.5
    ask_cost: float = 8.0  # talker: a question's cost in seconds of the person's time (answering, the interruption)
    max_questions: int = 4  # talker: questions after the opening one
    ask_cooldown: float = 15.0  # talker: s between questions
    voice_port: int = 8771  # talker: the person's voice server (helper/partnr/voice.py)
    recipe_prior: str = "llm"  # bipv2: llm (Gemini per class, families.recipe_prior) or uniform over the recipes
    beta_wrong_rel: float = 1.0  # bipv2: cost of doing a subgoal the task does not have, in mean subgoal durations
    beta_prec_rel: float = 2.0  # cost of an ordering violation, in mean subgoal durations
    kappa: float = 5.0
    robot_slowdown: float = 1.0  # robot time per unit of person time (both use PARTNR's oracle skills)
    # E[time left | time spent x] on a subgoal, in units of d, at x = 0, rem_step, ...; PARTNR human traces (hitl.residual)
    rem_curve: tuple = (1.0, 0.798, 0.654, 0.546, 0.483, 0.492, 0.467, 0.446, 0.493, 0.54, 0.608, 0.877, 0.797)
    rem_step: float = 0.25

    @property
    def run_dir(self) -> Path:
        return Path(self.runs) / self.name

    def to_dict(self) -> dict:
        return asdict(self)


def parse(args: list[str], cls=Config):
    """Apply key=value overrides; values are Python literals, anything else is a string."""
    cfg, names = cls(), {f.name for f in fields(cls)}
    for arg in args:
        key, _, value = arg.partition("=")
        if key not in names:
            raise SystemExit(f"unknown config key: {key}")
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        setattr(cfg, key, value)
    return cfg
