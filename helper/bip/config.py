"""Every setting of fast-slow Bayesian inverse planning (BIP) over task graphs, replayed on HD-EPIC.

Override fields on the command line as key=value, e.g.
    python -m helper.bip.run name=t_step_llm tier=step s2=llm
"""

import ast
from dataclasses import asdict, dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    name: str = "debug"
    runs: str = str(ROOT / "runs/bip")
    cache: str = str(ROOT / "data/bip")  # observation streams and S2 priors per video
    episodes: str = str(ROOT / "data/episodes/hd_epic_task")  # viewer episodes; estimates go next to them
    frame_cache: str = str(ROOT / "data/frames448")  # JPEG frames at `fps` (view.py)

    # Data: HD-EPIC videos with >= min_nodes annotated subgoals; the first `dev` of a fixed order fit parameters
    split: str = "test"  # dev | test | all
    truth: str = "annotated"  # scoring target: annotated | consistent | followed (score.py)
    dev: int = 30
    videos: int = 0  # cap on the number of videos (0 = all in the split)
    only_videos: tuple = ()
    min_nodes: int = 3
    grid: float = 10.0  # s between estimates written out
    trace: bool = False  # also keep the belief after every event (for clips)

    # Perception tier: "step" gives the subgoal of each action, "action" only its narration (both oracle);
    # "real" uses only detections (detect.py): handled objects as actions, gaze inside object boxes
    tier: str = "action"
    completion: str = "oracle"  # oracle (exact), latent (inferred by S1), noisy (a detector that misses and lags)
    p_done: float = 0.36  # P(leaving an unfinished-looking subgoal means it is complete), latent completion
    det_rate: float = 0.7  # noisy completion: share of completions detected
    det_delay: float = 10.0  # noisy completion: mean detection delay (s)

    # Real perception (detect.py, perceive.py)
    det_model: str = "yoloe-26l-seg.pt"  # open-vocabulary objects, prompted with Gemini's object names
    hand_model: str = "hand_yolov9c.pt"
    det_px: int = 640
    det_fps: float = 2.0  # detection rate; 2 reads the frame cache, other rates decode the video
    det_latency: float = 0.05  # s from a frame to its detections (21.7 ms median, 53 ms max, one RTX PRO 4000)
    det_conf: float = 0.2
    hand_conf: float = 0.3
    hand_margin: float = 0.15  # hand box grown by this share of its size to count an object as handled
    max_area: float = 0.2  # objects larger than this share of the frame (counters, stoves) are never handled
    window: float = 3.0  # s of detections pooled into one action event
    w_hand: float = 0.5  # handled-object mixture: weight of the current subgoal's objects (fit.py)

    # Subgoal model
    rule: str = "complete"  # an ordering u -> v lets v start once u is "complete" (Proposition 1) or "started"
    stay: float = 0.95  # P(next action keeps the current subgoal | it is incomplete)
    resume: float = 1.0  # weight of returning to a started subgoal, against 1 for an executable new one
    delta: float = 0.05  # weight of starting a subgoal whose predecessors are not done
    plan: float = 0.5  # P(a switch goes to the planned next subgoal)
    eps_step: float = 0.001  # error rate of the oracle step recognizer
    eps_done: float = 0.05  # P(a subgoal completes while another one is being worked on)
    beta_action: float = 20.0  # sharpness of the narration-to-subgoal similarity

    # Gaze mixture over fixated objects, weights fitted by EM on dev (fit.py); no avoid term: no labels for it
    gaze: bool = True
    w_cur: float = 0.17
    w_next: float = 0.0
    w_noise: float = 0.83
    gaze_temper: float = 1.0  # exponent on each fixation's likelihood
    fix_angle: float = 6.0  # degrees between the gaze ray and an object to count as fixated
    fix_min: float = 0.3  # s of steady gaze on one object to count as a fixation
    fov: float = 50.0  # half angle (degrees) of the visible cone
    gaze_hz: float = 10.0
    gaze_window: float = 1.0  # s of gaze painted on each frame shown to S2
    gaze_sigma: float = 0.04
    gaze_alpha: float = 0.55

    # S1 filter
    particles: int = 64  # per graph
    pool: int = 128  # graphs sampled from the S2 prior
    seed: int = 0

    # S2: "never" keeps the prior pool, "repair" drops falsified orderings without a model, "llm" asks the LLM
    s2: str = "llm"
    surprise: float = 1.4  # fire S2 when log P(event | empty graph) - log P(event | pool) exceeds this
    cooldown: float = 30.0  # s between S2 calls
    revisions: int = 3  # graphs per S2 revision
    model: str = "gemini-3.5-flash"  # Gemini on Vertex AI, a VLM: S2 also sees recent frames with gaze
    env: str = str(ROOT.parent / "MiniWeb-src/MiniWeb/.env")  # service account for Vertex AI
    thinking: str = "low"  # Gemini thinking level ("" for the model default)
    temperature: float = 0.0
    max_tokens: int = 8192
    s2_frames: int = 4  # images per S2 revision, the last at the call time
    s2_frame_gap: float = 3.0
    fps: float = 2.0  # frame cache rate
    image_px: int = 448
    streams: int = 8  # videos replayed concurrently

    @property
    def run_dir(self) -> Path:
        return Path(self.runs) / self.name

    def to_dict(self) -> dict:
        return asdict(self)


def parse(args: list[str]) -> Config:
    """Apply key=value overrides; values are Python literals, anything else is a string."""
    cfg, names = Config(), {f.name for f in fields(Config)}
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
