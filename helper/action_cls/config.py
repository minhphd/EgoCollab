"""Every setting of the action classifier, in one place. Override any field as key=value on the command line:

    python -m helper.action_cls.train name=no_gaze use_gaze=0
"""
from dataclasses import dataclass, field, fields
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    # data: HD-EPIC (Aria ego video, per-frame gaze, action segments with start and end)
    hd_epic: Path = REPO / "data/hd_epic"
    features: Path = REPO / "data/hd_epic/features/action_cls_448"  # one .pt per video, and pca.pt (extract.py)
    runs: Path = REPO / "runs/action_cls"
    name: str = "full"
    # the classes, as HD-EPIC verb class ids; every other verb is "other"
    classes: tuple = ("pick", "place", "open", "close", "other")
    verb_map: dict = field(default_factory=lambda: {0: "pick", 1: "place", 5: "place", 3: "open", 4: "close"})
    other_max: int = 12000  # segments of other verbs kept, sampled once with `seed`
    split: dict = field(default_factory=lambda: {"train": ("P01", "P02", "P03", "P04", "P05", "P06"),
                                                 "val": ("P07",), "test": ("P08", "P09")})
    # frames and features
    fps: float = 5.0  # frames sampled per second of video
    window: int = 30  # frames per window (6 s at 5 fps): gaze reaches a pick-up a median 3.1 s ahead
    image: int = 448  # the whole frame at this size, never cropped: context matters
    dino: str = "facebook/dinov2-base"
    dino_dim: int = 768  # the backbone's width: 1024 for dinov2-large (refeature.py)
    grid: int = 16  # DINO's patches (image / 14 a side) pooled to grid x grid
    pca_dim: int = 256  # patch features projected to this many dimensions (fit by extract.py pca=1); 0: keep 768
    raft: bool = True  # optical flow between consecutive sampled frames, pooled to the grid
    flow_residual: bool = True  # remove the head's own motion from the flow (dataset.camera_free)
    video_fps: float = 30.0
    gaze_px: float = 1408.0  # the Aria frame the gaze is given on
    # hands: boxes from the YOLO hand detector BIP's perception uses, on every frame of a cache at `fps` (hands.py)
    hand_model: Path = REPO / "models/yolo/hand_yolov9c.pt"
    hand_frames: Path = REPO / "data/frames448_5fps"  # every HD-EPIC video at fps (5), 448 px (helper/perception/frames.py)
    hands: Path = REPO / "data/hd_epic/features/hands_5fps"  # one [frames, 2, 5] array per video
    hand_conf: float = 0.3
    all_frames: bool = False  # extract every frame on the fps grid (for windows anywhere in a video), not only near actions
    # windows: each ends at a time in [start + end_lo * duration, end + end_hi]; the completion target is the
    # share of the action done by the window's end
    end_lo: float = 0.3
    end_hi: float = 0.5
    done_at: float = 1.0  # progress at or above this counts as done
    clip: float = 3.0  # s per clip of the stream (stream.py): labelled by the last step completed in it
    # model
    d: int = 256
    layers: int = 4
    heads: int = 4
    dropout: float = 0.1
    use_gaze: bool = True  # gaze as attention: a bias on the pooling over patches, and a token at the gaze point
    gaze_point: bool = True  # gaze also as a point: its coordinates, and whether it is there, feed each frame
    use_flow: bool = True
    use_patches: bool = True  # False: the DINO CLS token alone per frame
    gaze_sigma: float = 2.0  # grid cells, the spread of the gaze prior over patches
    gaze_drop: float = 0.2  # share of training windows with gaze removed
    use_hands: bool = False  # each hand as a token: the patches at it, and its box
    hold_w: float = 0.0  # weight of a per-frame loss on the objects in hand (HD-EPIC's object movements; 0: off)
    hold_max: int = 3  # objects in hand counted 0, 1, ..., hold_max or more: a pick adds one, a place removes one
    # tricks
    inpaint: float = 0.5  # weight of the masked-frame reconstruction loss (0: off)
    mask_ratio: float = 0.25
    flip: bool = True
    label_smoothing: float = 0.1
    progress_w: float = 1.0
    balanced: bool = True  # sample training windows so every class is equally likely
    # the VLM (vlm.py): the smallest multi-image VLM, fine-tuned end to end on the same windows, their frames read from
    # a cache with the gaze drawn as a point; heads on its last token give the class and the progress
    vlm: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
    vlm_frames: Path = REPO / "data/frames448_5fps"  # every HD-EPIC video at 5 fps, 448 px (helper/perception/frames.py)
    vlm_fps: float = 5.0
    vlm_n: int = 30  # frames per window: 6 s at 5 fps, as `window`
    vlm_lr: float = 2e-5  # the VLM's own weights; the heads train at lr
    vlm_batch: int = 8
    vlm_epochs: int = 3
    vlm_windows: int = 20000
    vlm_gaze: bool = True
    vlm_train_vision: bool = True  # the vision encoder trains too
    # training
    epochs: int = 30
    windows_per_epoch: int = 40000
    batch: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup: int = 500
    workers: int = 6
    seed: int = 0
    device: str = "cuda"


def parse(argv) -> Config:
    """Config() with key=value overrides, cast to each field's type."""
    cfg, types = Config(), {f.name: f.type for f in fields(Config)}
    for a in argv:
        k, v = a.split("=", 1)
        if k not in types:
            raise KeyError(f"unknown setting {k}")
        cur = getattr(cfg, k)
        setattr(cfg, k, (v not in ("0", "false", "False")) if isinstance(cur, bool) else Path(v) if isinstance(cur, Path)
                else type(cur)(v) if isinstance(cur, (int, float, str)) else eval(v))
    return cfg
