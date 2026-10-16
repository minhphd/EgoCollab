"""Windows of `window` sampled frames over the cached features (extract.py). A training window ends at a random time
in [start + end_lo * duration, end + end_hi]; an evaluation window ends at fixed points of the action. Each carries
the segment's class, the share of the action done by the window's end (progress), and whether it is done; with
`use_hands`, the hand boxes (hands.py), and with `hold_w`, how many objects are in hand at each frame."""
import numpy as np
import torch

from .segments import movements


def camera_free(flow, G, iters=3):
    """flow [n, G*G, 2] -> the flow minus a robust quadratic fit over the grid: the head's own motion (rotation and
    zoom bend the fisheye field) leaves, the hands and the objects they move stay."""
    ys, xs = torch.meshgrid(torch.arange(G), torch.arange(G), indexing="ij")
    x, y = (xs.flatten().float() + 0.5) / G - 0.5, (ys.flatten().float() + 0.5) / G - 0.5
    B = torch.stack([torch.ones_like(x), x, y, x * x, x * y, y * y], -1)
    w = torch.ones(flow.shape[0], G * G)
    for _ in range(iters):
        W = (w[..., None] * B).transpose(1, 2)
        r = flow - B @ torch.linalg.lstsq(W @ B, W @ flow).solution
        m = r.norm(dim=-1)
        w = 1 / torch.clamp(m / (1.5 * m.median(-1, keepdim=True).values + 1e-6), min=1)
    return r


EVAL_POINTS = (("frac", 0.3), ("frac", 0.6), ("frac", 0.9), ("after", 0.2), ("after", 0.5))  # where eval windows end


class Windows(torch.utils.data.Dataset):
    def __init__(self, cfg, seg, split, train):
        self.cfg, self.train = cfg, train
        seg = seg[(seg.split == split) & seg.video_id.map(lambda v: (cfg.features / f"{v}.pt").exists())]
        self.seg = seg.reset_index(drop=True)
        moves = movements(cfg) if cfg.hold_w > 0 else {}
        self.feat, self.keys, self.hands, self.hold = {}, {}, {}, {}
        for v in self.seg.video_id.unique():
            self.feat[v] = torch.load(cfg.features / f"{v}.pt", mmap=True)
            self.keys[v] = np.round(self.feat[v]["t"].numpy() * cfg.fps).astype(int)
            if cfg.use_hands:
                self.hands[v] = np.load(cfg.hands / f"{v}.npy")  # row k: the frame at k / fps
            if cfg.hold_w > 0:  # objects in hand: the movements spanning the frame
                t, m = self.keys[v][:, None] / cfg.fps, moves.get(v, np.zeros((0, 2)))
                self.hold[v] = torch.from_numpy(((t >= m[:, 0]) & (t <= m[:, 1])).sum(1).clip(max=cfg.hold_max))
        # training: one item per segment (ends drawn at random); evaluation: one item per segment and end point
        self.items = [(i, None) for i in range(len(self.seg))] if train else \
            [(i, p) for i in range(len(self.seg)) for p in EVAL_POINTS]

    def __len__(self):
        return len(self.items)

    def end_time(self, s, e, point):
        cfg = self.cfg
        if point is None:
            return np.random.uniform(s + cfg.end_lo * (e - s), e + cfg.end_hi)
        kind, x = point
        return s + x * (e - s) if kind == "frac" else e + x

    def __getitem__(self, j):
        cfg = self.cfg
        i, point = self.items[j]
        r = self.seg.iloc[i]
        k_end = int(round(self.end_time(r.start, r.end, point) * cfg.fps))
        t_end = k_end / cfg.fps
        progress = float(np.clip((t_end - r.start) / (r.end - r.start), 0, 1))
        return {**self.window(r.video_id, k_end), "y": int(r.y), "progress": progress, "done": float(t_end >= r.end - 1e-6),
                "point": EVAL_POINTS.index(point) if point else -1}

    def window(self, vid: str, k_end: int) -> dict:
        """The features of the `window` frames ending with frame k_end (augmented while training)."""
        cfg = self.cfg
        want = np.arange(k_end - cfg.window + 1, k_end + 1)
        keys, f = self.keys[vid], self.feat[vid]
        rows = np.minimum(np.searchsorted(keys, want), len(keys) - 1)
        valid = torch.from_numpy(keys[rows] == want)
        rows = torch.from_numpy(rows)
        cls, patch = f["cls"][rows].float(), f["patch"][rows].float()
        flow, gaze = f["flow"][rows].float(), f["gaze"][rows].clone()
        if cfg.flow_residual:
            res = camera_free(flow[..., :2], cfg.grid)
            flow = torch.cat([res, res.norm(dim=-1, keepdim=True)], -1) * (flow[..., 2:3].sum(1, keepdim=True) > 0)
        gaze[:, 2] *= valid
        if self.train and np.random.rand() < cfg.gaze_drop:
            gaze[:, 2] = 0
        extra = {}
        if cfg.use_hands:
            H = self.hands[vid]
            extra["hands"] = torch.from_numpy(H[np.clip(want, 0, len(H) - 1)].astype(np.float32)) * valid[:, None, None]
        if cfg.hold_w > 0:
            extra["holding"] = self.hold[vid][rows] * valid
        if self.train and cfg.flip and np.random.rand() < 0.5:  # mirror the grid, the flow's x, the gaze's x, the hands
            G = cfg.grid
            patch = patch.view(-1, G, G, patch.shape[-1]).flip(2).reshape(patch.shape)
            flow = flow.view(-1, G, G, 3).flip(2).reshape(flow.shape) * torch.tensor([-1.0, 1.0, 1.0])
            gaze[:, 0] = 1 - gaze[:, 0]
            if cfg.use_hands:  # left and right swap slots
                h = extra["hands"].flip(1)
                h[..., 1] = torch.where(h[..., 0] > 0, 1 - h[..., 1], h[..., 1])
                extra["hands"] = h
        return {"cls": cls, "patch": patch, "flow": flow, "gaze": gaze, "valid": valid, **extra}

    def class_weights(self):
        """Per-item sampling weights giving every class the same total weight."""
        y = self.seg.y.to_numpy()
        counts = np.bincount(y, minlength=len(self.cfg.classes)).astype(float)
        return torch.from_numpy(1.0 / counts[y])
