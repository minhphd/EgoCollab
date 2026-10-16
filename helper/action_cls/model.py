"""The action classifier: frozen DINOv2 features of the whole frame in. Per frame, an attention pooling over the
patches (each also carrying its hand-motion flow) that gaze biases, a token at the gaze point, and the gaze point
itself; with `use_hands`, a token per hand (the patches at it and its box); then a temporal transformer; heads for
the class, the action's progress (completion), with `hold_w` the number of objects in hand at each frame, and, while
training, the DINO features of masked frames. Gaze and hands steer attention but never select the input."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionNet(nn.Module):
    def __init__(self, cfg, dino_dim=None):
        super().__init__()
        dino_dim = dino_dim or cfg.dino_dim
        self.cfg, d, G = cfg, cfg.d, cfg.grid
        self.cls_in, self.patch_in = nn.Linear(dino_dim, d), nn.Linear(cfg.pca_dim or dino_dim, d)
        self.flow_in, self.gaze_in = nn.Linear(3, d), nn.Linear(3, d)
        self.pos2d = nn.Parameter(torch.randn(G * G, d) * 0.02)
        self.query = nn.Parameter(torch.randn(d) * 0.02)
        self.k, self.v = nn.Linear(d, d), nn.Linear(d, d)
        self.lam = nn.Parameter(torch.tensor(1.0))  # how strongly gaze steers the pooling
        ys, xs = torch.meshgrid(torch.arange(G), torch.arange(G), indexing="ij")
        self.register_buffer("cells", torch.stack([xs, ys], -1).flatten(0, 1).float() + 0.5)  # cell centres, grid units
        n_in = 4 + 2 * cfg.use_hands
        self.frame = nn.Sequential(nn.LayerNorm(n_in * d), nn.Linear(n_in * d, d), nn.GELU(), nn.Linear(d, d))
        self.hand_in = nn.Linear(5, d) if cfg.use_hands else None
        self.hold = nn.Linear(d, cfg.hold_max + 1) if cfg.hold_w > 0 else None
        self.mask_tok, self.cls_tok = nn.Parameter(torch.zeros(d)), nn.Parameter(torch.zeros(d))
        self.pos_t = nn.Parameter(torch.randn(cfg.window + 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, cfg.heads, 4 * d, cfg.dropout, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head, self.prog, self.recon = nn.Linear(d, len(cfg.classes)), nn.Linear(d, 1), nn.Linear(d, dino_dim)

    def gaze_logprior(self, gaze):
        """log of a Gaussian over the grid cells around the gaze point ([B, T, G*G]); 0 where gaze is missing."""
        g = gaze[..., :2] * self.cfg.grid
        d2 = ((self.cells - g[..., None, :]) ** 2).sum(-1)
        return -d2 / (2 * self.cfg.gaze_sigma ** 2) * gaze[..., 2:3]

    def hand_logprior(self, hands):
        """log of a Gaussian over the grid cells around each hand, half its box wide ([B, T, 2, G*G])."""
        G = self.cfg.grid
        sigma = (hands[..., 3:5].amax(-1, keepdim=True) * G / 2).clamp(min=0.75)
        d2 = ((self.cells - hands[..., None, 1:3] * G) ** 2).sum(-1)
        return -d2 / (2 * sigma ** 2)

    def frames(self, cls, patch, flow, gaze, hands=None):
        cfg = self.cfg
        c = self.cls_in(cls)
        zero = torch.zeros_like(c)
        point = self.gaze_in(gaze) if cfg.gaze_point else zero  # where gaze is, and whether it is there
        pooled = at_gaze = zero
        if cfg.use_patches:
            p = self.patch_in(patch) + self.pos2d + (self.flow_in(flow) if cfg.use_flow else 0)
            logits = self.k(p) @ self.query / math.sqrt(cfg.d)
            prior = self.gaze_logprior(gaze) if cfg.use_gaze else torch.zeros_like(logits)
            pooled = (logits + self.lam * prior).softmax(-1).unsqueeze(-1).mul(self.v(p)).sum(-2)
            if cfg.use_gaze:
                at_gaze = prior.softmax(-1).unsqueeze(-1).mul(p).sum(-2) * gaze[..., 2:3]
        toks = [c, pooled, at_gaze, point]
        if cfg.use_hands:  # per hand: its box, and the patches at it when it is seen
            h = self.hand_in(hands)
            if cfg.use_patches:
                h = h + (self.hand_logprior(hands).softmax(-1) @ p) * (hands[..., :1] > 0)
            toks += [h[..., 0, :], h[..., 1, :]]
        return self.frame(torch.cat(toks, -1))

    def forward(self, batch, mask_ratio=0.0):
        x = self.frames(batch["cls"], batch["patch"], batch["flow"], batch["gaze"], batch.get("hands"))  # [B, T, d]
        valid = batch["valid"]
        masked = (torch.rand(valid.shape, device=x.device) < mask_ratio) & valid if mask_ratio > 0 else torch.zeros_like(valid)
        x = torch.where(masked[..., None], self.mask_tok.expand_as(x), x)
        x = torch.cat([self.cls_tok.expand(len(x), 1, -1), x], 1) + self.pos_t
        pad = torch.cat([torch.zeros_like(valid[:, :1]), ~valid], 1)
        h = self.norm(self.temporal(x, src_key_padding_mask=pad))
        out = {"logits": self.head(h[:, 0]), "progress": self.prog(h[:, 0]).squeeze(-1)}
        if self.hold is not None:  # objects in hand, per frame: 0, 1, ..., hold_max or more
            out["holding"] = self.hold(h[:, 1:])
        if masked.any():  # inpainting: the masked frames' DINO CLS features
            out["inpaint"] = 1 - F.cosine_similarity(self.recon(h[:, 1:][masked]), batch["cls"][masked], dim=-1).mean()
        return out
