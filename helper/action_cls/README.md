# Action classifier: pick, place, open, close from the partner's view

The classifier reads the partner's egocentric video, their gaze and the motion of their hands, and says which
action they are doing (pick, place, open, close, or other) and how far through it they are. For BIP these are the
action reads (`helper/collab/README.md`, 2.6), and the progress is evidence of completion, which BIP otherwise only
infers. Every setting is a field of `Config` in `config.py`, overridable as `key=value`.

## Overview

- **Data** (section 1): HD-EPIC's Aria videos, per-frame gaze and action segments with start and end times.
- **Features** (section 2), computed once from the whole frame, never cropped (the scene's context matters): frozen
  DINOv2 at 448 px, optical flow with the head's own motion removed, the gaze point, and the hands' boxes.
- **Model** (section 3): per frame, attention pooling over the patches that gaze biases, the gaze point itself, and
  a token per hand; over frames, a temporal transformer; heads for the class, the progress, and the number of
  objects in hand at each frame. Gaze steers attention and is an
  input; it never selects what the model sees, since people often look at another object than the one in hand.
- **Training** (section 4): 6 s windows ending anywhere from 30% of the way through an action to 0.5 s after it, with
  masked-frame inpainting, gaze dropout, mirroring, label smoothing and class-balanced sampling.

```python
# Once
segments = HD-EPIC actions labelled pick (take), place (put, insert), open, close, other    # (1)
for video: features[t] = DINO(frame_t), flow(frame_t-1 -> frame_t), gaze_t at 5 fps          # (2)

# Each window: 30 frames (6 s) ending at t_end: long enough to catch the gaze arriving ahead of the hands
for frame: token = MLP([cls, gaze-biased pool(patches + flow), patch at the gaze, gaze point])  # (3.1)
h = TemporalTransformer([CLS] + tokens)                                                       # (3.2)
class = softmax(W_c h_CLS);  progress = sigmoid(w_p h_CLS)   # share of the action done by t_end (3.3)
```

## 1. Data

| | |
| --- | --- |
| Source | HD-EPIC (`data/hd_epic`): 156 Aria videos, 9 participants, gaze on 154 |
| Labels | the main action's HD-EPIC verb class: take as pick; put and insert as place; open; close; every other verb as other (`other_max` = 12000 sampled once) |
| Split | by participant: train P01 to P06, validation P07, test P08 and P09 |
| Segments | train: pick 9695, place 9140, open 2143, close 2026, other 7987; test: 2875, 2774, 563, 328, 3071 |

**Computed by** `segments.py`.

## 2. Features

Every frame some window needs, at `fps` (5): 647k frames, about 87 GB, one file per video (`extract.py`, one
process per GPU, CPU workers decoding the 1408 px video).

- **DINOv2** (`dino`, base, frozen) on the whole fisheye frame at 448 px (`image`): the CLS token, and the 32 x 32
  patches average-pooled to a 16 x 16 grid (`grid`, cells of 88 source pixels) and projected by PCA from 768 to 256
  dimensions (`pca_dim`; fit by `extract.py pca=1` on each participant's shortest video, 781k patches; it keeps
  92.8% of the variance).
- **Flow** (RAFT small) from the previous sampled frame, pooled to the grid, in image widths per step. The head's
  own motion is removed in `dataset.camera_free`: a quadratic flow field over the grid, fit robustly (three
  reweighted least-squares passes), is subtracted, so the hands and the objects they move remain. Averaging over a
  cell preserves a quadratic field, so the fit on the pooled flow matches a fit on the full one.
- **Gaze**: the frame's gaze point on the 1408 px frame, in [0, 1], with a flag where it is missing.
- **Hands** (`hands.py`): the YOLO hand detector BIP's perception uses (`hand_model`) on every frame of the 5 fps,
  448 px frame cache (`hand_frames`): per frame, the two most confident hands ordered left to right, each
  [confidence, cx, cy, w, h]; a lone hand takes the slot of its image half.

## 3. Model (`model.py`)

### 3.1 A frame

With patch tokens $p_j = W_p\,\mathrm{dino}_j + W_f\,\mathrm{flow}_j + e_j$ over the grid cells $j$:

$$
\log \gamma_j = -\frac{\lVert c_j - g \rVert^2}{2\sigma^2},
\qquad
\alpha_j = \mathrm{softmax}_j\Big(\frac{q^\top K p_j}{\sqrt d} + \lambda \log \gamma_j\Big),
\qquad
\text{pool} = \sum_j \alpha_j\, V p_j
$$

- $g$ is the gaze point and $c_j$ the cell centres, in grid units. $\sigma$ = `gaze_sigma` (2 cells), and
  $\lambda$ is learned (starts at 1), so the model can learn how far to trust gaze.
- The gaze token is $\sum_j \mathrm{softmax}_j(\log \gamma_j)\, p_j$, the patches at the gaze. Without gaze it is
  0, and so is the bias.
- The gaze point itself, $W_g\,(g_x, g_y, \text{present})$, is a fourth input (`gaze_point`).
- The frame token is $\mathrm{MLP}([W_c\,\mathrm{cls},\ \text{pool},\ \text{gaze token},\ W_g\,g])$.
- With `use_hands`, each hand adds a token: its box through a linear layer, plus the patches at it, pooled as the
  gaze token with a Gaussian half the box's longer side wide (at least 0.75 cells); 0 for a hand not seen.

### 3.2 Time

A pre-norm transformer encoder (`layers` 4, `heads` 4, `d` 256) over a learned [CLS] and the 30 frame tokens (6 s,
`window`), with learned time embeddings; frames missing from the cache are masked. The span is set by gaze: in
HD-EPIC, gaze first reaches a pick-up location a median 3.1 s before the pick (64% of picks) and a put-down location
1.8 s before (27%), so 6 s holds 71% and 92% of those leads where 2.4 s held 39% and 59%. What lies under the gaze
then (an object, or an empty spot or container) separates a pick from a place.

### 3.3 Heads and losses

$$
\mathcal{L} = \mathrm{CE}_{0.1}(y, W_c h) + \mathrm{BCE}\big(\rho, \sigma(w_p^\top h)\big) + 0.5\,\big(1 - \cos(W_r h_t,\ \mathrm{cls}_t)\big)_{t\ \text{masked}}
$$

- $\rho = \mathrm{clip}\big((t_{\text{end}} - t_{\text{start}}) / (t_{\text{stop}} - t_{\text{start}}), 0, 1\big)$, the share of the
  action done by the window's end; done means $\rho = 1$.
- **Inpainting:** 25% of the frames (`mask_ratio`) are replaced by a mask token, and the transformer reconstructs
  their DINO CLS features (weight `inpaint`).
- **Objects in hand** (weight `hold_w`): a head on each frame's output predicts how many objects are in hand (0, 1,
  2, or 3 and more), the number of HD-EPIC object movements (pick-up to placement) spanning the frame. Around a
  pick the count rises in 62% of segments and falls in 6%; around a place it falls in 66% and rises in 4%. The
  labels train the head only; at test time the model sees the video, gaze and hand boxes.

## 4. Training and evaluation (`train.py`)

- Windows end at a uniform time in $[t_{\text{start}} + 0.3\,\text{duration},\ t_{\text{stop}} + 0.5\ \text{s}]$
  (`end_lo`, `end_hi`). Sampling is class-balanced, 40k windows an epoch, 30 epochs.
- AdamW (lr 3e-4, weight decay 0.05), 500 warm-up steps, then cosine decay; bf16.
- Training augmentation: gaze removed in 20% of windows (`gaze_drop`), mirroring (the grid, the flow's x and the
  gaze's x).
- Evaluation windows end at 30%, 60% and 90% of each action and 0.2 s and 0.5 s after it. The checkpoint kept is
  the one with the best validation macro F1 just after the action.
- Reported on the held-out participants:
  - accuracy and macro F1 mid-action (60%) and just after the end (+0.2 s), and the confusion matrix;
  - completion: the AUROC of done against not done from the progress head, per class too, and the mean progress
    error.

## 5. The VLM baseline (`vlm.py`)

- SmolVLM2-256M-Video-Instruct (`vlm`), the smallest multi-image VLM, fine-tuned end to end, its vision encoder
  too (`vlm_train_vision`).
- Input: the window's 30 frames (`vlm_n`, 5 fps over the same 6 s), from a cache of every video at 448 px
  (`vlm_frames`), each with the gaze drawn as an orange ring; 64 tokens per frame, then the question.
- Output: a class head and a progress head on the last token's hidden state, with the same losses (no inpainting),
  windows, split and metrics as section 4.
- Training: AdamW, 2e-5 for the VLM and 3e-4 for the heads, batch 8, 3 epochs of 20k windows, bf16, gradient
  checkpointing.

## 6. The verb recognizer BIP reads (`stream.py`)

BIP (`helper/collab/bip.py`) scores clips of the stream, not windows anchored to actions. `stream.py` trains the same
network for that:

- Every clip of `clip` s (3) gets one label: the verb of the last pick, place, open or close that ends in it, or other
  when none does. In HD-EPIC 55% of 3 s clips complete no step, and 35% of those with a step hold two.
- The input is the `window` frames ending with the clip (6 s), from features of every frame (`all_frames`, 744k
  frames in `data/hd_epic/features/action_cls_448_all`).
- Training draws clip ends at any frame, class-balanced, so the network's prior is uniform and its probabilities
  divided by that prior are likelihoods; validation and test cut every video into consecutive clips.
- A temperature is fit on validation (class-balanced negative log-likelihood). The test clips' calibrated
  probabilities are saved as `bank.npz` (with each clip's video and end time): the simulator draws its recognizer
  outputs from it (`helper/collab/sense.py`), and `helper/collab/hdepic.py` runs BIP on it.
- Reported: accuracy, macro F1 and the confusion matrix at the natural class mix; the negative log-likelihood; and,
  per step verb, the mean $\log q(\text{verb}) / q(\text{other})$ on its clips, the evidence one clip gives BIP.

## 7. Files and use

| File | Role |
| --- | --- |
| `config.py` | every setting |
| `segments.py` | labelled segments, the split, the frames the windows need |
| `extract.py` | DINO, flow and gaze per video |
| `hands.py` | hand boxes per video |
| `dataset.py` | windows, the camera-motion removal, augmentation |
| `model.py` | the network |
| `train.py` | training, evaluation, `runs/action_cls/<name>/metrics.json` and `best.pt` |
| `vlm.py` | the VLM baseline, same outputs |
| `stream.py` | the verb recognizer on stream clips, calibrated, and its bank |

```bash
# on the compute node, one process per GPU
python -m helper.action_cls.extract pca=1
CUDA_VISIBLE_DEVICES=0 python -m helper.action_cls.extract part=0/2
CUDA_VISIBLE_DEVICES=1 python -m helper.action_cls.extract part=1/2
CUDA_VISIBLE_DEVICES=0 python -m helper.action_cls.train name=full
CUDA_VISIBLE_DEVICES=1 python -m helper.action_cls.train name=no_gaze use_gaze=0 gaze_point=0
python -m helper.perception.frames --shard 0/1 --fps 5 --px 448 --out data/frames448_5fps
CUDA_VISIBLE_DEVICES=0 python -m helper.action_cls.hands
CUDA_VISIBLE_DEVICES=0 python -m helper.action_cls.train name=hands use_hands=1 hold_w=0.5
CUDA_VISIBLE_DEVICES=1 python -m helper.action_cls.vlm name=vlm6
python -m helper.action_cls.extract part=0/2 all_frames=1 features=data/hd_epic/features/action_cls_448_all
CUDA_VISIBLE_DEVICES=0 python -m helper.action_cls.stream name=stream3 features=data/hd_epic/features/action_cls_448_all
```
