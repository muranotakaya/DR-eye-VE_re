# Rider gaze saliency prediction — DR(eye)VE PyTorch baseline

Predicting the gaze saliency map of motorcycle / two-wheeler riders from ego-centric video.
This repository starts with a clean PyTorch 2.x re-implementation of the four-wheeler baseline
**DR(eye)VE** (Palazzi et al., "Predicting the Driver's Focus of Attention: the DR(eye)VE Project",
IEEE TPAMI 41(7):1720–1733, as cited in the official repo README; arXiv: https://arxiv.org/abs/1705.03854).

## Repo name (must)
- Format: `yy-family-given-topic[-purpose]` (current repo name `DR-eye-VE_re` does not follow it yet — TODO)

## Overview
- Author: TODO <family given>
- FY (yy): TODO
- Topic keywords: rider gaze, saliency, video
- Upstream: https://github.com/ndrplz/dreyeve (MIT License, Keras 1 / Theano). Architecture ported, no code or weights copied.

## Environment (must)
- OS: Linux (verified in a Linux container)
- Python: 3.11
- Key libs: torch==2.14.1, torchvision==0.29.1, numpy==2.4.6, opencv-python-headless==5.0.0.93, PyYAML==6.0.1 (see `requirements.txt`)
- GPU/CUDA: optional (tests and the dummy run were verified on CPU only)

## Setup
### Option A: pip
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### Option B: conda
```bash
conda env create -f environment.yml
conda activate kameda-lab
```

> `data/` and `outputs/` are gitignored. Note: the top-level `datasets/` directory is **code**
> (re-included in `.gitignore`); put data under `data/`.

## Project layout
```
configs/                    YAML configs (`_base_` inheritance, CLI overrides via --opts key=value)
  dreyeve_c3d.yaml          reference hyper-parameters (T=16, 448x448, Adam 1e-4, KLD)
  dreyeve_r3d18.yaml        same head on Kinetics-400 R3D-18
  dreyeve_mc3_18.yaml       same head on Kinetics-400 MC3-18
  dummy_debug.yaml          small CPU setting for smoke tests
datasets/two_wheeler_dataset.py   real-data Dataset, synthetic Dataset (--dummy), GT rendering
models/baseline_dreyeve.py  DR(eye)VE saliency branch + multi-branch DreyeveNet
utils/metrics.py            KLD (loss), CC, SIM, NSS, IG
utils/visualization.py      overlays / prediction grids
utils/config.py             config loading, seeding, device
train.py, evaluate.py, test_baseline.py
```

## Model
One DR(eye)VE saliency branch (`models/baseline_dreyeve.py`), input `[B, C, T, H, W]` → output `[B, 1, H, W]`:

1. **Coarse path**: the clip is resized to `(H/4, W/4)` and encoded by a 3D-CNN — C3D up to `conv4b`
   (as in the paper) or a torchvision `r3d_18` / `mc3_18` / `r2plus1d_18` — then collapsed over time (max),
   bilinearly upsampled, `conv3x3 → 1` + ReLU, and upsampled to `(H, W)`.
2. **Refinement**: concatenated with the full-resolution last frame → conv 32-16-8-1 (LeakyReLU 0.001) → ReLU.
3. **Crop path (training only)**: a random `(H/4, W/4)` crop of the clip (taken from a 256×256 resize)
   goes through the *shared* encoder and its own head; both outputs are trained with KLD, as in the reference.

Intentional deviations from the reference code: size-based (not factor-based) upsampling so other backbones /
non-square inputs work; max over time instead of the fixed `pool4` reshape (identical for C3D, T=16);
per-channel Kinetics mean/std normalization instead of the dataset mean frame; Sports-1M C3D weights are not
ported (C3D trains from scratch — use `r3d_18` for pretrained features). Only the RGB branch is trained
for now; `DreyeveNet` (image + flow + semseg sum) is implemented but needs a dataset providing those inputs.

## Data policy (must)
### Public dataset: myEye2Wheeler
- Paper: https://arxiv.org/abs/2502.12723 (40 riders, Tobii Glasses 2 ego-centric video, 1920×1080).
- Do NOT copy the dataset into this repository; place it under `data/myEye2Wheeler/`.
- **The exact release format has not been verified yet.** The loader expects the layout below; convert
  the raw export into it (or adapt `data.gaze_columns` / `data.coords` in the config):
```
data/myEye2Wheeler/
  splits/{train,val,test}.txt   # sequence ids, one per line
  <seq_id>/video.mp4            # or <seq_id>/frames/000000.jpg ...
  <seq_id>/gaze.csv             # columns frame,x,y  (x,y normalized [0,1], top-left origin; empty = lost)
                                #   or timestamp,x,y (seconds; converted with video fps)
  <seq_id>/saliency/000000.png  # optional precomputed GT maps (used instead of gaze.csv)
```
- GT map for frame *t*: sum of Gaussians (σ = `data.sigma` × width) at the gaze points of frames
  `t ± data.gaze_window`, scaled to max 1. NSS uses the binary fixation map of frame *t*.

## How to run (reproducibility) (must)
```bash
# 1) unit tests (synthetic data only, ~10 s on CPU)
python -m pytest -q test_baseline.py
# 2) smoke test of the full pipeline without data
python train.py    --config configs/dummy_debug.yaml --dummy
python evaluate.py --config configs/dummy_debug.yaml --dummy --checkpoint outputs/dummy_debug_dummy/best.pt
# 3) real data
python train.py    --config configs/dreyeve_c3d.yaml   # or configs/dreyeve_r3d18.yaml
python evaluate.py --config configs/dreyeve_c3d.yaml --checkpoint outputs/dreyeve_c3d/best.pt --split test
```
Expected outputs (`outputs/<experiment>[_dummy]/`): `config.yaml`, `log.jsonl` (per-epoch train loss and
val KLD/CC/SIM/NSS), `best.pt`, `last.pt`, `vis/val_epochXXX.png`, and `eval_<split>/{metrics.json, per_sample.csv, samples.png}`.

`datasets.write_dummy_dataset(root)` writes a synthetic dataset in the on-disk layout above (mp4 + gaze.csv),
useful to test the real loader: `python train.py --config configs/dummy_debug.yaml --opts data.root=<root>`.

## Reproducibility check (must)
- Verified in a clean Linux container (CPU): `test_baseline.py` 17 passed; dummy train/evaluate runs end-to-end.
- Not yet verified: GPU training, real myEye2Wheeler data, pretrained Kinetics weights download.

## Manual steps (if any)
- Converting the raw myEye2Wheeler export to the layout above (TODO once the data is available).
