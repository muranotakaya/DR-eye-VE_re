"""Unit tests for the DR(eye)VE PyTorch port, using synthetic (dummy) data only.

Run:  python -m pytest -q test_baseline.py      (or simply: python test_baseline.py)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from datasets import (DummyTwoWheelerDataset, TwoWheelerGazeDataset, build_dataset,
                      gaze_to_saliency, write_dummy_dataset)
from models import DreyeveNet, DrEyeVEBaseline, build_model
from utils.config import load_config
from utils.metrics import cc, compute_metrics, kld, nss, sim

B, C, T = 2, 3, 16


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# ------------------------------------------------------------------------------ model
@pytest.mark.parametrize("backbone", ["c3d", "r3d_18", "mc3_18", "r2plus1d_18"])
def test_forward_shape(backbone):
    """[B, C, T, H, W] -> [B, 1, H, W], non-negative, finite."""
    h, w = 128, 128
    model = DrEyeVEBaseline(backbone=backbone, pretrained=False).eval()
    x = torch.randn(B, C, T, h, w)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (B, 1, h, w)
    assert torch.isfinite(y).all() and (y >= 0).all()


def test_forward_reference_resolution():
    """The reference setting: T=16, 448x448 full frame, 112x112 coarse / crop inputs."""
    model = DrEyeVEBaseline(backbone="c3d").eval()
    x = torch.randn(1, C, T, 448, 448)
    crop = torch.randn(1, C, T, 112, 112)
    with torch.no_grad():
        fine, crop_pred = model(x, crop)
        # C3D encoder: 112x112 -> 14x14 (stride 8), temporal 16 -> 4
        feat = model.coarse.encoder(torch.randn(1, C, T, 112, 112))
    assert fine.shape == (1, 1, 448, 448)
    assert crop_pred.shape == (1, 1, 112, 112)
    assert feat.shape == (1, 512, 4, 14, 14)


@pytest.mark.parametrize("hw", [(96, 160), (128, 224)])
def test_forward_non_square_and_short_clip(hw):
    model = DrEyeVEBaseline(backbone="c3d").eval()
    x = torch.randn(1, C, 8, *hw)
    with torch.no_grad():
        assert model(x).shape == (1, 1, *hw)


def test_multibranch_dreyevenet():
    model = DreyeveNet({"image": 3, "optical_flow": 3, "segmentation": 19}).eval()
    clips = {"image": torch.randn(1, 3, T, 64, 64), "optical_flow": torch.randn(1, 3, T, 64, 64),
             "segmentation": torch.randn(1, 19, T, 64, 64)}
    with torch.no_grad():
        assert model(clips).shape == (1, 1, 64, 64)


def test_backward_updates_all_parameters():
    model = DrEyeVEBaseline(backbone="c3d")
    x, crop = torch.randn(B, C, T, 64, 64), torch.randn(B, C, T, 16, 16)
    gt, gt_crop = torch.rand(B, 1, 64, 64), torch.rand(B, 1, 16, 16)
    fine, crop_pred = model(x, crop)
    loss = kld(fine, gt).mean() + kld(crop_pred, gt_crop).mean()
    loss.backward()
    assert torch.isfinite(loss)
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, missing


def test_build_model_from_configs():
    for path in ["configs/dreyeve_c3d.yaml", "configs/dreyeve_r3d18.yaml", "configs/dummy_debug.yaml"]:
        cfg = load_config(path, ["model.pretrained=false"])
        assert isinstance(build_model(cfg["model"]), torch.nn.Module)


# ------------------------------------------------------------------------------ data
def _dummy_kwargs(train):
    return dict(num_sequences=2, frames_per_sequence=40, stride=8, clip_len=T,
                frame_size=(64, 64), crop_before_size=(32, 32), train=train)


def test_gaze_to_saliency_peak():
    sal = gaze_to_saliency(np.array([[0.25, 0.75]]), (40, 80), sigma=0.05)
    y, x = np.unravel_index(sal.argmax(), sal.shape)
    assert sal.max() == pytest.approx(1.0)
    assert abs(x - 20) <= 1 and abs(y - 30) <= 1
    assert gaze_to_saliency(np.zeros((0, 2)), (8, 8), 0.05).sum() == 0


@pytest.mark.parametrize("train", [True, False])
def test_dummy_dataset_item(train):
    ds = DummyTwoWheelerDataset(**_dummy_kwargs(train))
    item = ds[0]
    assert item["clip"].shape == (C, T, 64, 64)
    assert item["saliency"].shape == (1, 64, 64)
    assert item["fixation"].shape == (1, 64, 64)
    assert item["saliency"].max() > 0 and item["fixation"].sum() >= 1
    if train:
        assert item["clip_crop"].shape == (C, T, 16, 16)
        assert item["saliency_crop"].shape == (1, 16, 16)
    else:
        assert "clip_crop" not in item
    # deterministic for a fixed epoch
    assert torch.equal(ds[1]["clip"], ds[1]["clip"])


def test_dummy_pipeline_end_to_end():
    """Dummy dataset -> DataLoader -> model -> loss/metrics -> one optimizer step."""
    ds = DummyTwoWheelerDataset(**_dummy_kwargs(True))
    batch = next(iter(DataLoader(ds, batch_size=B, shuffle=True)))
    model = DrEyeVEBaseline(backbone="c3d")
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    fine, crop = model(batch["clip"], batch["clip_crop"])
    assert fine.shape == (B, 1, 64, 64) and crop.shape == (B, 1, 16, 16)
    loss = kld(fine, batch["saliency"]).mean() + kld(crop, batch["saliency_crop"]).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    m = compute_metrics(fine.detach(), batch["saliency"], batch["fixation"])
    assert set(m) == {"kld", "cc", "sim", "nss"} and all(v.shape == (B,) for v in m.values())


@pytest.mark.parametrize("use_frames_dir", [False, True])
def test_on_disk_loader(tmp_path, use_frames_dir):
    """Write a synthetic dataset to disk (mp4 or frames + gaze.csv) and read it back."""
    root = write_dummy_dataset(tmp_path / "data", num_sequences=3, frames_per_sequence=24,
                               use_frames_dir=use_frames_dir)
    data_cfg = {"root": str(root), "clip_len": 8, "frame_size": [64, 64],
                "crop_before_size": [32, 32], "train_stride": 4}
    ds = build_dataset(data_cfg, "train")
    assert isinstance(ds, TwoWheelerGazeDataset) and len(ds) > 0
    # frames with lost tracking (t % 17 == 5) are skipped
    assert all(t % 17 != 5 for _, t in ds.samples)
    item = ds[0]
    assert item["clip"].shape == (C, 8, 64, 64)
    assert item["saliency"].max() == pytest.approx(1.0, abs=1e-5)
    test_ds = build_dataset(data_cfg, "test")
    assert "clip_crop" not in test_ds[0]


# ------------------------------------------------------------------------------ metrics
def test_metrics_sanity():
    gt = torch.rand(3, 1, 32, 32)
    other = torch.rand(3, 1, 32, 32)
    # the reference formula log(eps + Q / (eps + P)) gives ~-1e-4 (not exactly 0) for P == Q
    assert torch.allclose(kld(gt, gt), torch.zeros(3), atol=1e-3)
    assert (kld(other, gt) > 0).all()
    assert torch.allclose(cc(gt, gt), torch.ones(3), atol=1e-4)
    assert torch.allclose(sim(gt, gt), torch.ones(3), atol=1e-4)
    fix = torch.zeros(3, 1, 32, 32)
    fix[:, :, 5, 5] = 1
    peak = torch.zeros(3, 1, 32, 32)
    peak[:, :, 5, 5] = 1
    assert (nss(peak, fix) > 10).all()
    assert torch.isnan(nss(peak, torch.zeros_like(fix))).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
