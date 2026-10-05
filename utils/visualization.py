"""Visualization helpers: saliency overlays and prediction grids."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch


def denormalize_frame(frame: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """[3, H, W] normalized tensor -> uint8 RGB [H, W, 3]."""
    x = frame.detach().float().cpu().numpy()
    x = x * np.asarray(std).reshape(3, 1, 1) + np.asarray(mean).reshape(3, 1, 1)
    return (np.clip(x.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)


def saliency_to_color(sal: np.ndarray) -> np.ndarray:
    """[H, W] float map -> uint8 RGB JET colormap (min-max normalized)."""
    s = sal.astype(np.float32)
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    color = cv2.applyColorMap((s * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def overlay(frame: np.ndarray, sal: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a saliency map onto an RGB frame."""
    if sal.shape[:2] != frame.shape[:2]:
        sal = cv2.resize(sal, (frame.shape[1], frame.shape[0]))
    return cv2.addWeighted(frame, 1 - alpha, saliency_to_color(sal), alpha, 0)


def make_panel(frame: np.ndarray, gt: np.ndarray, pred: np.ndarray,
               fixation: Optional[np.ndarray] = None) -> np.ndarray:
    """Horizontal panel: frame | GT overlay | prediction overlay | prediction map."""
    gt_ov = overlay(frame, gt)
    if fixation is not None:
        for y, x in zip(*np.nonzero(fixation > 0.5)):
            cv2.circle(gt_ov, (int(x), int(y)), 4, (255, 255, 255), 1)
    panels = [frame, gt_ov, overlay(frame, pred), saliency_to_color(pred)]
    for p, name in zip(panels, ["input (last frame)", "GT", "prediction", "pred map"]):
        cv2.putText(p, name, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate(panels, axis=1)


def save_batch_visualization(path: str, clips: torch.Tensor, gts: torch.Tensor, preds: torch.Tensor,
                             mean: Sequence[float], std: Sequence[float],
                             fixations: Optional[torch.Tensor] = None, max_items: int = 4):
    """Save a grid (one row per sample) for a batch. Shapes: [B,C,T,H,W], [B,1,H,W], [B,1,H,W]."""
    rows = []
    for i in range(min(max_items, clips.shape[0])):
        frame = denormalize_frame(clips[i, :, -1], mean, std).copy()
        fix = fixations[i, 0].cpu().numpy() if fixations is not None else None
        rows.append(make_panel(frame, gts[i, 0].float().cpu().numpy(),
                               preds[i, 0].float().cpu().numpy(), fix))
    grid = np.concatenate(rows, axis=0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    return grid
