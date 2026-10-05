"""Saliency losses and metrics (batched, torch).

All functions take tensors of shape [B, 1, H, W] and return a tensor of shape [B]
(one value per sample) unless stated otherwise.

Definitions follow Bylinskii et al., "What do different evaluation metrics tell us about
saliency models?" (arXiv:1604.03605, https://arxiv.org/abs/1604.03605 -- the reference cited
by the DR(eye)VE code) and the DR(eye)VE reference loss
(https://github.com/ndrplz/dreyeve/blob/master/experiments/train/loss_functions.py).
"""

from __future__ import annotations

from typing import Dict

import torch

EPS = 1e-7  # Keras K.epsilon(), as used by the reference KLD loss


def _flat(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1).float()


def _to_distribution(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    x = _flat(x)
    return x / (x.sum(dim=1, keepdim=True) + eps)


def kld(pred: torch.Tensor, gt: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """KL divergence KL(GT || Pred) as in the DR(eye)VE reference loss (lower is better)."""
    p = _to_distribution(pred, eps)
    q = _to_distribution(gt, eps)
    return (q * torch.log(eps + q / (eps + p))).sum(dim=1)


def cc(pred: torch.Tensor, gt: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Pearson's linear correlation coefficient (higher is better)."""
    p = _flat(pred)
    q = _flat(gt)
    p = (p - p.mean(1, keepdim=True)) / (p.std(1, keepdim=True) + eps)
    q = (q - q.mean(1, keepdim=True)) / (q.std(1, keepdim=True) + eps)
    return (p * q).mean(1) * (p.shape[1] / max(p.shape[1] - 1, 1))


def sim(pred: torch.Tensor, gt: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Similarity / histogram intersection (higher is better)."""
    return torch.minimum(_to_distribution(pred, eps), _to_distribution(gt, eps)).sum(1)


def nss(pred: torch.Tensor, fixations: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Normalized Scanpath Saliency on a binary fixation map (higher is better).

    Samples without any fixation yield NaN so that they can be excluded by ``nanmean``.
    """
    p = _flat(pred)
    f = (_flat(fixations) > 0.5).float()
    p = (p - p.mean(1, keepdim=True)) / (p.std(1, keepdim=True) + eps)
    n = f.sum(1)
    out = (p * f).sum(1) / n.clamp_min(1)
    return torch.where(n > 0, out, torch.full_like(out, float("nan")))


def information_gain(pred: torch.Tensor, fixations: torch.Tensor, baseline: torch.Tensor,
                     eps: float = EPS) -> torch.Tensor:
    """Information gain (bits/fixation) of ``pred`` over ``baseline`` (higher is better)."""
    p = _to_distribution(pred, eps)
    b = _to_distribution(baseline, eps)
    f = (_flat(fixations) > 0.5).float()
    n = f.sum(1)
    out = (f * (torch.log2(eps + p) - torch.log2(eps + b))).sum(1) / n.clamp_min(1)
    return torch.where(n > 0, out, torch.full_like(out, float("nan")))


class KLDLoss(torch.nn.Module):
    """Mean KLD over the batch (the loss used by DR(eye)VE for both outputs)."""

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return kld(pred, gt).mean()


@torch.no_grad()
def compute_metrics(pred: torch.Tensor, gt: torch.Tensor, fixations: torch.Tensor = None) -> Dict[str, torch.Tensor]:
    """Return per-sample metrics ``{name: tensor[B]}``."""
    out = {"kld": kld(pred, gt), "cc": cc(pred, gt), "sim": sim(pred, gt)}
    if fixations is not None:
        out["nss"] = nss(pred, fixations)
    return out


class MetricAccumulator:
    """Accumulates per-sample metrics and reports means (NaNs are ignored)."""

    def __init__(self):
        self._values: Dict[str, list] = {}

    def update(self, metrics: Dict[str, torch.Tensor]):
        for k, v in metrics.items():
            self._values.setdefault(k, []).append(v.detach().float().cpu().reshape(-1))

    def compute(self) -> Dict[str, float]:
        return {k: torch.cat(v).nanmean().item() for k, v in self._values.items()}
