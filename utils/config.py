"""YAML config loading with ``key.sub=value`` command-line overrides."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import yaml


def _merge(base: Dict, other: Dict) -> Dict:
    out = copy.deepcopy(base)
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str, overrides: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Load a YAML config. ``_base_: other.yaml`` (relative path) is merged first.

    :param overrides: strings like ``train.epochs=5`` (values parsed as YAML).
    """
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    base = cfg.pop("_base_", None)
    if base:
        cfg = _merge(load_config(str(path.parent / base)), cfg)
    for item in overrides or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Override must be key=value, got {item!r}")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
