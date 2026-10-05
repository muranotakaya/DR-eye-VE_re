"""Evaluate a DR(eye)VE model on a split and report KLD / CC / SIM / NSS.

Examples:
    python evaluate.py --config configs/dreyeve_c3d.yaml --checkpoint outputs/dreyeve_c3d/best.pt
    python evaluate.py --config configs/dummy_debug.yaml --dummy --checkpoint outputs/dummy_debug_dummy/best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from datasets import build_dataset
from models import build_model
from utils.config import load_config, resolve_device
from utils.metrics import MetricAccumulator, compute_metrics
from utils.visualization import save_batch_visualization


@torch.no_grad()
def run_evaluation(model: torch.nn.Module, loader: DataLoader, device: torch.device,
                   max_steps: Optional[int] = None, vis_path: Optional[str] = None,
                   mean=None, std=None, per_sample_csv: Optional[str] = None,
                   amp: bool = False) -> Dict[str, float]:
    """Run the model over ``loader`` and return mean metrics.

    The first batch is visualized to ``vis_path`` if given.
    """
    model.eval()
    acc = MetricAccumulator()
    rows = []
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        clip = batch["clip"].to(device, non_blocking=True)
        gt = batch["saliency"].to(device)
        fix = batch["fixation"].to(device)
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            pred = model(clip)
        pred = pred.float()
        m = compute_metrics(pred, gt, fix)
        acc.update(m)
        if per_sample_csv is not None:
            for i in range(clip.shape[0]):
                rows.append({"seq": batch["seq"][i], "frame": int(batch["frame"][i]),
                             **{k: float(v[i]) for k, v in m.items()}})
        if step == 0 and vis_path is not None:
            save_batch_visualization(vis_path, clip, gt, pred, mean, std, fixations=fix)
    if per_sample_csv is not None and rows:
        Path(per_sample_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(per_sample_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return acc.compute()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="model checkpoint (.pt); random init if omitted")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--dummy", action="store_true", help="evaluate on synthetic data")
    parser.add_argument("--out-dir", default=None, help="where to write metrics / visualizations")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--opts", nargs="*", default=[], help="config overrides, e.g. train.batch_size=2")
    args = parser.parse_args()

    cfg = load_config(args.config, args.opts)
    device = resolve_device(cfg.get("device", "auto"))
    data_cfg, train_cfg = cfg["data"], cfg["train"]

    model = build_model(cfg["model"]).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded {args.checkpoint} (epoch {ckpt.get('epoch')})")
    else:
        print("WARNING: no checkpoint given, evaluating a randomly initialised model")

    dataset = build_dataset(data_cfg, args.split, dummy=args.dummy)
    loader = DataLoader(dataset, batch_size=train_cfg.get("batch_size", 8), shuffle=False,
                        num_workers=train_cfg.get("num_workers", 0), pin_memory=device.type == "cuda")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.checkpoint:
        out_dir = Path(args.checkpoint).parent / f"eval_{args.split}"
    else:
        out_dir = Path(cfg.get("output_dir", "outputs")) / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = run_evaluation(model, loader, device, max_steps=args.max_steps,
                             vis_path=str(out_dir / "samples.png"),
                             mean=data_cfg["mean"], std=data_cfg["std"],
                             per_sample_csv=str(out_dir / "per_sample.csv"),
                             amp=train_cfg.get("amp", False))
    result = {"split": args.split, "dummy": args.dummy, "checkpoint": args.checkpoint,
              "num_samples": len(dataset), "metrics": metrics}
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
