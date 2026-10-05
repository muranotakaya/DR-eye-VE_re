"""Train the DR(eye)VE baseline.

Examples:
    # smoke test on synthetic data (no dataset needed, runs on CPU)
    python train.py --config configs/dummy_debug.yaml --dummy
    # real data
    python train.py --config configs/dreyeve_c3d.yaml
    python train.py --config configs/dreyeve_r3d18.yaml --opts train.batch_size=8 data.root=/path/to/data
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from datasets import build_dataset
from evaluate import run_evaluation
from models import build_model, count_parameters
from utils.config import load_config, resolve_device, seed_everything
from utils.metrics import kld


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dummy", action="store_true", help="train on synthetic data (no dataset needed)")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument("--opts", nargs="*", default=[], help="config overrides, e.g. train.epochs=5")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.opts)
    seed_everything(cfg.get("seed", 0))
    device = resolve_device(cfg.get("device", "auto"))
    data_cfg, train_cfg = cfg["data"], cfg["train"]
    if cfg["model"].get("branches"):
        raise NotImplementedError("Multi-branch training needs a dataset providing flow / semseg clips.")

    run_name = cfg.get("experiment", Path(args.config).stem) + ("_dummy" if args.dummy else "")
    out_dir = Path(cfg.get("output_dir", "outputs")) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    train_set = build_dataset(data_cfg, "train", dummy=args.dummy)
    val_set = build_dataset(data_cfg, "val", dummy=args.dummy)
    loader_kw = dict(batch_size=train_cfg["batch_size"], num_workers=train_cfg.get("num_workers", 0),
                     pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_set, shuffle=True, drop_last=len(train_set) >= train_cfg["batch_size"],
                              persistent_workers=loader_kw["num_workers"] > 0, **loader_kw)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kw)

    model = build_model(cfg["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"], betas=tuple(train_cfg.get("betas", (0.9, 0.999))),
                                 weight_decay=train_cfg.get("weight_decay", 0.0))
    use_amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"device={device} train={len(train_set)} val={len(val_set)} "
          f"params={count_parameters(model) / 1e6:.2f}M -> {out_dir}")

    start_epoch, best = 0, float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, best = ckpt["epoch"] + 1, ckpt.get("best_val_kld", best)

    w_fine, w_crop = train_cfg.get("w_loss_fine", 1.0), train_cfg.get("w_loss_crop", 1.0)
    max_steps = train_cfg.get("max_steps_per_epoch")
    log_path = out_dir / "log.jsonl"

    for epoch in range(start_epoch, train_cfg["epochs"]):
        model.train()
        train_set.set_epoch(epoch)
        t0, running, n = time.time(), 0.0, 0
        for step, batch in enumerate(train_loader):
            if max_steps is not None and step >= max_steps:
                break
            clip = batch["clip"].to(device, non_blocking=True)
            gt = batch["saliency"].to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=use_amp):
                if "clip_crop" in batch:
                    fine, crop = model(clip, batch["clip_crop"].to(device, non_blocking=True))
                else:
                    fine, crop = model(clip), None
            # losses in fp32 (KLD uses eps=1e-7)
            loss = w_fine * kld(fine.float(), gt).mean()
            if crop is not None:
                loss = loss + w_crop * kld(crop.float(), batch["saliency_crop"].to(device)).mean()

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            n += 1
            if step % train_cfg.get("log_every", 20) == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

        vis = str(out_dir / "vis" / f"val_epoch{epoch:03d}.png") if train_cfg.get("vis_every_epoch", True) else None
        val = run_evaluation(model, val_loader, device, max_steps=train_cfg.get("max_val_steps"),
                             vis_path=vis, mean=data_cfg["mean"], std=data_cfg["std"], amp=use_amp)
        record = {"epoch": epoch, "train_loss": running / max(n, 1), "time_s": time.time() - t0,
                  **{f"val_{k}": v for k, v in val.items()}}
        print(json.dumps(record))
        with log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

        is_best = val["kld"] < best
        best = min(best, val["kld"])
        ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                "best_val_kld": best, "config": cfg}
        torch.save(ckpt, out_dir / "last.pt")
        if is_best:
            torch.save(ckpt, out_dir / "best.pt")

    print(f"done. best val KLD = {best:.4f}; checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
