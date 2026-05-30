"""Main training entry point.

Trains the multimodal zero-shot model on the AzSL seen split.
Only the azBERT text encoder and the linear projection heads are updated;
visual encoders stay frozen — paper Sec. V-C.

Usage
-----
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --batch_size 128
    python scripts/train.py --config configs/default.yaml --resume outputs/exp01/checkpoints/checkpoint.pt
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Make `src.*` importable when running this script directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data import (
    DescriptionLoader,
    ZeroShotSignDataset,
    collate_fn,
    load_split,
)
from src.models import MultimodalZSLModel
from src.utils import (
    MetricsLogger,
    TensorBoardLogger,
    WandbLogger,
    load_checkpoint,
    save_checkpoint,
    setup_logger,
)


# ----------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train ZSSLR-AzSL")
    p.add_argument("--config",      type=str, default="configs/default.yaml")
    p.add_argument("--resume",      type=str, default=None)
    p.add_argument("--exp_name",    type=str, default=None)
    p.add_argument("--seed",        type=int, default=None)
    p.add_argument("--batch_size",  type=int, default=None)
    p.add_argument("--lr",          type=float, default=None)
    p.add_argument("--epochs",      type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--device",      type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--use_wandb",        action="store_true")
    p.add_argument("--use_tensorboard",  action="store_true")
    p.add_argument("--output_dir", type=str, default="outputs")
    return p.parse_args()


# ----------------------------------------------------------------------
def load_config(path: str) -> Dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------------
def build_dataloaders(cfg: Dict, args) -> tuple:
    azsld_cfg = cfg["data"]["azsld"]
    pp_cfg = cfg["data"]["preprocessing"]

    # 1. Load descriptions
    desc_loader = DescriptionLoader(azsld_cfg["descriptions_file"])
    descriptions = dict(desc_loader.items())

    # 2. Load seen/unseen splits
    splits_dir = Path(azsld_cfg["splits_dir"])
    seen_glosses = load_split(splits_dir / "seen_glosses.txt")
    unseen_glosses_path = splits_dir / "unseen_glosses.txt"
    unseen_glosses = (
        load_split(unseen_glosses_path) if unseen_glosses_path.exists() else []
    )

    # Stable gloss-to-label mapping across the WHOLE vocabulary
    all_glosses = sorted(set(seen_glosses) | set(unseen_glosses) | set(descriptions.keys()))
    gloss_to_label = {g: i for i, g in enumerate(all_glosses)}

    # 3. Build train + val datasets restricted to seen glosses
    common_kwargs = dict(
        video_dir=azsld_cfg["video_dir"],
        descriptions=descriptions,
        gloss_to_label=gloss_to_label,
        skeleton_dir=azsld_cfg.get("skeleton_dir"),
        num_frames=pp_cfg["num_frames"],
        target_size=pp_cfg["spatial_size"],
        sampling=pp_cfg["sampling"],
        normalize_mean=pp_cfg["normalize_mean"],
        normalize_std=pp_cfg["normalize_std"],
    )

    train_ds = ZeroShotSignDataset(
        seen_glosses=seen_glosses,
        unseen_glosses=unseen_glosses,
        split="seen",
        **common_kwargs,
    )
    val_ds = ZeroShotSignDataset(
        seen_glosses=seen_glosses,
        unseen_glosses=unseen_glosses,
        split="seen",
        **common_kwargs,
    )

    # NOTE: in a real run, train/val are split by signer identity using
    # `splits/signer_map.json`. For the starter scaffold we simply use the
    # full seen split for both and rely on the in-script seed for repro.

    batch_size = args.batch_size or cfg["training"]["batch_size"]
    num_workers = args.num_workers or cfg["hardware"]["num_workers"]

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=cfg["hardware"]["pin_memory"],
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=cfg["hardware"]["pin_memory"],
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, gloss_to_label


# ----------------------------------------------------------------------
def build_model(cfg: Dict, device: str) -> MultimodalZSLModel:
    model = MultimodalZSLModel(
        embedding_dim=cfg["model"]["embedding_dim"],
        temperature=cfg["model"]["temperature_init"],
        temperature_learnable=cfg["model"]["temperature_learnable"],
        sapiens_config=cfg["model"]["visual"]["sapiens"],
        motionbert_config=cfg["model"]["visual"]["motionbert"],
        azbert_config=cfg["model"]["text"],
    )
    return model.to(device)


# ----------------------------------------------------------------------
def build_optimizer(model: nn.Module, cfg: Dict, args) -> optim.Optimizer:
    lr = args.lr or cfg["training"]["learning_rate"]
    wd = cfg["training"]["weight_decay"]
    trainable = [p for p in model.parameters() if p.requires_grad]
    return optim.AdamW(trainable, lr=lr, weight_decay=wd)


# ----------------------------------------------------------------------
def build_scheduler(opt: optim.Optimizer, cfg: Dict, total_epochs: int):
    name = cfg["training"]["scheduler"]
    warmup = cfg["training"]["warmup_epochs"]
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, total_epochs - warmup), eta_min=1e-6,
        )
    if name == "step":
        return optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)
    if name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=10)
    return None


# ----------------------------------------------------------------------
def amp_dtype_from_str(name: str) -> Optional[torch.dtype]:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return None


# ----------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    grad_clip: float,
    amp_dtype: Optional[torch.dtype],
    log_every: int,
    logger,
    wandb_logger: Optional[WandbLogger],
) -> Dict[str, float]:
    model.train()
    losses, accs = [], []
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    use_amp = amp_dtype is not None and device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    for step, batch in enumerate(pbar):
        frames = batch["frames"].to(device, non_blocking=True)
        skeleton = (batch["skeleton"].to(device, non_blocking=True)
                    if batch["skeleton"] is not None else None)
        descriptions = batch["description"]

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                loss, metrics = model(frames, descriptions, skeleton)
            if amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        else:
            loss, metrics = model(frames, descriptions, skeleton)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        losses.append(metrics["loss"])
        accs.append(metrics["acc"])
        pbar.set_postfix({
            "loss": f"{metrics['loss']:.3f}",
            "acc":  f"{metrics['acc']:.3f}",
            "τ":    f"{metrics['temperature']:.3f}",
        })

        if (step + 1) % log_every == 0 and wandb_logger is not None:
            wandb_logger.log(
                {f"train/{k}": v for k, v in metrics.items()
                 if isinstance(v, (int, float))},
                step=epoch * len(loader) + step,
            )

    avg_loss = float(np.mean(losses)) if losses else 0.0
    avg_acc = float(np.mean(accs)) if accs else 0.0
    logger.info("Epoch %d  train_loss=%.4f  train_acc=%.4f", epoch, avg_loss, avg_acc)
    return {"loss": avg_loss, "acc": avg_acc}


# ----------------------------------------------------------------------
@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    losses, accs = [], []
    for batch in tqdm(loader, desc="Val", leave=False):
        frames = batch["frames"].to(device, non_blocking=True)
        skeleton = (batch["skeleton"].to(device, non_blocking=True)
                    if batch["skeleton"] is not None else None)
        loss, metrics = model(frames, batch["description"], skeleton)
        losses.append(metrics["loss"])
        accs.append(metrics["acc"])
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "acc":  float(np.mean(accs)) if accs else 0.0,
    }


# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed)

    exp_name = args.exp_name or f"zsslr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = Path(args.output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("train", log_dir=exp_dir / "logs")
    logger.info("Experiment: %s", exp_name)
    logger.info("Device: %s", args.device)
    logger.info("Config:\n%s", yaml.safe_dump(cfg, sort_keys=False))

    # Loggers
    wandb_logger = None
    if args.use_wandb:
        wandb_logger = WandbLogger(
            project=cfg["logging"]["project_name"],
            name=exp_name,
            config=cfg,
        )
    tb_logger = TensorBoardLogger(exp_dir / "tensorboard", enabled=args.use_tensorboard)
    metrics_logger = MetricsLogger(exp_dir / "metrics.json")

    # Data + model
    logger.info("Building dataloaders...")
    train_loader, val_loader, _ = build_dataloaders(cfg, args)
    logger.info("Train batches: %d  Val batches: %d",
                len(train_loader), len(val_loader))

    logger.info("Building model...")
    model = build_model(cfg, args.device)
    counts = model.count_parameters()
    logger.info("Params: total=%d, trainable=%d",
                counts["total"], counts["trainable"])

    optimizer = build_optimizer(model, cfg, args)
    num_epochs = args.epochs or cfg["training"]["num_epochs"]
    scheduler = build_scheduler(optimizer, cfg, num_epochs)
    amp_dtype = amp_dtype_from_str(cfg["training"]["mixed_precision"])

    start_epoch = 0
    if args.resume:
        info = load_checkpoint(args.resume, model, optimizer, scheduler,
                               map_location=args.device)
        start_epoch = info["epoch"] + 1
        logger.info("Resumed from epoch %d", info["epoch"])

    best_val_acc = -1.0
    ckpt_dir = exp_dir / "checkpoints"

    for epoch in range(start_epoch, num_epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, args.device,
            epoch=epoch + 1,
            grad_clip=cfg["training"]["gradient_clip"],
            amp_dtype=amp_dtype,
            log_every=cfg["logging"]["log_every_n_steps"],
            logger=logger, wandb_logger=wandb_logger,
        )

        do_val = (
            (epoch + 1) % cfg["training"]["val_every_n_epochs"] == 0
            or (epoch + 1) == num_epochs
        )
        val_metrics: Dict[str, float] = {}
        if do_val:
            val_metrics = validate(model, val_loader, args.device)
            logger.info("Epoch %d  val_loss=%.4f  val_acc=%.4f",
                        epoch + 1, val_metrics["loss"], val_metrics["acc"])
            if wandb_logger is not None:
                wandb_logger.log(
                    {f"val/{k}": v for k, v in val_metrics.items()},
                    step=(epoch + 1) * len(train_loader),
                )

            is_best = val_metrics["acc"] > best_val_acc
            if is_best:
                best_val_acc = val_metrics["acc"]
                logger.info("New best val acc: %.4f", best_val_acc)

            save_checkpoint(
                model, optimizer, scheduler,
                epoch=epoch + 1,
                step=(epoch + 1) * len(train_loader),
                metrics={**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}},
                checkpoint_dir=ckpt_dir,
                filename="checkpoint.pt",
                is_best=is_best,
            )

        # Per-epoch metric log
        metrics_logger.log(
            {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}},
            step=(epoch + 1) * len(train_loader),
            epoch=epoch + 1,
        )
        tb_logger.log_scalars("train", train_metrics, epoch + 1)
        if val_metrics:
            tb_logger.log_scalars("val", val_metrics, epoch + 1)

        # LR scheduler
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                if val_metrics:
                    scheduler.step(val_metrics["acc"])
            else:
                scheduler.step()

    # Final save
    save_checkpoint(
        model, optimizer, scheduler,
        epoch=num_epochs, step=num_epochs * len(train_loader),
        metrics={}, checkpoint_dir=ckpt_dir, filename="final.pt",
    )
    if wandb_logger is not None:
        wandb_logger.finish()
    tb_logger.close()
    logger.info("Training complete. Best val acc: %.4f", best_val_acc)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
