"""Checkpoint saving / loading.

Robust to four key conventions encountered in the wild:
    {"model_state_dict": state}     # this repo (canonical)
    {"model_state":      state}     # earlier repo iteration
    {"model":            state}     # some HF / Lightning saves
    state                           # bare torch.save(model.state_dict())

Also strips DDP `module.` prefixes automatically.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    epoch: int,
    step: int,
    metrics: Dict[str, float],
    checkpoint_dir: Union[str, Path],
    filename: str = "checkpoint.pt",
    is_best: bool = False,
    extra: Optional[Dict] = None,
) -> str:
    """Save model + (optionally) optimizer / scheduler state."""
    cdir = Path(checkpoint_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    ckpt: Dict[str, Any] = {
        "epoch":            int(epoch),
        "step":             int(step),
        "metrics":          dict(metrics),
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    if extra:
        ckpt["extra"] = extra

    path = cdir / filename
    torch.save(ckpt, path)
    logger.info("Saved checkpoint to %s (epoch=%d, step=%d)", path, epoch, step)
    if is_best:
        best = cdir / "best.pt"
        shutil.copy2(path, best)
        logger.info("Marked best checkpoint at %s", best)
    return str(path)


# ----------------------------------------------------------------------
def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    """Robustly extract a state-dict regardless of which convention was used."""
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "model_state", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                logger.info("Extracted state_dict under key %r", key)
                return ckpt[key]
        # Heuristic: assume the dict itself IS a state_dict if it has tensor values
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            logger.info("Treating top-level dict as state_dict")
            return ckpt
    raise ValueError(
        "Could not find a state_dict in the checkpoint. "
        "Expected one of: model_state_dict / model_state / state_dict / model, "
        f"or a flat tensor dict. Got keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
    )


# ----------------------------------------------------------------------
def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = False,
) -> Dict[str, Any]:
    """Load weights into `model` (and optionally optimizer / scheduler).

    Returns a small dict with `{epoch, step, metrics}` from the checkpoint
    so callers can resume training.
    """
    p = Path(checkpoint_path)
    if not p.exists():
        raise FileNotFoundError(p)

    ckpt = torch.load(p, map_location=map_location)
    state = _extract_state_dict(ckpt)

    # Strip DDP prefix
    state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
             for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing:
        logger.warning("Missing %d keys (showing first 5): %s",
                       len(missing), missing[:5])
    if unexpected:
        logger.warning("Unexpected %d keys (showing first 5): %s",
                       len(unexpected), unexpected[:5])

    info: Dict[str, Any] = {"epoch": 0, "step": 0, "metrics": {}}
    if isinstance(ckpt, dict):
        info["epoch"] = int(ckpt.get("epoch", 0))
        info["step"]  = int(ckpt.get("step", 0))
        info["metrics"] = dict(ckpt.get("metrics", {}))

        if optimizer is not None and "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except (ValueError, KeyError) as e:
                logger.warning("Could not load optimizer state: %s", e)
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except (ValueError, KeyError) as e:
                logger.warning("Could not load scheduler state: %s", e)
    return info


# ----------------------------------------------------------------------
def resume_training(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    map_location: Union[str, torch.device] = "cpu",
) -> int:
    """Resume from a single checkpoint file. Returns the next epoch index."""
    if not Path(checkpoint_path).exists():
        logger.info("No checkpoint at %s — starting from scratch", checkpoint_path)
        return 0
    info = load_checkpoint(checkpoint_path, model, optimizer, scheduler, map_location)
    logger.info("Resumed from epoch %d", info["epoch"])
    return int(info["epoch"]) + 1
