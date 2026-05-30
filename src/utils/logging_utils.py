"""Logger + optional W&B / TensorBoard / JSON metrics."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import wandb
    _HAS_WANDB = True
except ImportError:                                  # pragma: no cover
    _HAS_WANDB = False

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except ImportError:                                  # pragma: no cover
    _HAS_TB = False


# ----------------------------------------------------------------------
def setup_logger(
    name: str = "zsslr_azsl",
    log_dir: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create a logger that writes to stdout (+ optionally a file)."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        fh = logging.FileHandler(log_dir / fname)
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # Also configure the root so 3rd-party modules log nicely.
    logging.basicConfig(level=level, handlers=logger.handlers, force=False)
    return logger


# ----------------------------------------------------------------------
class WandbLogger:
    """Thin wrapper that silently no-ops when wandb is unavailable / disabled."""

    def __init__(
        self,
        project: str = "zssl-azsl",
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled and _HAS_WANDB
        self.project = project
        self.name = name
        if self.enabled:
            try:
                wandb.init(project=project, name=name, config=config or {})
            except Exception as e:                   # pragma: no cover
                logging.getLogger(__name__).warning(
                    "wandb init failed (%s) — disabling wandb logging.", e
                )
                self.enabled = False

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self.enabled:
            wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self.enabled:
            wandb.finish()


# ----------------------------------------------------------------------
class TensorBoardLogger:
    """Thin wrapper around `torch.utils.tensorboard.SummaryWriter`."""

    def __init__(self, log_dir: Union[str, Path], enabled: bool = True):
        self.enabled = enabled and _HAS_TB
        if self.enabled:
            self.writer = SummaryWriter(str(log_dir))
        else:
            self.writer = None

    def log_scalars(self, prefix: str, metrics: Dict[str, float], step: int) -> None:
        if not self.enabled:
            return
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"{prefix}/{k}", v, step)

    def close(self) -> None:
        if self.enabled and self.writer is not None:
            self.writer.close()


# ----------------------------------------------------------------------
class MetricsLogger:
    """Append-only JSON log of every metric entry. Safe to tail."""

    def __init__(self, log_file: Union[str, Path]):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.history: list = []
        if self.log_file.exists():
            try:
                self.history = json.loads(self.log_file.read_text())
            except json.JSONDecodeError:
                self.history = []

    def log(self, metrics: Dict[str, Any], step: int, epoch: int) -> None:
        entry = {
            "step":      int(step),
            "epoch":     int(epoch),
            "timestamp": datetime.now().isoformat(),
            **{k: (float(v) if isinstance(v, (int, float)) else v)
               for k, v in metrics.items()},
        }
        self.history.append(entry)
        self.log_file.write_text(json.dumps(self.history, indent=2))

    def best(self, metric: str, mode: str = "max") -> Optional[Dict]:
        vals = [(i, e[metric]) for i, e in enumerate(self.history) if metric in e]
        if not vals:
            return None
        idx = (max if mode == "max" else min)(vals, key=lambda x: x[1])[0]
        return self.history[idx]
