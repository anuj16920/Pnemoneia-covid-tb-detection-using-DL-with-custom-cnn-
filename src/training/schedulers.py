"""
schedulers.py
Learning rate schedulers with linear warmup.

Implements cosine annealing with linear warmup for stable
training of the hybrid EfficientNet+Swin architecture.
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import List


class CosineAnnealingWithWarmup(_LRScheduler):
    """
    Cosine annealing learning rate schedule with linear warmup.

    Schedule:
      Epochs [0, warmup_epochs):     linear ramp from 0 → base_lr
      Epochs [warmup_epochs, T_max): cosine decay from base_lr → min_lr

    Args:
        optimizer:     PyTorch optimizer
        T_max:         Total number of epochs
        warmup_epochs: Number of warmup epochs
        min_lr:        Minimum learning rate (eta_min)
        last_epoch:    Last epoch index (-1 = start fresh)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_max: int,
        warmup_epochs: int = 10,
        min_lr: float = 1e-7,
        last_epoch: int = -1,
    ):
        self.T_max = T_max
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate at current epoch."""
        epoch = self.last_epoch

        if epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = (epoch + 1) / max(self.warmup_epochs, 1)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        else:
            # Cosine annealing
            cosine_epochs = epoch - self.warmup_epochs
            cosine_total  = self.T_max - self.warmup_epochs

            if cosine_total <= 0:
                return self.base_lrs

            factor = 0.5 * (1 + math.cos(math.pi * cosine_epochs / cosine_total))
            return [
                self.min_lr + (base_lr - self.min_lr) * factor
                for base_lr in self.base_lrs
            ]


def build_scheduler(optimizer, cfg: dict):
    """
    Build scheduler from config dict.

    Args:
        optimizer: PyTorch optimizer
        cfg:       Full config dict

    Returns:
        scheduler instance
    """
    train_cfg = cfg.get("training", {})
    sched_cfg = train_cfg.get("scheduler", {})
    name = sched_cfg.get("name", "cosine_warmup")
    epochs = train_cfg.get("epochs", 50)
    warmup = sched_cfg.get("warmup_epochs", 10)
    min_lr = sched_cfg.get("min_lr", 1e-7)

    if name == "cosine_warmup":
        return CosineAnnealingWithWarmup(
            optimizer=optimizer,
            T_max=epochs,
            warmup_epochs=warmup,
            min_lr=min_lr,
        )
    elif name == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    elif name == "step":
        from torch.optim.lr_scheduler import StepLR
        return StepLR(optimizer, step_size=sched_cfg.get("step_size", 15), gamma=0.1)
    else:
        raise ValueError(f"Unknown scheduler: {name}")


def build_optimizer(model, cfg: dict) -> torch.optim.Optimizer:
    """
    Build optimizer with differential learning rates.

    Backbone gets lower LR than attention/swin/head.
    """
    train_cfg = cfg.get("training", {})
    opt_cfg = train_cfg.get("optimizer", {})
    name = opt_cfg.get("name", "adamw")
    base_lr = opt_cfg.get("lr", 1e-4)
    weight_decay = opt_cfg.get("weight_decay", 1e-4)
    betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
    eps = opt_cfg.get("eps", 1e-8)

    lr_backbone = train_cfg.get("lr_backbone", base_lr * 0.1)

    # Get parameter groups from model
    if hasattr(model, "get_parameter_groups"):
        param_groups = model.get_parameter_groups(base_lr=base_lr)
        # Override backbone LR
        for group in param_groups:
            if group.get("name") == "backbone":
                group["lr"] = lr_backbone
    else:
        param_groups = [{"params": model.parameters(), "lr": base_lr}]

    # Add weight decay settings to groups (no wd for bias/BN)
    final_groups = []
    for group in param_groups:
        wd_params = []
        no_wd_params = []
        for p in group["params"]:
            if p.requires_grad:
                if p.ndim == 1:  # bias, BN weight/bias
                    no_wd_params.append(p)
                else:
                    wd_params.append(p)

        final_groups.append({
            "params": wd_params,
            "lr": group["lr"],
            "weight_decay": weight_decay,
            "name": group.get("name", "wd"),
        })
        final_groups.append({
            "params": no_wd_params,
            "lr": group["lr"],
            "weight_decay": 0.0,
            "name": group.get("name", "no_wd") + "_no_wd",
        })

    # Filter empty groups
    final_groups = [g for g in final_groups if len(g["params"]) > 0]

    if name == "adamw":
        return torch.optim.AdamW(final_groups, betas=betas, eps=eps)
    elif name == "adam":
        return torch.optim.Adam(final_groups, betas=betas, eps=eps)
    elif name == "sgd":
        return torch.optim.SGD(final_groups, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


if __name__ == "__main__":
    # Visualize LR schedule
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = torch.nn.Linear(10, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sched = CosineAnnealingWithWarmup(opt, T_max=50, warmup_epochs=10, min_lr=1e-7)

    lrs = []
    for _ in range(50):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()

    print(f"Warmup peak LR:  {max(lrs[:10]):.2e}")
    print(f"Final LR:        {lrs[-1]:.2e}")
    print(f"Schedule shape:  {len(lrs)} epochs ✓")
