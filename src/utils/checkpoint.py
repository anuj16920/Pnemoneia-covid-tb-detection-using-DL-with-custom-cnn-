"""
checkpoint.py
Model checkpoint saving and loading.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    path: str,
) -> None:
    """
    Save model checkpoint.

    Saves:
      - model state dict
      - optimizer state dict
      - epoch number
      - validation metrics
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, path)


def load_checkpoint(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict:
    """
    Load model from checkpoint.

    Args:
        model:     Model to load weights into
        path:      Checkpoint file path
        optimizer: Optional optimizer to restore state
        device:    Device to load onto

    Returns:
        checkpoint dict (contains epoch, metrics, etc.)
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", {})
    print(f"Loaded checkpoint from epoch {epoch+1} | "
          f"val_f1={metrics.get('f1_macro', 'N/A'):.4f}")

    return checkpoint


def load_model_for_inference(
    model: nn.Module,
    path: str,
    device: str = "cuda",
) -> nn.Module:
    """
    Load model checkpoint for inference only (no optimizer).
    Sets model to eval mode.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    load_checkpoint(model, path, device=str(dev))
    model = model.to(dev)
    model.eval()
    return model
