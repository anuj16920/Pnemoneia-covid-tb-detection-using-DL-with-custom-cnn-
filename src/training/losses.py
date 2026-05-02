"""
losses.py
Loss functions for pulmonary disease classification.

Includes:
  - LabelSmoothingCrossEntropy: Reduces overconfidence, improves calibration
  - MixUpCriterion: MixUp-compatible loss combining two labels
  - FocalLoss: Down-weights well-classified examples (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    Instead of one-hot targets, uses:
        y_smooth = (1 - ε) * y_hard + ε / K
    where K = num_classes, ε = smoothing.

    Benefits:
      - Reduces model overconfidence
      - Acts as regularization
      - Improves calibration and generalization

    Args:
        smoothing:     Label smoothing factor ε ∈ [0, 1) (default: 0.1)
        weight:        Per-class weights tensor [num_classes] (optional)
        reduction:     'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        assert 0 <= smoothing < 1.0
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, C] raw class scores
            targets: [B] integer class indices

        Returns:
            Scalar loss
        """
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)  # [B, C]

        # Smooth targets: (1-ε)*one_hot + ε/K
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / num_classes)
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        # Compute loss
        loss = -(smooth_targets * log_probs)   # [B, C]

        if self.weight is not None:
            # Apply class weights
            w = self.weight.to(logits.device)
            sample_weights = w[targets]         # [B]
            loss = loss.sum(dim=-1) * sample_weights  # [B]
        else:
            loss = loss.sum(dim=-1)             # [B]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class MixUpCriterion(nn.Module):
    """
    Loss function for MixUp augmentation.
    Computes: λ·loss(y_a) + (1-λ)·loss(y_b)

    Args:
        base_criterion: The underlying loss function to use
    """

    def __init__(self, base_criterion: nn.Module):
        super().__init__()
        self.base_criterion = base_criterion

    def forward(
        self,
        logits: torch.Tensor,
        targets_a: torch.Tensor,
        targets_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """
        Args:
            logits:    [B, C] model outputs
            targets_a: [B] labels for first mixed image
            targets_b: [B] labels for second mixed image
            lam:       MixUp lambda (interpolation ratio)

        Returns:
            Scalar mixed loss
        """
        loss_a = self.base_criterion(logits, targets_a)
        loss_b = self.base_criterion(logits, targets_b)
        return lam * loss_a + (1 - lam) * loss_b


class FocalLoss(nn.Module):
    """
    Focal loss (Lin et al., 2017) for handling class imbalance.
    Reduces relative loss for well-classified examples.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        gamma:  Focusing parameter γ ≥ 0 (default: 2.0)
        weight: Per-class alpha weights [num_classes] (optional)
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight,
                                  reduction="none")    # [B]
        pt = torch.exp(-ce_loss)                       # [B] probability of true class
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # [B]

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def build_criterion(cfg: dict, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """
    Build loss function from config.

    Args:
        cfg:           Full config dict
        class_weights: Inverse-frequency class weights tensor

    Returns:
        criterion: Loss module
    """
    loss_cfg = cfg.get("training", {}).get("loss", {})
    loss_name = loss_cfg.get("name", "label_smoothing_ce")
    smoothing = loss_cfg.get("smoothing", 0.1)
    use_weights = loss_cfg.get("use_class_weights", True)

    weight = class_weights if use_weights else None

    if loss_name == "label_smoothing_ce":
        return LabelSmoothingCrossEntropy(smoothing=smoothing, weight=weight)
    elif loss_name == "focal":
        return FocalLoss(gamma=2.0, weight=weight)
    elif loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=weight)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


if __name__ == "__main__":
    # Verify losses
    B, C = 8, 4
    logits = torch.randn(B, C)
    targets = torch.randint(0, C, (B,))
    weights = torch.tensor([1.0, 2.0, 1.5, 3.0])

    # Label smoothing CE
    ls_ce = LabelSmoothingCrossEntropy(smoothing=0.1, weight=weights)
    loss = ls_ce(logits, targets)
    print(f"LabelSmoothingCE: {loss.item():.4f}")

    # MixUp
    targets_b = torch.randint(0, C, (B,))
    mixup = MixUpCriterion(ls_ce)
    loss_mix = mixup(logits, targets, targets_b, lam=0.6)
    print(f"MixUpCE:          {loss_mix.item():.4f}")

    # Focal
    focal = FocalLoss(gamma=2.0, weight=weights)
    loss_f = focal(logits, targets)
    print(f"FocalLoss:        {loss_f.item():.4f}")

    print("Loss tests passed ✓")
