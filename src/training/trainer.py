"""
trainer.py
Training loop for the hybrid pulmonary disease classification model.

Features:
  - Mixed precision training (AMP)
  - MixUp augmentation
  - Gradient clipping
  - Early stopping
  - Checkpoint saving
  - Comprehensive metric tracking
  - Differential learning rates
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .losses import LabelSmoothingCrossEntropy, MixUpCriterion
from .schedulers import build_optimizer, build_scheduler
from ..data.preprocessing import MixUpTransform
from ..evaluation.metrics import compute_metrics
from ..utils.logger import get_logger
from ..utils.checkpoint import save_checkpoint, load_checkpoint


class EarlyStopping:
    """Early stopping with patience."""

    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score >= self.best_score + self.min_delta
        else:
            improved = score <= self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.stop = True

        return self.stop


class Trainer:
    """
    Trainer for PulmonaryDxModel.

    Args:
        model:         The PulmonaryDxModel instance
        cfg:           Config dict
        train_loader:  Training DataLoader
        val_loader:    Validation DataLoader
        fold_idx:      Current fold index (for checkpoint naming)
        class_weights: Per-class loss weights
        log_dir:       Directory for tensorboard/log files
        ckpt_dir:      Directory for checkpoints
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        fold_idx: int = 0,
        class_weights: Optional[torch.Tensor] = None,
        log_dir: str = "results/logs",
        ckpt_dir: str = "results/checkpoints",
        resume_epoch: int = 0,  # NEW: Resume from this epoch
    ):
        self.model = model
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.fold_idx = fold_idx
        self.log_dir = Path(log_dir)
        self.ckpt_dir = Path(ckpt_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Device
        self.device = torch.device(cfg.get("project", {}).get("device", "cuda")
                                   if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        if class_weights is not None:
            class_weights = class_weights.to(self.device)

        # Loss
        train_cfg = cfg.get("training", {})
        loss_cfg = train_cfg.get("loss", {})
        smoothing = loss_cfg.get("smoothing", 0.1)
        base_criterion = LabelSmoothingCrossEntropy(
            smoothing=smoothing,
            weight=class_weights,
        )
        self.criterion = base_criterion

        # MixUp
        mixup_alpha = (cfg.get("data", {}).get("augmentation", {})
                          .get("train", {}).get("mixup_alpha", 0.2))
        self.mixup = MixUpTransform(alpha=mixup_alpha) if mixup_alpha > 0 else None
        self.mixup_criterion = MixUpCriterion(base_criterion) if self.mixup else None

        # Optimizer & scheduler
        self.optimizer = build_optimizer(model, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg)

        # AMP
        self.use_amp = (cfg.get("project", {}).get("mixed_precision", True)
                        and self.device.type == "cuda")
        self.scaler = GradScaler(enabled=self.use_amp)

        # Training settings
        self.epochs = train_cfg.get("epochs", 50)
        self.grad_clip = train_cfg.get("gradient_clip_norm", 1.0)
        self.accum_steps = train_cfg.get("accumulation_steps", 1)

        # Early stopping
        es_cfg = train_cfg.get("early_stopping", {})
        self.early_stopping = EarlyStopping(
            patience=es_cfg.get("patience", 10),
            mode=es_cfg.get("mode", "max"),
        ) if es_cfg.get("enabled", True) else None

        # Logging
        self.logger = get_logger(
            f"trainer_fold{fold_idx}",
            log_file=str(self.log_dir / f"fold{fold_idx}_train.log"),
        )

        # History
        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
            "val_f1":     [], "lr": [],
            "val_precision": [], "val_recall": [],  # NEW
            "val_auc": [],  # NEW
        }
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.start_epoch = resume_epoch  # NEW: Start from this epoch

    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []
        
        # Progress logging interval
        log_interval = max(len(self.train_loader) // 10, 1)  # Log 10 times per epoch

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # MixUp augmentation
            use_mixup = self.mixup is not None and torch.rand(1).item() > 0.5
            if use_mixup:
                images, labels_a, labels_b, lam = self.mixup(images, labels)

            # Forward pass (AMP)
            with autocast(enabled=self.use_amp):
                logits, _ = self.model(images)

                if use_mixup:
                    loss = self.mixup_criterion(logits, labels_a, labels_b, lam)
                else:
                    loss = self.criterion(logits, labels)

                loss = loss / self.accum_steps

            # Backward
            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.accum_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.accum_steps

            # Track predictions (use original labels for accuracy)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Batch-level logging
            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(self.train_loader):
                batch_acc = sum(p == l for p, l in zip(
                    preds.cpu().numpy(), labels.cpu().numpy()
                )) / len(labels)
                
                # GPU memory stats
                if self.device.type == "cuda":
                    mem_allocated = torch.cuda.memory_allocated() / 1024**3
                    mem_reserved = torch.cuda.memory_reserved() / 1024**3
                    gpu_info = f"GPU: {mem_allocated:.2f}GB/{mem_reserved:.2f}GB"
                else:
                    gpu_info = ""
                
                self.logger.info(
                    f"  Batch [{batch_idx+1:>4}/{len(self.train_loader)}] "
                    f"loss: {loss.item() * self.accum_steps:.4f} "
                    f"acc: {batch_acc:.4f} {gpu_info}"
                )

        avg_loss = total_loss / len(self.train_loader)
        accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

        return {"loss": avg_loss, "acc": accuracy}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation epoch."""
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                logits, _ = self.model(images)
                loss = self.criterion(logits, labels)

            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        metrics = compute_metrics(all_labels, all_preds, all_probs)
        metrics["loss"] = avg_loss

        return metrics

    def fit(self) -> Dict:
        """
        Full training loop with early stopping and checkpointing.

        Returns:
            history: Training history dict
        """
        self.logger.info(f"Starting training | Fold {self.fold_idx+1} | "
                         f"Device: {self.device} | AMP: {self.use_amp}")
        self.logger.info(f"Epochs: {self.start_epoch+1}-{self.epochs} | Batches/epoch: {len(self.train_loader)}")

        for epoch in range(self.start_epoch, self.epochs):
            epoch_start = time.time()

            # ── Train ───────────────────────────────────────
            train_metrics = self.train_epoch()

            # ── Validate ─────────────────────────────────────
            val_metrics = self.validate()

            # ── LR step ──────────────────────────────────────
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # ── Log ──────────────────────────────────────────
            elapsed = time.time() - epoch_start
            
            # Per-class F1 scores
            f1_per_class = val_metrics.get('f1_per_class', [])
            class_names = self.cfg.get('data', {}).get('class_names', ['C0', 'C1', 'C2', 'C3'])
            f1_str = " | ".join([f"{name[:4]}:{f1:.3f}" for name, f1 in zip(class_names, f1_per_class)])
            
            self.logger.info(
                f"\n{'='*80}\n"
                f"Epoch [{epoch+1:03d}/{self.epochs}] Summary\n"
                f"{'='*80}\n"
                f"  Train: loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f}\n"
                f"  Val:   loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} "
                f"f1_macro={val_metrics['f1_macro']:.4f}\n"
                f"  Per-class F1: {f1_str}\n"
                f"  LR: {current_lr:.2e} | Time: {elapsed:.1f}s\n"
                f"{'='*80}"
            )

            # ── History ───────────────────────────────────────
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["acc"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["val_f1"].append(val_metrics["f1_macro"])
            self.history["val_precision"].append(val_metrics.get("precision_macro", 0.0))
            self.history["val_recall"].append(val_metrics.get("recall_macro", 0.0))
            self.history["val_auc"].append(val_metrics.get("auc_roc_macro", 0.0))
            self.history["lr"].append(current_lr)

            # ── Checkpoint ────────────────────────────────────
            val_f1 = val_metrics["f1_macro"]
            
            # Save every epoch
            epoch_ckpt_path = self.ckpt_dir / f"epoch_{epoch+1:03d}_fold{self.fold_idx+1}.pth"
            save_checkpoint(self.model, self.optimizer, epoch, val_metrics, str(epoch_ckpt_path))
            
            # Save best model
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_epoch = epoch + 1
                ckpt_path = self.ckpt_dir / f"best_fold{self.fold_idx+1}.pth"
                save_checkpoint(self.model, self.optimizer, epoch, val_metrics, str(ckpt_path))
                self.logger.info(f"  ✅ Best model saved (val_f1={val_f1:.4f})")

            # Save latest
            save_checkpoint(
                self.model, self.optimizer, epoch, val_metrics,
                str(self.ckpt_dir / f"latest_fold{self.fold_idx+1}.pth"),
            )
            
            self.logger.info(f"  💾 Checkpoints saved: epoch_{epoch+1:03d}, latest, {'best' if val_f1 > self.best_val_f1 else ''}")

            # ── Early stopping ────────────────────────────────
            if self.early_stopping and self.early_stopping(val_f1):
                self.logger.info(
                    f"Early stopping triggered at epoch {epoch+1} "
                    f"(best epoch: {self.best_epoch}, best F1: {self.best_val_f1:.4f})"
                )
                break

        self.logger.info(
            f"Training complete | Best epoch: {self.best_epoch} | "
            f"Best val F1: {self.best_val_f1:.4f}"
        )

        return self.history
