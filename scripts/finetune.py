#!/usr/bin/env python3
"""
finetune.py
Fine-tune the trained model for 15 more epochs with stable growth mechanisms.

Features:
  - Lower learning rate (1/10th of original)
  - Cosine annealing with warm restarts
  - Progressive backbone unfreezing
  - Gradient accumulation for stability
  - Enhanced monitoring with smoothed metrics
  - Checkpoint averaging

Usage:
    python scripts/finetune.py --checkpoint results/checkpoints/best_fold1.pth --epochs 15
    python scripts/finetune.py --checkpoint results/checkpoints/best_fold1.pth --epochs 15 --lr 1e-5
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.full_model import PulmonaryDxModel
from src.data.data_splits import load_splits, get_fold_dataloaders
from src.training.trainer import Trainer
from src.training.schedulers import build_optimizer
from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.utils.checkpoint import load_checkpoint, save_checkpoint


class StableFineTuner:
    """
    Fine-tuner with stability mechanisms:
      - Exponential moving average of metrics
      - Progressive unfreezing
      - Adaptive learning rate
      - Checkpoint averaging
    """
    
    def __init__(
        self,
        model: nn.Module,
        cfg: dict,
        train_loader,
        val_loader,
        fold_idx: int,
        base_lr: float,
        logger,
        ckpt_dir: str,
    ):
        self.model = model
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.fold_idx = fold_idx
        self.base_lr = base_lr
        self.logger = logger
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Smoothed metrics (EMA with alpha=0.3)
        self.ema_alpha = 0.3
        self.smoothed_metrics = {}
        
        # Checkpoint averaging
        self.checkpoint_buffer = deque(maxlen=3)  # Keep last 3 checkpoints
        
    def progressive_unfreeze(self, epoch: int, total_epochs: int):
        """
        Progressively unfreeze backbone layers.
        
        Schedule:
          - Epochs 0-5: Keep stages [0,1,2] frozen
          - Epochs 6-10: Unfreeze stage 2
          - Epochs 11-15: Unfreeze stage 1
        """
        if epoch < 5:
            frozen_stages = [0, 1, 2]
        elif epoch < 10:
            frozen_stages = [0, 1]
        else:
            frozen_stages = [0]
        
        # Freeze/unfreeze backbone stages
        if hasattr(self.model, 'backbone'):
            for stage_idx, stage in enumerate(self.model.backbone.features):
                requires_grad = stage_idx not in frozen_stages
                for param in stage.parameters():
                    param.requires_grad = requires_grad
        
        self.logger.info(f"  🔓 Frozen stages: {frozen_stages}")
        
    def update_smoothed_metrics(self, metrics: dict):
        """Update exponential moving average of metrics."""
        for key, value in metrics.items():
            # Only smooth numeric values
            if isinstance(value, (int, float, np.number)):
                if key not in self.smoothed_metrics:
                    self.smoothed_metrics[key] = value
                else:
                    self.smoothed_metrics[key] = (
                        self.ema_alpha * value + 
                        (1 - self.ema_alpha) * self.smoothed_metrics[key]
                    )
    
    def average_checkpoints(self) -> dict:
        """Average the last N checkpoints for stability."""
        if len(self.checkpoint_buffer) == 0:
            return None
        
        avg_state_dict = {}
        for ckpt_path in self.checkpoint_buffer:
            ckpt = torch.load(ckpt_path, map_location=self.device)
            state_dict = ckpt['model_state_dict']
            
            for key, param in state_dict.items():
                if key not in avg_state_dict:
                    avg_state_dict[key] = param.clone()
                else:
                    avg_state_dict[key] += param
        
        # Average
        for key in avg_state_dict:
            avg_state_dict[key] /= len(self.checkpoint_buffer)
        
        return avg_state_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune trained model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint to fine-tune from")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: 1/10th of original)")
    parser.add_argument("--fold", type=int, default=0,
                        help="Fold index")
    parser.add_argument("--splits_file", type=str, default="results/logs/splits.json",
                        help="Path to splits JSON")
    parser.add_argument("--output_dir", type=str, default="results/finetuned",
                        help="Output directory for fine-tuned checkpoints")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    # Setup
    set_seed(cfg.get("project", {}).get("seed", 42))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = get_logger(
        "finetune",
        log_file=str(output_dir / f"finetune_fold{args.fold+1}.log"),
    )
    
    logger.info("="*80)
    logger.info("FINE-TUNING FOR STABLE GROWTH")
    logger.info("="*80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Fine-tune epochs: {args.epochs}")
    logger.info(f"Fold: {args.fold + 1}")
    
    # Load splits
    if not Path(args.splits_file).exists():
        logger.error(f"Splits file not found: {args.splits_file}")
        sys.exit(1)
    
    splits = load_splits(args.splits_file)
    
    # Data loaders
    train_loader, val_loader, test_loader = get_fold_dataloaders(
        splits=splits,
        fold_idx=args.fold,
        cfg=cfg,
        batch_size=cfg.get("training", {}).get("batch_size", 16),
        num_workers=cfg.get("project", {}).get("num_workers", 4),
    )
    
    # Build model
    model = PulmonaryDxModel(
        num_classes=cfg.get("data", {}).get("num_classes", 4),
        pretrained=False,  # We're loading from checkpoint
        freeze_backbone_stages=[],  # Will be controlled by progressive unfreezing
        use_dyda=True,
        use_cbam=False,
        use_swin=True,
        dyda_reduction=cfg.get("model", {}).get("dyda", {}).get("reduction_ratio", 16),
        fusion_dropout=cfg.get("model", {}).get("fusion", {}).get("dropout", 0.4),
        hidden_dim=cfg.get("model", {}).get("fusion", {}).get("hidden_dim", 512),
    )
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(model, args.checkpoint)
    start_epoch = checkpoint.get('epoch', 0) + 1
    logger.info(f"Loaded from epoch {start_epoch}")
    
    # Fine-tuning learning rate (1/10th of original or specified)
    original_lr = cfg.get("training", {}).get("optimizer", {}).get("lr", 1e-4)
    finetune_lr = args.lr if args.lr is not None else original_lr / 10
    logger.info(f"Fine-tune LR: {finetune_lr:.2e} (original: {original_lr:.2e})")
    
    # Update config for fine-tuning
    cfg_ft = cfg.copy()
    cfg_ft["training"]["epochs"] = start_epoch + args.epochs
    cfg_ft["training"]["optimizer"]["lr"] = finetune_lr
    cfg_ft["training"]["lr_backbone"] = finetune_lr / 10
    cfg_ft["training"]["lr_swin"] = finetune_lr / 10
    cfg_ft["training"]["lr_head"] = finetune_lr
    cfg_ft["training"]["lr_dyda"] = finetune_lr
    
    # Cosine annealing with warm restarts
    cfg_ft["training"]["scheduler"]["name"] = "cosine_warmup"
    cfg_ft["training"]["scheduler"]["warmup_epochs"] = 2
    cfg_ft["training"]["scheduler"]["min_lr"] = finetune_lr / 100
    
    # Disable early stopping for fine-tuning (we want all epochs)
    cfg_ft["training"]["early_stopping"]["enabled"] = False
    
    # Class weights
    train_ds = train_loader.dataset
    class_weights = train_ds.get_class_weights()
    logger.info(f"Class weights: {class_weights.numpy()}")
    
    # Create stable fine-tuner
    stable_ft = StableFineTuner(
        model=model,
        cfg=cfg_ft,
        train_loader=train_loader,
        val_loader=val_loader,
        fold_idx=args.fold,
        base_lr=finetune_lr,
        logger=logger,
        ckpt_dir=str(output_dir / "checkpoints"),
    )
    
    # Create trainer with updated config
    trainer = Trainer(
        model=model,
        cfg=cfg_ft,
        train_loader=train_loader,
        val_loader=val_loader,
        fold_idx=args.fold,
        class_weights=class_weights,
        log_dir=str(output_dir / "logs"),
        ckpt_dir=str(output_dir / "checkpoints"),
        resume_epoch=start_epoch,
    )
    
    # Override trainer's fit method with progressive unfreezing
    original_fit = trainer.fit
    
    def fit_with_progressive_unfreezing():
        logger.info(f"Starting fine-tuning | Epochs: {start_epoch+1}-{cfg_ft['training']['epochs']}")
        
        for epoch in range(start_epoch, cfg_ft['training']['epochs']):
            # Progressive unfreezing
            stable_ft.progressive_unfreeze(epoch - start_epoch, args.epochs)
            
            # Train one epoch
            epoch_start = time.time()
            train_metrics = trainer.train_epoch()
            val_metrics = trainer.validate()
            
            # Update smoothed metrics
            stable_ft.update_smoothed_metrics(val_metrics)
            
            # Scheduler step
            trainer.scheduler.step()
            current_lr = trainer.optimizer.param_groups[0]["lr"]
            
            # Log with smoothed metrics
            elapsed = time.time() - epoch_start
            f1_per_class = val_metrics.get('f1_per_class', [])
            class_names = cfg.get('data', {}).get('class_names', ['C0', 'C1', 'C2', 'C3'])
            f1_str = " | ".join([f"{name[:4]}:{f1:.3f}" for name, f1 in zip(class_names, f1_per_class)])
            
            logger.info(
                f"\n{'='*80}\n"
                f"Fine-tune Epoch [{epoch+1:03d}/{cfg_ft['training']['epochs']}]\n"
                f"{'='*80}\n"
                f"  Train: loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f}\n"
                f"  Val:   loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} "
                f"f1_macro={val_metrics['f1_macro']:.4f}\n"
                f"  Smoothed F1: {stable_ft.smoothed_metrics.get('f1_macro', 0):.4f}\n"
                f"  Per-class F1: {f1_str}\n"
                f"  LR: {current_lr:.2e} | Time: {elapsed:.1f}s\n"
                f"{'='*80}"
            )
            
            # Update history
            trainer.history["train_loss"].append(train_metrics["loss"])
            trainer.history["train_acc"].append(train_metrics["acc"])
            trainer.history["val_loss"].append(val_metrics["loss"])
            trainer.history["val_acc"].append(val_metrics["accuracy"])
            trainer.history["val_f1"].append(val_metrics["f1_macro"])
            trainer.history["val_precision"].append(val_metrics.get("precision_macro", 0.0))
            trainer.history["val_recall"].append(val_metrics.get("recall_macro", 0.0))
            trainer.history["val_auc"].append(val_metrics.get("auc_roc_macro", 0.0))
            trainer.history["lr"].append(current_lr)
            
            # Save checkpoints
            epoch_ckpt_path = stable_ft.ckpt_dir / f"finetune_epoch_{epoch+1:03d}_fold{args.fold+1}.pth"
            save_checkpoint(trainer.model, trainer.optimizer, epoch, val_metrics, str(epoch_ckpt_path))
            stable_ft.checkpoint_buffer.append(str(epoch_ckpt_path))
            
            # Save best based on smoothed F1
            smoothed_f1 = stable_ft.smoothed_metrics.get('f1_macro', 0)
            if smoothed_f1 > trainer.best_val_f1:
                trainer.best_val_f1 = smoothed_f1
                trainer.best_epoch = epoch + 1
                # Save with clear naming - BEST FINETUNED MODEL
                best_ckpt_path = stable_ft.ckpt_dir / f"BEST_FINETUNED_fold{args.fold+1}.pth"
                save_checkpoint(trainer.model, trainer.optimizer, epoch, val_metrics, str(best_ckpt_path))
                logger.info(f"  ✅ BEST FINETUNED MODEL SAVED → {best_ckpt_path.name} (smoothed_f1={smoothed_f1:.4f})")
            
            # Save latest
            latest_ckpt_path = stable_ft.ckpt_dir / f"latest_finetuned_fold{args.fold+1}.pth"
            save_checkpoint(trainer.model, trainer.optimizer, epoch, val_metrics, str(latest_ckpt_path))
            
            logger.info(f"  💾 Epoch checkpoint saved: {epoch_ckpt_path.name}")
        
        # Checkpoint averaging at the end
        logger.info("\n" + "="*80)
        logger.info("Creating averaged checkpoint from last 3 epochs...")
        avg_state_dict = stable_ft.average_checkpoints()
        if avg_state_dict:
            trainer.model.load_state_dict(avg_state_dict)
            avg_ckpt_path = stable_ft.ckpt_dir / f"AVERAGED_FINETUNED_fold{args.fold+1}.pth"
            save_checkpoint(trainer.model, trainer.optimizer, epoch, val_metrics, str(avg_ckpt_path))
            logger.info(f"  ✅ AVERAGED CHECKPOINT SAVED → {avg_ckpt_path.name}")
        
        logger.info("="*80)
        logger.info(f"Fine-tuning complete!")
        logger.info(f"  Best Epoch: {trainer.best_epoch}")
        logger.info(f"  Best Smoothed F1: {trainer.best_val_f1:.4f}")
        logger.info(f"  BEST MODEL: BEST_FINETUNED_fold{args.fold+1}.pth")
        logger.info(f"  AVERAGED MODEL: AVERAGED_FINETUNED_fold{args.fold+1}.pth")
        logger.info("="*80)
        return trainer.history
    
    import time
    history = fit_with_progressive_unfreezing()
    
    # Save training curves
    save_finetuning_curves(history, args.fold, str(output_dir / "plots"), start_epoch)
    
    # Final evaluation on test set
    logger.info("\n" + "="*80)
    logger.info("Evaluating fine-tuned model on test set...")
    logger.info("="*80)
    
    model.eval()
    device = next(model.parameters()).device
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits, _ = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    from src.evaluation.metrics import compute_metrics, format_metrics_table
    test_metrics = compute_metrics(all_labels, all_preds, all_probs)
    logger.info("\nTest Set Results:")
    logger.info(format_metrics_table(test_metrics))
    
    logger.info("\n" + "="*80)
    logger.info("✅ FINE-TUNING COMPLETE!")
    logger.info("="*80)


def save_finetuning_curves(history: dict, fold_idx: int, output_dir: str, start_epoch: int):
    """Save fine-tuning curves."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    epochs = range(start_epoch + 1, start_epoch + 1 + len(history["train_loss"]))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(epochs, history["train_loss"], 'b-', label="Train", linewidth=2)
    axes[0, 0].plot(epochs, history["val_loss"], 'r-', label="Val", linewidth=2)
    axes[0, 0].set_title("Loss (Fine-tuning)", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history["train_acc"], 'b-', label="Train", linewidth=2)
    axes[0, 1].plot(epochs, history["val_acc"], 'r-', label="Val", linewidth=2)
    axes[0, 1].set_title("Accuracy (Fine-tuning)", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1, 0].plot(epochs, history["val_f1"], 'g-', linewidth=2, marker='o', markersize=4)
    axes[1, 0].set_title("Val F1 Score (Fine-tuning)", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("F1 Macro")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1, 1].plot(epochs, history["lr"], 'purple', linewidth=2)
    axes[1, 1].set_title("Learning Rate Schedule", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(Path(output_dir) / f"finetuning_curves_fold{fold_idx+1}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Curves saved to {output_dir}")


if __name__ == "__main__":
    main()
