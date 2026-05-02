#!/usr/bin/env python3
"""
train.py
Main training script for pulmonary disease classification.

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --config configs/config.yaml --fold 0
    python scripts/train.py --config configs/config.yaml --all_folds
    python scripts/train.py --config configs/config.yaml --resume results/checkpoints/latest_fold1.pth
"""

import os
import sys
import argparse
import json
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.full_model import PulmonaryDxModel, print_model_summary
from src.data.data_splits import create_cv_splits, get_fold_dataloaders, save_splits, load_splits
from src.training.trainer import Trainer
from src.evaluation.metrics import (
    compute_metrics, format_metrics_table, aggregate_fold_metrics, print_cv_summary
)
from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.utils.checkpoint import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train pulmonary disease classification model")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--fold", type=int, default=None,
                        help="Specific fold to train (0-indexed). Default: fold 0")
    parser.add_argument("--all_folds", action="store_true",
                        help="Train all folds for full cross-validation")
    parser.add_argument("--splits_file", type=str, default=None,
                        help="Path to pre-computed splits JSON (skips scanning)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from checkpoint")
    parser.add_argument("--no_swin", action="store_true",
                        help="Disable Swin Transformer (ablation)")
    parser.add_argument("--no_dyda", action="store_true",
                        help="Disable DyDA attention (ablation)")
    parser.add_argument("--use_cbam", action="store_true",
                        help="Use CBAM instead of DyDA (ablation)")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def train_fold(
    cfg: dict,
    splits: dict,
    fold_idx: int,
    use_dyda: bool = True,
    use_cbam: bool = False,
    use_swin: bool = True,
    resume_path: str = None,
    logger=None,
) -> dict:
    """Train model on a single fold."""
    if logger is None:
        logger = get_logger(f"train_fold{fold_idx}")

    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING FOLD {fold_idx + 1} / {splits['n_folds']}")
    logger.info(f"{'='*60}")

    # Data loaders
    train_loader, val_loader, test_loader = get_fold_dataloaders(
        splits=splits,
        fold_idx=fold_idx,
        cfg=cfg,
        batch_size=cfg.get("training", {}).get("batch_size", 32),
        num_workers=cfg.get("project", {}).get("num_workers", 4),
    )

    # Class weights
    from src.data.dataset import PulmonaryDataset
    train_ds = train_loader.dataset
    class_weights = train_ds.get_class_weights()
    logger.info(f"Class weights: {class_weights.numpy()}")

    # Build model
    model = PulmonaryDxModel(
        num_classes=cfg.get("data", {}).get("num_classes", 4),
        pretrained=cfg.get("model", {}).get("backbone", {}).get("pretrained", True),
        freeze_backbone_stages=cfg.get("model", {}).get("backbone", {}).get("freeze_stages", [0,1,2]),
        use_dyda=use_dyda,
        use_cbam=use_cbam,
        use_swin=use_swin,
        dyda_reduction=cfg.get("model", {}).get("dyda", {}).get("reduction_ratio", 16),
        fusion_dropout=cfg.get("model", {}).get("fusion", {}).get("dropout", 0.4),
        hidden_dim=cfg.get("model", {}).get("fusion", {}).get("hidden_dim", 512),
    )

    print_model_summary(model)

    # Resume from checkpoint if specified
    resume_epoch = 0
    if resume_path and Path(resume_path).exists():
        logger.info(f"Resuming from {resume_path}")
        from src.training.schedulers import build_optimizer
        optimizer = build_optimizer(model, cfg)
        checkpoint = load_checkpoint(model, resume_path, optimizer=optimizer)
        resume_epoch = checkpoint.get('epoch', 0) + 1  # Start from next epoch
        logger.info(f"Resuming from epoch {resume_epoch}")

    # Train
    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        fold_idx=fold_idx,
        class_weights=class_weights,
        log_dir=cfg.get("paths", {}).get("logs", "results/logs"),
        ckpt_dir=cfg.get("paths", {}).get("checkpoints", "results/checkpoints"),
        resume_epoch=resume_epoch,  # Pass resume epoch
    )

    history = trainer.fit()

    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    device = next(model.parameters()).device
    model.eval()

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

    test_metrics = compute_metrics(all_labels, all_preds, all_probs)
    logger.info(format_metrics_table(test_metrics))

    # Save training curves
    save_training_curves(history, fold_idx, cfg.get("paths", {}).get("plots", "results/plots"))

    return {
        "history": history,
        "test_metrics": test_metrics,
        "best_val_f1": trainer.best_val_f1,
        "best_epoch": trainer.best_epoch,
    }


def save_training_curves(history: dict, fold_idx: int, output_dir: str) -> None:
    """Save training/validation loss and metric curves."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title(f"Loss — Fold {fold_idx+1}")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"],   label="Val")
    axes[1].set_title(f"Accuracy — Fold {fold_idx+1}")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F1 + LR
    ax2 = axes[2].twinx()
    axes[2].plot(history["val_f1"], "b-", label="Val F1")
    ax2.plot(history["lr"], "r--", alpha=0.5, label="LR")
    axes[2].set_title(f"Val F1 & LR — Fold {fold_idx+1}")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1", color="b")
    ax2.set_ylabel("LR", color="r")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(Path(output_dir) / f"training_curves_fold{fold_idx+1}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Seed
    set_seed(cfg.get("project", {}).get("seed", 42))

    logger = get_logger(
        "main_train",
        log_file=str(Path(cfg.get("paths", {}).get("logs", "results/logs")) / "main.log"),
    )
    logger.info(f"Config: {args.config}")
    logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Determine ablation flags
    use_dyda = (not args.no_dyda) and (not args.use_cbam)
    use_cbam = args.use_cbam
    use_swin = not args.no_swin

    # Load or create splits
    if args.splits_file and Path(args.splits_file).exists():
        splits = load_splits(args.splits_file)
    else:
        splits = create_cv_splits(
            root_dir=cfg.get("data", {}).get("root_dir", "data/"),
            n_folds=cfg.get("cross_validation", {}).get("n_folds", 5),
            test_fraction=cfg.get("data", {}).get("test_fraction", 0.15),
            seed=cfg.get("project", {}).get("seed", 42),
        )
        splits_path = Path(cfg.get("paths", {}).get("logs", "results/logs")) / "splits.json"
        save_splits(splits, splits_path)

    # Determine which folds to train
    if args.all_folds:
        fold_indices = list(range(splits["n_folds"]))
    else:
        fold_idx = args.fold if args.fold is not None else 0
        fold_indices = [fold_idx]

    # Train
    all_results = []
    for fold_idx in fold_indices:
        result = train_fold(
            cfg=cfg,
            splits=splits,
            fold_idx=fold_idx,
            use_dyda=use_dyda,
            use_cbam=use_cbam,
            use_swin=use_swin,
            resume_path=args.resume if len(fold_indices) == 1 else None,
            logger=logger,
        )
        all_results.append(result["test_metrics"])

    # Aggregate CV results
    if len(all_results) > 1:
        aggregated = aggregate_fold_metrics(all_results)
        print_cv_summary(aggregated)

        # Save CV results
        results_path = Path(cfg.get("paths", {}).get("logs", "results/logs")) / "cv_results.json"
        with open(results_path, "w") as f:
            json.dump(aggregated, f, indent=2, default=str)
        logger.info(f"CV results saved to {results_path}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
