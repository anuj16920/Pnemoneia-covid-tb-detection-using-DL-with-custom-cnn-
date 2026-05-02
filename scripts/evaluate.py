#!/usr/bin/env python3
"""
evaluate.py
Evaluate a trained model on the test set.

Usage:
    python scripts/evaluate.py --config configs/config.yaml \
                                --checkpoint results/checkpoints/best_fold1.pth \
                                --splits results/logs/splits.json
"""

import sys
import argparse
import yaml
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.full_model import PulmonaryDxModel
from src.data.data_splits import create_cv_splits, get_fold_dataloaders, load_splits
from src.data.dataset import CLASS_TO_IDX
from src.evaluation.metrics import compute_metrics, format_metrics_table
from src.utils.checkpoint import load_model_for_inference
from src.utils.seed import set_seed


CLASS_NAMES = ["Normal", "COVID-19", "Pneumonia", "Tuberculosis"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--splits",     type=str, default=None)
    parser.add_argument("--fold",       type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="results/evaluation")
    return parser.parse_args()


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save a normalized confusion matrix."""
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title("Confusion Matrix (Counts)")
    ax1.set_ylabel("True Label")
    ax1.set_xlabel("Predicted Label")

    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt=".3f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title("Confusion Matrix (Normalized)")
    ax2.set_ylabel("True Label")
    ax2.set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved: {save_path}")


def plot_roc_curves(true_labels, pred_probs, class_names, save_path):
    """Plot multi-class ROC curves."""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    import numpy as np

    n_classes = len(class_names)
    true_bin = label_binarize(true_labels, classes=list(range(n_classes)))
    probs = np.array(pred_probs)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]

    for i, (cls_name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(true_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{cls_name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC curves saved: {save_path}")


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("project", {}).get("seed", 42))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load splits
    if args.splits and Path(args.splits).exists():
        splits = load_splits(args.splits)
    else:
        print("No splits file provided. Creating new splits (may differ from training splits).")
        splits = create_cv_splits(cfg["data"]["root_dir"])

    # Get test loader
    _, _, test_loader = get_fold_dataloaders(
        splits=splits,
        fold_idx=args.fold,
        cfg=cfg,
        batch_size=32,
        num_workers=4,
    )

    # Load model
    model = PulmonaryDxModel(
        num_classes=cfg.get("data", {}).get("num_classes", 4),
        pretrained=False,  # Will load from checkpoint
        use_dyda=True,
        use_swin=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model_for_inference(model, args.checkpoint, device=device)

    # Evaluate
    all_preds, all_labels, all_probs = [], [], []
    model.eval()

    print(f"\nEvaluating on {len(test_loader.dataset)} test samples...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits, _ = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    print(format_metrics_table(metrics))

    # Save metrics
    with open(Path(args.output_dir) / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Plots
    import numpy as np
    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(cm, CLASS_NAMES,
                          Path(args.output_dir) / "confusion_matrix.png")

    plot_roc_curves(all_labels, all_probs, CLASS_NAMES,
                    Path(args.output_dir) / "roc_curves.png")

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
