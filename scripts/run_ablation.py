#!/usr/bin/env python3
"""
run_ablation.py
Run the complete ablation study across all 4 model configurations.

Usage:
    python scripts/run_ablation.py --config configs/config.yaml
    python scripts/run_ablation.py --config configs/config.yaml --fold 0
"""

import sys
import argparse
import yaml
import json
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.full_model import PulmonaryDxModel
from src.data.data_splits import create_cv_splits, get_fold_dataloaders, load_splits, save_splits
from src.training.trainer import Trainer
from src.evaluation.metrics import compute_metrics, format_metrics_table
from src.evaluation.ablation import ABLATION_CONFIGS, print_ablation_summary
from src.utils.seed import set_seed
from src.utils.logger import get_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, default="configs/config.yaml")
    parser.add_argument("--fold",       type=int, default=0)
    parser.add_argument("--splits",     type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/ablation")
    return parser.parse_args()


def evaluate_loader(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits, _ = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    return compute_metrics(all_labels, all_preds, all_probs)


def plot_ablation_results(results, output_dir):
    """Bar chart comparing ablation configurations."""
    labels = [c["label"] for c in ABLATION_CONFIGS]
    short_labels = ["Backbone\nOnly", "+CBAM", "+DyDA\n(no Swin)", "Full\nModel"]
    metrics_to_plot = ["accuracy", "f1_macro", "auc_roc"]
    metric_labels = ["Accuracy", "F1 (Macro)", "AUC-ROC"]

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = ["#90CAF9", "#81C784", "#FFB74D", "#E57373"]

    for ax, metric, mlabel in zip(axes, metrics_to_plot, metric_labels):
        vals = []
        for config in ABLATION_CONFIGS:
            r = results.get(config["name"], {})
            vals.append(r.get(metric, 0))

        bars = ax.bar(x, vals, color=colors, width=0.6, edgecolor="white", linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=9)
        ax.set_ylim([max(0, min(vals) - 0.05), min(1.0, max(vals) + 0.05)])
        ax.set_title(mlabel, fontsize=13, fontweight="bold")
        ax.set_ylabel(mlabel)
        ax.grid(True, axis="y", alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.002,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Ablation Study Results", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(str(Path(output_dir) / "ablation_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Ablation plot saved to {output_dir}/ablation_comparison.png")


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("project", {}).get("seed", 42))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logger = get_logger("ablation_main",
                        log_file=str(Path(args.output_dir) / "ablation.log"))
    logger.info(f"Starting ablation study | Fold {args.fold}")

    # Load or create splits
    if args.splits and Path(args.splits).exists():
        splits = load_splits(args.splits)
    else:
        splits = create_cv_splits(
            root_dir=cfg["data"]["root_dir"],
            n_folds=cfg.get("cross_validation", {}).get("n_folds", 5),
            seed=cfg.get("project", {}).get("seed", 42),
        )

    # Data loaders
    train_loader, val_loader, test_loader = get_fold_dataloaders(
        splits=splits,
        fold_idx=args.fold,
        cfg=cfg,
        batch_size=cfg.get("training", {}).get("batch_size", 32),
        num_workers=cfg.get("project", {}).get("num_workers", 4),
    )

    class_weights = train_loader.dataset.get_class_weights()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}

    for config in ABLATION_CONFIGS:
        name = config["name"]
        label = config["label"]

        logger.info(f"\n{'='*55}")
        logger.info(f"  {label}")
        logger.info(f"{'='*55}")

        set_seed(cfg.get("project", {}).get("seed", 42))

        model = PulmonaryDxModel(
            num_classes=cfg.get("data", {}).get("num_classes", 4),
            pretrained=True,
            use_dyda=config["use_dyda"],
            use_cbam=config["use_cbam"],
            use_swin=config["use_swin"],
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable: {trainable:,} / Total: {total_params:,}")

        trainer = Trainer(
            model=model,
            cfg=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            fold_idx=0,
            class_weights=class_weights,
            log_dir=str(Path(args.output_dir) / "logs"),
            ckpt_dir=str(Path(args.output_dir) / "checkpoints"),
        )

        t0 = time.time()
        history = trainer.fit()
        elapsed = time.time() - t0

        metrics = evaluate_loader(model, test_loader, device)
        metrics["training_time_sec"] = elapsed
        metrics["total_params"] = total_params
        metrics["trainable_params"] = trainable
        metrics["best_val_f1"] = trainer.best_val_f1
        metrics["config"] = {"name": name, "label": label}

        results[name] = metrics
        logger.info(format_metrics_table(metrics))

    # Summary and plots
    print_ablation_summary(results)
    plot_ablation_results(results, args.output_dir)

    # Save JSON
    def to_serializable(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    save_path = Path(args.output_dir) / "ablation_results.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=to_serializable)

    logger.info(f"\nAblation complete. Results: {save_path}")


if __name__ == "__main__":
    main()
