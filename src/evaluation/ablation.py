"""
ablation.py
Ablation study runner.

Evaluates four model configurations to isolate component contributions:
  1. Backbone-Only      (EfficientNet-B3, no attention, no Swin)
  2. Backbone + CBAM   (sequential fixed attention)
  3. Backbone + DyDA   (dynamic dual attention, no Swin)
  4. Full Model        (EfficientNet-B3 + DyDA + Swin Transformer)
"""

import json
import copy
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from ..models.full_model import PulmonaryDxModel
from ..training.trainer import Trainer
from ..evaluation.metrics import compute_metrics, format_metrics_table, aggregate_fold_metrics
from ..utils.logger import get_logger
from ..utils.seed import set_seed


ABLATION_CONFIGS = [
    {
        "name": "backbone_only",
        "label": "Backbone-Only (EfficientNet-B3)",
        "use_dyda": False,
        "use_cbam": False,
        "use_swin": False,
    },
    {
        "name": "backbone_cbam",
        "label": "Backbone + CBAM",
        "use_dyda": False,
        "use_cbam": True,
        "use_swin": False,
    },
    {
        "name": "backbone_dyda",
        "label": "Backbone + DyDA (no Swin)",
        "use_dyda": True,
        "use_cbam": False,
        "use_swin": False,
    },
    {
        "name": "full_model",
        "label": "Full Model (EfficientNet + DyDA + Swin)",
        "use_dyda": True,
        "use_cbam": False,
        "use_swin": True,
    },
]


def run_ablation(
    cfg: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_weights=None,
    output_dir: str = "results/ablation",
    fold_idx: int = 0,
) -> Dict:
    """
    Run all ablation configurations and collect results.

    Args:
        cfg:           Master config dict
        train_loader:  Training DataLoader
        val_loader:    Validation DataLoader
        test_loader:   Test DataLoader
        class_weights: Class weights tensor
        output_dir:    Directory for ablation results
        fold_idx:      Fold index for reproducibility

    Returns:
        results: dict mapping config_name → metrics dict
    """
    logger = get_logger("ablation", log_file=str(Path(output_dir) / "ablation.log"))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = {}

    for config in ABLATION_CONFIGS:
        config_name = config["name"]
        config_label = config["label"]

        logger.info(f"\n{'='*60}")
        logger.info(f"ABLATION: {config_label}")
        logger.info(f"{'='*60}")

        # Set seed for reproducibility
        set_seed(cfg.get("project", {}).get("seed", 42))

        # Build model with this ablation config
        model = PulmonaryDxModel(
            num_classes=cfg.get("data", {}).get("num_classes", 4),
            pretrained=True,
            use_dyda=config["use_dyda"],
            use_cbam=config["use_cbam"],
            use_swin=config["use_swin"],
        )

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Parameters: {trainable_params:,} trainable / {total_params:,} total")

        # Train
        trainer = Trainer(
            model=model,
            cfg=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            fold_idx=fold_idx,
            class_weights=class_weights,
            log_dir=str(Path(output_dir) / "logs"),
            ckpt_dir=str(Path(output_dir) / "checkpoints"),
        )

        start_time = time.time()
        history = trainer.fit()
        train_time = time.time() - start_time

        # Evaluate on test set
        device = next(model.parameters()).device
        test_metrics = evaluate_on_loader(model, test_loader, device)

        test_metrics["training_time_sec"] = train_time
        test_metrics["total_params"] = total_params
        test_metrics["trainable_params"] = trainable_params
        test_metrics["best_val_f1"] = trainer.best_val_f1
        test_metrics["best_epoch"] = trainer.best_epoch
        test_metrics["config"] = config

        results[config_name] = test_metrics

        logger.info(f"Results for {config_label}:")
        logger.info(format_metrics_table(test_metrics))

    # Save results
    save_path = Path(output_dir) / "ablation_results.json"
    with open(save_path, "w") as f:
        # Convert numpy types to native Python for JSON serialization
        json.dump(_make_serializable(results), f, indent=2)

    logger.info(f"\nAblation results saved to {save_path}")
    print_ablation_summary(results)

    return results


@torch.no_grad()
def evaluate_on_loader(model, dataloader, device) -> Dict:
    """Evaluate model on a DataLoader and return metrics."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in dataloader:
        images = images.to(device)
        logits, _ = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return compute_metrics(all_labels, all_preds, all_probs)


def print_ablation_summary(results: Dict) -> None:
    """Print clean ablation summary table."""
    metrics_to_show = ["accuracy", "f1_macro", "auc_roc", "mean_sensitivity", "trainable_params"]
    label_map = {
        "accuracy": "Accuracy",
        "f1_macro": "F1 (Macro)",
        "auc_roc": "AUC-ROC",
        "mean_sensitivity": "Sensitivity",
        "trainable_params": "Params",
    }

    print("\n" + "=" * 80)
    print("  ABLATION STUDY SUMMARY")
    print("=" * 80)

    # Header
    header = f"  {'Configuration':<42}"
    for m in metrics_to_show:
        header += f" {label_map[m]:>10}"
    print(header)
    print("  " + "-" * 77)

    for config in ABLATION_CONFIGS:
        name = config["name"]
        label = config["label"]
        row = f"  {label:<42}"
        r = results.get(name, {})
        for m in metrics_to_show:
            val = r.get(m, float("nan"))
            if m == "trainable_params":
                row += f" {int(val) if not isinstance(val, float) else 0:>10,}"
            elif isinstance(val, float):
                row += f" {val:>10.4f}"
            else:
                row += f" {'N/A':>10}"
        print(row)

    print("=" * 80)


def _make_serializable(obj):
    """Recursively convert numpy types to Python native for JSON."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


if __name__ == "__main__":
    print("Ablation module ready.")
    print("Configurations:")
    for c in ABLATION_CONFIGS:
        print(f"  {c['name']}: DyDA={c['use_dyda']}, CBAM={c['use_cbam']}, Swin={c['use_swin']}")
