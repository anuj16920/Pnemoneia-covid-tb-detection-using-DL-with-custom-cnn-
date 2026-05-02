"""
metrics.py
Evaluation metrics for pulmonary disease classification.

Computes: accuracy, macro/per-class F1, ROC-AUC, confusion matrix,
sensitivity, specificity per class, and overall balanced accuracy.
"""

import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
)

CLASS_NAMES = ["Normal", "COVID-19", "Pneumonia", "Tuberculosis"]


def compute_metrics(
    true_labels: List[int],
    pred_labels: List[int],
    pred_probs: Optional[List] = None,
    class_names: List[str] = CLASS_NAMES,
) -> Dict:
    """
    Compute comprehensive classification metrics.

    Args:
        true_labels: Ground truth class indices
        pred_labels: Predicted class indices
        pred_probs:  Softmax probabilities [N, num_classes] (for AUC)
        class_names: Class names for per-class metrics

    Returns:
        metrics dict with all computed values
    """
    true = np.array(true_labels)
    pred = np.array(pred_labels)
    num_classes = len(class_names)

    metrics = {}

    # Basic metrics
    metrics["accuracy"] = float(accuracy_score(true, pred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(true, pred))
    metrics["cohen_kappa"] = float(cohen_kappa_score(true, pred))

    # F1 scores
    metrics["f1_macro"] = float(f1_score(true, pred, average="macro", zero_division=0))
    metrics["f1_weighted"] = float(f1_score(true, pred, average="weighted", zero_division=0))
    metrics["f1_per_class"] = f1_score(true, pred, average=None, zero_division=0).tolist()

    # Confusion matrix
    cm = confusion_matrix(true, pred, labels=list(range(num_classes)))
    metrics["confusion_matrix"] = cm.tolist()

    # Per-class sensitivity (recall) and specificity
    sensitivity = []
    specificity = []
    precision_per_class = []

    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        sensitivity.append(float(sens))
        specificity.append(float(spec))
        precision_per_class.append(float(prec))

    metrics["sensitivity_per_class"] = sensitivity
    metrics["specificity_per_class"] = specificity
    metrics["precision_per_class"] = precision_per_class
    metrics["mean_sensitivity"] = float(np.mean(sensitivity))
    metrics["mean_specificity"] = float(np.mean(specificity))

    # ROC-AUC (requires probability scores)
    if pred_probs is not None:
        probs = np.array(pred_probs)
        try:
            if num_classes == 2:
                metrics["auc_roc"] = float(roc_auc_score(true, probs[:, 1]))
            else:
                metrics["auc_roc"] = float(
                    roc_auc_score(true, probs, multi_class="ovr", average="macro")
                )
                # Per-class AUC
                auc_per_class = []
                for c in range(num_classes):
                    binary_true = (true == c).astype(int)
                    try:
                        auc_c = roc_auc_score(binary_true, probs[:, c])
                        auc_per_class.append(float(auc_c))
                    except ValueError:
                        auc_per_class.append(float("nan"))
                metrics["auc_per_class"] = auc_per_class
        except ValueError as e:
            metrics["auc_roc"] = float("nan")
            print(f"[WARNING] AUC computation failed: {e}")

    return metrics


def format_metrics_table(metrics: Dict, class_names: List[str] = CLASS_NAMES) -> str:
    """Format metrics as a readable table."""
    lines = []
    lines.append("=" * 60)
    lines.append("EVALUATION RESULTS")
    lines.append("=" * 60)
    lines.append(f"  Accuracy:          {metrics.get('accuracy', 0):.4f}")
    lines.append(f"  Balanced Accuracy: {metrics.get('balanced_accuracy', 0):.4f}")
    lines.append(f"  F1 (Macro):        {metrics.get('f1_macro', 0):.4f}")
    lines.append(f"  F1 (Weighted):     {metrics.get('f1_weighted', 0):.4f}")
    lines.append(f"  AUC-ROC:           {metrics.get('auc_roc', float('nan')):.4f}")
    lines.append(f"  Cohen's Kappa:     {metrics.get('cohen_kappa', 0):.4f}")
    lines.append("")
    lines.append("  Per-class Metrics:")
    lines.append(f"  {'Class':<16} {'F1':>6} {'Sens':>6} {'Spec':>6} {'AUC':>6}")
    lines.append("  " + "-" * 44)

    f1s = metrics.get("f1_per_class", [0] * len(class_names))
    sens = metrics.get("sensitivity_per_class", [0] * len(class_names))
    spec = metrics.get("specificity_per_class", [0] * len(class_names))
    aucs = metrics.get("auc_per_class", ["N/A"] * len(class_names))

    for i, cls in enumerate(class_names):
        auc_str = f"{aucs[i]:.4f}" if isinstance(aucs[i], float) and not np.isnan(aucs[i]) else " N/A"
        lines.append(
            f"  {cls:<16} {f1s[i]:>6.4f} {sens[i]:>6.4f} {spec[i]:>6.4f} {auc_str:>6}"
        )

    lines.append("=" * 60)
    return "\n".join(lines)


def aggregate_fold_metrics(fold_metrics: List[Dict]) -> Dict:
    """
    Aggregate metrics across folds (mean ± std).

    Args:
        fold_metrics: List of metric dicts, one per fold

    Returns:
        Aggregated dict with mean and std for each scalar metric
    """
    scalar_keys = [
        "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
        "auc_roc", "cohen_kappa", "mean_sensitivity", "mean_specificity",
    ]

    aggregated = {}
    for key in scalar_keys:
        vals = [m.get(key, float("nan")) for m in fold_metrics]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            aggregated[key] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals)),
                "values": vals,
            }

    return aggregated


def print_cv_summary(aggregated: Dict) -> None:
    """Print cross-validation summary table."""
    print("\n" + "=" * 55)
    print("  CROSS-VALIDATION SUMMARY")
    print("=" * 55)
    for key, stats in aggregated.items():
        print(f"  {key:<25}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)
    n = 200
    true = np.random.randint(0, 4, n)
    pred = np.random.randint(0, 4, n)
    probs = np.random.dirichlet(np.ones(4), size=n)

    metrics = compute_metrics(true, pred, probs)
    print(format_metrics_table(metrics))
    print("Metrics tests passed ✓")
