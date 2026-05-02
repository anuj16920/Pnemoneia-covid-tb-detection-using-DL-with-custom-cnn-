#!/usr/bin/env python3
"""
Comprehensive Model Evaluation with Plots
Generates confusion matrix, ROC curves, precision-recall, and all metrics
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import yaml

from src.models.full_model import PulmonaryDxModel
from src.data.data_splits import load_splits, get_fold_dataloaders
from src.utils.checkpoint import load_checkpoint

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)


def load_model_and_data(config_path, checkpoint_path, fold_idx=0):
    """Load trained model and test data"""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Load splits
    splits_path = Path(cfg['paths']['logs']) / 'splits.json'
    splits = load_splits(splits_path)
    
    # Get dataloaders
    _, _, test_loader = get_fold_dataloaders(
        splits=splits,
        fold_idx=fold_idx,
        cfg=cfg,
        batch_size=32,
        num_workers=4,
    )
    
    # Build model
    model = PulmonaryDxModel(
        num_classes=cfg['data']['num_classes'],
        pretrained=False,
        use_dyda=cfg['model']['dyda']['enabled'],
        use_swin=cfg['model']['swin']['enabled'],
    )
    
    # Load checkpoint
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    load_checkpoint(model, checkpoint_path, device=device)
    model = model.to(device)
    model.eval()
    
    return model, test_loader, cfg


@torch.no_grad()
def evaluate_model(model, test_loader, device='cuda'):
    """Run inference and collect predictions"""
    all_preds = []
    all_labels = []
    all_probs = []
    
    for images, labels in test_loader:
        images = images.to(device)
        logits, _ = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    
    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax2, cbar_kws={'label': 'Percentage'})
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    
    # Save as PNG
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Save as PDF
    pdf_path = str(save_path).replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    plt.close()
    print(f"✓ Confusion matrix saved: {save_path}")
    print(f"✓ Confusion matrix PDF saved: {pdf_path}")


def plot_roc_curves(y_true, y_probs, class_names, save_path):
    """Plot ROC curves for each class"""
    n_classes = len(class_names)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    # One-vs-rest ROC for each class
    for i, (class_name, ax) in enumerate(zip(class_names, axes)):
        y_true_binary = (y_true == i).astype(int)
        y_score = y_probs[:, i]
        
        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, linewidth=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(f'ROC Curve - {class_name}', fontsize=12, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save as PNG
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Save as PDF
    pdf_path = str(save_path).replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    plt.close()
    print(f"✓ ROC curves saved: {save_path}")
    print(f"✓ ROC curves PDF saved: {pdf_path}")


def plot_precision_recall_curves(y_true, y_probs, class_names, save_path):
    """Plot Precision-Recall curves"""
    n_classes = len(class_names)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    for i, (class_name, ax) in enumerate(zip(class_names, axes)):
        y_true_binary = (y_true == i).astype(int)
        y_score = y_probs[:, i]
        
        precision, recall, _ = precision_recall_curve(y_true_binary, y_score)
        ap = average_precision_score(y_true_binary, y_score)
        
        ax.plot(recall, precision, linewidth=2,
                label=f'PR curve (AP = {ap:.4f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=11)
        ax.set_ylabel('Precision', fontsize=11)
        ax.set_title(f'Precision-Recall - {class_name}', fontsize=12, fontweight='bold')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Precision-Recall curves saved: {save_path}")


def plot_per_class_metrics(y_true, y_pred, class_names, save_path):
    """Plot per-class metrics bar chart"""
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='skyblue')
    bars2 = ax.bar(x, recall, width, label='Recall', color='lightgreen')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='salmon')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save as PNG
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Save as PDF
    pdf_path = str(save_path).replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    plt.close()
    print(f"✓ Per-class metrics saved: {save_path}")
    print(f"✓ Per-class metrics PDF saved: {pdf_path}")


def create_summary_report(y_true, y_pred, y_probs, class_names, save_path):
    """Create comprehensive text report"""
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate per-class AUC
    auc_scores = []
    for i in range(len(class_names)):
        y_true_binary = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_probs[:, i])
        auc_scores.append(auc(fpr, tpr))
    
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PULMONARY DISEASE CLASSIFICATION - EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        
        f.write("Classification Report:\n")
        f.write("-"*80 + "\n")
        f.write(report)
        f.write("\n")
        
        f.write("Per-Class AUC Scores:\n")
        f.write("-"*80 + "\n")
        for name, score in zip(class_names, auc_scores):
            f.write(f"  {name:<20}: {score:.4f}\n")
        f.write(f"  {'Macro Average':<20}: {np.mean(auc_scores):.4f}\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
    
    print(f"✓ Summary report saved: {save_path}")
    print(f"\n📊 Overall Accuracy: {accuracy:.4f}")
    print(f"📊 Macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--checkpoint', default='results/checkpoints/best_fold1.pth')
    parser.add_argument('--output_dir', default='results/evaluation')
    parser.add_argument('--fold', type=int, default=0)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("  MODEL EVALUATION & VISUALIZATION")
    print("="*80)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Output dir: {output_dir}\n")
    
    # Load model and data
    print("Loading model and test data...")
    model, test_loader, cfg = load_model_and_data(
        args.config, args.checkpoint, args.fold
    )
    class_names = cfg['data']['class_names']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Evaluate
    print("Running inference on test set...")
    y_true, y_pred, y_probs = evaluate_model(model, test_loader, device)
    print(f"✓ Evaluated {len(y_true)} samples\n")
    
    # Generate plots
    print("Generating visualizations...")
    plot_confusion_matrix(y_true, y_pred, class_names, 
                         output_dir / 'confusion_matrix.png')
    
    plot_roc_curves(y_true, y_probs, class_names,
                   output_dir / 'roc_curves.png')
    
    plot_precision_recall_curves(y_true, y_probs, class_names,
                                output_dir / 'precision_recall_curves.png')
    
    plot_per_class_metrics(y_true, y_pred, class_names,
                          output_dir / 'per_class_metrics.png')
    
    create_summary_report(y_true, y_pred, y_probs, class_names,
                         output_dir / 'evaluation_report.txt')
    
    print("\n" + "="*80)
    print("✅ Evaluation complete! All plots saved to:", output_dir)
    print("="*80)


if __name__ == "__main__":
    main()
