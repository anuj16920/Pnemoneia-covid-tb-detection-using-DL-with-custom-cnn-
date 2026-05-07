#!/usr/bin/env python3
"""
Generate publication-quality separate plots for each metric
High-resolution PDFs suitable for papers/presentations
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
    precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import yaml

from src.models.full_model import PulmonaryDxModel
from src.data.data_splits import load_splits, get_fold_dataloaders
from src.utils.checkpoint import load_checkpoint

# High-quality plot settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
sns.set_style("whitegrid")


def load_model_and_evaluate(config_path, checkpoint_path, fold_idx=0):
    """Load model and get predictions"""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    splits_path = Path(cfg['paths']['logs']) / 'splits.json'
    splits = load_splits(splits_path)
    
    _, _, test_loader = get_fold_dataloaders(
        splits=splits, fold_idx=fold_idx, cfg=cfg,
        batch_size=32, num_workers=4,
    )
    
    model = PulmonaryDxModel(
        num_classes=cfg['data']['num_classes'],
        pretrained=False,
        use_dyda=cfg['model']['dyda']['enabled'],
        use_swin=cfg['model']['swin']['enabled'],
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    load_checkpoint(model, checkpoint_path, device=device)
    model = model.to(device)
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
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs), cfg['data']['class_names']


def plot_confusion_matrix_single(y_true, y_pred, class_names, output_dir):
    """Generate separate confusion matrix plots"""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot 1: Raw counts
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Count'}, linewidths=0.5,
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title('Confusion Matrix (Counts)', fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=16, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix_counts.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: confusion_matrix_counts.pdf")
    
    # Plot 2: Normalized percentages
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Percentage'}, linewidths=0.5,
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title('Confusion Matrix (Normalized)', fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=16, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix_normalized.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: confusion_matrix_normalized.pdf")


def plot_roc_curves_separate(y_true, y_probs, class_names, output_dir):
    """Generate separate ROC curve for each class"""
    for i, class_name in enumerate(class_names):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_true_binary = (y_true == i).astype(int)
        y_score = y_probs[:, i]
        
        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, linewidth=3, color='#2E86AB',
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        ax.fill_between(fpr, tpr, alpha=0.2, color='#2E86AB')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=16, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=16, fontweight='bold')
        ax.set_title(f'ROC Curve - {class_name}', fontsize=18, fontweight='bold', pad=20)
        ax.legend(loc="lower right", fontsize=14)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/roc_curve_{class_name.lower().replace(" ", "_")}.pdf', 
                   format='pdf', bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: roc_curve_{class_name.lower().replace(' ', '_')}.pdf")
    
    # Combined ROC curves
    fig, ax = plt.subplots(figsize=(12, 9))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        y_true_binary = (y_true == i).astype(int)
        y_score = y_probs[:, i]
        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=3, color=color,
                label=f'{class_name} (AUC = {roc_auc:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=16, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=16, fontweight='bold')
    ax.set_title('ROC Curves - All Classes', fontsize=18, fontweight='bold', pad=20)
    ax.legend(loc="lower right", fontsize=13)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curves_combined.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: roc_curves_combined.pdf")


def plot_metrics_comparison(y_true, y_pred, class_names, output_dir):
    """Generate metrics comparison plots as line charts"""
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    
    # Individual metric plots as line charts
    metrics = {
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    
    colors_map = {'Precision': '#2E86AB', 'Recall': '#A23B72', 'F1-Score': '#F18F01'}
    markers_map = {'Precision': 'o', 'Recall': 's', 'F1-Score': 'D'}
    
    for metric_name, values in metrics.items():
        fig, ax = plt.subplots(figsize=(12, 7))
        x = np.arange(len(class_names))
        
        # Line plot with markers
        ax.plot(x, values, color=colors_map[metric_name], linewidth=3, 
                marker=markers_map[metric_name], markersize=12, 
                markerfacecolor=colors_map[metric_name], 
                markeredgecolor='white', markeredgewidth=2,
                label=metric_name)
        
        # Add value labels
        for i, (xi, yi) in enumerate(zip(x, values)):
            ax.text(xi, yi + 0.03, f'{yi:.3f}', ha='center', va='bottom', 
                   fontsize=13, fontweight='bold')
        
        ax.set_xlabel('Class', fontsize=16, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=16, fontweight='bold')
        ax.set_title(f'{metric_name} by Class', fontsize=18, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, fontsize=14)
        ax.set_ylim([0.85, 1.05])
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.legend(fontsize=14, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{metric_name.lower().replace("-", "_")}_by_class.pdf', 
                   format='pdf', bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {metric_name.lower().replace('-', '_')}_by_class.pdf")
    
    # Combined metrics plot as line chart
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(class_names))
    
    # Plot lines for each metric
    ax.plot(x, precision, linewidth=3, marker='o', markersize=12, 
            color='#2E86AB', label='Precision', 
            markerfacecolor='#2E86AB', markeredgecolor='white', markeredgewidth=2)
    ax.plot(x, recall, linewidth=3, marker='s', markersize=12, 
            color='#A23B72', label='Recall',
            markerfacecolor='#A23B72', markeredgecolor='white', markeredgewidth=2)
    ax.plot(x, f1, linewidth=3, marker='D', markersize=12, 
            color='#F18F01', label='F1-Score',
            markerfacecolor='#F18F01', markeredgecolor='white', markeredgewidth=2)
    
    # Add value labels for F1 (to avoid clutter)
    for i, (xi, yi) in enumerate(zip(x, f1)):
        ax.text(xi, yi + 0.01, f'{yi:.3f}', ha='center', va='bottom', 
               fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Class', fontsize=16, fontweight='bold')
    ax.set_ylabel('Score', fontsize=16, fontweight='bold')
    ax.set_title('Performance Metrics Comparison', fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=14)
    ax.legend(fontsize=14, loc='lower right')
    ax.set_ylim([0.85, 1.05])
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metrics_comparison_combined.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: metrics_comparison_combined.pdf")


def plot_overall_accuracy(y_true, y_pred, output_dir):
    """Generate overall accuracy visualization"""
    accuracy = accuracy_score(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Create a gauge-like visualization
    categories = ['Accuracy']
    values = [accuracy * 100]
    
    bars = ax.barh(categories, values, color='#2E86AB', alpha=0.8, 
                   edgecolor='black', linewidth=2, height=0.5)
    
    ax.set_xlim([0, 100])
    ax.set_xlabel('Percentage (%)', fontsize=16, fontweight='bold')
    ax.set_title(f'Overall Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)', 
                fontsize=18, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x', linewidth=0.5)
    
    # Add value label
    ax.text(values[0] + 1, 0, f'{values[0]:.2f}%', 
           va='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/overall_accuracy.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: overall_accuracy.pdf")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--checkpoint', default='results/checkpoints/best_fold1.pth')
    parser.add_argument('--output_dir', default='results/publication_plots')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("  GENERATING PUBLICATION-QUALITY PLOTS")
    print("="*80)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Output: {output_dir}\n")
    
    # Load and evaluate
    print("Loading model and evaluating...")
    y_true, y_pred, y_probs, class_names = load_model_and_evaluate(
        args.config, args.checkpoint
    )
    print(f"✓ Evaluated {len(y_true)} samples\n")
    
    # Generate all plots
    print("Generating plots...\n")
    
    plot_confusion_matrix_single(y_true, y_pred, class_names, output_dir)
    print()
    
    plot_roc_curves_separate(y_true, y_probs, class_names, output_dir)
    print()
    
    plot_metrics_comparison(y_true, y_pred, class_names, output_dir)
    print()
    
    plot_overall_accuracy(y_true, y_pred, output_dir)
    
    print("\n" + "="*80)
    print(f"✅ All publication-quality plots saved to: {output_dir}")
    print("="*80)
    print(f"\nGenerated {len(list(output_dir.glob('*.pdf')))} PDF files")


if __name__ == "__main__":
    main()
