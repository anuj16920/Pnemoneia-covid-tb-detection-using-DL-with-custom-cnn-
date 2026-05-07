#!/usr/bin/env python3
"""
Plot training history across epochs
Shows accuracy, loss, F1, and other metrics over time
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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


def parse_training_log(log_path):
    """Parse training log to extract epoch-wise metrics"""
    metrics = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'lr': [],
        'time': []
    }
    
    if not Path(log_path).exists():
        print(f"Log file not found: {log_path}")
        return metrics
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Find all epoch summaries
    pattern = r'Epoch \[(\d+)/\d+\] Summary.*?Train: loss=([\d.]+) acc=([\d.]+).*?Val:.*?loss=([\d.]+) acc=([\d.]+) f1_macro=([\d.]+).*?LR: ([\d.e+-]+).*?Time: ([\d.]+)s'
    
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        metrics['epoch'].append(int(match[0]))
        metrics['train_loss'].append(float(match[1]))
        metrics['train_acc'].append(float(match[2]))
        metrics['val_loss'].append(float(match[3]))
        metrics['val_acc'].append(float(match[4]))
        metrics['val_f1'].append(float(match[5]))
        metrics['lr'].append(float(match[6]))
        metrics['time'].append(float(match[7]))
    
    return metrics


def plot_accuracy_over_epochs(metrics, output_path):
    """Plot training and validation accuracy"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    epochs = metrics['epoch']
    
    # Plot lines
    ax.plot(epochs, metrics['train_acc'], linewidth=3, marker='o', markersize=8,
            color='#2E86AB', label='Training Accuracy', 
            markerfacecolor='#2E86AB', markeredgecolor='white', markeredgewidth=2)
    ax.plot(epochs, metrics['val_acc'], linewidth=3, marker='s', markersize=8,
            color='#A23B72', label='Validation Accuracy',
            markerfacecolor='#A23B72', markeredgecolor='white', markeredgewidth=2)
    
    # Highlight best validation accuracy
    best_idx = np.argmax(metrics['val_acc'])
    best_epoch = epochs[best_idx]
    best_acc = metrics['val_acc'][best_idx]
    
    ax.axvline(best_epoch, color='red', linestyle='--', linewidth=2, alpha=0.5,
               label=f'Best Val Acc: {best_acc:.4f} @ Epoch {best_epoch}')
    ax.plot(best_epoch, best_acc, 'r*', markersize=20, markeredgecolor='white', markeredgewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=16, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=16, fontweight='bold')
    ax.set_title('Training & Validation Accuracy Over Epochs', fontsize=18, fontweight='bold', pad=20)
    ax.legend(fontsize=13, loc='lower right')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_ylim([0.4, 1.0])
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_loss_over_epochs(metrics, output_path):
    """Plot training and validation loss"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    epochs = metrics['epoch']
    
    ax.plot(epochs, metrics['train_loss'], linewidth=3, marker='o', markersize=8,
            color='#2E86AB', label='Training Loss',
            markerfacecolor='#2E86AB', markeredgecolor='white', markeredgewidth=2)
    ax.plot(epochs, metrics['val_loss'], linewidth=3, marker='s', markersize=8,
            color='#A23B72', label='Validation Loss',
            markerfacecolor='#A23B72', markeredgecolor='white', markeredgewidth=2)
    
    # Highlight best validation loss
    best_idx = np.argmin(metrics['val_loss'])
    best_epoch = epochs[best_idx]
    best_loss = metrics['val_loss'][best_idx]
    
    ax.axvline(best_epoch, color='red', linestyle='--', linewidth=2, alpha=0.5,
               label=f'Best Val Loss: {best_loss:.4f} @ Epoch {best_epoch}')
    ax.plot(best_epoch, best_loss, 'r*', markersize=20, markeredgecolor='white', markeredgewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=16, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=16, fontweight='bold')
    ax.set_title('Training & Validation Loss Over Epochs', fontsize=18, fontweight='bold', pad=20)
    ax.legend(fontsize=13, loc='upper right')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_f1_over_epochs(metrics, output_path):
    """Plot validation F1 score"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    epochs = metrics['epoch']
    
    ax.plot(epochs, metrics['val_f1'], linewidth=3, marker='D', markersize=10,
            color='#F18F01', label='Validation F1 Score',
            markerfacecolor='#F18F01', markeredgecolor='white', markeredgewidth=2)
    
    # Highlight best F1
    best_idx = np.argmax(metrics['val_f1'])
    best_epoch = epochs[best_idx]
    best_f1 = metrics['val_f1'][best_idx]
    
    ax.axvline(best_epoch, color='red', linestyle='--', linewidth=2, alpha=0.5,
               label=f'Best F1: {best_f1:.4f} @ Epoch {best_epoch}')
    ax.plot(best_epoch, best_f1, 'r*', markersize=20, markeredgecolor='white', markeredgewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=16, fontweight='bold')
    ax.set_ylabel('F1 Score (Macro)', fontsize=16, fontweight='bold')
    ax.set_title('Validation F1 Score Over Epochs', fontsize=18, fontweight='bold', pad=20)
    ax.legend(fontsize=13, loc='lower right')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_ylim([0.75, 1.0])
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_learning_rate(metrics, output_path):
    """Plot learning rate schedule"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    epochs = metrics['epoch']
    
    ax.plot(epochs, metrics['lr'], linewidth=3, marker='o', markersize=8,
            color='#C73E1D', label='Learning Rate',
            markerfacecolor='#C73E1D', markeredgecolor='white', markeredgewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=16, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=16, fontweight='bold')
    ax.set_title('Learning Rate Schedule', fontsize=18, fontweight='bold', pad=20)
    ax.set_yscale('log')
    ax.legend(fontsize=13, loc='upper right')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_combined_metrics(metrics, output_path):
    """Plot all key metrics in one figure"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    epochs = metrics['epoch']
    
    # Accuracy
    ax = axes[0, 0]
    ax.plot(epochs, metrics['train_acc'], linewidth=2.5, marker='o', markersize=6,
            color='#2E86AB', label='Train Acc')
    ax.plot(epochs, metrics['val_acc'], linewidth=2.5, marker='s', markersize=6,
            color='#A23B72', label='Val Acc')
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Accuracy', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Loss
    ax = axes[0, 1]
    ax.plot(epochs, metrics['train_loss'], linewidth=2.5, marker='o', markersize=6,
            color='#2E86AB', label='Train Loss')
    ax.plot(epochs, metrics['val_loss'], linewidth=2.5, marker='s', markersize=6,
            color='#A23B72', label='Val Loss')
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax.set_title('Loss', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # F1 Score
    ax = axes[1, 0]
    ax.plot(epochs, metrics['val_f1'], linewidth=2.5, marker='D', markersize=6,
            color='#F18F01', label='Val F1')
    best_idx = np.argmax(metrics['val_f1'])
    ax.axvline(epochs[best_idx], color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=14, fontweight='bold')
    ax.set_title('F1 Score (Macro)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Learning Rate
    ax = axes[1, 1]
    ax.plot(epochs, metrics['lr'], linewidth=2.5, marker='o', markersize=6,
            color='#C73E1D', label='Learning Rate')
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=14, fontweight='bold')
    ax.set_title('Learning Rate Schedule', fontsize=16, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default='results/logs/fold0_train.log')
    parser.add_argument('--output_dir', default='results/publication_plots')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("  GENERATING TRAINING HISTORY PLOTS")
    print("="*80)
    print(f"\nLog file: {args.log}")
    print(f"Output: {output_dir}\n")
    
    # Parse log
    print("Parsing training log...")
    metrics = parse_training_log(args.log)
    
    if len(metrics['epoch']) == 0:
        print("❌ No training data found in log file!")
        return
    
    print(f"✓ Found {len(metrics['epoch'])} epochs\n")
    
    # Generate plots
    print("Generating plots...\n")
    
    plot_accuracy_over_epochs(metrics, output_dir / 'training_accuracy_over_epochs.pdf')
    plot_loss_over_epochs(metrics, output_dir / 'training_loss_over_epochs.pdf')
    plot_f1_over_epochs(metrics, output_dir / 'training_f1_over_epochs.pdf')
    plot_learning_rate(metrics, output_dir / 'learning_rate_schedule.pdf')
    plot_combined_metrics(metrics, output_dir / 'training_metrics_combined.pdf')
    
    print("\n" + "="*80)
    print(f"✅ All training history plots saved to: {output_dir}")
    print("="*80)
    
    # Print summary
    print(f"\n📊 Training Summary:")
    print(f"  Total Epochs: {len(metrics['epoch'])}")
    print(f"  Best Val Accuracy: {max(metrics['val_acc']):.4f} @ Epoch {metrics['epoch'][np.argmax(metrics['val_acc'])]}")
    print(f"  Best Val F1: {max(metrics['val_f1']):.4f} @ Epoch {metrics['epoch'][np.argmax(metrics['val_f1'])]}")
    print(f"  Best Val Loss: {min(metrics['val_loss']):.4f} @ Epoch {metrics['epoch'][np.argmin(metrics['val_loss'])]}")


if __name__ == "__main__":
    main()
