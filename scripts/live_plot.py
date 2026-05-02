#!/usr/bin/env python3
"""
Live training visualization - updates plots in real-time
Monitors training logs and creates live graphs
"""

import os
import sys
import time
import json
import re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

sns.set_style("whitegrid")

def parse_log_file(log_path):
    """Parse training log to extract metrics"""
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
    
    if not os.path.exists(log_path):
        return metrics
    
    with open(log_path, 'r') as f:
        for line in f:
            # Match epoch summary lines
            match = re.search(
                r'Epoch \[(\d+)/\d+\] Summary.*?'
                r'Train: loss=([\d.]+) acc=([\d.]+).*?'
                r'Val:.*?loss=([\d.]+) acc=([\d.]+) f1_macro=([\d.]+).*?'
                r'LR: ([\d.e+-]+).*?Time: ([\d.]+)s',
                line, re.DOTALL
            )
            
            if match:
                metrics['epoch'].append(int(match.group(1)))
                metrics['train_loss'].append(float(match.group(2)))
                metrics['train_acc'].append(float(match.group(3)))
                metrics['val_loss'].append(float(match.group(4)))
                metrics['val_acc'].append(float(match.group(5)))
                metrics['val_f1'].append(float(match.group(6)))
                metrics['lr'].append(float(match.group(7)))
                metrics['time'].append(float(match.group(8)))
    
    return metrics

def create_live_plots(metrics, output_dir='results/plots'):
    """Create comprehensive training plots"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if len(metrics['epoch']) == 0:
        print("No metrics to plot yet...")
        return
    
    epochs = metrics['epoch']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Loss curves
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(epochs, metrics['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, metrics['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy curves
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(epochs, metrics['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, metrics['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. F1 Score
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(epochs, metrics['val_f1'], 'g-', linewidth=2, marker='o')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1 Score (Macro)')
    ax3.set_title('Validation F1 Score', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    best_f1_idx = metrics['val_f1'].index(max(metrics['val_f1']))
    ax3.axvline(epochs[best_f1_idx], color='r', linestyle='--', alpha=0.5, label=f'Best: {max(metrics["val_f1"]):.4f}')
    ax3.legend()
    
    # 4. Learning Rate
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(epochs, metrics['lr'], 'purple', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    # 5. Training Time per Epoch
    ax5 = plt.subplot(3, 3, 5)
    ax5.bar(epochs, metrics['time'], color='orange', alpha=0.7)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Time (seconds)')
    ax5.set_title('Training Time per Epoch', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Loss Improvement
    ax6 = plt.subplot(3, 3, 6)
    if len(metrics['val_loss']) > 1:
        loss_improvement = [0] + [metrics['val_loss'][i-1] - metrics['val_loss'][i] 
                                   for i in range(1, len(metrics['val_loss']))]
        colors = ['g' if x > 0 else 'r' for x in loss_improvement]
        ax6.bar(epochs, loss_improvement, color=colors, alpha=0.7)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Loss Improvement')
    ax6.set_title('Validation Loss Improvement', fontsize=12, fontweight='bold')
    ax6.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Overfitting Monitor (Train vs Val Accuracy Gap)
    ax7 = plt.subplot(3, 3, 7)
    gap = [t - v for t, v in zip(metrics['train_acc'], metrics['val_acc'])]
    ax7.plot(epochs, gap, 'r-', linewidth=2, marker='o')
    ax7.set_xlabel('Epoch')
    ax7.set_ylabel('Train - Val Accuracy')
    ax7.set_title('Overfitting Monitor', fontsize=12, fontweight='bold')
    ax7.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax7.grid(True, alpha=0.3)
    ax7.fill_between(epochs, 0, gap, where=[g > 0 for g in gap], alpha=0.3, color='red', label='Overfitting')
    ax7.legend()
    
    # 8. Cumulative Training Time
    ax8 = plt.subplot(3, 3, 8)
    cumulative_time = [sum(metrics['time'][:i+1])/60 for i in range(len(metrics['time']))]
    ax8.plot(epochs, cumulative_time, 'b-', linewidth=2)
    ax8.set_xlabel('Epoch')
    ax8.set_ylabel('Cumulative Time (minutes)')
    ax8.set_title('Total Training Time', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    
    # 9. Summary Stats
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    best_epoch = epochs[best_f1_idx]
    best_f1 = max(metrics['val_f1'])
    best_val_acc = metrics['val_acc'][best_f1_idx]
    current_epoch = epochs[-1]
    total_time = sum(metrics['time']) / 60
    
    summary_text = f"""
    📊 TRAINING SUMMARY
    {'='*40}
    
    Current Epoch:     {current_epoch}
    Best Epoch:        {best_epoch}
    
    Best Val F1:       {best_f1:.4f}
    Best Val Acc:      {best_val_acc:.4f}
    
    Current Train Acc: {metrics['train_acc'][-1]:.4f}
    Current Val Acc:   {metrics['val_acc'][-1]:.4f}
    Current Val F1:    {metrics['val_f1'][-1]:.4f}
    
    Total Time:        {total_time:.1f} min
    Avg Time/Epoch:    {total_time/len(epochs):.1f} min
    
    Current LR:        {metrics['lr'][-1]:.2e}
    """
    
    ax9.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/live_training_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Updated plots at epoch {current_epoch} | Best F1: {best_f1:.4f} @ epoch {best_epoch}")

def monitor_training(log_path='results/logs/fold0_train.log', interval=30):
    """Monitor training and update plots periodically"""
    print(f"🔍 Monitoring training log: {log_path}")
    print(f"📊 Updating plots every {interval} seconds...")
    print(f"📁 Plots saved to: results/plots/live_training_dashboard.png")
    print("\nPress Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            metrics = parse_log_file(log_path)
            if len(metrics['epoch']) > 0:
                create_live_plots(metrics)
            else:
                print("Waiting for training to start...")
            
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n✓ Monitoring stopped")
        print("Final plots saved!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default='results/logs/fold0_train.log', help='Training log file')
    parser.add_argument('--interval', type=int, default=30, help='Update interval in seconds')
    parser.add_argument('--once', action='store_true', help='Generate plots once and exit')
    args = parser.parse_args()
    
    if args.once:
        metrics = parse_log_file(args.log)
        create_live_plots(metrics)
        print("✓ Plots generated!")
    else:
        monitor_training(args.log, args.interval)
