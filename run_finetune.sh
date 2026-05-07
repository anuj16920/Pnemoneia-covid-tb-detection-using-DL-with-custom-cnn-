#!/bin/bash
# ============================================================
# Fine-tune Model for 15 More Epochs with Stable Growth
# ============================================================

set -e  # Exit on error

echo "=========================================="
echo "Fine-tuning for Stable Growth"
echo "=========================================="
echo ""

# Configuration
CHECKPOINT="${1:-results/checkpoints/best_fold1.pth}"
CONFIG="${2:-configs/finetune_config.yaml}"
EPOCHS="${3:-15}"
FOLD="${4:-0}"
LR="${5:-1e-5}"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint not found: $CHECKPOINT"
    echo ""
    echo "Available checkpoints:"
    ls -lh results/checkpoints/*.pth 2>/dev/null || echo "  No checkpoints found"
    exit 1
fi

echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT"
echo "  Config: $CONFIG"
echo "  Epochs: $EPOCHS"
echo "  Fold: $((FOLD + 1))"
echo "  Learning Rate: $LR"
echo ""

# Create output directory
mkdir -p results/finetuned/{checkpoints,logs,plots}

echo "Starting fine-tuning..."
echo ""

# Run fine-tuning
python scripts/finetune.py \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --epochs "$EPOCHS" \
    --fold "$FOLD" \
    --lr "$LR" \
    --splits_file results/logs/splits.json \
    --output_dir results/finetuned

echo ""
echo "=========================================="
echo "✅ Fine-tuning Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  Checkpoints: results/finetuned/checkpoints/"
echo "  Logs: results/finetuned/logs/"
echo "  Plots: results/finetuned/plots/"
echo ""
echo "Best models:"
ls -lh results/finetuned/checkpoints/best_*.pth 2>/dev/null || echo "  No best checkpoint yet"
echo ""
echo "Averaged checkpoint:"
ls -lh results/finetuned/checkpoints/averaged_*.pth 2>/dev/null || echo "  No averaged checkpoint yet"
echo ""
