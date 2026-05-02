# Hybrid EfficientNet-DyDA-Swin Transformer for Pulmonary Disease Classification

> **Research-Grade Implementation**  
> Multi-class chest X-ray classification: COVID-19 | Pneumonia | Tuberculosis | Normal

---

## 📌 Abstract

This project implements a novel hybrid deep learning architecture for pulmonary disease classification from chest radiographs. The model integrates:

1. **EfficientNet-B3** — pretrained CNN backbone for hierarchical local feature extraction  
2. **Dynamic Dual Attention (DyDA)** — parallel channel + spatial attention with input-dependent learnable gates (α, β) constrained by softmax (α + β = 1)  
3. **Swin Transformer** — shifted-window global context modeling  
4. **Feature Fusion** — concatenation of CNN and Transformer embeddings → FC classification head

---

## 📂 Project Structure

```
pulmonary_dx/
├── configs/
│   └── config.yaml              # All hyperparameters & paths
├── src/
│   ├── models/
│   │   ├── efficientnet_backbone.py
│   │   ├── dyda_module.py        # Dynamic Dual Attention
│   │   ├── swin_transformer.py   # Swin Transformer branch
│   │   ├── fusion_head.py        # Concatenation + classifier
│   │   └── full_model.py         # Assembled hybrid model
│   ├── data/
│   │   ├── dataset.py            # Multi-source dataset loader
│   │   ├── preprocessing.py      # CLAHE + augmentation
│   │   └── data_splits.py        # Stratified K-Fold
│   ├── training/
│   │   ├── trainer.py            # Training loop
│   │   ├── losses.py             # Label smoothing cross-entropy
│   │   └── schedulers.py         # Cosine annealing + warmup
│   ├── evaluation/
│   │   ├── metrics.py            # Acc, F1, AUC, confusion matrix
│   │   ├── gradcam.py            # Grad-CAM saliency maps
│   │   └── ablation.py           # Ablation study runner
│   └── utils/
│       ├── logger.py
│       ├── checkpoint.py
│       └── seed.py
├── scripts/
│   ├── train.py                  # Main training entry point
│   ├── evaluate.py               # Evaluation on test set
│   ├── run_ablation.py           # Full ablation study
│   └── visualize_gradcam.py      # Generate Grad-CAM maps
├── notebooks/
│   └── EDA_and_Results.ipynb     # Exploratory analysis notebook
├── tests/
│   ├── test_dyda.py
│   ├── test_model_forward.py
│   └── test_dataset.py
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🗄️ Datasets

| Dataset | Classes | Source |
|---------|---------|--------|
| COVID-19 Radiography Database | COVID-19, Normal | Kaggle |
| Chest X-Ray Pneumonia | Pneumonia, Normal | Kaggle |
| Montgomery County TB | TB, Normal | NIH/NLM |
| Shenzhen TB Collection | TB, Normal | NIH/NLM |

**Merged Label Space:** `{Normal, COVID-19, Pneumonia, Tuberculosis}`

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare data
```bash
# Edit configs/config.yaml → set data.root_dir to your dataset path
# Expected layout:
# data/
#   Normal/       *.png / *.jpg
#   COVID-19/
#   Pneumonia/
#   Tuberculosis/
```

### 3. Train (5-fold cross-validation)
```bash
python scripts/train.py --config configs/config.yaml
```

### 4. Evaluate
```bash
python scripts/evaluate.py --config configs/config.yaml --checkpoint results/checkpoints/best_fold1.pth
```

### 5. Ablation study
```bash
python scripts/run_ablation.py --config configs/config.yaml
```

### 6. Grad-CAM visualization
```bash
python scripts/visualize_gradcam.py --config configs/config.yaml --checkpoint results/checkpoints/best_fold1.pth --image path/to/xray.jpg
```

---

## 🏗️ Architecture

```
Input (224×224×3)
       │
  EfficientNet-B3 Backbone
  (pretrained ImageNet, frozen stages 1-3)
       │
  Feature Maps [B, C, H, W]
       │
  ┌────┴─────────────────────────┐
  │   Dynamic Dual Attention     │
  │   (DyDA Module)              │
  │                              │
  │  Channel Path → fC           │
  │  Spatial Path → fS           │
  │                              │
  │  [α, β] = softmax(MLP(fC⊕fS))│
  │  out = α·fC + β·fS           │
  └────────────┬─────────────────┘
               │ attended features
       ┌───────┴───────┐
       │               │
  CNN GAP           Swin Transformer
  vector            (window=7, 4 stages)
       │               │
       └───────┬───────┘
        Concat [CNN ‖ Swin]
               │
          Linear → BN → ReLU → Dropout
               │
          Linear → 4 classes
               │
           Softmax
```

---

## 📊 Ablation Configurations

| Config | Backbone | DyDA | Swin | Expected |
|--------|----------|------|------|----------|
| Backbone-Only | ✓ | ✗ | ✗ | ~88% |
| +CBAM | ✓ | CBAM | ✗ | ~90% |
| +DyDA (no Swin) | ✓ | ✓ | ✗ | ~92% |
| Full Model | ✓ | ✓ | ✓ | ~95%+ |

---

## 🔬 Key Design Choices

### DyDA vs CBAM vs DANet

| Property | CBAM | DANet | **DyDA (Ours)** |
|----------|------|-------|-----------------|
| Channel attn | Sequential | ✓ | Parallel |
| Spatial attn | Sequential | ✓ | Parallel |
| Gating | Fixed | Global scalar | Input-dependent softmax |
| α+β constraint | N/A | N/A | ✓ (normalized) |

### Training Protocol
- **Optimizer:** AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler:** Cosine Annealing with Linear Warmup (10 epochs)
- **Loss:** Label Smoothing Cross-Entropy (ε=0.1)
- **Augmentation:** RandomHorizontalFlip, RandomRotation(±10°), ColorJitter, CLAHE
- **Regularization:** Dropout(0.4) in classifier head, MixUp(α=0.2)
- **Epochs:** 50 | **Batch size:** 32 | **Early stopping:** patience=10

---

## 📋 Citation

If you use this work, please cite:
```bibtex
@article{pulmonary_hybrid_2025,
  title={Hybrid EfficientNet-DyDA-Swin Architecture for Pulmonary Disease Classification from Chest Radiographs},
  author={MLRIT Research},
  year={2025}
}
```
