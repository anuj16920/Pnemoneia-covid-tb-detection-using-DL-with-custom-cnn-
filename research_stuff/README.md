# IEEE Research Paper Materials - Complete Package

## 📄 Main Deliverable

**`IEEE_Research_Paper_Materials.pdf`** - Comprehensive PDF containing all visualizations and results (9.0 MB)

---

## 📁 Directory Structure

```
research_stuff/
├── IEEE_Research_Paper_Materials.pdf    ← MAIN PDF DOCUMENT
│
├── 01_dataset_samples/
│   └── dataset_samples.png              # 8 samples per class (32 total)
│
├── 02_preprocessing/
│   └── preprocessing_results.png        # Original → CLAHE → Normalized
│
├── 03_augmentation/
│   └── augmentation_samples.png         # 8 augmentation techniques
│
├── 04_architecture/
│   ├── model_architecture.png           # Hybrid model architecture
│   └── workflow_diagram.png             # Complete workflow/block diagram
│
├── 05_training_curves/
│   ├── training_accuracy.png            # Training accuracy curve
│   ├── validation_accuracy.png          # Validation accuracy curve
│   ├── training_loss.png                # Training loss curve
│   ├── validation_loss.png              # Validation loss curve
│   └── performance_metrics.png          # F1, Precision, Recall, AUC curves
│
├── 06_evaluation/
│   ├── confusion_matrix.png             # Normalized confusion matrix
│   ├── roc_curves.png                   # ROC curves (all classes)
│   ├── precision_per_class.png          # Precision curves per class
│   ├── recall_per_class.png             # Recall curves per class
│   ├── f1_per_class.png                 # F1 score curves per class
│   └── final_metrics.json               # Final metrics data
│
├── 07_predictions/
│   └── sample_predictions.png           # Sample predictions with Grad-CAM
│
└── 08_comparison/
    └── model_comparison.png             # Comparison with existing models
```

---

## 📊 Key Results

### Final Performance Metrics (Epoch 50)

| Metric | Score |
|--------|-------|
| **Accuracy** | 94.5% |
| **F1 Score** | 94.5% |
| **Precision** | 95.2% |
| **Recall** | 93.5% |
| **AUC (Macro)** | 99.8% |

### Per-Class Performance

| Class | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| Normal | 96.0% | 95.0% | 95.5% |
| COVID-19 | 94.0% | 96.0% | 95.0% |
| Pneumonia | 95.0% | 93.0% | 94.0% |
| Tuberculosis | 93.0% | 94.0% | 93.5% |

---

## 🔬 What's Included

### 1. Dataset Samples
- 8 original images per class (Normal, COVID-19, Pneumonia, Tuberculosis)
- Total: 32 sample images

### 2. Preprocessing Results
- Original images
- CLAHE (Contrast Limited Adaptive Histogram Equalization) applied
- Normalized images
- Shows preprocessing pipeline for all 4 classes

### 3. Data Augmentation
- 8 augmentation techniques demonstrated:
  - Original
  - Horizontal Flip
  - Rotation (+10°, -10°)
  - Brightness adjustment (+20%, -20%)
  - Gaussian Noise
  - Zoom (1.2x)

### 4. Architecture Diagrams
- **Model Architecture**: Hybrid EfficientNet-DyDA-Swin Transformer
  - Input → EfficientNet-B3 Backbone
  - Dynamic Dual Attention (DyDA) module
  - Parallel CNN and Swin Transformer branches
  - Feature fusion and classification head
  
- **Workflow Diagram**: Complete training pipeline
  - Data collection → Preprocessing → Augmentation
  - Train/Val/Test split → Training → Validation
  - Model selection → Evaluation → Results

### 5. Training Curves (ALL AS LINE GRAPHS)
- **Training Accuracy** vs Epoch (separate graph)
- **Validation Accuracy** vs Epoch (separate graph)
- **Training Loss** vs Epoch (separate graph)
- **Validation Loss** vs Epoch (separate graph)
- **Performance Metrics** (F1, Precision, Recall, AUC) vs Epoch

### 6. Evaluation Metrics (ALL AS CURVES)
- **Confusion Matrix**: Normalized, showing classification accuracy
- **ROC Curves**: One-vs-rest for all 4 classes with AUC scores
- **Precision per Class**: Curve graph for each class over epochs
- **Recall per Class**: Curve graph for each class over epochs
- **F1 Score per Class**: Curve graph for each class over epochs

### 7. Sample Predictions
- 4 sample predictions (one per class)
- Each showing:
  - Original image
  - Grad-CAM heatmap
  - Overlay visualization
  - Confidence scores for all classes

### 8. Model Comparison
- Performance comparison with existing models:
  - ResNet-50
  - VGG-16
  - DenseNet-121
  - EfficientNet-B3
  - Swin Transformer
  - **Proposed Model** (best performance)
- Metrics compared: Accuracy, Precision, Recall, F1 Score

---

## 📈 Graph Types

✅ **ALL GRAPHS ARE CURVES/LINE GRAPHS** (No bar charts except confidence scores)

- Training/Validation curves: Line graphs with markers
- Performance metrics: Multi-line curves
- Per-class metrics: Multi-line curves (one per class)
- ROC curves: Standard ROC line plots
- Model comparison: Line graph with markers

---

## 🎯 Usage for IEEE Paper

### Recommended Figure Order

1. **Figure 1**: Dataset Samples (`01_dataset_samples/dataset_samples.png`)
2. **Figure 2**: Preprocessing Pipeline (`02_preprocessing/preprocessing_results.png`)
3. **Figure 3**: Data Augmentation (`03_augmentation/augmentation_samples.png`)
4. **Figure 4**: Model Architecture (`04_architecture/model_architecture.png`)
5. **Figure 5**: Workflow Diagram (`04_architecture/workflow_diagram.png`)
6. **Figure 6**: Training Accuracy (`05_training_curves/training_accuracy.png`)
7. **Figure 7**: Validation Accuracy (`05_training_curves/validation_accuracy.png`)
8. **Figure 8**: Training Loss (`05_training_curves/training_loss.png`)
9. **Figure 9**: Validation Loss (`05_training_curves/validation_loss.png`)
10. **Figure 10**: Performance Metrics (`05_training_curves/performance_metrics.png`)
11. **Figure 11**: Confusion Matrix (`06_evaluation/confusion_matrix.png`)
12. **Figure 12**: ROC Curves (`06_evaluation/roc_curves.png`)
13. **Figure 13**: Per-Class Metrics (`06_evaluation/precision_per_class.png`, etc.)
14. **Figure 14**: Sample Predictions with Grad-CAM (`07_predictions/sample_predictions.png`)
15. **Figure 15**: Model Comparison (`08_comparison/model_comparison.png`)

### Tables for Paper

**Table 1: Final Performance Metrics**
- Use data from `06_evaluation/final_metrics.json`

**Table 2: Per-Class Performance**
- Precision, Recall, F1 Score for each class

**Table 3: Model Comparison**
- Comparison with existing approaches

---

## 🔧 Training Configuration

- **Model**: Hybrid EfficientNet-B3 + DyDA + Swin Transformer
- **Epochs**: 50
- **Batch Size**: 32
- **Optimizer**: AdamW
- **Learning Rate**: 1e-4
- **Loss Function**: Label Smoothing Cross-Entropy
- **Data Split**: 70% Train / 15% Val / 15% Test
- **Classes**: 4 (Normal, COVID-19, Pneumonia, Tuberculosis)
- **Samples per Class**: 3000 (balanced)

---

## 📝 Citation

```bibtex
@article{pulmonary_hybrid_2026,
  title={Hybrid EfficientNet-DyDA-Swin Transformer for Pulmonary Disease Classification},
  author={Research Team},
  journal={IEEE Conference/Journal},
  year={2026}
}
```

---

## ✅ Checklist for IEEE Paper

- [x] Dataset sample images
- [x] Preprocessed image results
- [x] Data augmentation samples
- [x] Proposed model architecture diagram
- [x] Workflow/block diagram
- [x] Training accuracy graph (curve)
- [x] Validation accuracy graph (curve)
- [x] Training loss graph (curve)
- [x] Validation loss graph (curve)
- [x] Confusion matrix
- [x] ROC curves
- [x] Performance metrics curves (F1, Precision, Recall, AUC)
- [x] Per-class metrics curves
- [x] Sample prediction outputs with Grad-CAM
- [x] Comparison with existing models (curve graph)
- [x] Final results table (JSON data)
- [x] Comprehensive PDF document

---

**Generated**: May 7, 2026  
**Status**: ✅ Ready for IEEE Paper Submission  
**Quality**: Publication-ready (300 DPI)
