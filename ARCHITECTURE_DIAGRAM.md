# Pulmonary Disease Classification Model Architecture

## Full Model Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INPUT: Chest X-Ray Image                             │
│                            [B, 3, 224, 224]                                  │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
    ┌───────────────────────────┐   ┌──────────────────────┐
    │   EfficientNet-B3         │   │  Swin Transformer    │
    │   Backbone                │   │  Branch              │
    │   (Pretrained)            │   │  (Pretrained)        │
    │                           │   │                      │
    │  • Stages 0-8             │   │  • img_size: 224     │
    │  • Freeze: [0,1,2]        │   │  • Hierarchical      │
    │  • Output: 1536 channels  │   │    attention         │
    │  • Spatial: 7×7           │   │  • Output: 768-dim   │
    └────────────┬──────────────┘   └──────────┬───────────┘
                 │                              │
                 │ [B, 1536, 7, 7]              │ [B, 768]
                 ▼                              │
    ┌────────────────────────────┐              │
    │   DyDA Module              │              │
    │   (Dynamic Dual Attention) │              │
    │                            │              │
    │  ┌──────────────────────┐  │              │
    │  │ Channel Attention    │  │              │
    │  │ • AvgPool + MaxPool  │  │              │
    │  │ • Shared MLP         │  │              │
    │  │ • Sigmoid activation │  │              │
    │  │ Output: [B,C,1,1]    │  │              │
    │  └──────────┬───────────┘  │              │
    │             │               │              │
    │             ├─────► FC      │              │
    │             │    [B,C,H,W]  │              │
    │  ┌──────────┴───────────┐  │              │
    │  │ Spatial Attention    │  │              │
    │  │ • Channel pooling    │  │              │
    │  │   (avg + max)        │  │              │
    │  │ • Conv2d + BN        │  │              │
    │  │ • Sigmoid activation │  │              │
    │  │ Output: [B,1,H,W]    │  │              │
    │  └──────────┬───────────┘  │              │
    │             │               │              │
    │             └─────► FS      │              │
    │                  [B,C,H,W]  │              │
    │                             │              │
    │  ┌──────────────────────┐  │              │
    │  │ Input-Dependent Gate │  │              │
    │  │                      │  │              │
    │  │  g = [GAP(FC) ‖      │  │              │
    │  │       GAP(FS)]       │  │              │
    │  │       ↓              │  │              │
    │  │  [α, β] = softmax(  │  │              │
    │  │      MLP(g))         │  │              │
    │  │                      │  │              │
    │  │  Constraint: α+β=1   │  │              │
    │  └──────────┬───────────┘  │              │
    │             │               │              │
    │             ▼               │              │
    │  ┌──────────────────────┐  │              │
    │  │ Dynamic Fusion       │  │              │
    │  │                      │  │              │
    │  │  out = α·FC + β·FS   │  │              │
    │  │      + residual·x    │  │              │
    │  └──────────┬───────────┘  │              │
    └─────────────┼───────────────┘              │
                  │                              │
                  │ [B, 1536, 7, 7]              │
                  ▼                              │
    ┌─────────────────────────┐                  │
    │  Global Average Pool    │                  │
    │  (GAP)                  │                  │
    └─────────────┬───────────┘                  │
                  │                              │
                  │ [B, 1536]                    │
                  │                              │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────┐
                  │   Fusion Head            │
                  │                          │
                  │  CNN Features: [B,1536]  │
                  │  Swin Features: [B,768]  │
                  │         ↓                │
                  │  Concatenate → [B,2304]  │
                  │         ↓                │
                  │  Linear(2304 → 512)      │
                  │         ↓                │
                  │  BatchNorm1d             │
                  │         ↓                │
                  │  ReLU                    │
                  │         ↓                │
                  │  Dropout(0.4)            │
                  │         ↓                │
                  │  Linear(512 → 4)         │
                  └──────────┬───────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   OUTPUT: Logits     │
                  │   [B, 4]             │
                  │                      │
                  │   Classes:           │
                  │   • COVID-19         │
                  │   • Lung Opacity     │
                  │   • Normal           │
                  │   • Viral Pneumonia  │
                  └──────────────────────┘
```

## DyDA Module Detailed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DyDA Module (Core Innovation)                 │
│                                                                  │
│  Input: F ∈ R^(B×C×H×W)                                         │
│                                                                  │
│  ┌────────────────────────┐    ┌────────────────────────┐      │
│  │  Channel Attention     │    │  Spatial Attention     │      │
│  │  Pathway               │    │  Pathway               │      │
│  │                        │    │                        │      │
│  │  AvgPool(F) ──┐        │    │  AvgPool_C(F) ──┐     │      │
│  │               ├─► MLP  │    │                 │     │      │
│  │  MaxPool(F) ──┘    ↓   │    │  MaxPool_C(F) ──┴─►   │      │
│  │              Sigmoid   │    │  Conv2d([·,·])        │      │
│  │                 ↓      │    │        ↓              │      │
│  │  fC: [B,C,1,1]         │    │  BatchNorm            │      │
│  │                 ↓      │    │        ↓              │      │
│  │  FC = fC ⊗ F           │    │  Sigmoid              │      │
│  │     [B,C,H,W]          │    │        ↓              │      │
│  │                        │    │  fS: [B,1,H,W]        │      │
│  └────────┬───────────────┘    │        ↓              │      │
│           │                    │  FS = fS ⊗ F          │      │
│           │                    │     [B,C,H,W]         │      │
│           │                    └────────┬───────────────┘      │
│           │                             │                      │
│           └──────────┬──────────────────┘                      │
│                      │                                         │
│                      ▼                                         │
│           ┌──────────────────────┐                            │
│           │  Input-Dependent     │                            │
│           │  Gating Network      │                            │
│           │                      │                            │
│           │  GAP(FC) → [B,C]     │                            │
│           │  GAP(FS) → [B,C]     │                            │
│           │       ↓              │                            │
│           │  Concat → [B,2C]     │                            │
│           │       ↓              │                            │
│           │  Linear(2C → 64)     │                            │
│           │       ↓              │                            │
│           │  ReLU                │                            │
│           │       ↓              │                            │
│           │  Linear(64 → 2)      │                            │
│           │       ↓              │                            │
│           │  Softmax             │                            │
│           │       ↓              │                            │
│           │  [α, β] ∈ [0,1]²     │                            │
│           │  α + β = 1           │                            │
│           └──────────┬───────────┘                            │
│                      │                                         │
│                      ▼                                         │
│           ┌──────────────────────┐                            │
│           │  Dynamic Fusion      │                            │
│           │                      │                            │
│           │  out = α·FC + β·FS   │                            │
│           │      + λ·F           │                            │
│           │                      │                            │
│           │  (λ: residual scale) │                            │
│           └──────────────────────┘                            │
│                                                                │
│  Output: Attended features [B,C,H,W]                          │
│  Aux: {alpha: [B], beta: [B]}                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Model Variants (Ablation Study)

| Variant | Backbone | Attention | Swin | Description |
|---------|----------|-----------|------|-------------|
| **Ablation 1** | ✓ | ✗ | ✗ | Baseline: EfficientNet-B3 only |
| **Ablation 2** | ✓ | CBAM | ✗ | Sequential attention (channel→spatial) |
| **Ablation 3** | ✓ | DyDA | ✗ | Parallel attention with dynamic gating |
| **Full Model** | ✓ | DyDA | ✓ | Complete hybrid architecture |

## Key Components

### 1. EfficientNet-B3 Backbone
- **Purpose**: Extract hierarchical CNN features
- **Output**: 1536 channels at 7×7 spatial resolution
- **Pretrained**: ImageNet weights
- **Frozen stages**: [0, 1, 2] for transfer learning

### 2. DyDA Module (Novel Contribution)
- **Purpose**: Adaptive attention mechanism
- **Innovation**: Input-dependent gating (α, β) with softmax constraint
- **Advantage**: Dynamically balances channel vs spatial attention per image
- **Parameters**: ~100K (lightweight)

### 3. Swin Transformer Branch
- **Purpose**: Capture long-range dependencies via self-attention
- **Output**: 768-dimensional feature vector
- **Architecture**: Hierarchical shifted windows
- **Pretrained**: ImageNet-1K weights

### 4. Fusion Head
- **Input**: Concatenated CNN (1536) + Swin (768) = 2304 dims
- **Architecture**: 
  - Linear(2304 → 512)
  - BatchNorm + ReLU + Dropout(0.4)
  - Linear(512 → 4)
- **Output**: 4-class logits

## Mathematical Formulation

### DyDA Forward Pass

Given input features **F** ∈ ℝ^(B×C×H×W):

1. **Channel Attention**:
   ```
   f_C = σ(MLP(AvgPool(F))) + σ(MLP(MaxPool(F)))
   F_C = f_C ⊗ F
   ```

2. **Spatial Attention**:
   ```
   f_S = σ(Conv(AvgPool_channel(F) ‖ MaxPool_channel(F)))
   F_S = f_S ⊗ F
   ```

3. **Input-Dependent Gating**:
   ```
   g = [GAP(F_C) ‖ GAP(F_S)] ∈ ℝ^(2C)
   [α, β] = softmax(MLP(g))
   ```
   **Constraint**: α + β = 1

4. **Dynamic Fusion**:
   ```
   out = α · F_C + β · F_S + λ · F
   ```

## Training Configuration

- **Input Size**: 224×224 RGB
- **Batch Size**: 32
- **Optimizer**: AdamW
- **Learning Rates**:
  - Backbone: 1e-5 (10× lower)
  - Attention + Swin + Head: 1e-4
- **Loss**: Cross-Entropy with class weights
- **Augmentation**: RandomRotation, ColorJitter, RandomHorizontalFlip

## Performance Characteristics

- **Total Parameters**: ~40M
- **Trainable Parameters**: ~25M (with frozen stages)
- **Input**: [B, 3, 224, 224]
- **Output**: [B, 4] logits
- **Inference Time**: ~50ms per image (GPU)

## Key Innovations

1. **Dynamic Dual Attention (DyDA)**:
   - Parallel channel + spatial pathways
   - Input-dependent gating with softmax normalization
   - Adaptive feature emphasis per image

2. **Hybrid Architecture**:
   - CNN (local features) + Transformer (global context)
   - Complementary feature extraction

3. **Differential Learning Rates**:
   - Preserves pretrained knowledge
   - Faster convergence on medical domain
