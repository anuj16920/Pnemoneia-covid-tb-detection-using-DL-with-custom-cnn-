"""
fusion_head.py
Feature Fusion and Classification Head.

Concatenates CNN (EfficientNet+DyDA GAP) and Swin Transformer
feature vectors, then passes through a classification MLP with
BatchNorm and Dropout regularization.

Architecture:
    [CNN_feat (1536) ‖ Swin_feat (768)] → Linear(2304→512) → BN → ReLU → Dropout(0.4)
                                        → Linear(512→num_classes)
"""

import torch
import torch.nn as nn
from typing import Optional


class FusionHead(nn.Module):
    """
    Multi-modal feature fusion and classification head.

    Supports three operational modes:
      - "full":       CNN + Swin concatenation (both branches active)
      - "cnn_only":   Only CNN features (Swin disabled / ablation)
      - "swin_only":  Only Swin features (for experimental use)

    Args:
        cnn_feat_dim:   Dimension of CNN (EfficientNet) feature vector
        swin_feat_dim:  Dimension of Swin Transformer feature vector
        hidden_dim:     Hidden layer dimension in classifier MLP
        num_classes:    Number of output classes
        dropout:        Dropout probability
        mode:           Fusion mode — "full" | "cnn_only" | "swin_only"
    """

    def __init__(
        self,
        cnn_feat_dim: int = 1536,
        swin_feat_dim: int = 768,
        hidden_dim: int = 512,
        num_classes: int = 4,
        dropout: float = 0.4,
        mode: str = "full",
    ):
        super().__init__()
        assert mode in ("full", "cnn_only", "swin_only"), \
            f"Invalid mode: {mode}. Choose from 'full', 'cnn_only', 'swin_only'"

        self.mode = mode
        self.cnn_feat_dim = cnn_feat_dim
        self.swin_feat_dim = swin_feat_dim

        # Determine input dimension based on mode
        if mode == "full":
            in_dim = cnn_feat_dim + swin_feat_dim
        elif mode == "cnn_only":
            in_dim = cnn_feat_dim
        else:  # swin_only
            in_dim = swin_feat_dim

        self.in_dim = in_dim

        # Classification MLP
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        cnn_features: Optional[torch.Tensor],
        swin_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            cnn_features:  CNN feature vector [B, cnn_feat_dim] or None
            swin_features: Swin feature vector [B, swin_feat_dim] or None

        Returns:
            logits: Class logits [B, num_classes]
        """
        if self.mode == "full":
            assert cnn_features is not None and swin_features is not None
            fused = torch.cat([cnn_features, swin_features], dim=1)  # [B, C+S]
        elif self.mode == "cnn_only":
            assert cnn_features is not None
            fused = cnn_features
        else:
            assert swin_features is not None
            fused = swin_features

        logits = self.classifier(fused)
        return logits

    def extra_repr(self) -> str:
        return (
            f"mode={self.mode}, "
            f"in_dim={self.in_dim}, "
            f"cnn={self.cnn_feat_dim}, swin={self.swin_feat_dim}"
        )


if __name__ == "__main__":
    head = FusionHead(cnn_feat_dim=1536, swin_feat_dim=768,
                      hidden_dim=512, num_classes=4, mode="full")
    cnn_f = torch.randn(4, 1536)
    swn_f = torch.randn(4, 768)
    logits = head(cnn_f, swn_f)
    print(f"Logits shape: {logits.shape}")  # [4, 4]

    # Test ablation mode
    head_cnn = FusionHead(cnn_feat_dim=1536, swin_feat_dim=768, mode="cnn_only")
    logits2 = head_cnn(cnn_f, None)
    print(f"CNN-only logits: {logits2.shape}")
    print("Test passed ✓")
