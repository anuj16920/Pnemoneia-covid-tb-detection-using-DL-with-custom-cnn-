"""
efficientnet_backbone.py
EfficientNet-B3 backbone with configurable stage freezing.
Extracts multi-scale feature maps for the DyDA attention module.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B3_Weights
from typing import List, Optional


class EfficientNetBackbone(nn.Module):
    """
    EfficientNet-B3 backbone adapted for feature extraction.

    Architecture overview (EfficientNet-B3 MBConv stages):
        Stage 0: stem conv            (112×112, 40ch)
        Stage 1: MBConv1 3x3         (112×112, 24ch)
        Stage 2: MBConv6 3x3 ×2     (56×56,  32ch)
        Stage 3: MBConv6 5x5 ×3     (28×28,  48ch)
        Stage 4: MBConv6 3x3 ×3     (14×14,  96ch)
        Stage 5: MBConv6 5x5 ×4     (14×14, 136ch)
        Stage 6: MBConv6 5x5 ×5     (7×7,  232ch)
        Stage 7: MBConv6 3x3 ×2     (7×7,  384ch)
        Head:    Conv1x1 + Pool      (1×1, 1536ch)

    Args:
        pretrained: Load ImageNet weights
        freeze_stages: List of stage indices to freeze (0-indexed)
        out_channels: Expected output channels (384 for stage-7 features)
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_stages: Optional[List[int]] = None,
        out_channels: int = 384,
    ):
        super().__init__()
        self.out_channels = out_channels

        # Load pretrained EfficientNet-B3
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = models.efficientnet_b3(weights=weights)

        # Extract feature extractor (all stages except final classifier)
        # base_model.features: Sequential of 9 stages (0-8)
        self.features = base_model.features  # nn.Sequential

        # Global Average Pooling for the CNN branch vector
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Freeze specified stages
        if freeze_stages is not None:
            self._freeze_stages(freeze_stages)

    def _freeze_stages(self, stages: List[int]) -> None:
        """Freeze parameters in the specified EfficientNet stages."""
        for stage_idx in stages:
            if stage_idx < len(self.features):
                for param in self.features[stage_idx].parameters():
                    param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze all backbone parameters (for fine-tuning later)."""
        for param in self.features.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            feature_maps: Spatial feature maps [B, C, h, w] for attention
            gap_vector:   GAP-pooled 1D vector [B, 1536] for fusion
        """
        # Pass through all stages
        feature_maps = self.features(x)        # [B, 1536, 7, 7] for 224 input

        # Global average pooling for the classification branch
        gap_vector = self.gap(feature_maps)    # [B, 1536, 1, 1]
        gap_vector = gap_vector.flatten(1)     # [B, 1536]

        return feature_maps, gap_vector

    def get_intermediate_features(self, x: torch.Tensor, stage: int):
        """Extract features at a specific intermediate stage (for Grad-CAM)."""
        for i, stage_module in enumerate(self.features):
            x = stage_module(x)
            if i == stage:
                return x
        return x


if __name__ == "__main__":
    # Quick shape verification
    backbone = EfficientNetBackbone(pretrained=False, freeze_stages=[0, 1, 2])
    dummy = torch.randn(2, 3, 224, 224)
    feat_maps, gap_vec = backbone(dummy)
    print(f"Feature maps: {feat_maps.shape}")  # [2, 1536, 7, 7]
    print(f"GAP vector:   {gap_vec.shape}")    # [2, 1536]
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    total = sum(p.numel() for p in backbone.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")
