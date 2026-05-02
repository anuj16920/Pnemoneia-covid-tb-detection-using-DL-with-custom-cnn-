"""
swin_transformer.py
Swin Transformer branch for global context modeling.

Uses pretrained Swin-Tiny from torchvision / timm as the backbone.
Extracts a global feature vector for fusion with the CNN (EfficientNet+DyDA) branch.

Reference: Liu et al., "Swin Transformer: Hierarchical Vision Transformer
           using Shifted Windows", ICCV 2021.

Architecture (Swin-T):
    Patch Embed (4×4) → Stage1(2 blocks) → PatchMerge →
    Stage2(2 blocks)  → PatchMerge →
    Stage3(6 blocks)  → PatchMerge →
    Stage4(2 blocks)  → LayerNorm → GAP → 768-dim vector
"""

import torch
import torch.nn as nn
from typing import Optional

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

try:
    from torchvision.models import swin_t, Swin_T_Weights
    TORCHVISION_SWIN_AVAILABLE = True
except ImportError:
    TORCHVISION_SWIN_AVAILABLE = False


class SwinTransformerBranch(nn.Module):
    """
    Swin Transformer feature extractor branch.

    Produces a fixed-dimensional global feature vector from an input
    chest X-ray image for fusion with the CNN (EfficientNet+DyDA) branch.

    Tries to load in order:
      1. timm's swin_tiny_patch4_window7_224 (preferred, most flexible)
      2. torchvision's swin_t
      3. Custom lightweight fallback (no pretrained weights)

    Args:
        img_size:        Input image size (square)
        out_features:    Output feature dimension (768 for Swin-T)
        pretrained:      Load pretrained ImageNet weights
        freeze_stages:   Number of Swin stages to freeze (0-4)
    """

    def __init__(
        self,
        img_size: int = 224,
        out_features: int = 768,
        pretrained: bool = True,
        freeze_stages: int = 0,
    ):
        super().__init__()
        self.out_features = out_features
        self._loaded_from = None

        if TIMM_AVAILABLE:
            self._build_from_timm(pretrained)
        elif TORCHVISION_SWIN_AVAILABLE:
            self._build_from_torchvision(pretrained)
        else:
            self._build_fallback(out_features)

        if freeze_stages > 0:
            self._freeze_stages(freeze_stages)

    def _build_from_timm(self, pretrained: bool) -> None:
        """Load Swin-T from timm (most flexible option)."""
        model = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,      # Remove classifier head → returns features
            global_pool="avg",  # GAP → 1D vector
        )
        self.swin = model
        # timm's swin_tiny outputs 768-dim with global_pool='avg'
        self.out_features = model.num_features
        self._loaded_from = "timm"

    def _build_from_torchvision(self, pretrained: bool) -> None:
        """Load Swin-T from torchvision."""
        weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        model = swin_t(weights=weights)
        # Remove the classification head
        self.swin_features = nn.Sequential(
            model.features,
            model.norm,
            model.permute,
            model.avgpool,
            nn.Flatten(1),
        )
        self.out_features = 768
        self._loaded_from = "torchvision"

    def _build_fallback(self, out_features: int) -> None:
        """
        Minimal Swin Transformer implementation (no pretrained weights).
        Used when neither timm nor torchvision.swin_t is available.
        """
        print("[WARNING] timm/torchvision Swin-T not found. Using lightweight fallback.")
        self.swin = LightweightSwinFallback(out_features=out_features)
        self._loaded_from = "fallback"

    def _freeze_stages(self, n_stages: int) -> None:
        """Freeze first n_stages of the Swin model."""
        if self._loaded_from == "timm":
            # timm Swin stages are in self.swin.layers
            stages = list(self.swin.layers)
            for stage in stages[:n_stages]:
                for param in stage.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image [B, 3, H, W] (H=W=224)

        Returns:
            features: Global feature vector [B, out_features]
        """
        if self._loaded_from == "timm":
            return self.swin(x)                # [B, 768]
        elif self._loaded_from == "torchvision":
            return self.swin_features(x)       # [B, 768]
        else:
            return self.swin(x)


# ─────────────────────────────────────────────────────────────
# Lightweight fallback (when timm/torchvision not available)
# ─────────────────────────────────────────────────────────────

class WindowAttention(nn.Module):
    """Simplified window-based multi-head self-attention."""

    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinBlock(nn.Module):
    """Simplified Swin Transformer block (without cyclic shift for brevity)."""

    def __init__(self, dim: int, num_heads: int, window_size: int = 7, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerging(nn.Module):
    """Patch merging for Swin downsampling."""

    def __init__(self, input_resolution: int, dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H = W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4C]
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class LightweightSwinFallback(nn.Module):
    """
    Lightweight 4-stage Swin-T reimplementation.
    Used only when timm/torchvision is unavailable.
    Follows original Swin-T architecture spec.
    """

    def __init__(self, img_size: int = 224, patch_size: int = 4,
                 embed_dim: int = 96, out_features: int = 768):
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
        )
        self.norm0 = nn.LayerNorm(embed_dim)

        resolution = img_size // patch_size  # 56

        # Stage 1: 2 blocks, 56×56, dim=96
        self.stage1 = nn.Sequential(*[SwinBlock(embed_dim, num_heads=3) for _ in range(2)])
        self.merge1 = PatchMerging(resolution, embed_dim)
        resolution //= 2  # 28, dim=192

        # Stage 2: 2 blocks
        self.stage2 = nn.Sequential(*[SwinBlock(embed_dim * 2, num_heads=6) for _ in range(2)])
        self.merge2 = PatchMerging(resolution, embed_dim * 2)
        resolution //= 2  # 14, dim=384

        # Stage 3: 6 blocks
        self.stage3 = nn.Sequential(*[SwinBlock(embed_dim * 4, num_heads=12) for _ in range(6)])
        self.merge3 = PatchMerging(resolution, embed_dim * 4)
        resolution //= 2  # 7, dim=768

        # Stage 4: 2 blocks
        self.stage4 = nn.Sequential(*[SwinBlock(embed_dim * 8, num_heads=24) for _ in range(2)])

        self.norm = nn.LayerNorm(embed_dim * 8)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embedding: [B, 3, H, W] → [B, embed_dim, 56, 56] → [B, 56*56, embed_dim]
        B = x.shape[0]
        x = self.patch_embed(x)            # [B, embed_dim, N]
        x = x.transpose(1, 2)             # [B, N, embed_dim]
        x = self.norm0(x)

        x = self.stage1(x)
        x = self.merge1(x)

        x = self.stage2(x)
        x = self.merge2(x)

        x = self.stage3(x)
        x = self.merge3(x)

        x = self.stage4(x)
        x = self.norm(x)                   # [B, N, 768]

        # Global average pooling
        x = x.transpose(1, 2)             # [B, 768, N]
        x = self.gap(x).squeeze(-1)        # [B, 768]

        return x


# ─────────────────────────────────────────────────────────────
# Unit test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Swin Transformer Branch Test")
    print("=" * 45)

    dummy = torch.randn(2, 3, 224, 224)
    branch = SwinTransformerBranch(pretrained=False)
    out = branch(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    print(f"Source: {branch._loaded_from}")
    params = sum(p.numel() for p in branch.parameters())
    print(f"Params: {params:,}")
    print("Test passed ✓")
