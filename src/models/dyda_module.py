"""
dyda_module.py
Dynamic Dual Attention (DyDA) Module — Core Novel Contribution

Description:
    DyDA applies channel and spatial attention pathways in PARALLEL
    and combines their outputs via a pair of input-dependent learnable
    gates α and β, constrained by softmax normalization such that α + β = 1.

    This enables the model to dynamically emphasize either:
      - Channel-wise features (α)  — "what" features are relevant
      - Spatially-localized features (β) — "where" features are relevant
    ...according to the pathological content of each individual image.

    Contrast with:
      - CBAM: fixed sequential (channel first, then spatial)
      - DANet: globally-constant scalar weighting

Mathematical formulation:
    Given input features F ∈ R^(B×C×H×W):

    1. Channel attention:
       fC = MLP(AvgPool(F)) · sigmoid + MLP(MaxPool(F)) · sigmoid → scale C dim
       FC = fC ⊗ F   (channel-scaled feature map)

    2. Spatial attention:
       fS_avg = AvgPool_channel(F)    → [B,1,H,W]
       fS_max = MaxPool_channel(F)    → [B,1,H,W]
       fS = sigmoid(Conv([fS_avg, fS_max]))  → spatial map [B,1,H,W]
       FS = fS ⊗ F   (spatially-scaled feature map)

    3. Input-dependent gating:
       g_in = [GAP(FC) ‖ GAP(FS)]   → [B, 2C] compressed descriptor
       [α, β] = softmax(MLP(g_in))   → [B, 2], constrained α+β=1

    4. Fused output:
       out = α · FC + β · FS          (broadcast over H,W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ─────────────────────────────────────────────────────────────
# Sub-modules
# ─────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel attention.
    Uses both avg-pool and max-pool branches (as in CBAM) for richer context.
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        mid = max(in_channels // reduction_ratio, 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP across both pooling branches
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            channel_attention_map: [B, C, 1, 1]  (values in [0,1])
        """
        avg_out = self.mlp(self.avg_pool(x))   # [B, C]
        max_out = self.mlp(self.max_pool(x))   # [B, C]
        attn = torch.sigmoid(avg_out + max_out)
        return attn.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]


class SpatialAttention(nn.Module):
    """
    Spatial attention using channel-pooled avg+max concatenation.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            spatial_attention_map: [B, 1, H, W]  (values in [0,1])
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        combined = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        return torch.sigmoid(self.bn(self.conv(combined)))  # [B, 1, H, W]


class InputDependentGate(nn.Module):
    """
    Computes input-dependent softmax gates [α, β] from attended features.

    The gate is computed from a compressed global descriptor that combines
    information from both the channel-attended and spatially-attended maps,
    ensuring α + β = 1 by construction via softmax.

    Architecture:
        g_in = [GAP(FC) ‖ GAP(FS)] ∈ R^(2C)
        [α, β] = softmax(Linear(ReLU(Linear(g_in))))
    """

    def __init__(self, in_channels: int, hidden_dim: int = 64):
        super().__init__()
        # Input: 2*C (from concat of two GAP-pooled attended features)
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2, bias=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(
        self,
        fc_feat: torch.Tensor,   # Channel-attended: [B, C, H, W]
        fs_feat: torch.Tensor,   # Spatially-attended: [B, C, H, W]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            alpha: [B, 1, 1, 1] gate for channel-attended branch
            beta:  [B, 1, 1, 1] gate for spatial-attended branch
            alpha + beta = 1 by construction
        """
        # Global average pool each attended feature map
        fc_pooled = self.gap(fc_feat).flatten(1)   # [B, C]
        fs_pooled = self.gap(fs_feat).flatten(1)   # [B, C]

        # Concatenate for joint gate descriptor
        g_in = torch.cat([fc_pooled, fs_pooled], dim=1)  # [B, 2C]

        # Compute gates via softmax (guarantees α+β=1)
        gates = torch.softmax(self.gate_mlp(g_in), dim=-1)  # [B, 2]

        alpha = gates[:, 0].view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        beta  = gates[:, 1].view(-1, 1, 1, 1)  # [B, 1, 1, 1]

        return alpha, beta


# ─────────────────────────────────────────────────────────────
# Main DyDA Module
# ─────────────────────────────────────────────────────────────

class DyDAModule(nn.Module):
    """
    Dynamic Dual Attention (DyDA) Module.

    Parallel channel + spatial attention with input-dependent
    softmax-normalized gating: output = α·FC + β·FS, α+β=1.

    Args:
        in_channels:        Number of input feature channels (C)
        reduction_ratio:    Bottleneck ratio for channel attention MLP
        spatial_kernel_size: Kernel size for spatial attention conv
        gate_hidden_dim:    Hidden dimension of the gate MLP
    """

    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
        spatial_kernel_size: int = 7,
        gate_hidden_dim: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels

        # Parallel attention pathways
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

        # Input-dependent gate predictor
        self.gate = InputDependentGate(in_channels, gate_hidden_dim)

        # Optional residual projection for stability
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            x: Input feature maps [B, C, H, W]

        Returns:
            out: Attended feature maps [B, C, H, W]
            aux: Dictionary with gate values for logging/visualization
                 {'alpha': [B], 'beta': [B]}
        """
        # ── Channel attention pathway ──────────────────────────
        c_attn = self.channel_attention(x)   # [B, C, 1, 1]
        fc = c_attn * x                      # [B, C, H, W]

        # ── Spatial attention pathway ──────────────────────────
        s_attn = self.spatial_attention(x)   # [B, 1, H, W]
        fs = s_attn * x                      # [B, C, H, W]

        # ── Input-dependent gating ─────────────────────────────
        alpha, beta = self.gate(fc, fs)      # [B,1,1,1] each, α+β=1

        # ── Dynamic fusion ─────────────────────────────────────
        out = alpha * fc + beta * fs         # [B, C, H, W]

        # Residual connection for training stability
        out = out + self.residual_scale * x

        aux = {
            "alpha": alpha.squeeze().detach(),
            "beta":  beta.squeeze().detach(),
        }

        return out, aux

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"channel_reduction=1/{16}, "
            f"gating=softmax(α+β=1)"
        )


# ─────────────────────────────────────────────────────────────
# CBAM (for ablation comparison)
# ─────────────────────────────────────────────────────────────

class CBAMModule(nn.Module):
    """
    Convolutional Block Attention Module (Woo et al., ECCV 2018).
    Sequential: channel attention → spatial attention.
    Included for ablation comparison against DyDA.
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # Sequential application
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x, {"alpha": None, "beta": None}


# ─────────────────────────────────────────────────────────────
# Unit test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    batch, C, H, W = 4, 1536, 7, 7
    x = torch.randn(batch, C, H, W)

    print("=" * 55)
    print("DyDA Module Test")
    print("=" * 55)
    dyda = DyDAModule(in_channels=C, reduction_ratio=16, gate_hidden_dim=64)
    out, aux = dyda(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Alpha (mean): {aux['alpha'].mean():.4f}")
    print(f"Beta  (mean): {aux['beta'].mean():.4f}")
    print(f"α+β check:    {(aux['alpha'] + aux['beta']).mean():.6f}  (should be 1.0)")
    assert out.shape == x.shape, "Shape mismatch!"
    alpha_beta_sum = (aux['alpha'] + aux['beta'])
    assert torch.allclose(alpha_beta_sum, torch.ones_like(alpha_beta_sum), atol=1e-5), \
        "Softmax constraint violated!"

    params = sum(p.numel() for p in dyda.parameters())
    print(f"DyDA params:  {params:,}")
    print()

    print("CBAM Module Test")
    print("=" * 55)
    cbam = CBAMModule(in_channels=C)
    out_cbam, _ = cbam(x)
    print(f"CBAM output shape: {out_cbam.shape}")
    cbam_params = sum(p.numel() for p in cbam.parameters())
    print(f"CBAM params: {cbam_params:,}")
    print("\nAll tests passed ✓")
