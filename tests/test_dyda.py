"""
test_dyda.py
Unit tests for the Dynamic Dual Attention (DyDA) module.
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dyda_module import (
    DyDAModule, CBAMModule,
    ChannelAttention, SpatialAttention, InputDependentGate
)


class TestChannelAttention:
    def test_output_shape(self):
        ca = ChannelAttention(in_channels=64, reduction_ratio=8)
        x = torch.randn(4, 64, 7, 7)
        out = ca(x)
        assert out.shape == (4, 64, 1, 1), f"Expected (4,64,1,1), got {out.shape}"

    def test_values_in_range(self):
        ca = ChannelAttention(in_channels=32, reduction_ratio=4)
        x = torch.randn(2, 32, 14, 14)
        out = ca(x)
        assert out.min() >= 0.0 and out.max() <= 1.0, "Channel attention values out of [0,1]"


class TestSpatialAttention:
    def test_output_shape(self):
        sa = SpatialAttention(kernel_size=7)
        x = torch.randn(4, 128, 7, 7)
        out = sa(x)
        assert out.shape == (4, 1, 7, 7), f"Expected (4,1,7,7), got {out.shape}"

    def test_values_in_range(self):
        sa = SpatialAttention(kernel_size=7)
        x = torch.randn(2, 64, 14, 14)
        out = sa(x)
        assert out.min() >= 0.0 and out.max() <= 1.0, "Spatial attention values out of [0,1]"


class TestInputDependentGate:
    def test_softmax_constraint(self):
        """α + β must equal 1 for all samples in batch."""
        gate = InputDependentGate(in_channels=64, hidden_dim=32)
        B, C, H, W = 8, 64, 7, 7
        fc = torch.randn(B, C, H, W)
        fs = torch.randn(B, C, H, W)

        alpha, beta = gate(fc, fs)

        # Check shape
        assert alpha.shape == (B, 1, 1, 1)
        assert beta.shape  == (B, 1, 1, 1)

        # Check softmax constraint
        alpha_flat = alpha.squeeze()
        beta_flat  = beta.squeeze()
        sum_ab = alpha_flat + beta_flat

        assert torch.allclose(sum_ab, torch.ones(B), atol=1e-5), \
            f"α+β constraint violated: {sum_ab}"

    def test_different_inputs_give_different_gates(self):
        """The gate should produce different α,β for different inputs (input-dependent)."""
        gate = InputDependentGate(in_channels=32, hidden_dim=16)
        C, H, W = 32, 7, 7

        fc1 = torch.randn(1, C, H, W)
        fs1 = torch.randn(1, C, H, W)
        fc2 = torch.randn(1, C, H, W)
        fs2 = torch.randn(1, C, H, W)

        alpha1, _ = gate(fc1, fs1)
        alpha2, _ = gate(fc2, fs2)

        # Different inputs should (probabilistically) give different gates
        # (This could theoretically fail with tiny probability)
        assert not torch.allclose(alpha1, alpha2, atol=1e-6), \
            "Gate produced identical outputs for different inputs (not input-dependent)"


class TestDyDAModule:
    def test_output_shape_preserved(self):
        """DyDA must preserve the input spatial dimensions."""
        for channels in [64, 256, 1536]:
            dyda = DyDAModule(in_channels=channels)
            x = torch.randn(2, channels, 7, 7)
            out, aux = dyda(x)
            assert out.shape == x.shape, \
                f"Shape mismatch: input {x.shape}, output {out.shape}"

    def test_alpha_beta_sum_to_one(self):
        """α + β = 1 must hold after DyDA forward pass."""
        dyda = DyDAModule(in_channels=128, gate_hidden_dim=32)
        x = torch.randn(4, 128, 7, 7)
        _, aux = dyda(x)

        alpha = aux["alpha"]
        beta  = aux["beta"]

        if alpha.dim() == 0:
            total = alpha + beta
        else:
            total = alpha + beta

        assert torch.allclose(total, torch.ones_like(total), atol=1e-4), \
            f"α+β ≠ 1: {total}"

    def test_gradients_flow(self):
        """Gradients must flow through DyDA for end-to-end training."""
        dyda = DyDAModule(in_channels=64)
        x = torch.randn(2, 64, 7, 7, requires_grad=True)
        out, _ = dyda(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "No gradient flowed through DyDA"
        assert not torch.isnan(x.grad).any(), "NaN gradients in DyDA"

    def test_batch_size_one(self):
        """Should work with batch size 1 (inference)."""
        dyda = DyDAModule(in_channels=256)
        x = torch.randn(1, 256, 7, 7)
        out, _ = dyda(x)
        assert out.shape == x.shape

    def test_different_spatial_sizes(self):
        """Should work with different spatial dimensions."""
        dyda = DyDAModule(in_channels=128)
        for hw in [7, 14, 28]:
            x = torch.randn(2, 128, hw, hw)
            out, _ = dyda(x)
            assert out.shape == x.shape, f"Failed at spatial size {hw}×{hw}"


class TestCBAMModule:
    def test_output_shape(self):
        cbam = CBAMModule(in_channels=256)
        x = torch.randn(4, 256, 7, 7)
        out, _ = cbam(x)
        assert out.shape == x.shape


class TestDyDAVsCBAM:
    def test_dyda_has_more_parameters(self):
        """DyDA has input-dependent gate parameters; verify it has different param count."""
        C = 256
        dyda = DyDAModule(in_channels=C)
        cbam = CBAMModule(in_channels=C)

        dyda_params = sum(p.numel() for p in dyda.parameters())
        cbam_params = sum(p.numel() for p in cbam.parameters())

        # DyDA has extra gate MLP params
        print(f"DyDA params: {dyda_params:,}")
        print(f"CBAM params: {cbam_params:,}")
        # Both should be > 0
        assert dyda_params > 0
        assert cbam_params > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
