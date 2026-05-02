"""
test_model_forward.py
Integration tests for PulmonaryDxModel forward pass.
Tests all ablation configurations and verifies output shapes/properties.
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.full_model import PulmonaryDxModel, print_model_summary
from src.models.dyda_module import DyDAModule
from src.models.swin_transformer import SwinTransformerBranch
from src.models.fusion_head import FusionHead


NUM_CLASSES = 4
BATCH = 2
IMG_SIZE = 224


@pytest.fixture
def dummy_input():
    return torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE)


class TestFullModel:
    def _build_model(self, use_dyda=True, use_cbam=False, use_swin=True):
        return PulmonaryDxModel(
            num_classes=NUM_CLASSES,
            pretrained=False,
            use_dyda=use_dyda,
            use_cbam=use_cbam,
            use_swin=use_swin,
        )

    def test_full_model_output_shape(self, dummy_input):
        model = self._build_model()
        with torch.no_grad():
            logits, aux = model(dummy_input)
        assert logits.shape == (BATCH, NUM_CLASSES), \
            f"Expected ({BATCH},{NUM_CLASSES}), got {logits.shape}"

    def test_backbone_only_output_shape(self, dummy_input):
        model = self._build_model(use_dyda=False, use_swin=False)
        with torch.no_grad():
            logits, _ = model(dummy_input)
        assert logits.shape == (BATCH, NUM_CLASSES)

    def test_cbam_ablation(self, dummy_input):
        model = self._build_model(use_dyda=False, use_cbam=True, use_swin=False)
        with torch.no_grad():
            logits, _ = model(dummy_input)
        assert logits.shape == (BATCH, NUM_CLASSES)

    def test_dyda_no_swin(self, dummy_input):
        model = self._build_model(use_dyda=True, use_swin=False)
        with torch.no_grad():
            logits, _ = model(dummy_input)
        assert logits.shape == (BATCH, NUM_CLASSES)

    def test_aux_dict_has_gate_values(self, dummy_input):
        model = self._build_model(use_dyda=True)
        with torch.no_grad():
            _, aux = model(dummy_input)
        assert "alpha" in aux, "DyDA aux should contain 'alpha'"
        assert "beta"  in aux, "DyDA aux should contain 'beta'"

    def test_softmax_gates_sum_to_one(self, dummy_input):
        model = self._build_model(use_dyda=True)
        with torch.no_grad():
            _, aux = model(dummy_input)
        alpha = aux["alpha"]
        beta  = aux["beta"]
        total = alpha + beta
        assert torch.allclose(total, torch.ones_like(total), atol=1e-4), \
            "α+β must equal 1"

    def test_logits_are_finite(self, dummy_input):
        model = self._build_model()
        with torch.no_grad():
            logits, _ = model(dummy_input)
        assert torch.isfinite(logits).all(), "Model output contains NaN or Inf"

    def test_gradients_end_to_end(self, dummy_input):
        """Backpropagation should succeed through the full model."""
        model = self._build_model()
        logits, _ = model(dummy_input)
        loss = logits.sum()
        loss.backward()

        # Check backbone gradients (unfrozen layers)
        grad_found = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), \
                    f"NaN gradient in {name}"
                grad_found = True

        assert grad_found, "No gradients found after backward pass"

    def test_parameter_groups(self):
        model = self._build_model()
        groups = model.get_parameter_groups(base_lr=1e-4)
        assert len(groups) >= 2
        # Backbone should have lower LR
        backbone_group = next(g for g in groups if g["name"] == "backbone")
        other_group    = next(g for g in groups if g["name"] != "backbone")
        assert backbone_group["lr"] < other_group["lr"], \
            "Backbone should have lower LR than head"

    def test_count_parameters(self):
        model = self._build_model()
        stats = model.count_parameters()
        for component, counts in stats.items():
            assert counts["total"] >= 0
            assert counts["trainable"] <= counts["total"]

    def test_eval_mode_consistency(self, dummy_input):
        """Model should produce same output in eval mode (deterministic)."""
        model = self._build_model()
        model.eval()
        with torch.no_grad():
            out1, _ = model(dummy_input)
            out2, _ = model(dummy_input)
        assert torch.allclose(out1, out2), "Non-deterministic eval mode output"


class TestFusionHead:
    def test_full_fusion(self):
        head = FusionHead(cnn_feat_dim=1536, swin_feat_dim=768,
                          num_classes=4, mode="full")
        cnn = torch.randn(4, 1536)
        swn = torch.randn(4, 768)
        out = head(cnn, swn)
        assert out.shape == (4, 4)

    def test_cnn_only(self):
        head = FusionHead(cnn_feat_dim=1536, swin_feat_dim=768,
                          num_classes=4, mode="cnn_only")
        cnn = torch.randn(4, 1536)
        out = head(cnn, None)
        assert out.shape == (4, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
