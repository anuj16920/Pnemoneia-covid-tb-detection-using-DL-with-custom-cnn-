"""
full_model.py
Full Hybrid Pulmonary Disease Classification Model.

Assembles:
  EfficientNet-B3 → DyDA → [CNN GAP ‖ Swin-T] → FusionHead → 4-class output

Supports ablation variants via config flags.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .efficientnet_backbone import EfficientNetBackbone
from .dyda_module import DyDAModule, CBAMModule
from .swin_transformer import SwinTransformerBranch
from .fusion_head import FusionHead


class PulmonaryDxModel(nn.Module):
    """
    Hybrid EfficientNet-B3 + DyDA + Swin Transformer model for
    pulmonary disease classification from chest radiographs.

    Args:
        num_classes:      Number of output classes (default: 4)
        pretrained:       Load pretrained weights for backbone components
        freeze_backbone_stages: Stages to freeze in EfficientNet-B3
        use_dyda:         Enable DyDA attention module
        use_cbam:         Enable CBAM module (for ablation; overrides use_dyda)
        use_swin:         Enable Swin Transformer branch
        dyda_reduction:   Channel reduction ratio in DyDA
        swin_freeze_stages: Number of Swin stages to freeze
        fusion_dropout:   Dropout in classification head
        hidden_dim:       Hidden dim in classification head

    Architecture modes:
        use_dyda=F, use_cbam=F, use_swin=F → Backbone-only (ablation 1)
        use_dyda=F, use_cbam=T, use_swin=F → Backbone+CBAM (ablation 2)
        use_dyda=T, use_cbam=F, use_swin=F → Backbone+DyDA (ablation 3)
        use_dyda=T, use_cbam=F, use_swin=T → Full model
    """

    # EfficientNet-B3 output channels (from features[-1])
    CNN_OUT_CHANNELS = 1536   # After EfficientNet head conv
    CNN_FEAT_CHANNELS = 384   # Stage-7 spatial features (for DyDA)

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        freeze_backbone_stages: list = None,
        use_dyda: bool = True,
        use_cbam: bool = False,
        use_swin: bool = True,
        dyda_reduction: int = 16,
        dyda_gate_hidden: int = 64,
        swin_freeze_stages: int = 0,
        fusion_dropout: float = 0.4,
        hidden_dim: int = 512,
    ):
        super().__init__()

        if freeze_backbone_stages is None:
            freeze_backbone_stages = [0, 1, 2]

        self.use_dyda = use_dyda
        self.use_cbam = use_cbam
        self.use_swin = use_swin

        # ── 1. EfficientNet-B3 Backbone ───────────────────────
        self.backbone = EfficientNetBackbone(
            pretrained=pretrained,
            freeze_stages=freeze_backbone_stages,
        )

        # ── 2. Attention Module (DyDA or CBAM or None) ────────
        # EfficientNet-B3 features[-1] outputs 1536 channels at 7×7
        attn_channels = self.CNN_OUT_CHANNELS

        if use_dyda and not use_cbam:
            self.attention = DyDAModule(
                in_channels=attn_channels,
                reduction_ratio=dyda_reduction,
                spatial_kernel_size=7,
                gate_hidden_dim=dyda_gate_hidden,
            )
        elif use_cbam:
            self.attention = CBAMModule(
                in_channels=attn_channels,
                reduction_ratio=dyda_reduction,
            )
        else:
            self.attention = None

        # ── 3. Swin Transformer Branch ─────────────────────────
        if use_swin:
            self.swin_branch = SwinTransformerBranch(
                img_size=224,
                pretrained=pretrained,
                freeze_stages=swin_freeze_stages,
            )
            swin_feat_dim = self.swin_branch.out_features
        else:
            self.swin_branch = None
            swin_feat_dim = 768  # unused

        # ── 4. Fusion Head ────────────────────────────────────
        fusion_mode = "full" if use_swin else "cnn_only"
        self.fusion_head = FusionHead(
            cnn_feat_dim=self.CNN_OUT_CHANNELS,
            swin_feat_dim=swin_feat_dim if use_swin else 768,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=fusion_dropout,
            mode=fusion_mode,
        )

        # Auxiliary gate statistics accumulator (for logging)
        self._gate_stats = {"alpha": [], "beta": []}

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: Input chest X-ray images [B, 3, 224, 224]

        Returns:
            logits: Class logits [B, num_classes]
            aux:    Auxiliary outputs {
                        'alpha': gate values (DyDA only),
                        'beta':  gate values (DyDA only),
                        'cnn_features': CNN feature vector,
                        'swin_features': Swin feature vector (if enabled)
                    }
        """
        aux = {}

        # ── Backbone forward ──────────────────────────────────
        feature_maps, cnn_gap = self.backbone(x)
        # feature_maps: [B, 1536, 7, 7]
        # cnn_gap:      [B, 1536]  (before attention)

        # ── Attention ─────────────────────────────────────────
        if self.attention is not None:
            attended_maps, attn_aux = self.attention(feature_maps)
            aux.update(attn_aux)

            # Re-pool after attention
            from torch import nn as _nn
            gap = _nn.AdaptiveAvgPool2d(1)
            cnn_features = gap(attended_maps).flatten(1)   # [B, 1536]
        else:
            cnn_features = cnn_gap   # [B, 1536]

        aux["cnn_features"] = cnn_features.detach()

        # ── Swin branch ───────────────────────────────────────
        swin_features = None
        if self.use_swin and self.swin_branch is not None:
            swin_features = self.swin_branch(x)   # [B, 768]
            aux["swin_features"] = swin_features.detach()

        # ── Fusion + Classification ───────────────────────────
        logits = self.fusion_head(cnn_features, swin_features)

        return logits, aux

    def get_parameter_groups(self, base_lr: float = 1e-4) -> list:
        """
        Return parameter groups with differential learning rates.
        Backbone gets 10x lower LR; attention + swin + head get full LR.
        """
        backbone_params = list(self.backbone.parameters())
        backbone_ids = set(id(p) for p in backbone_params)

        other_params = [
            p for p in self.parameters()
            if id(p) not in backbone_ids
        ]

        return [
            {"params": backbone_params, "lr": base_lr * 0.1, "name": "backbone"},
            {"params": other_params,    "lr": base_lr,        "name": "head+attention+swin"},
        ]

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable and total parameters per component."""
        components = {
            "backbone": self.backbone,
            "attention": self.attention,
            "swin_branch": self.swin_branch,
            "fusion_head": self.fusion_head,
        }
        stats = {}
        for name, module in components.items():
            if module is None:
                stats[name] = {"trainable": 0, "total": 0}
                continue
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            stats[name] = {"trainable": trainable, "total": total}
        return stats

    def freeze_swin(self) -> None:
        """Freeze Swin branch (useful for warmup training)."""
        if self.swin_branch is not None:
            for p in self.swin_branch.parameters():
                p.requires_grad = False

    def unfreeze_swin(self) -> None:
        """Unfreeze Swin branch."""
        if self.swin_branch is not None:
            for p in self.swin_branch.parameters():
                p.requires_grad = True

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone stages."""
        self.backbone.unfreeze_all()

    @classmethod
    def from_config(cls, cfg: dict) -> "PulmonaryDxModel":
        """
        Instantiate model from a config dict.
        Compatible with YAML config loaded via OmegaConf/PyYAML.
        """
        model_cfg = cfg.get("model", cfg)
        return cls(
            num_classes=cfg.get("data", {}).get("num_classes", 4),
            pretrained=model_cfg.get("backbone", {}).get("pretrained", True),
            freeze_backbone_stages=model_cfg.get("backbone", {}).get("freeze_stages", [0, 1, 2]),
            use_dyda=model_cfg.get("dyda", {}).get("enabled", True),
            use_cbam=model_cfg.get("cbam", {}).get("enabled", False),
            use_swin=model_cfg.get("swin", {}).get("enabled", True),
            dyda_reduction=model_cfg.get("dyda", {}).get("reduction_ratio", 16),
            dyda_gate_hidden=model_cfg.get("dyda", {}).get("gate_hidden_dim", 64),
            fusion_dropout=model_cfg.get("fusion", {}).get("dropout", 0.4),
            hidden_dim=model_cfg.get("fusion", {}).get("hidden_dim", 512),
        )


# ─────────────────────────────────────────────────────────────
# Quick model summary
# ─────────────────────────────────────────────────────────────

def print_model_summary(model: PulmonaryDxModel) -> None:
    stats = model.count_parameters()
    total_trainable = sum(v["trainable"] for v in stats.values())
    total_all = sum(v["total"] for v in stats.values())

    print("\n" + "=" * 55)
    print("  PulmonaryDxModel — Parameter Summary")
    print("=" * 55)
    for component, counts in stats.items():
        tr = counts["trainable"]
        tot = counts["total"]
        print(f"  {component:<20} {tr:>10,} trainable / {tot:>10,} total")
    print("-" * 55)
    print(f"  {'TOTAL':<20} {total_trainable:>10,} trainable / {total_all:>10,} total")
    print("=" * 55)
    print(f"\n  Attention:  {'DyDA' if model.use_dyda else ('CBAM' if model.use_cbam else 'None')}")
    print(f"  Swin:       {'Enabled' if model.use_swin else 'Disabled'}")
    print()


if __name__ == "__main__":
    # Full model test
    print("Full Model (EfficientNet-B3 + DyDA + Swin)")
    model = PulmonaryDxModel(
        num_classes=4,
        pretrained=False,  # Skip download in test
        use_dyda=True,
        use_swin=True,
    )
    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits, aux = model(dummy)

    print(f"Input:   {dummy.shape}")
    print(f"Logits:  {logits.shape}")
    print(f"Aux keys: {list(aux.keys())}")
    if "alpha" in aux:
        print(f"Alpha (mean): {aux['alpha'].mean():.4f}")
        print(f"β    (mean):  {aux['beta'].mean():.4f}")

    print_model_summary(model)

    # Ablation: backbone only
    print("\nBackbone-only ablation model:")
    model_ablation = PulmonaryDxModel(
        num_classes=4, pretrained=False,
        use_dyda=False, use_swin=False
    )
    with torch.no_grad():
        logits2, _ = model_ablation(dummy)
    print(f"Ablation logits: {logits2.shape}")
    print("All tests passed ✓")
