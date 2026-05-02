"""
gradcam.py
Gradient-weighted Class Activation Mapping (Grad-CAM) for chest X-rays.

Generates saliency maps over the EfficientNet-B3 backbone's final
convolutional feature maps to confirm predictions align with clinically
relevant radiological regions (consolidations, infiltrates, nodules, etc.)

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
           via Gradient-based Localization", ICCV 2017.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2


CLASS_NAMES = ["Normal", "COVID-19", "Pneumonia", "Tuberculosis"]


class GradCAM:
    """
    Grad-CAM implementation for PulmonaryDxModel.

    Hooks into a target convolutional layer to capture activations
    and gradients for generating class activation maps.

    Args:
        model:        PulmonaryDxModel instance
        target_layer: nn.Module — the target conv layer (typically last conv in backbone)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Compute Grad-CAM for an input image.

        Args:
            input_tensor: [1, 3, H, W] preprocessed image tensor
            class_idx:    Target class (None → use predicted class)

        Returns:
            cam:        Grad-CAM heatmap [H, W] in [0, 1]
            pred_class: Predicted class index
            confidence: Predicted class probability
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        logits, _ = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)

        pred_class = logits.argmax(dim=1).item()
        if class_idx is None:
            class_idx = pred_class

        confidence = probs[0, pred_class].item()

        # Backward for target class
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1.0
        logits.backward(gradient=one_hot)

        # Compute Grad-CAM
        gradients = self.gradients    # [1, C, h, w]
        activations = self.activations  # [1, C, h, w]

        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Weighted combination of feature maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, h, w]
        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam, pred_class, confidence

    def remove_hooks(self):
        """Remove registered hooks (call when done)."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def get_target_layer(model) -> nn.Module:
    """
    Get the target layer for Grad-CAM from PulmonaryDxModel.
    Uses the last convolutional block in EfficientNet-B3.
    """
    # EfficientNet-B3 features[-2] is the last MBConv block
    # features[-1] is the final Conv + BN + SiLU head
    return model.backbone.features[-2]


def overlay_heatmap(
    original_img: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image.

    Args:
        original_img: [H, W, 3] uint8 numpy array
        cam:          [h, w] float32 heatmap in [0, 1]
        alpha:        Heatmap transparency
        colormap:     OpenCV colormap

    Returns:
        Overlaid image [H, W, 3] uint8
    """
    H, W = original_img.shape[:2]

    # Resize CAM to image size
    cam_resized = cv2.resize(cam, (W, H))
    cam_uint8 = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(cam_uint8, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlaid = np.uint8((1 - alpha) * original_img + alpha * heatmap)
    return overlaid


def visualize_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    original_image: np.ndarray,
    true_label: Optional[int] = None,
    save_path: Optional[str] = None,
    class_names: List[str] = CLASS_NAMES,
) -> Dict:
    """
    Generate and optionally save Grad-CAM visualization.

    Args:
        model:          PulmonaryDxModel
        image_tensor:   [1, 3, 224, 224] preprocessed tensor
        original_image: [H, W, 3] uint8 numpy for display
        true_label:     Ground truth class (optional)
        save_path:      Path to save the figure (None → display only)
        class_names:    List of class names

    Returns:
        dict with 'cam', 'pred_class', 'confidence'
    """
    target_layer = get_target_layer(model)
    gradcam = GradCAM(model, target_layer)

    cam, pred_class, confidence = gradcam(image_tensor)
    gradcam.remove_hooks()

    overlaid = overlay_heatmap(original_image, cam)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_image, cmap="gray" if original_image.ndim == 2 else None)
    axes[0].set_title("Original X-Ray", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(original_image, cmap="gray" if original_image.ndim == 2 else None)
    axes[1].imshow(cam, cmap="jet", alpha=0.5,
                   extent=[0, original_image.shape[1], original_image.shape[0], 0])
    axes[1].set_title("Grad-CAM Heatmap", fontsize=14)
    axes[1].axis("off")

    axes[2].imshow(overlaid)
    pred_name = class_names[pred_class] if pred_class < len(class_names) else str(pred_class)
    title = f"Overlay\nPred: {pred_name} ({confidence:.1%})"
    if true_label is not None:
        true_name = class_names[true_label] if true_label < len(class_names) else str(true_label)
        correct = "✓" if pred_class == true_label else "✗"
        title += f"\nTrue: {true_name} {correct}"
    axes[2].set_title(title, fontsize=13)
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return {"cam": cam, "pred_class": pred_class, "confidence": confidence}


def generate_class_gradcam_grid(
    model: nn.Module,
    dataloader,
    output_dir: str,
    num_per_class: int = 5,
    class_names: List[str] = CLASS_NAMES,
    device: str = "cuda",
) -> None:
    """
    Generate Grad-CAM visualizations for multiple samples per class.
    Saves a grid of images organized by class and prediction correctness.

    Args:
        model:         PulmonaryDxModel
        dataloader:    DataLoader (val/test set)
        output_dir:    Directory to save visualizations
        num_per_class: Number of samples to visualize per class
        class_names:   Class name list
        device:        Device string
    """
    from ..data.preprocessing import denormalize

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model.eval()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev)

    # Collect samples per class
    class_samples = {i: [] for i in range(len(class_names))}
    for images, labels in dataloader:
        for img, lbl in zip(images, labels):
            c = lbl.item()
            if len(class_samples[c]) < num_per_class:
                class_samples[c].append((img, c))
        if all(len(v) >= num_per_class for v in class_samples.values()):
            break

    target_layer = get_target_layer(model)

    for class_idx, samples in class_samples.items():
        class_name = class_names[class_idx]
        print(f"  Generating Grad-CAM for {class_name} ({len(samples)} samples)...")

        for sample_idx, (img_tensor, true_label) in enumerate(samples):
            inp = img_tensor.unsqueeze(0).to(dev)

            # Denormalize for display
            display_img = denormalize(img_tensor).permute(1, 2, 0).numpy()
            display_img = np.uint8(display_img * 255)

            save_path = output_path / f"{class_name}_sample{sample_idx+1}.png"

            gradcam = GradCAM(model, target_layer)
            cam, pred_class, confidence = gradcam(inp)
            gradcam.remove_hooks()

            overlaid = overlay_heatmap(display_img, cam)

            # Save individual image
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(display_img)
            axes[0].set_title(f"Original\n(True: {class_name})", fontsize=11)
            axes[0].axis("off")

            axes[1].imshow(display_img)
            axes[1].imshow(cam, cmap="jet", alpha=0.5,
                           extent=[0, 224, 224, 0])
            axes[1].set_title("Grad-CAM", fontsize=11)
            axes[1].axis("off")

            pred_name = class_names[pred_class]
            correct = "Correct ✓" if pred_class == true_label else f"Wrong ✗ ({pred_name})"
            axes[2].imshow(overlaid)
            axes[2].set_title(f"Overlay\n{correct} ({confidence:.1%})", fontsize=11)
            axes[2].axis("off")

            plt.suptitle(f"Grad-CAM: {class_name}", fontsize=13, fontweight="bold")
            plt.tight_layout()
            plt.savefig(str(save_path), dpi=120, bbox_inches="tight")
            plt.close()

    print(f"Grad-CAM visualizations saved to {output_dir}")


if __name__ == "__main__":
    print("Grad-CAM module loaded successfully ✓")
    print("Use visualize_gradcam() or generate_class_gradcam_grid()")
