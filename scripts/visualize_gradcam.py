#!/usr/bin/env python3
"""
visualize_gradcam.py
Generate Grad-CAM saliency visualizations for model interpretability.

Usage:
    # Single image
    python scripts/visualize_gradcam.py \
        --config configs/config.yaml \
        --checkpoint results/checkpoints/best_fold1.pth \
        --image path/to/xray.jpg \
        --class_name COVID-19

    # Batch from dataset (num_per_class images per class)
    python scripts/visualize_gradcam.py \
        --config configs/config.yaml \
        --checkpoint results/checkpoints/best_fold1.pth \
        --splits results/logs/splits.json \
        --num_per_class 5
"""

import sys
import argparse
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")

from src.models.full_model import PulmonaryDxModel
from src.evaluation.gradcam import (
    GradCAM, get_target_layer,
    visualize_gradcam, generate_class_gradcam_grid
)
from src.data.preprocessing import get_gradcam_transforms, denormalize
from src.data.data_splits import load_splits, get_fold_dataloaders
from src.utils.checkpoint import load_model_for_inference
from src.utils.seed import set_seed


CLASS_NAMES = ["Normal", "COVID-19", "Pneumonia", "Tuberculosis"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        type=str, required=True)
    parser.add_argument("--checkpoint",    type=str, required=True)
    parser.add_argument("--image",         type=str, default=None, help="Single image path")
    parser.add_argument("--class_name",    type=str, default=None, help="True class for single image")
    parser.add_argument("--splits",        type=str, default=None)
    parser.add_argument("--fold",          type=int, default=0)
    parser.add_argument("--num_per_class", type=int, default=5)
    parser.add_argument("--output_dir",    type=str, default="results/gradcam")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("project", {}).get("seed", 42))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = PulmonaryDxModel(
        num_classes=4, pretrained=False, use_dyda=True, use_swin=True
    )
    model = load_model_for_inference(model, args.checkpoint, device=device)

    if args.image:
        # Single image mode
        print(f"Processing: {args.image}")
        transform = get_gradcam_transforms(image_size=224)

        pil_img = Image.open(args.image).convert("RGB")
        img_array = np.array(pil_img.resize((224, 224)))

        tensor = transform(pil_img).unsqueeze(0)
        tensor = tensor.to(device)

        true_label = CLASS_TO_IDX.get(args.class_name) if args.class_name else None

        save_path = Path(args.output_dir) / f"gradcam_{Path(args.image).stem}.png"
        result = visualize_gradcam(
            model=model,
            image_tensor=tensor,
            original_image=img_array,
            true_label=true_label,
            save_path=str(save_path),
            class_names=CLASS_NAMES,
        )

        pred_name = CLASS_NAMES[result["pred_class"]]
        print(f"Prediction: {pred_name} ({result['confidence']:.1%})")
        print(f"Saved: {save_path}")

    else:
        # Batch mode from dataset
        if not args.splits:
            print("ERROR: --splits is required for batch mode. "
                  "Use --image for single image mode.")
            sys.exit(1)

        splits = load_splits(args.splits)
        _, val_loader, _ = get_fold_dataloaders(
            splits=splits, fold_idx=args.fold, cfg=cfg, batch_size=16, num_workers=2
        )

        print(f"Generating Grad-CAM for {args.num_per_class} samples per class...")
        generate_class_gradcam_grid(
            model=model,
            dataloader=val_loader,
            output_dir=args.output_dir,
            num_per_class=args.num_per_class,
            class_names=CLASS_NAMES,
            device=device,
        )


if __name__ == "__main__":
    main()
