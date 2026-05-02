#!/usr/bin/env python3
"""
Dataset Organization Script - BALANCED VERSION with Augmentation
Merges chest_xray (Normal, Pneumonia) + TB dataset (Normal, TB) + COVID-19 dataset
Output: data/Normal, data/Pneumonia, data/Tuberculosis, data/COVID-19
All classes balanced to 3000 samples (with augmentation for small classes)
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from PIL import Image

def count_files(directory):
    """Count image files in directory"""
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

def get_image_files(src_dir):
    """Get list of image files from directory"""
    if not os.path.exists(src_dir):
        return []
    
    images = []
    for img_file in os.listdir(src_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            images.append(os.path.join(src_dir, img_file))
    return images

def augment_image(img_path):
    """Apply random augmentation to image"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    # Random augmentation choice
    aug_type = random.choice(['flip', 'rotate', 'brightness', 'contrast', 'noise'])
    
    if aug_type == 'flip':
        img = cv2.flip(img, 1)  # Horizontal flip
    elif aug_type == 'rotate':
        angle = random.choice([-10, -5, 5, 10])
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    elif aug_type == 'brightness':
        factor = random.uniform(0.8, 1.2)
        img = np.clip(img * factor, 0, 255).astype(np.uint8)
    elif aug_type == 'contrast':
        factor = random.uniform(0.8, 1.2)
        mean = img.mean()
        img = np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)
    elif aug_type == 'noise':
        noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    return img

def copy_images_balanced(src_paths, dst_dir, prefix="", max_samples=None):
    """
    Copy images from multiple sources to dst with optional limit
    If available images < max_samples, augment to reach target
    src_paths: list of source directories
    max_samples: target number of images (will augment if needed)
    """
    os.makedirs(dst_dir, exist_ok=True)
    
    # Collect all images from all sources
    all_images = []
    for src_dir in src_paths:
        all_images.extend(get_image_files(src_dir))
    
    if len(all_images) == 0:
        print(f"⚠️  No images found in sources")
        return 0
    
    available = len(all_images)
    
    # Case 1: More images than needed - random sample
    if max_samples and available >= max_samples:
        random.shuffle(all_images)
        all_images = all_images[:max_samples]
        print(f"   Sampled {max_samples} from {available} available")
        
        # Copy original images
        count = 0
        for src_path in all_images:
            img_file = os.path.basename(src_path)
            dst_filename = f"{prefix}_{img_file}" if prefix else img_file
            dst_path = os.path.join(dst_dir, dst_filename)
            shutil.copy2(src_path, dst_path)
            count += 1
        return count
    
    # Case 2: Need augmentation
    else:
        print(f"   Available: {available}, Target: {max_samples}")
        print(f"   Copying originals + augmenting {max_samples - available} images...")
        
        # Copy all original images first
        count = 0
        for src_path in all_images:
            img_file = os.path.basename(src_path)
            dst_filename = f"{prefix}_orig_{img_file}" if prefix else f"orig_{img_file}"
            dst_path = os.path.join(dst_dir, dst_filename)
            shutil.copy2(src_path, dst_path)
            count += 1
        
        # Augment to reach target
        needed = max_samples - available
        aug_count = 0
        
        while aug_count < needed:
            # Pick random image to augment
            src_path = random.choice(all_images)
            img_file = os.path.basename(src_path)
            base, ext = os.path.splitext(img_file)
            
            # Augment
            aug_img = augment_image(src_path)
            if aug_img is not None:
                dst_filename = f"{prefix}_aug{aug_count}_{base}.png" if prefix else f"aug{aug_count}_{base}.png"
                dst_path = os.path.join(dst_dir, dst_filename)
                cv2.imwrite(dst_path, aug_img)
                aug_count += 1
                count += 1
                
                if aug_count % 100 == 0:
                    print(f"      Augmented {aug_count}/{needed}...")
        
        print(f"   ✓ Created {available} originals + {aug_count} augmented = {count} total")
        return count

def main():
    print("=" * 70)
    print("  BALANCED Dataset Organization - Pulmonary Disease Classification")
    print("=" * 70)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Output directory
    output_dir = "data_balanced"
    
    # Remove old data_balanced if exists
    if os.path.exists(output_dir):
        print(f"\n🗑️  Removing old {output_dir}/...")
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    stats = defaultdict(int)
    available_counts = {}
    
    # ── Step 1: Count available images per class ─────────────
    print("\n📊 Counting available images...")
    
    # Normal sources
    normal_sources = [
        "chest_xray/chest_xray/train/NORMAL",
        "chest_xray/chest_xray/val/NORMAL",
        "chest_xray/chest_xray/test/NORMAL",
        "TB_Chest_Radiography_Database/Normal",
        "COVID-19_Radiography_Dataset/Normal/images",
    ]
    available_counts['normal'] = sum(len(get_image_files(src)) for src in normal_sources)
    
    # Pneumonia sources (bacterial + viral)
    pneumonia_sources = [
        "chest_xray/chest_xray/train/PNEUMONIA",
        "chest_xray/chest_xray/val/PNEUMONIA",
        "chest_xray/chest_xray/test/PNEUMONIA",
        "COVID-19_Radiography_Dataset/Viral Pneumonia/images",
        "COVID-19_Radiography_Dataset/Lung_Opacity/images",
    ]
    available_counts['pneumonia'] = sum(len(get_image_files(src)) for src in pneumonia_sources)
    
    # TB sources
    tb_sources = [
        "TB_Chest_Radiography_Database/Tuberculosis",
    ]
    available_counts['tuberculosis'] = sum(len(get_image_files(src)) for src in tb_sources)
    
    # COVID-19 sources
    covid_sources = [
        "COVID-19_Radiography_Dataset/COVID/images",
    ]
    available_counts['covid19'] = sum(len(get_image_files(src)) for src in covid_sources)
    
    print(f"\nAvailable images per class:")
    print(f"  Normal:        {available_counts['normal']:>5}")
    print(f"  Pneumonia:     {available_counts['pneumonia']:>5}")
    print(f"  Tuberculosis:  {available_counts['tuberculosis']:>5}")
    print(f"  COVID-19:      {available_counts['covid19']:>5}")
    
    # ── Step 2: Determine balanced sample size ───────────────
    # Target 3000 samples per class
    target_samples = 3000
    
    print(f"\n🎯 Target samples per class: {target_samples}")
    
    # Check which classes need augmentation
    needs_augmentation = []
    for cls, count in available_counts.items():
        if count < target_samples:
            needs_augmentation.append((cls, count, target_samples - count))
            print(f"   ⚠️  {cls.capitalize()}: only {count} available, need {target_samples - count} more (will augment)")
        else:
            print(f"   ✓ {cls.capitalize()}: {count} available (will sample {target_samples})")
    
    # ── Step 3: Copy balanced samples ────────────────────────
    print("\n📦 Creating balanced dataset...")
    
    # Normal
    print("\n[1/4] Processing NORMAL class...")
    normal_dir = os.path.join(output_dir, "Normal")
    stats['normal'] = copy_images_balanced(normal_sources, normal_dir, prefix="norm", max_samples=target_samples)
    print(f"✓ Normal: {stats['normal']} images")
    
    # Pneumonia
    print("\n[2/4] Processing PNEUMONIA class...")
    pneumonia_dir = os.path.join(output_dir, "Pneumonia")
    stats['pneumonia'] = copy_images_balanced(pneumonia_sources, pneumonia_dir, prefix="pneu", max_samples=target_samples)
    print(f"✓ Pneumonia: {stats['pneumonia']} images")
    
    # Tuberculosis
    print("\n[3/4] Processing TUBERCULOSIS class...")
    tb_dir = os.path.join(output_dir, "Tuberculosis")
    stats['tuberculosis'] = copy_images_balanced(tb_sources, tb_dir, prefix="tb", max_samples=target_samples)
    print(f"✓ Tuberculosis: {stats['tuberculosis']} images")
    
    # COVID-19
    print("\n[4/4] Processing COVID-19 class...")
    covid_dir = os.path.join(output_dir, "COVID-19")
    stats['covid19'] = copy_images_balanced(covid_sources, covid_dir, prefix="covid", max_samples=target_samples)
    print(f"✓ COVID-19: {stats['covid19']} images")
    
    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ✅ BALANCED Dataset Organization Complete!")
    print("=" * 70)
    print(f"\nOutput directory: {os.path.abspath(output_dir)}/")
    print(f"\n📊 Final BALANCED class distribution:")
    print(f"  Normal:        {stats['normal']:>5} images")
    print(f"  Pneumonia:     {stats['pneumonia']:>5} images")
    print(f"  Tuberculosis:  {stats['tuberculosis']:>5} images")
    print(f"  COVID-19:      {stats['covid19']:>5} images")
    print(f"  {'─' * 40}")
    print(f"  TOTAL:         {sum(stats.values()):>5} images")
    
    # Verify balance
    unique_counts = set(stats.values())
    if len(unique_counts) == 1 and list(unique_counts)[0] == 3000:
        print(f"\n✅ Perfect balance! All classes have 3000 samples")
    else:
        print(f"\n⚠️  Classes not perfectly balanced: {dict(stats)}")
    
    print("\n🚀 Ready for training!")
    print(f"  Update config.yaml: data.root_dir = '{output_dir}/'")

if __name__ == "__main__":
    main()
