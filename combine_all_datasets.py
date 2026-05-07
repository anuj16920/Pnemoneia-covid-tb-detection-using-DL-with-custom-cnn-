#!/usr/bin/env python3
"""
Combine ALL Dataset Images - Complete Copy (No Deletion)
Copies all images from COVID-19 and TB datasets into organized structure
Output: combined_dataset/Normal, combined_dataset/Pneumonia, combined_dataset/Tuberculosis, combined_dataset/COVID-19
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict

def count_files(directory):
    """Count image files in directory"""
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

def copy_images(src_dir, dst_dir, prefix=""):
    """Copy all images from src to dst with optional prefix"""
    if not os.path.exists(src_dir):
        print(f"   ⚠️  Source not found: {src_dir}")
        return 0
    
    os.makedirs(dst_dir, exist_ok=True)
    
    count = 0
    for img_file in os.listdir(src_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            src_path = os.path.join(src_dir, img_file)
            
            # Add prefix to avoid name conflicts
            if prefix:
                dst_filename = f"{prefix}_{img_file}"
            else:
                dst_filename = img_file
            
            dst_path = os.path.join(dst_dir, dst_filename)
            
            # Handle duplicate names
            if os.path.exists(dst_path):
                base, ext = os.path.splitext(dst_filename)
                counter = 1
                while os.path.exists(dst_path):
                    dst_filename = f"{base}_dup{counter}{ext}"
                    dst_path = os.path.join(dst_dir, dst_filename)
                    counter += 1
            
            shutil.copy2(src_path, dst_path)
            count += 1
    
    return count

def main():
    print("=" * 80)
    print("  COMBINING ALL DATASET IMAGES - Complete Copy")
    print("=" * 80)
    
    # Output directory
    output_dir = "combined_dataset"
    
    # Create output directory
    if os.path.exists(output_dir):
        print(f"\n⚠️  Output directory '{output_dir}' already exists!")
        response = input("Do you want to overwrite it? (yes/no): ").strip().lower()
        if response != 'yes':
            print("❌ Operation cancelled.")
            return
        print(f"🗑️  Removing old {output_dir}/...")
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    stats = defaultdict(int)
    
    print("\n" + "=" * 80)
    print("  STEP 1: Copying NORMAL Images")
    print("=" * 80)
    
    normal_dir = os.path.join(output_dir, "Normal")
    
    # TB Normal
    print("\n[1/2] TB Dataset - Normal...")
    src = "TB_Chest_Radiography_Database/Normal"
    available = count_files(src)
    print(f"   Found: {available} images")
    copied = copy_images(src, normal_dir, prefix="TB_Normal")
    stats['normal'] += copied
    print(f"   ✓ Copied: {copied} images")
    
    # COVID-19 Normal
    print("\n[2/2] COVID-19 Dataset - Normal...")
    src = "COVID-19_Radiography_Dataset/Normal/images"
    available = count_files(src)
    print(f"   Found: {available} images")
    copied = copy_images(src, normal_dir, prefix="COVID_Normal")
    stats['normal'] += copied
    print(f"   ✓ Copied: {copied} images")
    
    print(f"\n✅ Total Normal: {stats['normal']} images")
    
    print("\n" + "=" * 80)
    print("  STEP 2: Copying PNEUMONIA Images")
    print("=" * 80)
    
    pneumonia_dir = os.path.join(output_dir, "Pneumonia")
    
    # COVID-19 Viral Pneumonia
    print("\n[1/2] COVID-19 Dataset - Viral Pneumonia...")
    src = "COVID-19_Radiography_Dataset/Viral Pneumonia/images"
    available = count_files(src)
    print(f"   Found: {available} images")
    copied = copy_images(src, pneumonia_dir, prefix="COVID_ViralPneumonia")
    stats['pneumonia'] += copied
    print(f"   ✓ Copied: {copied} images")
    
    # COVID-19 Lung Opacity
    print("\n[2/2] COVID-19 Dataset - Lung Opacity...")
    src = "COVID-19_Radiography_Dataset/Lung_Opacity/images"
    available = count_files(src)
    print(f"   Found: {available} images")
    copied = copy_images(src, pneumonia_dir, prefix="COVID_LungOpacity")
    stats['pneumonia'] += copied
    print(f"   ✓ Copied: {copied} images")
    
    print(f"\n✅ Total Pneumonia: {stats['pneumonia']} images")
    
    print("\n" + "=" * 80)
    print("  STEP 3: Copying TUBERCULOSIS Images")
    print("=" * 80)
    
    tb_dir = os.path.join(output_dir, "Tuberculosis")
    
    # TB Dataset
    print("\n[1/1] TB Dataset - Tuberculosis...")
    src = "TB_Chest_Radiography_Database/Tuberculosis"
    available = count_files(src)
    print(f"   Found: {available} images")
    copied = copy_images(src, tb_dir, prefix="TB")
    stats['tuberculosis'] += copied
    print(f"   ✓ Copied: {copied} images")
    
    print(f"\n✅ Total Tuberculosis: {stats['tuberculosis']} images")
    
    print("\n" + "=" * 80)
    print("  STEP 4: Copying COVID-19 Images")
    print("=" * 80)
    
    covid_dir = os.path.join(output_dir, "COVID-19")
    
    # COVID-19 Dataset
    print("\n[1/1] COVID-19 Dataset - COVID-19...")
    src = "COVID-19_Radiography_Dataset/COVID/images"
    available = count_files(src)
    print(f"   Found: {available} images")
    copied = copy_images(src, covid_dir, prefix="COVID")
    stats['covid19'] += copied
    print(f"   ✓ Copied: {copied} images")
    
    print(f"\n✅ Total COVID-19: {stats['covid19']} images")
    
    # ── Final Summary ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  ✅ ALL DATASETS COMBINED SUCCESSFULLY!")
    print("=" * 80)
    
    print(f"\n📁 Output directory: {os.path.abspath(output_dir)}/")
    print(f"\n📊 Complete Dataset Statistics:")
    print(f"  ├─ Normal:        {stats['normal']:>6} images")
    print(f"  ├─ Pneumonia:     {stats['pneumonia']:>6} images")
    print(f"  ├─ Tuberculosis:  {stats['tuberculosis']:>6} images")
    print(f"  └─ COVID-19:      {stats['covid19']:>6} images")
    print(f"  {'─' * 50}")
    print(f"  TOTAL:            {sum(stats.values()):>6} images")
    
    # Class distribution analysis
    total = sum(stats.values())
    print(f"\n📈 Class Distribution:")
    for cls, count in stats.items():
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  {cls.capitalize():15} {count:>6} images ({percentage:>5.2f}%)")
    
    # Check for imbalance
    max_count = max(stats.values())
    min_count = min(stats.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"\n⚖️  Class Balance Analysis:")
    print(f"  Largest class:  {max_count} images")
    print(f"  Smallest class: {min_count} images")
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}x")
    
    if imbalance_ratio > 2.0:
        print(f"\n  ⚠️  WARNING: Significant class imbalance detected!")
        print(f"  Consider using:")
        print(f"    - Class weights during training")
        print(f"    - Data augmentation for smaller classes")
        print(f"    - Balanced sampling strategy")
    else:
        print(f"\n  ✓ Classes are reasonably balanced")
    
    print(f"\n🚀 Dataset ready for training!")
    print(f"  Original datasets remain untouched")
    print(f"  All images copied to: {output_dir}/")

if __name__ == "__main__":
    main()
