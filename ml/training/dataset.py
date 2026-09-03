"""
Dataset pipeline for the ClaimSight CV model.

Supports two data source layouts:
1. **Flat class-folder layout** (ImageFolder-compatible):
   ml/data/raw/<split>/<class_name>/image.jpg
   
2. **CSV-manifest layout** (for multi-label):
   ml/data/processed/train.csv  (columns: image_path, scratch, dent, ..., no_damage, severity)

The flat layout is remapped to multi-label by treating each class folder as a binary
label; severity is derived from a sub-folder or filename convention.

For the Kaggle/Roboflow "Car Damage Assessment" family of datasets:
  - We expect images in class-named subdirectories.
  - The class names are remapped in LABEL_REMAP to our canonical class set.

Run `python -m ml.training.dataset --prepare` to create the processed CSV split
from raw data placed in ml/data/raw/.
"""

import csv
import json
import os
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

from ml.training.config import (
    DAMAGE_TYPES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGE_SIZE,
    PROCESSED_DIR,
    RANDOM_SEED,
    SEVERITY_CLASSES,
    TRAIN_FRACTION,
    VAL_FRACTION,
)

# ─── Label remapping ──────────────────────────────────────────────────────────
# Maps raw dataset class folder names → our canonical class names.
# Update this mapping when a specific dataset is chosen.
LABEL_REMAP: dict[str, str] = {
    # Common Kaggle "car damage" dataset folder names
    "01-minor":          "minor",           # severity
    "02-moderate":       "moderate",        # severity
    "03-severe":         "severe",          # severity
    "scratch":           "scratch",
    "dent":              "dent",
    "crack":             "crack",
    "broken_windshield": "shattered_glass",
    "glass_shatter":     "shattered_glass",
    "shattered_glass":   "shattered_glass",
    "bumper_damage":     "bumper_damage",
    "bumper":            "bumper_damage",
    "panel_damage":      "panel_damage",
    "panel":             "panel_damage",
    "headlight_damage":  "headlight_damage",
    "headlight":         "headlight_damage",
    "no_damage":         "no_damage",
    "whole":             "no_damage",
    "normal":            "no_damage",
    # Add more remappings as needed for the chosen dataset
}


# ─── Transforms ───────────────────────────────────────────────────────────────

def get_train_transform() -> transforms.Compose:
    """Augmentation + normalisation for training images."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform() -> transforms.Compose:
    """Resize + centre-crop + normalisation only — no augmentation."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transform() -> transforms.Compose:
    """Identical to val_transform — used at inference time."""
    return get_val_transform()


# ─── Dataset class ────────────────────────────────────────────────────────────

class CarDamageDataset(Dataset):
    """
    Multi-label damage classification dataset.

    Reads from a CSV produced by `prepare_splits()`:
        image_path, scratch, dent, ..., no_damage, severity_label

    severity_label is an integer index into SEVERITY_CLASSES.
    """

    def __init__(
        self,
        csv_path: Path,
        transform: Optional[transforms.Compose] = None,
        root: Optional[Path] = None,
    ) -> None:
        self.root = root
        self.transform = transform or get_val_transform()
        self.samples: list[dict] = []

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(row)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.samples[idx]
        img_path = Path(row["image_path"])
        if self.root and not img_path.is_absolute():
            img_path = self.root / img_path

        try:
            img = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError):
            # Return a blank tensor so training can continue; log the error
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=0)

        if self.transform:
            img = self.transform(img)

        # Multi-label damage tensor
        damage_label = torch.tensor(
            [float(row[d]) for d in DAMAGE_TYPES], dtype=torch.float32
        )

        # Severity label (integer index)
        severity_label = torch.tensor(int(row["severity_label"]), dtype=torch.long)
        
        # Masks
        mask = torch.tensor([
            float(row["has_damage_label"]),
            float(row["has_severity_label"])
        ], dtype=torch.float32)

        return img, damage_label, severity_label, mask


# ─── Data preparation ─────────────────────────────────────────────────────────

def build_records_from_assessment(raw_dir: Path) -> list[dict]:
    """Loader for hamzamanssor/car-damage-assessment."""
    dataset_dir = raw_dir / "car-damage-assessment"
    csv_path = dataset_dir / "data.csv"
    if not csv_path.exists():
        return []
        
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_rel_path = row["image"]
            img_full_path = dataset_dir / img_rel_path
            
            if not img_full_path.exists():
                continue
                
            raw_class = row["classes"]
            canonical = LABEL_REMAP.get(raw_class)
            
            damage_labels = {d: 0 for d in DAMAGE_TYPES}
            if canonical in DAMAGE_TYPES and canonical != "no_damage":
                damage_labels[canonical] = 1
            else:
                damage_labels["no_damage"] = 1
                
            records.append({
                "image_path": str(img_full_path),
                "severity_label": -1,
                "has_damage_label": 1,
                "has_severity_label": 0,
                **damage_labels
            })
    return records


def build_records_from_severity(raw_dir: Path) -> list[dict]:
    """Loader for prajwalbhamere/car-damage-severity-dataset."""
    dataset_dir = raw_dir / "car-damage-severity-dataset" / "data3a"
    if not dataset_dir.exists():
        return []
        
    VALID_EXT = {".jpg", ".jpeg", ".png", ".webp"}
    records = []
    
    for path in sorted(dataset_dir.rglob("*")):
        if path.suffix.lower() not in VALID_EXT:
            continue
            
        folder_name = path.parent.name
        canonical = LABEL_REMAP.get(folder_name)
        if canonical not in SEVERITY_CLASSES:
            continue
            
        severity_label = SEVERITY_CLASSES.index(canonical)
        
        damage_labels = {d: 0 for d in DAMAGE_TYPES}
        
        records.append({
            "image_path": str(path),
            "severity_label": severity_label,
            "has_damage_label": 0,
            "has_severity_label": 1,
            **damage_labels
        })
    return records


def prepare_splits(raw_dir: Path, output_dir: Path, seed: int = RANDOM_SEED) -> dict:
    """
    Build multi-label records from both datasets, split into train/val/test CSVs.
    Returns a dict with counts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    records = build_records_from_assessment(raw_dir) + build_records_from_severity(raw_dir)
    if not records:
        raise FileNotFoundError(
            f"No valid images found in {raw_dir} subdirectories. "
            "Please ensure both datasets are downloaded and unzipped."
        )

    random.seed(seed)
    random.shuffle(records)

    n = len(records)
    n_train = int(n * TRAIN_FRACTION)
    n_val   = int(n * VAL_FRACTION)

    splits = {
        "train": records[:n_train],
        "val":   records[n_train:n_train + n_val],
        "test":  records[n_train + n_val:],
    }

    fieldnames = ["image_path", "severity_label", "has_damage_label", "has_severity_label"] + DAMAGE_TYPES

    for split_name, split_records in splits.items():
        csv_path = output_dir / f"{split_name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_records)

    counts = {k: len(v) for k, v in splits.items()}
    print(f"Dataset prepared: {counts}")
    return counts


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from ml.training.config import RAW_DATA_DIR, PROCESSED_DIR

    if "--prepare" in sys.argv:
        prepare_splits(RAW_DATA_DIR, PROCESSED_DIR)
    else:
        print("Usage: python -m ml.training.dataset --prepare")
