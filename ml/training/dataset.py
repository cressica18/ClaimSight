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
# 
# Canonical damage classes: scratch, dent, shattered_glass, 
#                           headlight_damage, no_damage
# Canonical severity classes: minor, moderate, severe
LABEL_REMAP: dict[str, str] = {
    # ─── Severity dataset (car-damage-severity-dataset) ───
    "01-minor":          "minor",
    "02-moderate":       "moderate",
    "03-severe":         "severe",

    # ─── Damage dataset (car-damage-assessment) ───
    # These are the actual folder/class names from the dataset
    "door_dent":         "dent",
    "bumper_scratch":    "scratch",
    "door_scratch":      "scratch",
    "glass_shatter":     "shattered_glass",
    "tail_lamp":         "headlight_damage",
    "head_lamp":         "headlight_damage",
    "bumper_dent":       "dent",
    "unknown":           "no_damage",

    # ─── Existing mappings (kept for compatibility) ───
    "01-minor":          "minor",
    "02-moderate":       "moderate",
    "03-severe":         "severe",
    "scratch":           "scratch",
    "dent":              "dent",
    "broken_windshield": "shattered_glass",
    "shattered_glass":   "shattered_glass",
    "headlight_damage":  "headlight_damage",
    "headlight":         "headlight_damage",
    "no_damage":         "no_damage",
    "whole":             "no_damage",
    "normal":            "no_damage",
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
                "_raw_class": raw_class,  # for validation
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
            "_raw_class": folder_name,  # for validation
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

    # Validate label mapping before splitting
    _validate_label_mapping(records)

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
        # Remove _raw_class field before writing (it's only for validation)
        clean_records = [{k: v for k, v in r.items() if k != "_raw_class"} for r in split_records]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(clean_records)

    counts = {k: len(v) for k, v in splits.items()}
    print(f"Dataset prepared: {counts}")
    return counts


def _validate_label_mapping(records: list[dict]) -> None:
    """
    Validate that label mapping is correct and no silent label collapse occurs.
    
    Checks:
    1. Every record with has_damage_label=1 maps to a valid canonical damage class (not no_damage unless intentional)
    2. No canonical damage class that should have data ends up with zero samples (warning only)
    3. Unexpected raw classes are reported
    4. Label IDs remain consistent with DAMAGE_TYPES and SEVERITY_CLASSES ordering
    """
    from collections import Counter
    
    damage_counter = Counter()
    severity_counter = Counter()
    raw_classes_seen = set()
    no_damage_from_known = 0
    
    for r in records:
        raw_classes_seen.add(r.get("_raw_class", "unknown"))
        
        if r.get("has_damage_label") == 1:
            # Find which damage class is set to 1
            damage_class = None
            for d in DAMAGE_TYPES:
                if r.get(d, 0) == 1:
                    damage_class = d
                    break
            if damage_class:
                damage_counter[damage_class] += 1
                if damage_class == "no_damage" and r.get("_raw_class") not in ("unknown", "no_damage", "whole", "normal"):
                    no_damage_from_known += 1
        
        if r.get("has_severity_label") == 1:
            sev_idx = r.get("severity_label", -1)
            if 0 <= sev_idx < len(SEVERITY_CLASSES):
                severity_counter[SEVERITY_CLASSES[sev_idx]] += 1
    
    # Check 1: No known damage classes silently mapped to no_damage
    if no_damage_from_known > 0:
        raise ValueError(
            f"Label mapping error: {no_damage_from_known} samples from known damage classes "
            f"were mapped to 'no_damage'. Check LABEL_REMAP for missing mappings."
        )
    
    # Check 2: Report class distribution
    print(f"\nDamage class distribution (all splits):")
    for cls in DAMAGE_TYPES:
        count = damage_counter.get(cls, 0)
        print(f"  {cls}: {count}")
    
    print(f"\nSeverity class distribution (all splits):")
    for cls in SEVERITY_CLASSES:
        count = severity_counter.get(cls, 0)
        print(f"  {cls}: {count}")
    
    # Check 3: Warn about canonical classes with zero samples (but don't fail - some classes may not exist in dataset)
    zero_damage = [cls for cls in DAMAGE_TYPES if damage_counter.get(cls, 0) == 0 and cls != "no_damage"]
    if zero_damage:
        print(f"\nWARNING: The following damage classes have ZERO samples: {zero_damage}")
        print("  These classes cannot be learned. Consider adding data or removing from DAMAGE_TYPES.")
    
    zero_severity = [cls for cls in SEVERITY_CLASSES if severity_counter.get(cls, 0) == 0]
    if zero_severity:
        print(f"WARNING: The following severity classes have ZERO samples: {zero_severity}")
    
    # Check 4: Validate that all raw classes seen are in LABEL_REMAP (or are intentionally unmapped)
    # We track raw classes in build_records functions by adding _raw_class field
    # This is a soft check - just warn about unexpected classes
    expected_raw_classes = set(LABEL_REMAP.keys())
    unexpected = raw_classes_seen - expected_raw_classes
    if unexpected:
        print(f"\nWARNING: Unexpected raw classes not in LABEL_REMAP: {unexpected}")
        print("  These may be silently mapped to defaults. Add to LABEL_REMAP if needed.")
    
print()  # spacing
 
 
# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from ml.training.config import RAW_DATA_DIR, PROCESSED_DIR

    if "--prepare" in sys.argv:
        prepare_splits(RAW_DATA_DIR, PROCESSED_DIR)
    else:
        print("Usage: python -m ml.training.dataset --prepare")
