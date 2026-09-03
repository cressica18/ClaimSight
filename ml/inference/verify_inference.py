"""
Inference verification for ClaimSight CV model.

Selects 3 deterministic test images (seed=42) and runs predictor on them.
"""

import random
from pathlib import Path

from ml.inference.predictor import VehicleDamagePredictor, get_predictor


def main():
    # Set seed for reproducibility
    random.seed(42)
    
    # Load test CSV
    test_csv_path = Path("/Users/dell/Documents/ClaimSight/ml/data/processed/test.csv")
    
    import csv
    test_samples = []
    with open(test_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_samples.append(row)
    
    # Select 3 deterministic samples
    selected = random.sample(test_samples, 3)
    
    # Initialize predictor
    predictor = get_predictor()
    
    print("="*70)
    print("INFERENCE VERIFICATION (seed=42)")
    print("="*70)
    
    for i, sample in enumerate(selected, 1):
        img_path = sample["image_path"]
        print(f"\n--- Sample {i} ---")
        print(f"Filename: {Path(img_path).name}")
        print(f"Full path: {img_path}")
        print(f"Ground truth damage: has_label={sample['has_damage_label']}")
        print(f"Ground truth severity: has_label={sample['has_severity_label']}, label={sample['severity_label']}")
        
        # Run prediction
        result = predictor.predict_from_path(img_path)
        
        print(f"\nPredicted damage types:")
        for d in result.damage_types:
            print(f"  - {d.label}: {d.confidence:.4f}")
        
        print(f"\nPredicted severity: {result.severity.label} (confidence: {result.severity.confidence:.4f})")
        print(f"Low confidence flag: {result.low_confidence}")
        
        if result.error:
            print(f"ERROR: {result.error}")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
