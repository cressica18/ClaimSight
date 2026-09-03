"""
Test-set evaluation for ClaimSight CV model.

Respects the masked-loss design:
- Damage metrics ONLY on samples where has_damage_label == 1
- Severity metrics ONLY on samples where has_severity_label == 1
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.training.config import (
    CHECKPOINT_PATH,
    DAMAGE_THRESHOLD,
    DAMAGE_TYPES,
    PROCESSED_DIR,
    SEVERITY_CLASSES,
)
from ml.training.dataset import CarDamageDataset, get_val_transform
from ml.training.model import DualHeadResNet50, load_checkpoint


def compute_damage_f1(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict:
    """Macro-F1 for multi-label damage prediction with per-class breakdown."""
    binary_preds = (torch.sigmoid(preds) > threshold).float()
    tp = (binary_preds * targets).sum(dim=0)
    fp = (binary_preds * (1 - targets)).sum(dim=0)
    fn = ((1 - binary_preds) * targets).sum(dim=0)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1_per_class = 2 * precision * recall / (precision + recall + 1e-8)
    
    per_class = {}
    for i, label in enumerate(DAMAGE_TYPES):
        per_class[label] = {
            "f1": f1_per_class[i].item(),
            "precision": precision[i].item(),
            "recall": recall[i].item(),
            "tp": int(tp[i].item()),
            "fp": int(fp[i].item()),
            "fn": int(fn[i].item()),
        }
    
    return {
        "macro_f1": f1_per_class.mean().item(),
        "per_class": per_class
    }


def compute_severity_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    """Accuracy and per-class metrics for single-label severity classification."""
    predicted = preds.argmax(dim=-1)
    correct = (predicted == targets).float()
    
    per_class = {}
    for i, label in enumerate(SEVERITY_CLASSES):
        mask = (targets == i)
        if mask.any():
            per_class[label] = {
                "accuracy": correct[mask].mean().item(),
                "count": int(mask.sum().item()),
            }
        else:
            per_class[label] = {"accuracy": 0.0, "count": 0}
    
    return {
        "accuracy": correct.mean().item(),
        "per_class": per_class
    }


def evaluate() -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    # Load test data
    test_csv = PROCESSED_DIR / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"{test_csv} not found. Run dataset preparation first.")

    test_ds = CarDamageDataset(test_csv, transform=get_val_transform())
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    # Load model
    model = load_checkpoint(CHECKPOINT_PATH).to(device)
    model.eval()

    criterion_damage = nn.BCEWithLogitsLoss()
    criterion_severity = nn.CrossEntropyLoss()

    # Accumulators
    all_damage_logits = []
    all_damage_labels = []
    all_severity_logits = []
    all_severity_labels = []
    total_loss = total_d_loss = total_s_loss = 0.0
    n_batches = 0
    n_damage_samples = 0
    n_severity_samples = 0

    with torch.no_grad():
        for images, damage_labels, severity_labels, masks in test_loader:
            images = images.to(device)
            damage_labels = damage_labels.to(device)
            severity_labels = severity_labels.to(device)
            masks = masks.to(device)

            damage_logits, severity_logits = model(images)

            dmask = masks[:, 0].bool()
            smask = masks[:, 1].bool()

            # Loss (only on valid samples)
            loss_d = criterion_damage(damage_logits[dmask], damage_labels[dmask]) if dmask.any() else torch.tensor(0.0, device=device)
            loss_s = criterion_severity(severity_logits[smask], severity_labels[smask]) if smask.any() else torch.tensor(0.0, device=device)
            loss = loss_d + loss_s

            total_loss += loss.item()
            total_d_loss += loss_d.item()
            total_s_loss += loss_s.item()
            n_batches += 1

            # Accumulate for metrics (only masked samples)
            if dmask.any():
                all_damage_logits.append(damage_logits[dmask].cpu())
                all_damage_labels.append(damage_labels[dmask].cpu())
                n_damage_samples += dmask.sum().item()

            if smask.any():
                all_severity_logits.append(severity_logits[smask].cpu())
                all_severity_labels.append(severity_labels[smask].cpu())
                n_severity_samples += smask.sum().item()

    # Compute metrics
    results = {
        "test_loss": total_loss / n_batches if n_batches > 0 else 0,
        "test_damage_loss": total_d_loss / n_batches if n_batches > 0 else 0,
        "test_severity_loss": total_s_loss / n_batches if n_batches > 0 else 0,
        "n_damage_samples": n_damage_samples,
        "n_severity_samples": n_severity_samples,
    }

    if all_damage_logits:
        damage_logits_cat = torch.cat(all_damage_logits, dim=0)
        damage_labels_cat = torch.cat(all_damage_labels, dim=0)
        damage_metrics = compute_damage_f1(damage_logits_cat, damage_labels_cat, DAMAGE_THRESHOLD)
        results["damage_macro_f1"] = damage_metrics["macro_f1"]
        results["damage_per_class"] = damage_metrics["per_class"]

    if all_severity_logits:
        severity_logits_cat = torch.cat(all_severity_logits, dim=0)
        severity_labels_cat = torch.cat(all_severity_labels, dim=0)
        severity_metrics = compute_severity_metrics(severity_logits_cat, severity_labels_cat)
        results["severity_accuracy"] = severity_metrics["accuracy"]
        results["severity_per_class"] = severity_metrics["per_class"]

    return results


if __name__ == "__main__":
    results = evaluate()
    
    print("\n" + "="*60)
    print("TEST SET EVALUATION RESULTS")
    print("="*60)
    print(f"Test Loss:              {results['test_loss']:.4f}")
    print(f"  Damage Loss:          {results['test_damage_loss']:.4f}")
    print(f"  Severity Loss:        {results['test_severity_loss']:.4f}")
    print(f"Damage samples evaluated: {results['n_damage_samples']}")
    print(f"Severity samples evaluated: {results['n_severity_samples']}")
    print()
    print(f"DAMAGE MACRO-F1:        {results.get('damage_macro_f1', 0):.4f}")
    print(f"SEVERITY ACCURACY:      {results.get('severity_accuracy', 0):.4f}")
    print()
    
    if "damage_per_class" in results:
        print("Per-class Damage F1:")
        for label, metrics in results["damage_per_class"].items():
            print(f"  {label:20s}: F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  (TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']})")
    
    if "severity_per_class" in results:
        print()
        print("Per-class Severity Accuracy:")
        for label, metrics in results["severity_per_class"].items():
            print(f"  {label:10s}: Acc={metrics['accuracy']:.4f}  (n={metrics['count']})")

    # Save results
    output_path = PROCESSED_DIR.parent / "results" / "test_evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
