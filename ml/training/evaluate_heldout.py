"""
Held-out test evaluation for ClaimSight CV model — both 5-class checkpoints.

Respects the masked-loss design:
- Damage metrics ONLY on samples where has_damage_label == 1
- Severity metrics ONLY on samples where has_severity_label == 1

Outputs:
- Per-class precision, recall, F1, support for damage (multi-label)
- Macro precision, recall, F1 for damage
- Per-class accuracy, macro F1 for severity (single-label)
- Confusion matrices for both tasks
- Machine-readable JSON + human-readable report
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from ml.training.config import (
    DAMAGE_THRESHOLD,
    DAMAGE_TYPES,
    PROCESSED_DIR,
    SEVERITY_CLASSES,
)
from ml.training.dataset import CarDamageDataset, get_val_transform
from ml.training.model import DualHeadResNet50


def compute_damage_metrics(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict:
    """Multi-label damage metrics with per-class breakdown and confusion matrices."""
    binary_preds = (torch.sigmoid(preds) > threshold).float()
    y_true = targets.cpu().numpy()
    y_pred = binary_preds.cpu().numpy()

    # Per-class metrics
    per_class = {}
    for i, label in enumerate(DAMAGE_TYPES):
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average='binary', zero_division=0
        )
        # support is None with average='binary', compute manually
        support_val = int((y_true[:, i] == 1).sum())
        per_class[label] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "support": support_val,
        }

    # Macro averages
    macro_prec = np.mean([per_class[l]["precision"] for l in DAMAGE_TYPES])
    macro_rec = np.mean([per_class[l]["recall"] for l in DAMAGE_TYPES])
    macro_f1 = np.mean([per_class[l]["f1"] for l in DAMAGE_TYPES])

    # Confusion matrices per class (2x2 each)
    confusion = {}
    for i, label in enumerate(DAMAGE_TYPES):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1])
        confusion[label] = cm.tolist()  # [[TN, FP], [FN, TP]]

    return {
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion": confusion,
    }


def compute_severity_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    """Single-label severity metrics with per-class breakdown and confusion matrix."""
    predicted = preds.argmax(dim=-1).cpu().numpy()
    y_true = targets.cpu().numpy()

    # Overall accuracy
    accuracy = float((predicted == y_true).mean())

    # Per-class metrics
    per_class = {}
    for i, label in enumerate(SEVERITY_CLASSES):
        mask = (y_true == i)
        if mask.any():
            per_class[label] = {
                "accuracy": float((predicted[mask] == y_true[mask]).mean()),
                "support": int(mask.sum()),
            }
        else:
            per_class[label] = {"accuracy": 0.0, "support": 0}

    # Macro F1
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, predicted, average='macro', zero_division=0)
    macro_f1 = float(f1)

    # Confusion matrix (3x3)
    cm = confusion_matrix(y_true, predicted, labels=[0, 1, 2])
    confusion = cm.tolist()

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion": confusion,
    }


def evaluate_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    """Evaluate a single checkpoint on the held-out test set."""
    print(f"\nEvaluating: {checkpoint_path.name}")
    print(f"Device: {device}")

    # Load test data
    test_csv = PROCESSED_DIR / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"{test_csv} not found. Run dataset preparation first.")

    test_ds = CarDamageDataset(test_csv, transform=get_val_transform())
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    # Load model
    model = DualHeadResNet50(pretrained=False).to(device)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
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
        "checkpoint": str(checkpoint_path),
        "epoch": state.get("epoch"),
        "stage": state.get("stage"),
        "best_val_f1": state.get("best_val_f1"),
        "test_loss": total_loss / n_batches if n_batches > 0 else 0,
        "test_damage_loss": total_d_loss / n_batches if n_batches > 0 else 0,
        "test_severity_loss": total_s_loss / n_batches if n_batches > 0 else 0,
        "n_damage_samples": n_damage_samples,
        "n_severity_samples": n_severity_samples,
    }

    if all_damage_logits:
        damage_logits_cat = torch.cat(all_damage_logits, dim=0)
        damage_labels_cat = torch.cat(all_damage_labels, dim=0)
        damage_metrics = compute_damage_metrics(damage_logits_cat, damage_labels_cat, DAMAGE_THRESHOLD)
        results["damage_macro_precision"] = damage_metrics["macro_precision"]
        results["damage_macro_recall"] = damage_metrics["macro_recall"]
        results["damage_macro_f1"] = damage_metrics["macro_f1"]
        results["damage_per_class"] = damage_metrics["per_class"]
        results["damage_confusion"] = damage_metrics["confusion"]

    if all_severity_logits:
        severity_logits_cat = torch.cat(all_severity_logits, dim=0)
        severity_labels_cat = torch.cat(all_severity_labels, dim=0)
        severity_metrics = compute_severity_metrics(severity_logits_cat, severity_labels_cat)
        results["severity_accuracy"] = severity_metrics["accuracy"]
        results["severity_macro_f1"] = severity_metrics["macro_f1"]
        results["severity_per_class"] = severity_metrics["per_class"]
        results["severity_confusion"] = severity_metrics["confusion"]

    return results


def print_results(results: dict):
    """Print human-readable evaluation results."""
    print("\n" + "="*70)
    print(f"TEST SET EVALUATION — {results['checkpoint']}")
    print("="*70)
    print(f"Checkpoint epoch:     {results['epoch']} (stage {results['stage']})")
    print(f"Best val F1 (train):  {results['best_val_f1']:.4f}")
    print(f"Test Loss:            {results['test_loss']:.4f}")
    print(f"  Damage Loss:        {results['test_damage_loss']:.4f}")
    print(f"  Severity Loss:      {results['test_severity_loss']:.4f}")
    print(f"Damage samples:       {results['n_damage_samples']}")
    print(f"Severity samples:     {results['n_severity_samples']}")
    print()

    print(f"DAMAGE (multi-label, threshold={DAMAGE_THRESHOLD})")
    print(f"  Macro Precision:    {results.get('damage_macro_precision', 0):.4f}")
    print(f"  Macro Recall:       {results.get('damage_macro_recall', 0):.4f}")
    print(f"  Macro F1:           {results.get('damage_macro_f1', 0):.4f}")
    print()
    print("  Per-class:")
    for label, m in results.get("damage_per_class", {}).items():
        print(f"    {label:20s}: P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}  (support={m['support']})")
    
    print()
    print("  Damage Confusion Matrices (rows=true, cols=pred):")
    for label, cm in results.get("damage_confusion", {}).items():
        print(f"    {label}: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")

    print()
    print(f"SEVERITY (single-label)")
    print(f"  Accuracy:           {results.get('severity_accuracy', 0):.4f}")
    print(f"  Macro F1:           {results.get('severity_macro_f1', 0):.4f}")
    print()
    print("  Per-class:")
    for label, m in results.get("severity_per_class", {}).items():
        print(f"    {label:10s}: Acc={m['accuracy']:.4f}  (n={m['support']})")
    
    print()
    print("  Severity Confusion Matrix (rows=true, cols=pred):")
    for i, row in enumerate(results.get("severity_confusion", [])):
        true_label = SEVERITY_CLASSES[i]
        print(f"    {true_label:10s}: " + " ".join(f"{v:4d}" for v in row))

    print("="*70)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoints = [
        Path("ml/weights/claimsight_cv_v2_5class_stage1_epoch5.pt"),
        Path("ml/weights/claimsight_cv_v2_5class_latest.pt"),
    ]

    all_results = {}
    RESULTS_DIR = PROCESSED_DIR.parent / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for ckpt_path in checkpoints:
        if not ckpt_path.exists():
            print(f"Checkpoint not found: {ckpt_path}")
            continue

        results = evaluate_checkpoint(ckpt_path, device)
        all_results[ckpt_path.stem] = results
        print_results(results)

        # Save machine-readable JSON
        output_path = RESULTS_DIR / f"test_evaluation_{ckpt_path.stem}.json"
        with open(output_path, "w") as f:
            # Convert numpy types for JSON
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {output_path}")

    # Save combined comparison
    combined_path = RESULTS_DIR / "test_evaluation_comparison.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nCombined comparison saved to: {combined_path}")


if __name__ == "__main__":
    main()