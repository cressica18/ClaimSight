"""
Training script for the ClaimSight CV model.

Two-stage training per blueprint Section 2.2:
  Stage 1: Freeze backbone, train heads only  (5 epochs, lr=1e-3)
  Stage 2: Unfreeze layer3+layer4, fine-tune (15 epochs, lr=1e-5)

Usage:
    cd /Users/dell/Documents/ClaimSight
    source backend/.venv/bin/activate
    python -m ml.training.train

Requires processed CSVs from:
    python -m ml.training.dataset --prepare
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ml.training.config import (
    CHECKPOINT_PATH,
    PROCESSED_DIR,
    RESULTS_DIR,
    STAGE1_BATCH,
    STAGE1_EPOCHS,
    STAGE1_LR,
    STAGE2_BATCH,
    STAGE2_EPOCHS,
    STAGE2_LR,
    RANDOM_SEED,
    WEIGHT_DECAY,
    WEIGHTS_DIR,
)
from ml.training.dataset import CarDamageDataset, get_train_transform, get_val_transform
from ml.training.model import DualHeadResNet50, build_model


# ─── Reproducibility ──────────────────────────────────────────────────────────

torch.manual_seed(RANDOM_SEED)


# ─── Metrics helpers ──────────────────────────────────────────────────────────

def compute_damage_f1(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Macro-F1 for multi-label damage prediction."""
    binary_preds = (torch.sigmoid(preds) > threshold).float()
    tp = (binary_preds * targets).sum(dim=0)
    fp = (binary_preds * (1 - targets)).sum(dim=0)
    fn = ((1 - binary_preds) * targets).sum(dim=0)

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1_per_class = 2 * precision * recall / (precision + recall + 1e-8)
    return f1_per_class.mean().item()


def compute_severity_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Accuracy for single-label severity classification."""
    predicted = preds.argmax(dim=-1)
    return (predicted == targets).float().mean().item()


# ─── Training loops ───────────────────────────────────────────────────────────

def run_epoch(
    model: DualHeadResNet50,
    loader: DataLoader,
    criterion_damage: nn.BCEWithLogitsLoss,
    criterion_severity: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool = True,
) -> dict:
    """Single training or validation epoch. Returns a dict of metrics."""
    model.train(train)

    total_loss = total_d_loss = total_s_loss = 0.0
    total_f1   = total_acc   = 0.0
    n_batches   = 0
    n_batches_d = 0
    n_batches_s = 0

    for images, damage_labels, severity_labels, masks in loader:
        images         = images.to(device)
        damage_labels  = damage_labels.to(device)
        severity_labels = severity_labels.to(device)
        masks          = masks.to(device)

        damage_logits, severity_logits = model(images)

        dmask = masks[:, 0].bool()
        smask = masks[:, 1].bool()

        loss_d = criterion_damage(damage_logits[dmask], damage_labels[dmask]) if dmask.any() else torch.tensor(0.0, device=device, requires_grad=True)
        loss_s = criterion_severity(severity_logits[smask], severity_labels[smask]) if smask.any() else torch.tensor(0.0, device=device, requires_grad=True)
        loss   = loss_d + loss_s

        if train and optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss   += loss.item()
        total_d_loss += loss_d.item()
        total_s_loss += loss_s.item()
        
        if dmask.any():
            total_f1 += compute_damage_f1(damage_logits[dmask].detach(), damage_labels[dmask])
            n_batches_d += 1
            
        if smask.any():
            total_acc += compute_severity_accuracy(severity_logits[smask].detach(), severity_labels[smask])
            n_batches_s += 1
            
        n_batches    += 1

    if n_batches == 0:
        return {"loss": 0, "damage_loss": 0, "severity_loss": 0, "damage_f1": 0, "severity_acc": 0}

    return {
        "loss":         total_loss   / n_batches,
        "damage_loss":  total_d_loss / n_batches,
        "severity_loss":total_s_loss / n_batches,
        "damage_f1":    total_f1     / n_batches_d if n_batches_d > 0 else 0.0,
        "severity_acc": total_acc    / n_batches_s if n_batches_s > 0 else 0.0,
    }


# ─── Checkpoint ───────────────────────────────────────────────────────────────

def save_checkpoint(model: DualHeadResNet50, metrics: dict, epoch: int) -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":             epoch,
            "model_state_dict":  model.state_dict(),
            "metrics":           metrics,
        },
        CHECKPOINT_PATH,
    )
    print(f"  ✓ Checkpoint saved → {CHECKPOINT_PATH}")


# ─── Main training routine ────────────────────────────────────────────────────

def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_csv = PROCESSED_DIR / "train.csv"
    val_csv   = PROCESSED_DIR / "val.csv"

    if not train_csv.exists():
        raise FileNotFoundError(
            f"{train_csv} not found.\n"
            "Run: python -m ml.training.dataset --prepare\n"
            "after placing the dataset in ml/data/raw/"
        )

    train_ds = CarDamageDataset(train_csv, transform=get_train_transform())
    val_ds   = CarDamageDataset(val_csv,   transform=get_val_transform())

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(pretrained=True).to(device)

    criterion_damage   = nn.BCEWithLogitsLoss()
    criterion_severity = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    history: list[dict] = []

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 1: train heads only
    # ══════════════════════════════════════════════════════════════════════════
    print("\n═══ Stage 1: Training heads only ═══")
    optimizer1 = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=STAGE1_LR, weight_decay=WEIGHT_DECAY
    )
    scheduler1 = CosineAnnealingLR(optimizer1, T_max=STAGE1_EPOCHS)

    loader1_train = DataLoader(train_ds, batch_size=STAGE1_BATCH, shuffle=True,  num_workers=0, pin_memory=False)
    loader1_val   = DataLoader(val_ds,   batch_size=STAGE1_BATCH, shuffle=False, num_workers=0, pin_memory=False)

    for epoch in range(1, STAGE1_EPOCHS + 1):
        t0 = time.time()
        train_m = run_epoch(model, loader1_train, criterion_damage, criterion_severity, optimizer1, device, train=True)
        val_m   = run_epoch(model, loader1_val,   criterion_damage, criterion_severity, None,       device, train=False)
        scheduler1.step()

        row = {"stage": 1, "epoch": epoch, "train": train_m, "val": val_m}
        history.append(row)

        print(
            f"  S1 E{epoch:02d} | "
            f"train loss={train_m['loss']:.4f} f1={train_m['damage_f1']:.4f} sev_acc={train_m['severity_acc']:.4f} | "
            f"val   loss={val_m['loss']:.4f}  f1={val_m['damage_f1']:.4f}  sev_acc={val_m['severity_acc']:.4f} | "
            f"{time.time()-t0:.1f}s"
        )

        if val_m["damage_f1"] > best_val_f1:
            best_val_f1 = val_m["damage_f1"]
            save_checkpoint(model, val_m, epoch)

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 2: unfreeze layer3+layer4, fine-tune
    # ══════════════════════════════════════════════════════════════════════════
    print("\n═══ Stage 2: Fine-tuning layer3+layer4 ═══")
    model.unfreeze_stage2()
    optimizer2 = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=STAGE2_LR, weight_decay=WEIGHT_DECAY
    )
    scheduler2 = CosineAnnealingLR(optimizer2, T_max=STAGE2_EPOCHS)

    loader2_train = DataLoader(train_ds, batch_size=STAGE2_BATCH, shuffle=True,  num_workers=0, pin_memory=False)
    loader2_val   = DataLoader(val_ds,   batch_size=STAGE2_BATCH, shuffle=False, num_workers=0, pin_memory=False)

    for epoch in range(1, STAGE2_EPOCHS + 1):
        t0 = time.time()
        train_m = run_epoch(model, loader2_train, criterion_damage, criterion_severity, optimizer2, device, train=True)
        val_m   = run_epoch(model, loader2_val,   criterion_damage, criterion_severity, None,       device, train=False)
        scheduler2.step()

        row = {"stage": 2, "epoch": epoch, "train": train_m, "val": val_m}
        history.append(row)

        print(
            f"  S2 E{epoch:02d} | "
            f"train loss={train_m['loss']:.4f} f1={train_m['damage_f1']:.4f} sev_acc={train_m['severity_acc']:.4f} | "
            f"val   loss={val_m['loss']:.4f}  f1={val_m['damage_f1']:.4f}  sev_acc={val_m['severity_acc']:.4f} | "
            f"{time.time()-t0:.1f}s"
        )

        if val_m["damage_f1"] > best_val_f1:
            best_val_f1 = val_m["damage_f1"]
            save_checkpoint(model, val_m, epoch)

    # ── Save training history ─────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val damage F1: {best_val_f1:.4f}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()
