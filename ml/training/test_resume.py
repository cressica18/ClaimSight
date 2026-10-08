#!/usr/bin/env python3
"""
Focused resume smoke test for ClaimSight CV model.
Verifies that the existing Epoch 3 checkpoint can be loaded and resume works correctly.
"""

import torch
from pathlib import Path

from ml.training.config import CHECKPOINT_PATH_5CLASS
from ml.training.model import build_model
from ml.training.train import load_checkpoint_resume


def test_resume_smoke():
    """Test that checkpoint loads and resume logic works."""
    print("=" * 60)
    print("RESUME SMOKE TEST")
    print("=" * 60)
    
    # 1. Verify checkpoint exists
    assert CHECKPOINT_PATH_5CLASS.exists(), f"Checkpoint not found: {CHECKPOINT_PATH_5CLASS}"
    print(f"✓ Checkpoint exists: {CHECKPOINT_PATH_5CLASS}")
    print(f"  Size: {CHECKPOINT_PATH_5CLASS.stat().st_size / 1e6:.1f} MB")
    
    # 2. Load checkpoint and verify contents
    state = torch.load(CHECKPOINT_PATH_5CLASS, map_location="cpu")
    print(f"\n✓ Checkpoint loaded successfully")
    print(f"  Keys: {list(state.keys())}")
    print(f"  Epoch: {state['epoch']}")
    print(f"  Metrics: {state['metrics']}")
    
    # 3. Verify model architecture matches checkpoint
    model = build_model(pretrained=False)
    model.load_state_dict(state["model_state_dict"])
    print(f"\n✓ Model state_dict loaded successfully")
    
    # 4. Verify output dimensions
    damage_head = model.head_damage
    severity_head = model.head_severity
    
    # Check final layer output dimensions
    damage_out = damage_head[-1].out_features  # head_damage.3
    severity_out = severity_head[-1].out_features  # head_severity.3
    
    print(f"  Damage output dim: {damage_out} (expected 5)")
    print(f"  Severity output dim: {severity_out} (expected 3)")
    
    assert damage_out == 5, f"Damage output dim mismatch: {damage_out} != 5"
    assert severity_out == 3, f"Severity output dim mismatch: {severity_out} != 3"
    print(f"  ✓ Output dimensions correct")
    
    # 5. Test load_checkpoint_resume function
    print(f"\n--- Testing load_checkpoint_resume() ---")
    model2 = build_model(pretrained=False)
    optimizer = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    
    completed_epoch, loaded_stage, best_val_f1, history = load_checkpoint_resume(
        model2, optimizer, scheduler, str(CHECKPOINT_PATH_5CLASS)
    )
    
    print(f"  completed_epoch: {completed_epoch}")
    print(f"  loaded_stage: {loaded_stage}")
    print(f"  best_val_f1: {best_val_f1:.4f}")
    print(f"  history length: {len(history)}")
    
    # 6. Verify resume starts from AFTER Epoch 3
    assert completed_epoch == 3, f"Expected epoch 3, got {completed_epoch}"
    assert loaded_stage == 1, f"Expected stage 1, got {loaded_stage}"
    
    # Next epoch should be 4 (after epoch 3)
    next_epoch = completed_epoch + 1
    print(f"  Next epoch to run: {next_epoch} (Stage {loaded_stage})")
    assert next_epoch == 4, f"Expected next epoch 4, got {next_epoch}"
    print(f"  ✓ Resume will start from Epoch 4 (Stage 1)")
    
    # 7. Verify optimizer/scheduler state is NOT in checkpoint (fresh init)
    has_opt_state = "optimizer_state_dict" in state
    has_sched_state = "scheduler_state_dict" in state
    print(f"\n  Optimizer state in checkpoint: {has_opt_state}")
    print(f"  Scheduler state in checkpoint: {has_sched_state}")
    assert not has_opt_state, "Optimizer state should NOT be in checkpoint (fresh init expected)"
    assert not has_sched_state, "Scheduler state should NOT be in checkpoint (fresh init expected)"
    print(f"  ✓ Optimizer/scheduler will be freshly initialized (as expected)")
    
    # 8. Verify v1 checkpoint is untouched
    v1_path = CHECKPOINT_PATH_5CLASS.parent / "claimsight_cv_v1.pt"
    assert v1_path.exists(), "v1 checkpoint missing!"
    v1_state = torch.load(v1_path, map_location="cpu")
    assert v1_state["epoch"] == 4, f"v1 checkpoint modified! epoch={v1_state['epoch']}"
    print(f"\n✓ Original v1 checkpoint untouched (epoch={v1_state['epoch']})")
    
    # 9. Verify no fresh ImageNet model in resume mode
    # The model should have loaded weights from checkpoint, not pretrained
    # Check that model weights differ from a fresh pretrained model
    fresh_model = build_model(pretrained=True)
    checkpoint_model = build_model(pretrained=False)
    checkpoint_model.load_state_dict(state["model_state_dict"])
    
    # Compare a few layers - they should differ from pretrained
    diff_found = False
    for (name1, p1), (name2, p2) in zip(fresh_model.named_parameters(), checkpoint_model.named_parameters()):
        if not torch.allclose(p1, p2):
            diff_found = True
            break
    assert diff_found, "Resumed model weights should differ from fresh pretrained model"
    print(f"  ✓ Resumed model weights differ from fresh pretrained (correctly loaded)")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
    print("\nSUMMARY:")
    print(f"  Checkpoint: {CHECKPOINT_PATH_5CLASS.name}")
    print(f"  Last epoch: 3 (Stage 1)")
    print(f"  Resume starts from: Epoch 4, Stage 1")
    print(f"  Optimizer/scheduler: Freshly initialized (not in checkpoint)")
    print(f"  v1 checkpoint: Untouched")
    print(f"  No training started")
    
    return True


if __name__ == "__main__":
    test_resume_smoke()