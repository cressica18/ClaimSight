# Data Card — CV Model

**Phase**: Not yet trained (Phase 4).

This file will be completed in Phase 4 with:

| Field | Value |
|-------|-------|
| Dataset name | (to be determined at training time) |
| Dataset source | Kaggle / Roboflow Universe |
| Dataset version/hash | (record exact version) |
| License | (verify and record) |
| Number of images | (to be filled) |
| Class mapping | scratch, dent, crack, shattered_glass, bumper_damage, panel_damage, headlight_damage, no_damage |
| Severity classes | minor, moderate, severe |
| Train / Val / Test split | 70% / 15% / 15% |
| Preprocessing | Resize 224×224, ImageNet mean/std normalization |
| Augmentation (train) | H-flip, ±15° rotation, color jitter ±0.2, random crop w/ padding |
| Model architecture | ResNet-50 (ImageNet pretrained) — dual-head fine-tune |
| Training hardware | (to be filled) |
| Test macro-F1 (Head A) | (to be filled — target ≥ 0.65) |
| Severity accuracy (Head B) | (to be filled) |
| Weights file | `ml/weights/claimsight_cv_v1.pt` |

> **Note**: This dataset documents findings from public datasets only. ClaimSight does not claim fraud-detection model training on proprietary insurer data.
