"""
Training configuration for the ClaimSight CV model.

All hyperparameters and paths are centralised here.
Import this module instead of hardcoding values anywhere else.
"""

from pathlib import Path
from dataclasses import dataclass, field

# ─── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # ClaimSight/
ML_DIR       = PROJECT_ROOT / "ml"
DATA_DIR     = ML_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
WEIGHTS_DIR  = ML_DIR / "weights"
RESULTS_DIR  = ML_DIR / "results"

# ─── Classes ──────────────────────────────────────────────────────────────────

DAMAGE_TYPES = [
    "scratch",
    "dent",
    "crack",
    "shattered_glass",
    "bumper_damage",
    "panel_damage",
    "headlight_damage",
    "no_damage",
]

SEVERITY_CLASSES = ["minor", "moderate", "severe"]

NUM_DAMAGE_CLASSES = len(DAMAGE_TYPES)   # 8
NUM_SEVERITY_CLASSES = len(SEVERITY_CLASSES)   # 3

# ─── Preprocessing ────────────────────────────────────────────────────────────

IMAGE_SIZE    = 224       # pixels (ImageNet standard)
# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─── Training (stage 1 — heads only) ─────────────────────────────────────────

STAGE1_EPOCHS   = 5
STAGE1_LR       = 1e-3
STAGE1_BATCH    = 32

# ─── Training (stage 2 — unfreeze layer3/4 + heads) ─────────────────────────

STAGE2_EPOCHS   = 15
STAGE2_LR       = 1e-5
STAGE2_BATCH    = 16

# ─── Optimizer / Loss ────────────────────────────────────────────────────────

WEIGHT_DECAY    = 1e-4
LR_SCHEDULER    = "cosine"   # or "step"

# Head A: Binary Cross Entropy (multi-label)
# Head B: Cross Entropy (single-label)

# ─── Evaluation ──────────────────────────────────────────────────────────────

DAMAGE_THRESHOLD  = 0.5     # sigmoid probability to call a damage type present
LOW_CONF_DAMAGE   = 0.4     # below this → low_confidence flag
LOW_CONF_SEVERITY = 0.5     # severity softmax confidence below this → low_confidence

# ─── Misc ─────────────────────────────────────────────────────────────────────

RANDOM_SEED   = 42
CHECKPOINT_NAME = "claimsight_cv_v1.pt"
CHECKPOINT_PATH = WEIGHTS_DIR / CHECKPOINT_NAME

# ─── Dataset split ────────────────────────────────────────────────────────────

TRAIN_FRACTION = 0.70
VAL_FRACTION   = 0.15
TEST_FRACTION  = 0.15
