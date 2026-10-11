"""
Inference predictor for the ClaimSight CV model.

Loads the trained ResNet-50 checkpoint and runs inference on a single image
or a list of images, returning structured CVPrediction objects.

Blueprint Section 2.6 inference pipeline:
  1. Load image → validate format/size → preprocess
  2. Forward pass → sigmoid probabilities (damage types) + softmax (severity)
  3. Apply thresholds → structured output + low_confidence flag
  4. low_confidence = True when top damage conf < 0.4 OR severity conf < 0.5

Supports configurable checkpoint via backend settings (CV_CHECKPOINT_PATH).
Validates checkpoint architecture at load time.
"""

from __future__ import annotations

import io
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from PIL import Image, UnidentifiedImageError

# ─── Imports from ML package ──────────────────────────────────────────────────

from ml.training.config import (
    DAMAGE_THRESHOLD,
    DAMAGE_TYPES,
    IMAGE_SIZE,
    LOW_CONF_DAMAGE,
    LOW_CONF_SEVERITY,
    SEVERITY_CLASSES,
)
from ml.training.dataset import get_inference_transform
from ml.training.model import DualHeadResNet50


# ─── Checkpoint validation ────────────────────────────────────────────────────

EXPECTED_DAMAGE_CLASSES_5CLASS = ["scratch", "dent", "shattered_glass", "headlight_damage", "no_damage"]
EXPECTED_DAMAGE_CLASSES_8CLASS = ["scratch", "dent", "crack", "shattered_glass", "bumper_damage", "panel_damage", "headlight_damage", "no_damage"]
EXPECTED_SEVERITY_CLASSES = ["minor", "moderate", "severe"]


def validate_checkpoint(checkpoint_path: Path) -> tuple[str, list[str], list[str]]:
    """
    Validate checkpoint architecture and return (model_version, damage_types, severity_types).
    
    Raises:
        ValueError: If checkpoint is incompatible or missing expected keys.
        FileNotFoundError: If checkpoint file doesn't exist.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    
    # Check required keys
    if "model_state_dict" not in state:
        raise ValueError("Checkpoint missing required key: model_state_dict")
    
    # Infer architecture from head output dimensions
    model_state = state["model_state_dict"]
    damage_head_weight = model_state.get("head_damage.3.weight")
    severity_head_weight = model_state.get("head_severity.3.weight")
    
    if damage_head_weight is None or severity_head_weight is None:
        raise ValueError("Checkpoint missing expected head weights (head_damage.3.weight, head_severity.3.weight)")
    
    num_damage = damage_head_weight.shape[0]
    num_severity = severity_head_weight.shape[0]
    
    if num_severity != 3:
        raise ValueError(f"Expected 3 severity classes, got {num_severity}")
    
    # Determine model version and class mapping
    # Priority: 1) model_version in checkpoint, 2) damage_types in checkpoint, 3) filename heuristic, 4) class count
    model_version = state.get("model_version")
    damage_types = state.get("damage_types")
    
    if num_damage == 5:
        if damage_types is None:
            damage_types = EXPECTED_DAMAGE_CLASSES_5CLASS
        if model_version is None:
            # Heuristic: check filename for version hints
            filename = checkpoint_path.name.lower()
            if "v1" in filename and "5class" not in filename:
                model_version = "claimsight_cv_v1"
            elif "5class" in filename or "v2" in filename:
                model_version = "claimsight_cv_v2_5class"
            else:
                model_version = "claimsight_cv_v2_5class"
    elif num_damage == 8:
        if damage_types is None:
            damage_types = EXPECTED_DAMAGE_CLASSES_8CLASS
        if model_version is None:
            model_version = "claimsight_cv_v1"
    else:
        raise ValueError(f"Unsupported number of damage classes: {num_damage} (expected 5 or 8)")
    
    # Validate checkpoint's class mapping matches our expectations if available
    if damage_types is not None and "damage_types" in state:
        saved_damage_types = state["damage_types"]
        if saved_damage_types != damage_types:
            warnings.warn(
                f"Checkpoint damage_types {saved_damage_types} differs from expected {damage_types}",
                UserWarning
            )
    
    return model_version or "unknown", damage_types, EXPECTED_SEVERITY_CLASSES


# ─── Result types ─────────────────────────────────────────────────────────────

@dataclass
class DamageTypePrediction:
    label: str
    confidence: float


@dataclass
class SeverityPrediction:
    label: str
    confidence: float


@dataclass
class CVPrediction:
    """
    Structured prediction for a single image (blueprint Section 2.6).
    """
    damage_types: list[DamageTypePrediction]      # labels with conf > threshold
    severity: SeverityPrediction
    low_confidence: bool                           # True if model is uncertain
    model_version: str
    source_image: Optional[str] = None            # original file reference
    timestamp: Optional[str] = None
    error: Optional[str] = None                   # set if inference failed


# ─── Predictor class ──────────────────────────────────────────────────────────

class VehicleDamagePredictor:
    """
    Predictor that loads the model once and runs inference.
    
    Usage:
        predictor = VehicleDamagePredictor(checkpoint_path="/path/to/model.pt")
        result = predictor.predict_from_path("path/to/image.jpg")
    """

    # Minimum acceptable image dimension (pixels)
    MIN_DIM = 32

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        damage_types: Optional[list[str]] = None,
        severity_classes: Optional[list[str]] = None,
        model_version: Optional[str] = None,
    ) -> None:
        """
        Initialize predictor with optional checkpoint override.
        
        Args:
            checkpoint_path: Path to model checkpoint. If None, uses default from config.
            damage_types: List of damage class labels. If None, inferred from checkpoint.
            severity_classes: List of severity class labels. If None, uses defaults.
            model_version: Human-readable model version string. If None, inferred from checkpoint.
        """
        # Import here to avoid circular dependency with backend config
        try:
            from app.core.config import settings
            default_ckpt = Path(settings.cv_checkpoint_path)
            # Resolve relative to project root (where this file's grandparent is)
            project_root = Path(__file__).resolve().parents[2]
            if not default_ckpt.is_absolute():
                default_ckpt = project_root / default_ckpt
        except Exception:
            # Fallback if backend config not available
            project_root = Path(__file__).resolve().parents[2]
            default_ckpt = project_root / "ml/weights/claimsight_cv_v1.pt"
        
        self._checkpoint_path = checkpoint_path or default_ckpt
        self._damage_types_override = damage_types
        self._severity_classes_override = severity_classes
        self._model_version_override = model_version
        
        # Deferred validation - will run on first model load
        self._model_version: Optional[str] = None
        self._damage_types: Optional[list[str]] = None
        self._severity_classes: Optional[list[str]] = None
        self._validated = False
        
        self._model: Optional[DualHeadResNet50] = None
        self._transform = get_inference_transform()
        self._device = torch.device("cpu")   # CPU-only for local dev

# ── Model loading ─────────────────────────────────────────────────────────

    def _validate_and_load_metadata(self) -> None:
        """Validate checkpoint and load metadata (lazy, on first load)."""
        if self._validated:
            return
        self._model_version, self._damage_types, self._severity_classes = validate_checkpoint(self._checkpoint_path)
        
        # Apply overrides if provided
        if self._damage_types_override is not None:
            self._damage_types = self._damage_types_override
        if self._severity_classes_override is not None:
            self._severity_classes = self._severity_classes_override
        if self._model_version_override is not None:
            self._model_version = self._model_version_override
            
        self._validated = True

    def _load_model(self) -> None:
        """Load model from checkpoint (lazy, only on first prediction call)."""
        self._validate_and_load_metadata()
        
        if not self._checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at {self._checkpoint_path}. "
                "Train the model first with: python -m ml.training.train"
            )
        model = DualHeadResNet50(pretrained=False)
        state = torch.load(self._checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        self._model = model

    def _ensure_model_loaded(self) -> None:
        if self._model is None:
            self._load_model()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def model_version(self) -> str:
        if not self._validated:
            self._validate_and_load_metadata()
        return self._model_version or "unknown"

    @property
    def damage_types(self) -> list[str]:
        if not self._validated:
            self._validate_and_load_metadata()
        return self._damage_types or []

    @property
    def severity_classes(self) -> list[str]:
        if not self._validated:
            self._validate_and_load_metadata()
        return self._severity_classes or []

    # ── Image loading ─────────────────────────────────────────────────────────

    def _load_image(self, source: str | Path | bytes) -> Image.Image:
        """Load and basic-validate a PIL image from path, bytes, or file-like."""
        if isinstance(source, (str, Path)):
            img = Image.open(source)
        elif isinstance(source, bytes):
            img = Image.open(io.BytesIO(source))
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        img = img.convert("RGB")

        # Size validation
        w, h = img.size
        if w < self.MIN_DIM or h < self.MIN_DIM:
            raise ValueError(f"Image too small ({w}×{h}). Minimum: {self.MIN_DIM}×{self.MIN_DIM}")

        return img

    # ── Inference core ────────────────────────────────────────────────────────

    def _run_inference(
        self, img: Image.Image
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw sigmoid probs (damage) and softmax probs (severity)."""
        tensor = self._transform(img).unsqueeze(0).to(self._device)   # (1, 3, 224, 224)
        damage_probs, severity_probs = self._model.predict(tensor)     # type: ignore
        return damage_probs.squeeze(0), severity_probs.squeeze(0)

    # ── Build result ──────────────────────────────────────────────────────────

    def _build_prediction(
        self,
        damage_probs: torch.Tensor,
        severity_probs: torch.Tensor,
        source_image: Optional[str] = None,
    ) -> CVPrediction:
        # Damage types above threshold
        detected: list[DamageTypePrediction] = []
        top_damage_conf = 0.0
        for i, label in enumerate(self._damage_types):
            conf = float(damage_probs[i])
            if conf > top_damage_conf:
                top_damage_conf = conf
            if conf >= DAMAGE_THRESHOLD:
                detected.append(DamageTypePrediction(label=label, confidence=round(conf, 4)))

        # If nothing detected above threshold, report no_damage
        if not detected:
            no_damage_idx = self._damage_types.index("no_damage") if "no_damage" in self._damage_types else -1
            if no_damage_idx >= 0:
                detected = [DamageTypePrediction(label="no_damage", confidence=round(float(damage_probs[no_damage_idx]), 4))]
            else:
                detected = []

        # Sort detected by confidence descending
        detected.sort(key=lambda x: x.confidence, reverse=True)

        # Severity
        sev_idx  = int(severity_probs.argmax())
        sev_conf = float(severity_probs[sev_idx])
        severity = SeverityPrediction(
            label=self._severity_classes[sev_idx],
            confidence=round(sev_conf, 4),
        )

        # Low confidence flag
        low_confidence = (top_damage_conf < LOW_CONF_DAMAGE) or (sev_conf < LOW_CONF_SEVERITY)

        return CVPrediction(
            damage_types=detected,
            severity=severity,
            low_confidence=low_confidence,
            model_version=self._model_version,
            source_image=source_image,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def predict_from_path(self, image_path: str | Path) -> CVPrediction:
        """Run inference on an image file."""
        self._ensure_model_loaded()
        try:
            img = self._load_image(image_path)
            d_probs, s_probs = self._run_inference(img)
            return self._build_prediction(d_probs, s_probs, source_image=str(image_path))
        except (FileNotFoundError, ValueError, UnidentifiedImageError) as exc:
            return CVPrediction(
                damage_types=[],
                severity=SeverityPrediction(label="unknown", confidence=0.0),
                low_confidence=True,
                model_version=self._model_version,
                source_image=str(image_path),
                error=str(exc),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )

    def predict_from_bytes(self, data: bytes, filename: str = "") -> CVPrediction:
        """Run inference on raw image bytes (e.g. from file upload)."""
        self._ensure_model_loaded()
        try:
            img = self._load_image(data)
            d_probs, s_probs = self._run_inference(img)
            return self._build_prediction(d_probs, s_probs, source_image=filename)
        except (ValueError, UnidentifiedImageError) as exc:
            return CVPrediction(
                damage_types=[],
                severity=SeverityPrediction(label="unknown", confidence=0.0),
                low_confidence=True,
                model_version=self._model_version,
                source_image=filename,
                error=str(exc),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )

    def predict_batch(self, image_paths: list[str | Path]) -> list[CVPrediction]:
        """Run inference on multiple images, returns one CVPrediction per image."""
        self._ensure_model_loaded()
        return [self.predict_from_path(p) for p in image_paths]


# ─── Module-level singleton ───────────────────────────────────────────────────

_predictor: Optional[VehicleDamagePredictor] = None


def get_predictor() -> VehicleDamagePredictor:
    """Return the module-level predictor singleton (lazy load)."""
    global _predictor
    if _predictor is None:
        _predictor = VehicleDamagePredictor()
    return _predictor
