"""
Inference predictor for the ClaimSight CV model.

Loads the trained ResNet-50 checkpoint and runs inference on a single image
or a list of images, returning structured CVPrediction objects.

Blueprint Section 2.6 inference pipeline:
  1. Load image → validate format/size → preprocess
  2. Forward pass → sigmoid probabilities (damage types) + softmax (severity)
  3. Apply thresholds → structured output + low_confidence flag
  4. low_confidence = True when top damage conf < 0.4 OR severity conf < 0.5
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from PIL import Image, UnidentifiedImageError

# ─── Imports from ML package ──────────────────────────────────────────────────

from ml.training.config import (
    CHECKPOINT_PATH,
    DAMAGE_THRESHOLD,
    DAMAGE_TYPES,
    IMAGE_SIZE,
    LOW_CONF_DAMAGE,
    LOW_CONF_SEVERITY,
    SEVERITY_CLASSES,
)
from ml.training.dataset import get_inference_transform
from ml.training.model import DualHeadResNet50


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
    model_version: str = "claimsight_cv_v1"
    source_image: Optional[str] = None            # original file reference
    timestamp: Optional[str] = None
    error: Optional[str] = None                   # set if inference failed


# ─── Predictor class ──────────────────────────────────────────────────────────

class VehicleDamagePredictor:
    """
    Singleton-style predictor that loads the model once and runs inference.
    
    Usage:
        predictor = VehicleDamagePredictor()
        result = predictor.predict_from_path("path/to/image.jpg")
    """

    # Minimum acceptable image dimension (pixels)
    MIN_DIM = 32

    def __init__(self, checkpoint_path: Optional[Path] = None) -> None:
        self._checkpoint_path = checkpoint_path or CHECKPOINT_PATH
        self._model: Optional[DualHeadResNet50] = None
        self._transform = get_inference_transform()
        self._device = torch.device("cpu")   # CPU-only for local dev

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load model from checkpoint (lazy, only on first prediction call)."""
        if not Path(self._checkpoint_path).exists():
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
        for i, label in enumerate(DAMAGE_TYPES):
            conf = float(damage_probs[i])
            if conf > top_damage_conf:
                top_damage_conf = conf
            if conf >= DAMAGE_THRESHOLD:
                detected.append(DamageTypePrediction(label=label, confidence=round(conf, 4)))

        # If nothing detected above threshold, report no_damage
        if not detected:
            detected = [DamageTypePrediction(label="no_damage", confidence=round(float(damage_probs[DAMAGE_TYPES.index("no_damage")]), 4))]

        # Sort detected by confidence descending
        detected.sort(key=lambda x: x.confidence, reverse=True)

        # Severity
        sev_idx  = int(severity_probs.argmax())
        sev_conf = float(severity_probs[sev_idx])
        severity = SeverityPrediction(
            label=SEVERITY_CLASSES[sev_idx],
            confidence=round(sev_conf, 4),
        )

        # Low confidence flag
        low_confidence = (top_damage_conf < LOW_CONF_DAMAGE) or (sev_conf < LOW_CONF_SEVERITY)

        return CVPrediction(
            damage_types=detected,
            severity=severity,
            low_confidence=low_confidence,
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
