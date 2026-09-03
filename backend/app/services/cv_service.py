"""
CV service — bridges the inference predictor with the ClaimSight database layer.

Responsibilities:
- Accept an uploaded image path + claim context
- Run VehicleDamagePredictor (real trained checkpoint)
- Persist results as Damage rows
- Return a list of persisted Damage objects

Does NOT:
- Run the consistency engine
- Compute risk scores
- Generate risk signals
- Call Gemini
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from app.models.damage import Damage

if TYPE_CHECKING:
    from ml.inference.predictor import CVPrediction

logger = logging.getLogger(__name__)


# ─── Demo CV predictor (Phase 13) ──────────────────────────────────────────
#
# When `settings.use_demo_cv` is true, the service substitutes a small
# deterministic predictor in place of the trained model. The output
# (damage type, severity, confidence) is derived from the image
# filename so the demo is reproducible and the resulting claim graph
# feeds the R1–R9 rules the way the blueprint intends. The
# deterministic table covers the 5 demo scenarios plus a small number
# of common keywords; unknown filenames fall through to a sensible
# "scratch/minor" default.
_DEMO_CV_TABLE: dict[str, tuple[str, str, float]] = {
    # filename token → (damage_type, severity, confidence)
    "scratch":      ("scratch",        "minor",     0.92),
    "rear-scratch": ("scratch",        "minor",     0.93),
    "small-dent":   ("dent",           "minor",     0.91),
    "dent":         ("dent",           "minor",     0.90),
    "bumper-dent":  ("bumper_damage",  "moderate",  0.88),
    "bumper":       ("bumper_damage",  "moderate",  0.88),
    "front-damage": ("dent",           "minor",     0.89),
    "panel":        ("panel_damage",   "moderate",  0.87),
    "glass":        ("shattered_glass","severe",    0.94),
    "headlight":    ("headlight_damage","minor",    0.90),
}
_DEMO_CV_DEFAULT = ("scratch", "minor", 0.85)


def _lookup_demo_cv(filename: str) -> tuple[str, str, float]:
    """Map an image filename to a (damage_type, severity, confidence)
    tuple using substring matches. Filename comparison is lowercased
    for case insensitivity.
    """
    base = filename.lower()
    for token, signature in _DEMO_CV_TABLE.items():
        if token in base:
            return signature
    return _DEMO_CV_DEFAULT


class _DemoCVPredictor:
    """Deterministic stand-in for `ml.inference.predictor.get_predictor()`.

    Activated by `settings.use_demo_cv = True` (or the `USE_DEMO_CV`
    env var). Produces duck-typed `CVPrediction` objects so the rest
    of the service (and the consistency rules that read the
    persisted Damage rows) treat the output exactly the same as the
    real model.
    """

    def __init__(self):
        self.model_version = "demo_deterministic_v1"

    def predict_from_path(self, image_path):  # noqa: D401
        path = Path(image_path)
        damage_type, severity, confidence = _lookup_demo_cv(path.name)
        ts = "2026-02-15T00:00:00"

        class _Pred:
            def __init__(self, label, confidence):
                self.label = label
                self.confidence = confidence

        class _Result:
            pass

        r = _Result()
        r.damage_types = [_Pred(damage_type, confidence)]
        r.severity = _Pred(severity, confidence - 0.04)
        r.low_confidence = False
        r.model_version = self.model_version
        r.source_image = str(image_path)
        r.timestamp = ts
        r.error = None
        return r


def _extract_image_path(region_ref: str) -> str:
    """Extract the actual image path from region_ref (which may be JSON or plain path)."""
    if not region_ref:
        return ""
    try:
        parsed = json.loads(region_ref)
        return parsed.get("image_path", region_ref)
    except json.JSONDecodeError:
        return region_ref


def run_cv_on_image(
    db: Session,
    claim_id: int,
    image_path: str,
    predictor=None,
) -> list[Damage]:
    """
    Run CV inference on a single image and persist Damage rows.

    Args:
        db:           SQLAlchemy session
        claim_id:     The claim the image belongs to
        image_path:   region_ref from Damage record (JSON with "image_path" or plain path)
        predictor:    Optional injected predictor (for testing); defaults to module singleton

    Returns:
        List of newly persisted Damage rows (may be empty on error).
    """
    # Resolve the predictor: caller-injected first (tests, demo data
    # script), then the optional demo predictor (USE_DEMO_CV=1), then
    # the real model. The real package is optional — `ml/` lives
    # outside the backend in the same repo but is not on sys.path in
    # many runtimes. The injection point must work even when `ml` is
    # unavailable.
    if predictor is None:
        from app.core.config import settings
        if settings.use_demo_cv:
            predictor = _DemoCVPredictor()
        else:
            try:
                from ml.inference.predictor import get_predictor  # noqa: WPS433
            except ImportError:
                logger.exception(
                    "ml.inference.predictor is not importable and no "
                    "predictor was injected; writing a cv_error row."
                )
                failed_dmg = Damage(
                    claim_id=claim_id,
                    source="image",
                    damage_type="cv_error",
                    severity="unknown",
                    confidence=0.0,
                    region_ref=image_path,
                )
                db.add(failed_dmg)
                db.flush()
                return [failed_dmg]
            predictor = get_predictor()

    # Extract actual image path from region_ref (may be JSON)
    actual_image_path = _extract_image_path(image_path)

    # Resolve full path
    from app.core.config import settings
    upload_base = Path(settings.upload_dir)
    if not upload_base.is_absolute():
        import os
        upload_base = Path(os.getcwd()) / upload_base

    # Handle both "uploads/claim_id/filename" and "claim_id/filename" formats
    path_parts = actual_image_path.split("/")
    if path_parts[0] == "uploads":
        relative_path = Path(*path_parts[1:])
    else:
        relative_path = Path(*path_parts)

    full_path = upload_base / relative_path

    if not full_path.exists():
        logger.error("Image file not found: %s (resolved from %s)", full_path, actual_image_path)
        failed_dmg = Damage(
            claim_id=claim_id,
            source="image",
            damage_type="cv_error",
            severity="unknown",
            confidence=0.0,
            region_ref=image_path,
        )
        db.add(failed_dmg)
        db.flush()
        return [failed_dmg]

    try:
        result: CVPrediction = predictor.predict_from_path(full_path)
    except FileNotFoundError as e:
        logger.warning("CV model checkpoint not found: %s", e)
        failed_dmg = Damage(
            claim_id=claim_id,
            source="image",
            damage_type="cv_error",
            severity="unknown",
            confidence=0.0,
            region_ref=image_path,
        )
        db.add(failed_dmg)
        db.flush()
        return [failed_dmg]
    except Exception as e:
        logger.exception("CV inference failed for %s: %s", actual_image_path, e)
        failed_dmg = Damage(
            claim_id=claim_id,
            source="image",
            damage_type="cv_error",
            severity="unknown",
            confidence=0.0,
            region_ref=image_path,
        )
        db.add(failed_dmg)
        db.flush()
        return [failed_dmg]

    if result.error:
        logger.warning("CV inference returned error for %s: %s", actual_image_path, result.error)
        failed_dmg = Damage(
            claim_id=claim_id,
            source="image",
            damage_type="cv_error",
            severity="unknown",
            confidence=0.0,
            region_ref=image_path,
        )
        db.add(failed_dmg)
        db.flush()
        return [failed_dmg]

    damages: list[Damage] = []

    # Store all detected damage types (not just the first one)
    for dt in result.damage_types:
        dmg = Damage(
            claim_id=claim_id,
            source="image",
            damage_type=dt.label,
            severity=result.severity.label,
            confidence=dt.confidence,
            # Store metadata including low_confidence flag in region_ref as JSON
            region_ref=json.dumps({
                "image_path": actual_image_path,
                "low_confidence": result.low_confidence,
                "severity_confidence": result.severity.confidence,
                "model_version": result.model_version,
                "timestamp": result.timestamp,
            }),
        )
        db.add(dmg)
        damages.append(dmg)

    # If we created at least one, flush to get IDs
    if damages:
        db.flush()

    logger.info(
        "CV result for claim %d image %s: %d damage type(s), severity=%s (conf=%.3f), low_conf=%s",
        claim_id, actual_image_path, len(damages), result.severity.label, result.severity.confidence, result.low_confidence
    )

    return damages
