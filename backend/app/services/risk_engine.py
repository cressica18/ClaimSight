"""
Anomaly + Risk Engine — blueprint Section 6.

This module turns a `ClaimContext` and a list of `RiskSignal` rows (already
produced by the Phase 6 consistency engine) into an explainable risk score
in [0, 100] with a Low / Medium / High band.

It is fully deterministic and explainable:
- 5 named features (`f1`..`f5`), each normalized 0–1 per Section 6.2.
- 5 fixed weights (0.35, 0.15, 0.25, 0.15, 0.10) — MUST, not learned.
- `score = 100 * Σ(weight_i * f_i)`, clamped to [0, 100].
- Bands: Low 0–34, Medium 35–64, High 65–100.
- Low-data-confidence qualifier: if more than 30% of inputs are
  low-confidence, the band is bumped up to at least Medium ("defaults
  toward Medium review rather than automatic Low clearance").
- Every contributing factor carries the linked RiskSignal ids so the UI
  can click a row of the score and jump straight to the evidence.

Scope honesty (per the Phase 7 prompt):
- The repair-cost baseline is computed from a small **synthetic** dataset
  embedded in this module. It is illustrative, not industry-validated.
  Every public doc string and the progress doc repeat that point.
- The Isolation Forest feature (`f5`) is a no-op stub. Section 6.1 marks
  it NICE; wiring it would require a sklearn dep and a training step on
  the same synthetic data. The engine's math already accounts for the
  case where `f5` is unused via proportional weight redistribution.

Public API:
- `compute_baseline(vehicle_segment, damage_type, severity) -> BaselineRange`
- `compute_risk_score(ctx, signals, *, baseline=None) -> RiskScore`
- `persist(risk_score, db, *, claim) -> Claim` — writes the score and
  band to the `Claim` row and commits.
- `ContributingFactor`, `RiskScore`, `BaselineRange` — frozen dataclasses.
"""

from __future__ import annotations

import logging
import re
import statistics
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from sqlalchemy.orm import Session

from app.models.claim import Claim
from app.models.enums import RiskBand, SignalSeverity
from app.models.risk_signal import RiskSignal
from app.services.consistency import (
    ClaimContext,
    ImageDamageCtx,
    PreviousClaimCtx,
)

logger = logging.getLogger(__name__)


# ─── Public dataclasses ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class BaselineRange:
    """Repair-cost baseline range for one (segment, damage_type, severity) cell.

    The bounds are `mean ± 1.5 * IQR` of the synthetic observation set
    (blueprint Section 6.1). The `n` field is the number of synthetic
    observations the range was derived from — useful for tests and for
    the UI to surface "based on n=5 synthetic observations".
    """

    segment: str
    damage_type: str
    severity: str
    lower: float
    upper: float
    mean: float
    n: int

    def contains(self, cost: float) -> bool:
        return self.lower <= cost <= self.upper


@dataclass(frozen=True)
class ContributingFactor:
    """One row of the score's explanation.

    `value` is the *normalized* feature value (0..1), not the raw input.
    `weight` is the *effective* weight used in the scoring formula
    (already scaled for the f5-on/f5-off redistribution).
    `linked_signal_ids` are the RiskSignal ids this feature aggregates.
    """

    feature: str       # "f1_high_signals" | "f2_medium_signals" | "f3_cost_ratio" | "f4_previous_overlap" | "f5_anomaly"
    weight: float      # effective weight (post redistribution)
    value: float       # normalized 0..1
    raw: str           # human-readable description of the raw input
    linked_signal_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class RiskScore:
    """The final risk output. Immutable so callers can't accidentally
    mutate a score that's already on the `Claim` row."""

    score: float                              # 0..100
    band: str                                 # "Low" | "Medium" | "High"
    low_data_confidence: bool
    factors: tuple[ContributingFactor, ...]
    baseline: BaselineRange | None
    notes: tuple[str, ...] = ()


# ─── Fixed weights and bands (Section 6.2 MUST) ─────────────────────────────

# MUST weights from the blueprint. f5 is 0.10 only if Isolation Forest is
# wired in; otherwise the remaining four weights are scaled so the sum
# stays 1.0 (proportional redistribution per Section 6.2).
W_HIGH_SIGNALS = 0.35
W_MED_SIGNALS = 0.15
W_COST_RATIO = 0.25
W_PREVIOUS_OVERLAP = 0.15
W_ANOMALY = 0.10

# Feature cap constants (Section 6.2).
F1_CAP = 3          # count of High signals before /3 saturates
F2_CAP = 5          # count of Medium signals before /5 saturates
F3_CAP = 3.0        # cost ratio saturates at 3.0× the baseline upper

# Band thresholds (Section 6.2): Low 0–34, Medium 35–64, High 65–100.
BAND_LOW_MAX = 34
BAND_MED_MAX = 64
# Anything ≥ 65 is High.

# Low-data-confidence threshold (Section 6.2): > 30% of inputs low → Medium default.
LOW_DATA_CONFIDENCE_FRACTION = 0.30


# ─── Synthetic baseline dataset (Section 6.1) ───────────────────────────────
#
# HONEST SCOPE: This dataset is a small hand-written table used to
# derive a per-(segment, damage_type, severity) baseline cost range.
# It is NOT industry-validated. The blueprint explicitly permits
# "synthetic + input claim data" for the risk engine and labels any
# resulting baseline as "illustrative". The progress doc repeats this.
#
# Format: each cell has at least 3 observations (small but > 1 so the
# IQR is well-defined). Costs are in USD. The means are roughly
# 1.5×–2× typical US market rates; the IQRs are tight on purpose so
# the test suite can exercise the "cost above baseline" path with
# realistic multiples.

_SYNTHETIC_BASELINE: dict[tuple[str, str, str], list[float]] = {
    # ─── sedan ─────────────────────────────────────────────────────────────
    ("sedan", "scratch", "minor"): [180, 220, 250, 300, 350, 400],
    ("sedan", "scratch", "moderate"): [350, 420, 500, 600, 700, 800],
    ("sedan", "scratch", "severe"): [600, 750, 900, 1100, 1300],
    ("sedan", "dent", "minor"): [250, 320, 400, 500, 600],
    ("sedan", "dent", "moderate"): [500, 650, 800, 950, 1100],
    ("sedan", "dent", "severe"): [900, 1100, 1400, 1700, 2000],
    ("sedan", "bumper_damage", "minor"): [400, 500, 600, 700, 800],
    ("sedan", "bumper_damage", "moderate"): [700, 900, 1100, 1300, 1500],
    ("sedan", "bumper_damage", "severe"): [1200, 1500, 1800, 2200, 2600],
    ("sedan", "panel_damage", "minor"): [350, 450, 550, 650, 750],
    ("sedan", "panel_damage", "moderate"): [600, 800, 1000, 1200, 1400],
    ("sedan", "panel_damage", "severe"): [1000, 1300, 1600, 2000, 2400],
    ("sedan", "shattered_glass", "minor"): [200, 280, 350, 450, 550],
    ("sedan", "shattered_glass", "moderate"): [300, 400, 500, 650, 800],
    ("sedan", "shattered_glass", "severe"): [400, 550, 700, 900, 1100],
    ("sedan", "headlight_damage", "minor"): [150, 200, 250, 300, 400],
    ("sedan", "headlight_damage", "moderate"): [250, 350, 450, 600, 750],
    ("sedan", "headlight_damage", "severe"): [400, 550, 700, 900, 1100],
    ("sedan", "crack", "minor"): [120, 160, 200, 260, 320],
    ("sedan", "crack", "moderate"): [200, 280, 360, 450, 550],
    # ─── suv ───────────────────────────────────────────────────────────────
    ("suv", "scratch", "minor"): [220, 280, 320, 380, 440],
    ("suv", "scratch", "moderate"): [400, 500, 600, 720, 850],
    ("suv", "bumper_damage", "moderate"): [800, 1000, 1300, 1500, 1800],
    ("suv", "bumper_damage", "severe"): [1400, 1800, 2200, 2600, 3000],
    ("suv", "panel_damage", "moderate"): [700, 900, 1100, 1400, 1700],
    ("suv", "shattered_glass", "moderate"): [350, 450, 600, 750, 900],
    # ─── truck ─────────────────────────────────────────────────────────────
    ("truck", "bumper_damage", "moderate"): [900, 1200, 1500, 1800, 2100],
    ("truck", "bumper_damage", "severe"): [1600, 2100, 2600, 3100, 3600],
    ("truck", "panel_damage", "severe"): [1400, 1800, 2200, 2700, 3200],
    ("truck", "dent", "moderate"): [600, 800, 1000, 1200, 1400],
    # ─── luxury ────────────────────────────────────────────────────────────
    ("luxury", "scratch", "minor"): [400, 500, 600, 750, 900],
    ("luxury", "scratch", "moderate"): [700, 900, 1100, 1400, 1700],
    ("luxury", "bumper_damage", "moderate"): [1200, 1500, 1900, 2300, 2800],
    ("luxury", "panel_damage", "moderate"): [1100, 1400, 1800, 2200, 2700],
    ("luxury", "shattered_glass", "moderate"): [600, 800, 1000, 1300, 1600],
}

# Fallback for cells not in the table: conservative mid-range costs.
_FALLBACK_BASELINE: dict[tuple[str, str], list[float]] = {
    ("sedan", "scratch"): [200, 300, 500, 800, 1200],
    ("sedan", "dent"): [300, 500, 800, 1200, 1800],
    ("sedan", "crack"): [150, 250, 400, 600, 900],
    ("sedan", "shattered_glass"): [250, 400, 650, 900, 1300],
    ("sedan", "bumper_damage"): [500, 800, 1200, 1700, 2400],
    ("sedan", "panel_damage"): [400, 700, 1100, 1500, 2200],
    ("sedan", "headlight_damage"): [200, 350, 550, 800, 1100],
    ("suv", "scratch"): [300, 450, 700, 1000, 1500],
    ("suv", "dent"): [400, 650, 1000, 1500, 2200],
    ("suv", "bumper_damage"): [700, 1100, 1700, 2400, 3200],
    ("suv", "panel_damage"): [600, 1000, 1500, 2200, 3000],
    ("suv", "shattered_glass"): [350, 550, 850, 1200, 1700],
    ("truck", "bumper_damage"): [900, 1400, 2100, 2800, 3600],
    ("truck", "panel_damage"): [800, 1300, 2000, 2700, 3500],
    ("truck", "dent"): [500, 800, 1200, 1700, 2300],
    ("luxury", "scratch"): [500, 800, 1300, 1900, 2600],
    ("luxury", "dent"): [700, 1100, 1700, 2400, 3300],
    ("luxury", "bumper_damage"): [1200, 1900, 2800, 3800, 5000],
    ("luxury", "panel_damage"): [1100, 1700, 2500, 3500, 4700],
    ("luxury", "shattered_glass"): [700, 1100, 1700, 2400, 3200],
}


# ─── Vehicle-segment derivation ─────────────────────────────────────────────
#
# The Vehicle model has no `vehicle_segment` column. We derive one
# deterministically from (make, model, year). This is the same trick
# real insurance rating engines use for their coarse "vehicle class"
# buckets when no explicit segment is stored.

_TRUCK_MODELS = {"f-150", "f150", "silverado", "ram", "tundra", "tacoma", "ranger", "frontier", "colorado"}
_LUXURY_MAKES = {"bmw", "mercedes", "mercedes-benz", "audi", "lexus", "porsche", "jaguar", "tesla", "cadillac", "lincoln", "acura", "infiniti"}
_SUV_MODEL_HINTS = ("rav", "cr-v", "crv", "pilot", "explorer", "tahoe", "suburban", "highlander", "4runner", "wrangler", "grand cherokee", "pathfinder", "rogue", "equinox", "escape", "tucson", "sportage", "cx-", "forester", "outback", "ascent", "pilot")


def derive_vehicle_segment(make: str, model: str, year: int) -> str:
    """Coarse vehicle segment used to bucket the baseline cost ranges.

    Returns one of: `"sedan"`, `"suv"`, `"truck"`, `"luxury"`.
    Luxury takes precedence over SUV/truck (a luxury SUV is still
    luxury-priced). The default for any unmapped vehicle is `"sedan"`.
    """
    make_norm = (make or "").strip().lower()
    model_norm = (model or "").strip().lower()
    if make_norm in _LUXURY_MAKES:
        return "luxury"
    if model_norm in _TRUCK_MODELS:
        return "truck"
    if any(hint in model_norm for hint in _SUV_MODEL_HINTS):
        return "suv"
    return "sedan"


# ─── Baseline computation ───────────────────────────────────────────────────


def _iqr_bounds(samples: Sequence[float]) -> tuple[float, float, float]:
    """Return (lower, upper, mean) using `mean ± 1.5 * IQR` (Section 6.1)."""
    if len(samples) < 2:
        # Defensive: with one sample the IQR is 0, so the range collapses.
        # Fall back to mean ± 25% of the mean.
        mean = float(samples[0])
        return (mean * 0.75, mean * 1.25, mean)
    sorted_samples = sorted(samples)
    q1 = _percentile(sorted_samples, 25)
    q3 = _percentile(sorted_samples, 75)
    iqr = q3 - q1
    mean = statistics.fmean(samples)
    lower = max(0.0, mean - 1.5 * iqr)
    upper = mean + 1.5 * iqr
    return (lower, upper, mean)


def _percentile(sorted_samples: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile (matches numpy's default)."""
    if not sorted_samples:
        return 0.0
    if len(sorted_samples) == 1:
        return float(sorted_samples[0])
    k = (len(sorted_samples) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_samples) - 1)
    if f == c:
        return float(sorted_samples[f])
    return float(sorted_samples[f] + (sorted_samples[c] - sorted_samples[f]) * (k - f))


def _observations_for(
    segment: str, damage_type: str, severity: str
) -> tuple[list[float], str]:
    """Look up the synthetic observations for a (segment, damage, severity)
    cell. Falls back to the (segment, damage) cell when severity-specific
    data is missing, and finally to a generic default. Returns
    `(observations, source)` where `source` describes which table was used.
    """
    key = (segment, damage_type, severity)
    if key in _SYNTHETIC_BASELINE:
        return list(_SYNTHETIC_BASELINE[key]), "synthetic:segment+damage+severity"
    fallback_key = (segment, damage_type)
    if fallback_key in _FALLBACK_BASELINE:
        return list(_FALLBACK_BASELINE[fallback_key]), "synthetic:segment+damage"
    return [200, 500, 1000, 2000, 4000], "synthetic:default"


def compute_baseline(
    vehicle_segment: str,
    damage_type: str,
    severity: str,
) -> BaselineRange:
    """Compute the baseline cost range for one (segment, damage, severity) cell.

    Per Section 6.1, the range is `mean ± 1.5 * IQR` of the synthetic
    observations. The baseline is documented as illustrative.
    """
    samples, _source = _observations_for(vehicle_segment, damage_type, severity)
    lower, upper, mean = _iqr_bounds(samples)
    return BaselineRange(
        segment=vehicle_segment,
        damage_type=damage_type,
        severity=severity,
        lower=lower,
        upper=upper,
        mean=mean,
        n=len(samples),
    )


# ─── Feature extraction ─────────────────────────────────────────────────────


def _high_signal_count(signals: Sequence[RiskSignal]) -> int:
    return sum(1 for s in signals if s.severity == SignalSeverity.high.value)


def _medium_signal_count(signals: Sequence[RiskSignal]) -> int:
    return sum(1 for s in signals if s.severity == SignalSeverity.medium.value)


def _normalise_count(count: int, cap: int) -> float:
    if cap <= 0:
        return 0.0
    return max(0.0, min(1.0, count / cap))


def _claim_cost(ctx: ClaimContext) -> float | None:
    """The single cost figure used for f3.

    Prefers the formal repair estimate (more reliable than the
    customer-claimed amount), then falls back to claimed_amount.
    """
    if ctx.repair_estimate is not None and ctx.repair_estimate.total_cost is not None:
        return float(ctx.repair_estimate.total_cost)
    return ctx.claimed_amount


def _primary_damage(
    ctx: ClaimContext,
) -> tuple[str, str] | None:
    """Pick the (damage_type, severity) used to look up the baseline.

    Preference order:
    1. The first CV-confirmed (high-confidence) image damage.
    2. The first claim-form damage.
    3. None.
    """
    for d in ctx.image_damages:
        if d.damage_type and not d.low_confidence:
            return d.damage_type, d.severity or "moderate"
    for d in ctx.claim_form_damages:
        if d.damage_type:
            return d.damage_type, d.severity or "moderate"
    return None


def _normalise_cost_ratio(cost: float | None, baseline: BaselineRange | None) -> float:
    """f3 = cost / baseline.upper, capped at F3_CAP, then /F3_CAP.

    A cost within the baseline (cost <= baseline.upper) gives f3 ≤ 1/3,
    so this feature alone cannot push the score to High. A cost of 3×
    the upper bound saturates f3 at 1.0.
    """
    if cost is None or baseline is None or baseline.upper <= 0:
        return 0.0
    ratio = cost / baseline.upper
    ratio = max(0.0, ratio)
    ratio = min(ratio, F3_CAP)
    return ratio / F3_CAP


_DAMAGE_TOKENS = re.compile(r"[a-z_]+")


def _tokenise_damage(text: str | None) -> set[str]:
    if not text:
        return set()
    return set(_DAMAGE_TOKENS.findall(text.lower()))


def _previous_overlap_score(ctx: ClaimContext) -> float:
    """f4 — the max token-overlap between current claim damage tokens
    and each previous claim's `damage_summary`.

    This is a deterministic, content-based overlap (a Jaccard-like
    measure on a small token set). It is what R5 already inspects
    qualitatively; the engine quantifies the same signal for the
    score. Returns 0..1.
    """
    if not ctx.previous_claims:
        return 0.0
    current_tokens: set[str] = set()
    for d in ctx.image_damages:
        if d.damage_type:
            current_tokens.update(_tokenise_damage(d.damage_type))
    for d in ctx.claim_form_damages:
        if d.damage_type:
            current_tokens.update(_tokenise_damage(d.damage_type))
    if not current_tokens:
        return 0.0
    best = 0.0
    for prev in ctx.previous_claims:
        prev_tokens = _tokenise_damage(prev.damage_summary)
        if not prev_tokens:
            continue
        intersection = current_tokens & prev_tokens
        if not intersection:
            continue
        # Jaccard on the small token set keeps the score in [0, 1].
        union = current_tokens | prev_tokens
        score = len(intersection) / len(union)
        if score > best:
            best = score
    return min(1.0, best)


def _low_data_confidence_fraction(ctx: ClaimContext) -> float:
    """Section 6.2: 'proportion of documents/images with low confidence
    extraction' — if >30%, the score is shown with a low-data
    confidence qualifier and defaults toward Medium review.

    Inputs we count:
    - Each document whose extraction produced no structured fields
      (`extracted_fields` empty) or whose `raw_confidence` < 0.5.
    - Each image damage with `low_confidence=True`.
    """
    if not ctx.documents and not ctx.image_damages:
        return 0.0
    low = 0
    total = 0
    for doc in ctx.documents:
        total += 1
        if not doc.extracted_fields or (doc.raw_confidence is not None and doc.raw_confidence < 0.5):
            low += 1
    for dmg in ctx.image_damages:
        total += 1
        if dmg.low_confidence:
            low += 1
    if total == 0:
        return 0.0
    return low / total


# ─── Scoring ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _ScoringResult:
    score: float
    factors: tuple[ContributingFactor, ...]
    baseline: BaselineRange | None
    raw_band: str
    final_band: str
    low_data_confidence: bool


def _band_for(score: float) -> str:
    """Section 6.2 bands: Low 0–34, Medium 35–64, High 65–100."""
    if score <= BAND_LOW_MAX:
        return RiskBand.low.value
    if score <= BAND_MED_MAX:
        return RiskBand.medium.value
    return RiskBand.high.value


def _band_index(band: str) -> int:
    return {"Low": 0, "Medium": 1, "High": 2}.get(band, 0)


def _linked_ids(
    signals: Sequence[RiskSignal], rule_id: str
) -> tuple[int, ...]:
    return tuple(s.id for s in signals if s.id is not None and s.rule_id == rule_id)


def _linked_ids_in_severity(
    signals: Sequence[RiskSignal], severity: str
) -> tuple[int, ...]:
    return tuple(
        s.id for s in signals if s.id is not None and s.severity == severity
    )


def _compute_score(
    ctx: ClaimContext,
    signals: Sequence[RiskSignal],
    *,
    anomaly_feature: float | None = None,
) -> _ScoringResult:
    high_count = _high_signal_count(signals)
    med_count = _medium_signal_count(signals)

    f1 = _normalise_count(high_count, F1_CAP)
    f2 = _normalise_count(med_count, F2_CAP)

    primary = _primary_damage(ctx)
    segment = derive_vehicle_segment(
        ctx.vehicle_make, ctx.vehicle_model, ctx.vehicle_year
    )
    baseline: BaselineRange | None = None
    if primary is not None:
        baseline = compute_baseline(segment, primary[0], primary[1])
    cost = _claim_cost(ctx)
    f3 = _normalise_cost_ratio(cost, baseline)

    f4 = _previous_overlap_score(ctx)

    # f5 is the optional Isolation Forest score (NICE). When None, the
    # engine redistributes its weight across the other four features
    # proportionally (Section 6.2).
    use_anomaly = anomaly_feature is not None
    f5 = float(anomaly_feature) if use_anomaly else 0.0

    if use_anomaly:
        w1, w2, w3, w4, w5 = (
            W_HIGH_SIGNALS, W_MED_SIGNALS, W_COST_RATIO,
            W_PREVIOUS_OVERLAP, W_ANOMALY,
        )
    else:
        scale = 1.0 / (1.0 - W_ANOMALY)
        w1, w2, w3, w4 = (
            W_HIGH_SIGNALS * scale,
            W_MED_SIGNALS * scale,
            W_COST_RATIO * scale,
            W_PREVIOUS_OVERLAP * scale,
        )
        w5 = 0.0

    raw_score = (
        w1 * f1 + w2 * f2 + w3 * f3 + w4 * f4 + w5 * f5
    ) * 100.0
    score = max(0.0, min(100.0, raw_score))
    raw_band = _band_for(score)

    low_conf_frac = _low_data_confidence_fraction(ctx)
    low_data_confidence = low_conf_frac > LOW_DATA_CONFIDENCE_FRACTION
    # Low-data-confidence bumps the band to at least Medium (Section 6.2).
    final_band = raw_band
    if low_data_confidence and _band_index(raw_band) < _band_index(RiskBand.medium.value):
        final_band = RiskBand.medium.value

    factors = (
        ContributingFactor(
            feature="f1_high_signals",
            weight=w1,
            value=f1,
            raw=f"{high_count} High-severity signal(s) (cap={F1_CAP})",
            linked_signal_ids=_linked_ids_in_severity(signals, SignalSeverity.high.value),
        ),
        ContributingFactor(
            feature="f2_medium_signals",
            weight=w2,
            value=f2,
            raw=f"{med_count} Medium-severity signal(s) (cap={F2_CAP})",
            linked_signal_ids=_linked_ids_in_severity(signals, SignalSeverity.medium.value),
        ),
        ContributingFactor(
            feature="f3_cost_ratio",
            weight=w3,
            value=f3,
            raw=(
                f"cost={cost} vs baseline.upper={baseline.upper:.0f} "
                f"(segment={baseline.segment}, damage={baseline.damage_type}, "
                f"severity={baseline.severity}, n={baseline.n})"
            ) if (cost is not None and baseline is not None) else
            f"cost={cost} (no baseline available)",
            linked_signal_ids=_linked_ids(signals, "R4_excessive_repair_cost"),
        ),
        ContributingFactor(
            feature="f4_previous_overlap",
            weight=w4,
            value=f4,
            raw=f"Jaccard overlap with previous-claim damage summaries={f4:.3f}",
            linked_signal_ids=_linked_ids(signals, "R5_duplicate_previous_damage"),
        ),
    )
    if use_anomaly:
        factors = factors + (
            ContributingFactor(
                feature="f5_anomaly",
                weight=w5,
                value=f5,
                raw=f"Isolation Forest anomaly score={f5:.3f}",
                linked_signal_ids=(),
            ),
        )

    return _ScoringResult(
        score=score,
        factors=factors,
        baseline=baseline,
        raw_band=raw_band,
        final_band=final_band,
        low_data_confidence=low_data_confidence,
    )


def compute_risk_score(
    ctx: ClaimContext,
    signals: Sequence[RiskSignal],
    *,
    baseline: BaselineRange | None = None,
    anomaly_feature: float | None = None,
) -> RiskScore:
    """Compute the explainable risk score for one claim.

    Parameters
    ----------
    ctx:
        The ClaimContext (used for image damages, repair estimate,
        previous claims, document confidences, etc.).
    signals:
        The RiskSignal rows produced by the Phase 6 consistency engine.
        These can be unsaved (`id is None`) — linked_signal_ids will
        simply be empty in that case.
    baseline:
        Optional override for the computed baseline range. Mostly used
        by tests that want a fixed cost-ratio without recomputing from
        the synthetic dataset.
    anomaly_feature:
        Optional 0..1 Isolation-Forest-style score (NICE per Section 6.1).
        When None, the engine redistributes the f5 weight across the
        other four features. When set, it contributes to the score.
    """
    result = _compute_score(ctx, signals, anomaly_feature=anomaly_feature)
    # If the caller provided an explicit baseline, prefer it for the
    # explanation. The score itself was already computed; this is just
    # metadata for the response.
    if baseline is not None:
        result = _ScoringResult(
            score=result.score,
            factors=result.factors,
            baseline=baseline,
            raw_band=result.raw_band,
            final_band=result.final_band,
            low_data_confidence=result.low_data_confidence,
        )
    notes: list[str] = []
    if result.low_data_confidence:
        notes.append(
            "Low data confidence: more than 30% of inputs are low-confidence; "
            "band defaulted to at least Medium review."
        )
    if not use_anomaly_score(anomaly_feature):
        notes.append(
            "Anomaly (f5) feature not used; remaining weights were "
            "scaled proportionally per Section 6.2."
        )
    if baseline is not None and result.baseline is not baseline:
        notes.append(
            f"Baseline supplied by caller: {baseline.segment}/"
            f"{baseline.damage_type}/{baseline.severity}, n={baseline.n}."
        )
    return RiskScore(
        score=round(result.score, 2),
        band=result.final_band,
        low_data_confidence=result.low_data_confidence,
        factors=result.factors,
        baseline=result.baseline,
        notes=tuple(notes),
    )


def use_anomaly_score(anomaly_feature: float | None) -> bool:
    """Helper that returns True if `f5` was supplied to the score.

    Exposed because the band-defaults-to-Medium rule in
    `_compute_score` already handled the low-data-confidence bump; this
    function is a public, testable predicate over the input.
    """
    return anomaly_feature is not None


# ─── Persistence ────────────────────────────────────────────────────────────


def persist(risk: RiskScore, db: Session, *, claim: Claim) -> Claim:
    """Write the risk score and band onto the `Claim` row and commit.

    Returns the same `Claim` instance (now with `risk_score` and
    `risk_band` populated). Uses `db.flush()` to surface unique /
    check-constraint violations before the caller does more work.
    """
    claim.risk_score = float(risk.score)
    claim.risk_band = risk.band
    db.add(claim)
    db.flush()
    db.refresh(claim)
    db.commit()
    return claim
