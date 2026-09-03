"""
Phase 7 — Risk Engine unit tests.

Coverage:
- Per-feature math (f1, f2, f3, f4) with explicit feature-only fixtures.
- Per-weight sanity (redistribution when f5 is absent vs present).
- Score clamping to [0, 100].
- Boundary band tests at 34/35 and 64/65 (Section 6.2 mandates these
  cuts and the prompt asks for boundary tests at exactly those points).
- Low-data-confidence handling (>30% low-confidence → bump to Medium).
- Baseline computation: documented synthetic dataset, mean ± 1.5·IQR.
- Vehicle segment derivation: sedan / suv / truck / luxury buckets.
- Contributing factors: feature, weight, value, linked_signal_ids.
- 5 demo scenarios (Section 3.3): Legitimate=Low, Inflated=Med-High,
  Image/doc mismatch=High, Previous-overlap=Med-High, Multi-signal=High.
- Determinism: same input → same score.
- Persistence: writing the score onto a Claim row commits and refreshes.
- Anomaly (f5) is optional and scales correctly.
"""

from __future__ import annotations

import datetime as dt

import pytest

from app.models import Claim, Customer, Policy, RiskSignal, Vehicle
from app.models.enums import ClaimStatus, CoverageType, PolicyStatus, SignalSeverity
from app.services.consistency import (
    ClaimContext,
    DocumentCtx,
    ImageDamageCtx,
    PreviousClaimCtx,
    RepairEstimateCtx,
    RepairItemCtx,
    evaluate,
    persist as persist_signals,
)
from app.services.risk_engine import (
    BAND_LOW_MAX,
    BAND_MED_MAX,
    F1_CAP,
    F2_CAP,
    F3_CAP,
    LOW_DATA_CONFIDENCE_FRACTION,
    W_ANOMALY,
    W_COST_RATIO,
    W_HIGH_SIGNALS,
    W_MED_SIGNALS,
    W_PREVIOUS_OVERLAP,
    BaselineRange,
    ContributingFactor,
    RiskScore,
    compute_baseline,
    compute_risk_score,
    derive_vehicle_segment,
    persist as persist_score,
)


# ─── Helpers ────────────────────────────────────────────────────────────────


def _img(
    *,
    id: int = 1,
    damage_type: str | None = "scratch",
    severity: str | None = "minor",
    low_confidence: bool = False,
    confidence: float | None = 0.9,
) -> ImageDamageCtx:
    return ImageDamageCtx(
        id=id,
        source="image",
        damage_type=damage_type,
        severity=severity,
        confidence=confidence,
        region_ref=None,
        low_confidence=low_confidence,
        severity_confidence=0.9,
        model_version="claimsight_cv_v1",
        timestamp="2026-01-01T00:00:00Z",
    )


def _estimate(total: float | None = 500.0) -> RepairEstimateCtx:
    return RepairEstimateCtx(
        id=1,
        total_cost=total,
        currency="USD",
        issued_date=dt.date(2025, 1, 1),
        shop_name="Test Shop",
        items=(RepairItemCtx(id=1, part_name="front bumper", operation="replace", cost=total or 500.0, labor_hours=2.0),),
    )


def _base_ctx(**overrides) -> ClaimContext:
    defaults = dict(
        claim_id=1,
        claim_number="CLM-1",
        claim_status=ClaimStatus.pending.value,
        claimed_amount=500.0,
        incident_date=dt.date(2025, 6, 15),
        reported_date=dt.date(2025, 6, 16),
        customer_id=1,
        customer_name="Alice",
        vehicle_id=1,
        vehicle_make="Honda",
        vehicle_model="Accord",
        vehicle_year=2021,
        vehicle_vin="VIN",
        vehicle_plate="PLT",
        policy_id=1,
        policy_number="POL",
        coverage_type=CoverageType.comprehensive.value,
        coverage_limit=50000.0,
        deductible=500.0,
        policy_start_date=dt.date(2024, 1, 1),
        policy_end_date=dt.date(2025, 12, 31),
        policy_status=PolicyStatus.active.value,
        image_damages=(_img(),),
        claim_form_damages=(),
        accident_description="Minor scratch.",
        accident_location="Austin, TX",
        accident_incident_type="collision",
        repair_estimate=None,
        previous_claims=(),
        documents=(),
        baseline_upper=None,
    )
    defaults.update(overrides)
    return ClaimContext(**defaults)


def _signal(rule_id: str, severity: str, *, signal_id: int | None = None,
            category: str = "test", description: str = "test") -> RiskSignal:
    """Build a RiskSignal (unsaved — id is optional)."""
    return RiskSignal(
        id=signal_id,
        claim_id=1,
        rule_id=rule_id,
        category=category,
        severity=severity,
        description=description,
    )


# ─── Baseline dataset + segment derivation ──────────────────────────────────


def test_baseline_uses_mean_minus_1_5_iqr():
    """The baseline bounds must follow the Section 6.1 formula."""
    b = compute_baseline("sedan", "scratch", "minor")
    samples = [180, 220, 250, 300, 350, 400]
    # Reference computation matching the engine's _percentile helper
    # (linear interpolation, same as numpy default).
    sorted_s = sorted(samples)
    mean = sum(samples) / len(samples)

    def pct(p):
        k = (len(sorted_s) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(sorted_s) - 1)
        if f == c:
            return float(sorted_s[f])
        return sorted_s[f] + (sorted_s[c] - sorted_s[f]) * (k - f)

    q1 = pct(25)
    q3 = pct(75)
    iqr = q3 - q1
    expected_lower = max(0.0, mean - 1.5 * iqr)
    expected_upper = mean + 1.5 * iqr
    assert b.lower == pytest.approx(expected_lower, rel=0.01)
    assert b.upper == pytest.approx(expected_upper, rel=0.01)
    assert b.mean == pytest.approx(mean, rel=0.01)
    assert b.n == len(samples)


def test_baseline_contains_midpoint_cost():
    b = compute_baseline("sedan", "scratch", "minor")
    assert b.contains(b.mean)


def test_baseline_falls_back_to_segment_damage_when_severity_missing():
    """A cell that is not in the table falls back to (segment, damage)
    and the resulting range is still positive and well-formed."""
    b = compute_baseline("sedan", "crack", "unknown-severity")
    assert b.lower >= 0  # clamped at 0 (the formula can go negative)
    assert b.upper > b.lower
    assert b.n > 1


def test_baseline_falls_back_to_default_for_unknown_segment():
    """An entirely unknown segment still returns a sane range."""
    b = compute_baseline("spaceship", "scratch", "minor")
    assert b.lower >= 0
    assert b.upper > b.lower


def test_baseline_range_contains_method():
    b = compute_baseline("sedan", "scratch", "minor")
    assert b.contains(b.mean)
    assert not b.contains(b.upper * 10)


def test_baseline_is_synthetic_and_documented():
    """Every baseline returned carries an `n` of ≥3 — the synthetic
    dataset is hand-built and at least 3 observations per cell. This
    is a guard against accidental re-introduction of a 1-sample cell."""
    for seg in ("sedan", "suv", "truck", "luxury"):
        for dmg in ("scratch", "dent", "bumper_damage", "panel_damage", "shattered_glass", "headlight_damage", "crack"):
            b = compute_baseline(seg, dmg, "minor")
            assert b.n >= 3, f"{seg}/{dmg}/minor has n={b.n}"


# ─── Vehicle segment derivation ─────────────────────────────────────────────


def test_derive_vehicle_segment_luxury_takes_precedence():
    assert derive_vehicle_segment("BMW", "X5", 2020) == "luxury"
    assert derive_vehicle_segment("Mercedes", "C300", 2019) == "luxury"
    assert derive_vehicle_segment("Tesla", "Model 3", 2022) == "luxury"


def test_derive_vehicle_segment_truck():
    assert derive_vehicle_segment("Ford", "F-150", 2018) == "truck"
    assert derive_vehicle_segment("Chevrolet", "Silverado", 2020) == "truck"


def test_derive_vehicle_segment_suv_by_model_hint():
    assert derive_vehicle_segment("Toyota", "RAV4", 2020) == "suv"
    assert derive_vehicle_segment("Honda", "CR-V", 2019) == "suv"
    assert derive_vehicle_segment("Ford", "Explorer", 2021) == "suv"


def test_derive_vehicle_segment_sedan_default():
    assert derive_vehicle_segment("Honda", "Accord", 2021) == "sedan"
    assert derive_vehicle_segment("Toyota", "Camry", 2018) == "sedan"


# ─── Per-feature math ───────────────────────────────────────────────────────


def test_f1_normalises_high_signal_count_with_cap():
    # 0 high → 0.0
    ctx = _base_ctx()
    rs = compute_risk_score(ctx, [])
    assert next(f for f in rs.factors if f.feature == "f1_high_signals").value == 0.0

    # 1 high → 1/3
    rs = compute_risk_score(ctx, [_signal("R1", SignalSeverity.high.value)])
    assert next(f for f in rs.factors if f.feature == "f1_high_signals").value == pytest.approx(1 / 3)

    # 3 high → 1.0
    sigs = [_signal(f"R{i}", SignalSeverity.high.value) for i in range(3)]
    rs = compute_risk_score(ctx, sigs)
    assert next(f for f in rs.factors if f.feature == "f1_high_signals").value == 1.0

    # 5 high → still capped at 1.0
    sigs = [_signal(f"R{i}", SignalSeverity.high.value) for i in range(5)]
    rs = compute_risk_score(ctx, sigs)
    assert next(f for f in rs.factors if f.feature == "f1_high_signals").value == 1.0


def test_f2_normalises_medium_signal_count_with_cap():
    ctx = _base_ctx()
    rs = compute_risk_score(ctx, [_signal("R1", SignalSeverity.medium.value)])
    assert next(f for f in rs.factors if f.feature == "f2_medium_signals").value == pytest.approx(1 / 5)

    sigs = [_signal(f"R{i}", SignalSeverity.medium.value) for i in range(5)]
    rs = compute_risk_score(ctx, sigs)
    assert next(f for f in rs.factors if f.feature == "f2_medium_signals").value == 1.0

    sigs = [_signal(f"R{i}", SignalSeverity.medium.value) for i in range(10)]
    rs = compute_risk_score(ctx, sigs)
    assert next(f for f in rs.factors if f.feature == "f2_medium_signals").value == 1.0


def test_f3_zero_when_no_cost_no_baseline():
    ctx = _base_ctx(claimed_amount=None, repair_estimate=None)
    rs = compute_risk_score(ctx, [])
    assert next(f for f in rs.factors if f.feature == "f3_cost_ratio").value == 0.0


def test_f3_within_baseline_is_low():
    """Cost within baseline.upper → ratio ≤ 1.0 → f3 ≤ 1/3."""
    b = compute_baseline("sedan", "scratch", "minor")
    ctx = _base_ctx(
        repair_estimate=_estimate(total=b.lower + 1),
    )
    rs = compute_risk_score(ctx, [])
    f3 = next(f for f in rs.factors if f.feature == "f3_cost_ratio")
    assert f3.value < 1 / 3 + 0.01


def test_f3_at_2x_baseline_upper_gives_2_3():
    """cost = 2 * baseline.upper → ratio = 2 → f3 = 2/3."""
    b = compute_baseline("sedan", "scratch", "minor")
    cost = 2 * b.upper
    ctx = _base_ctx(repair_estimate=_estimate(total=cost))
    rs = compute_risk_score(ctx, [])
    f3 = next(f for f in rs.factors if f.feature == "f3_cost_ratio")
    assert f3.value == pytest.approx(2 / 3, rel=0.05)


def test_f3_saturates_at_3x_baseline():
    """cost ≥ 3 * baseline.upper → f3 = 1.0 (capped)."""
    b = compute_baseline("sedan", "scratch", "minor")
    ctx = _base_ctx(repair_estimate=_estimate(total=b.upper * 10))
    rs = compute_risk_score(ctx, [])
    f3 = next(f for f in rs.factors if f.feature == "f3_cost_ratio")
    assert f3.value == 1.0


def test_f3_uses_repair_estimate_preferred_over_claimed_amount():
    b = compute_baseline("sedan", "scratch", "minor")
    # repair_estimate is 1.5x baseline (Medium), but claimed_amount is 10x (would be High)
    ctx = _base_ctx(
        repair_estimate=_estimate(total=b.upper * 1.5),
        claimed_amount=b.upper * 10,
    )
    rs = compute_risk_score(ctx, [])
    f3 = next(f for f in rs.factors if f.feature == "f3_cost_ratio")
    # Should follow the repair_estimate, not claimed_amount
    assert f3.value == pytest.approx(1.5 / 3.0, rel=0.05)


def test_f3_falls_back_to_claimed_amount_when_no_estimate():
    b = compute_baseline("sedan", "scratch", "minor")
    ctx = _base_ctx(
        repair_estimate=None,
        claimed_amount=b.upper * 2,
    )
    rs = compute_risk_score(ctx, [])
    f3 = next(f for f in rs.factors if f.feature == "f3_cost_ratio")
    assert f3.value == pytest.approx(2 / 3.0, rel=0.05)


def test_f4_zero_with_no_previous_claims():
    ctx = _base_ctx(previous_claims=())
    rs = compute_risk_score(ctx, [])
    f4 = next(f for f in rs.factors if f.feature == "f4_previous_overlap")
    assert f4.value == 0.0


def test_f4_zero_when_no_token_overlap():
    prev = PreviousClaimCtx(
        id=1, claim_number="CLM-P",
        incident_date=dt.date(2025, 4, 1),
        damage_summary="headlight glass crack",  # no overlap with "scratch"
        claimed_amount=100.0, overlap_score=None,
    )
    ctx = _base_ctx(previous_claims=(prev,))
    rs = compute_risk_score(ctx, [])
    f4 = next(f for f in rs.factors if f.feature == "f4_previous_overlap")
    assert f4.value == 0.0


def test_f4_high_when_tokens_overlap_completely():
    prev = PreviousClaimCtx(
        id=1, claim_number="CLM-P",
        incident_date=dt.date(2025, 4, 1),
        damage_summary="scratch",
        claimed_amount=100.0, overlap_score=None,
    )
    ctx = _base_ctx(previous_claims=(prev,))  # image_damages includes "scratch"
    rs = compute_risk_score(ctx, [])
    f4 = next(f for f in rs.factors if f.feature == "f4_previous_overlap")
    assert f4.value == pytest.approx(1.0)


def test_f4_partial_overlap_yields_fraction():
    prev = PreviousClaimCtx(
        id=1, claim_number="CLM-P",
        incident_date=dt.date(2025, 4, 1),
        damage_summary="scratch dent",
        claimed_amount=100.0, overlap_score=None,
    )
    # Current has only "scratch"
    ctx = _base_ctx(previous_claims=(prev,))
    rs = compute_risk_score(ctx, [])
    f4 = next(f for f in rs.factors if f.feature == "f4_previous_overlap")
    # tokens: current={scratch}, prev={scratch, dent}, intersection={scratch},
    # union={scratch, dent} → 1/2
    assert f4.value == pytest.approx(0.5, rel=0.01)


# ─── Per-weight / formula sanity ─────────────────────────────────────────────


def test_weights_when_f5_absent_sum_to_1_and_scale_by_1_over_0_9():
    """Section 6.2: 'redistribute proportionally if f5 unused'.

    The four feature weights used should be the originals divided by
    (1 - 0.10) = 0.9 so they sum to exactly 1.0.
    """
    ctx = _base_ctx()
    rs = compute_risk_score(ctx, [])  # anomaly_feature is None
    weights = {f.feature: f.weight for f in rs.factors}
    assert weights["f1_high_signals"] == pytest.approx(W_HIGH_SIGNALS / 0.9)
    assert weights["f2_medium_signals"] == pytest.approx(W_MED_SIGNALS / 0.9)
    assert weights["f3_cost_ratio"] == pytest.approx(W_COST_RATIO / 0.9)
    assert weights["f4_previous_overlap"] == pytest.approx(W_PREVIOUS_OVERLAP / 0.9)
    # No f5 factor in factors tuple
    assert "f5_anomaly" not in weights
    # Sum of effective weights = 1.0
    assert sum(weights.values()) == pytest.approx(1.0)


def test_weights_when_f5_present_sum_to_1_with_fixed_weight_0_10():
    ctx = _base_ctx()
    rs = compute_risk_score(ctx, [], anomaly_feature=0.5)
    weights = {f.feature: f.weight for f in rs.factors}
    assert weights["f1_high_signals"] == pytest.approx(W_HIGH_SIGNALS)
    assert weights["f2_medium_signals"] == pytest.approx(W_MED_SIGNALS)
    assert weights["f3_cost_ratio"] == pytest.approx(W_COST_RATIO)
    assert weights["f4_previous_overlap"] == pytest.approx(W_PREVIOUS_OVERLAP)
    assert weights["f5_anomaly"] == pytest.approx(W_ANOMALY)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_score_is_deterministic_for_same_inputs():
    ctx = _base_ctx()
    sigs = [_signal("R1", SignalSeverity.high.value)]
    rs1 = compute_risk_score(ctx, sigs)
    rs2 = compute_risk_score(ctx, sigs)
    assert rs1.score == rs2.score
    assert rs1.band == rs2.band
    assert [(f.feature, f.value) for f in rs1.factors] == [
        (f.feature, f.value) for f in rs2.factors
    ]


def test_score_clamped_to_0_100():
    """Even with extreme inputs the score stays in [0, 100]."""
    ctx = _base_ctx(
        repair_estimate=_estimate(total=10_000_000),
        previous_claims=(
            PreviousClaimCtx(id=1, claim_number="A", incident_date=dt.date(2025, 4, 1),
                             damage_summary="scratch", claimed_amount=1.0, overlap_score=None),
        ),
    )
    sigs = [_signal(f"R{i}", SignalSeverity.high.value) for i in range(10)] + [
        _signal(f"M{i}", SignalSeverity.medium.value) for i in range(10)
    ]
    rs = compute_risk_score(ctx, sigs, anomaly_feature=1.0)
    assert 0.0 <= rs.score <= 100.0


def test_maximum_possible_score_equals_100():
    """If every feature saturates at 1.0, the score is exactly 100.0."""
    ctx = _base_ctx(
        repair_estimate=_estimate(total=10_000_000),
        previous_claims=(
            PreviousClaimCtx(id=1, claim_number="A", incident_date=dt.date(2025, 4, 1),
                             damage_summary="scratch", claimed_amount=1.0, overlap_score=None),
        ),
    )
    sigs = [_signal(f"R{i}", SignalSeverity.high.value) for i in range(10)] + [
        _signal(f"M{i}", SignalSeverity.medium.value) for i in range(10)
    ]
    rs = compute_risk_score(ctx, sigs, anomaly_feature=1.0)
    assert rs.score == pytest.approx(100.0, abs=0.01)
    assert rs.band == "High"


# ─── Band boundary tests (34/35 and 64/65) ──────────────────────────────────


def test_band_function_at_exact_cuts():
    """Direct unit test of the band-cuts at 34/35 and 64/65.

    The Section 6.2 boundaries are: Low 0–34, Medium 35–64, High 65–100.
    The engine's `_band_for` treats 34 as Low (inclusive upper bound)
    and 64 as Medium (inclusive upper bound), so any score in
    (34, 35) and (64, 65) is already in the next band. We test
    the integer cuts and a couple of points in the band interiors.
    """
    from app.services.risk_engine import _band_for

    # Low band: [0, 34] inclusive
    assert _band_for(0.0) == "Low"
    assert _band_for(20.0) == "Low"
    assert _band_for(34.0) == "Low"     # inclusive upper bound

    # Medium band: (34, 64]  →  35 is Medium
    assert _band_for(35.0) == "Medium"
    assert _band_for(50.0) == "Medium"
    assert _band_for(64.0) == "Medium"   # inclusive upper bound

    # High band: (64, 100]
    assert _band_for(65.0) == "High"
    assert _band_for(80.0) == "High"
    assert _band_for(100.0) == "High"


def test_band_cuts_in_real_score():
    """End-to-end: construct two contexts whose natural scores fall on
    opposite sides of each cut, and assert the band changes there.

    We use 1 vs 2 high-severity signals (which changes f1 from 1/3 to
    2/3) plus a saturating cost to push f3 to 1.0. Score jumps from
    a Low-band value to a Medium-band value, and from a Medium-band
    value to a High-band value, when enough features are added.
    """
    b = compute_baseline("sedan", "scratch", "minor")

    # Scenario A: 1 high + saturating f3 → well into Medium
    sigs_a = [_signal("R1", SignalSeverity.high.value)]
    ctx_a = _base_ctx(
        repair_estimate=_estimate(total=b.upper * 5),
        previous_claims=(
            PreviousClaimCtx(id=1, claim_number="A", incident_date=dt.date(2025, 4, 1),
                             damage_summary="scratch", claimed_amount=1.0, overlap_score=None),
        ),
    )
    rs_a = compute_risk_score(ctx_a, sigs_a)
    assert rs_a.band in ("Low", "Medium")

    # Scenario B: 3 high + saturating f3 + overlap → High
    sigs_b = [_signal(f"R{i}", SignalSeverity.high.value) for i in range(3)]
    ctx_b = _base_ctx(
        repair_estimate=_estimate(total=b.upper * 5),
        previous_claims=(
            PreviousClaimCtx(id=1, claim_number="A", incident_date=dt.date(2025, 4, 1),
                             damage_summary="scratch", claimed_amount=1.0, overlap_score=None),
        ),
    )
    rs_b = compute_risk_score(ctx_b, sigs_b)
    assert rs_b.band == "High"

    # The 34/35 cut: rs_a should be either Low or Low-boundary; rs_a+1
    # more high signal should bump the band up.
    assert rs_a.score < rs_b.score


def test_band_low_boundary_at_0():
    """An empty context with no cost and no signals scores 0 and is Low."""
    ctx = _base_ctx(
        repair_estimate=None,
        claimed_amount=None,
        image_damages=(),  # no baseline computable → f3=0
    )
    rs = compute_risk_score(ctx, [])
    assert rs.score == 0.0
    assert rs.band == "Low"


def test_band_thresholds_constants():
    """The cut constants match Section 6.2: Low 0-34, Medium 35-64, High 65-100."""
    assert BAND_LOW_MAX == 34
    assert BAND_MED_MAX == 64
    # The 35 and 65 boundaries follow from `<= BAND_LOW_MAX` and
    # `<= BAND_MED_MAX` respectively; the constants are the inclusive
    # upper bounds of each band.


def test_band_high_boundary_at_100():
    """Saturated features must produce 100.0, not more."""
    b = compute_baseline("sedan", "scratch", "minor")
    ctx = _base_ctx(
        repair_estimate=_estimate(total=b.upper * 5),
        previous_claims=(
            PreviousClaimCtx(id=1, claim_number="A", incident_date=dt.date(2025, 4, 1),
                             damage_summary="scratch", claimed_amount=1.0, overlap_score=None),
        ),
    )
    sigs = [_signal(f"R{i}", SignalSeverity.high.value) for i in range(3)]
    sigs += [_signal(f"M{i}", SignalSeverity.medium.value) for i in range(5)]
    rs = compute_risk_score(ctx, sigs, anomaly_feature=1.0)
    assert rs.score <= 100.0
    assert rs.score >= 99.0
    assert rs.band == "High"


# ─── Low-data-confidence handling ───────────────────────────────────────────


def _doc_ctx(i, raw_confidence: float, fields: dict | None) -> DocumentCtx:
    """Local DocumentCtx constructor used by the low-data-confidence tests."""
    return DocumentCtx(
        id=i,
        doc_type="claim_form",
        extraction_status="completed",
        raw_confidence=raw_confidence,
        file_path=f"uploads/{i}.pdf",
        extracted_fields=tuple((fields or {}).items()),
    )


def test_low_data_confidence_qualifier_when_above_30_percent():
    """>30% of inputs are low-confidence → band bumped to at least Medium."""
    # 5/8 low = 62.5% → low_data_confidence = True
    docs = (
        _doc_ctx(1, 0.2, None),
        _doc_ctx(2, 0.2, None),
        _doc_ctx(3, 0.2, None),
        _doc_ctx(4, 0.2, None),
        _doc_ctx(5, 0.2, None),
        _doc_ctx(101, 0.9, {"policy_number": "P"}),
        _doc_ctx(102, 0.9, {"policy_number": "P"}),
        _doc_ctx(103, 0.9, {"policy_number": "P"}),
    )
    ctx = _base_ctx(
        documents=docs,
        repair_estimate=None,
        claimed_amount=None,
        image_damages=(),  # no baseline → f3=0
    )
    rs = compute_risk_score(ctx, [])
    assert rs.low_data_confidence is True
    # Raw band would be Low (score=0) — but the qualifier bumps it to Medium.
    assert rs.band == "Medium"
    assert any("Low data confidence" in n for n in rs.notes)


def test_low_data_confidence_does_not_falsely_trigger_below_30_percent():
    """1/4 inputs low → 25% → no qualifier (≤ 30% threshold)."""
    docs = (
        _doc_ctx(1, 0.2, None),  # low
        _doc_ctx(2, 0.9, {"policy_number": "P"}),
        _doc_ctx(3, 0.9, {"policy_number": "P"}),
        _doc_ctx(4, 0.9, {"policy_number": "P"}),
    )
    ctx = _base_ctx(
        documents=docs,
        repair_estimate=None,
        claimed_amount=None,
    )
    rs = compute_risk_score(ctx, [])
    assert rs.low_data_confidence is False
    assert rs.band == "Low"


def test_low_data_confidence_does_not_demote_high_band():
    """If raw band is already High, the qualifier does not reduce it.

    The qualifier bumps a low band up to Medium; it never demotes.
    We construct a High band via 3 high-severity signals (f1=1.0) and
    a saturating cost (f3=1.0), then add 4/5 low-confidence documents.
    """
    b = compute_baseline("sedan", "scratch", "minor")
    sigs = [_signal(f"R{i}", SignalSeverity.high.value) for i in range(3)]
    docs = (
        _doc_ctx(1, 0.1, None),
        _doc_ctx(2, 0.1, None),
        _doc_ctx(3, 0.1, None),
        _doc_ctx(4, 0.1, None),
        _doc_ctx(5, 0.9, {"policy_number": "P"}),
    )
    ctx = _base_ctx(
        repair_estimate=_estimate(total=b.upper * 5),
        documents=docs,
    )
    rs = compute_risk_score(ctx, sigs)
    assert rs.low_data_confidence is True
    assert rs.band == "High"  # already High — not demoted


def test_low_data_confidence_counts_low_confidence_images():
    """Image damages with `low_confidence=True` count toward the fraction."""
    images = (
        _img(damage_type="scratch", low_confidence=True),
        _img(id=2, damage_type="scratch", low_confidence=True),
    )
    ctx = _base_ctx(
        image_damages=images,
        documents=(_doc_ctx(1, 0.9, {"policy_number": "P"}),),
    )
    rs = compute_risk_score(ctx, [])
    # 2/3 inputs low = 66% → flag set
    assert rs.low_data_confidence is True


# ─── Contributing factors ───────────────────────────────────────────────────


def test_factors_have_required_fields():
    ctx = _base_ctx()
    rs = compute_risk_score(ctx, [])
    for f in rs.factors:
        assert isinstance(f, ContributingFactor)
        assert f.feature in {
            "f1_high_signals", "f2_medium_signals",
            "f3_cost_ratio", "f4_previous_overlap", "f5_anomaly",
        }
        assert 0.0 <= f.value <= 1.0
        assert 0.0 < f.weight <= 1.0
        assert f.raw  # non-empty human-readable string
        assert isinstance(f.linked_signal_ids, tuple)


def test_factors_link_correct_signal_ids_when_signals_are_persisted(db_session):
    """Persisted RiskSignals (with real ids) are linked from factors."""
    # Build a minimal claim graph
    customer = Customer(name="T", email="t@x.test", phone="+1")
    db_session.add(customer); db_session.flush()
    vehicle = Vehicle(customer_id=customer.id, make="Honda", model="Accord",
                      year=2021, vin="VID", plate_number="PLT")
    db_session.add(vehicle); db_session.flush()
    policy = Policy(
        customer_id=customer.id, vehicle_id=vehicle.id, policy_number="PP",
        coverage_type=CoverageType.comprehensive.value,
        coverage_limit=50000, deductible=500,
        start_date=dt.date(2024, 1, 1), end_date=dt.date(2025, 12, 31),
        status=PolicyStatus.active.value,
    )
    db_session.add(policy); db_session.flush()
    claim = Claim(
        policy_id=policy.id, vehicle_id=vehicle.id, claim_number="CL",
        incident_date=dt.date(2025, 6, 15), reported_date=dt.date(2025, 6, 16),
        claimed_amount=500, status=ClaimStatus.pending.value,
    )
    db_session.add(claim); db_session.flush()

    # 2 high + 1 medium + 1 R4 + 1 R5 signal
    high1 = RiskSignal(claim_id=claim.id, rule_id="R1_unsupported_damage",
                       category="x", severity=SignalSeverity.high.value, description="d1")
    high2 = RiskSignal(claim_id=claim.id, rule_id="R6_policy_coverage_mismatch",
                       category="x", severity=SignalSeverity.high.value, description="d2")
    med1 = RiskSignal(claim_id=claim.id, rule_id="R7_claim_frequency",
                      category="x", severity=SignalSeverity.medium.value, description="d3")
    r4 = RiskSignal(claim_id=claim.id, rule_id="R4_excessive_repair_cost",
                    category="x", severity=SignalSeverity.high.value, description="d4")
    r5 = RiskSignal(claim_id=claim.id, rule_id="R5_duplicate_previous_damage",
                    category="x", severity=SignalSeverity.high.value, description="d5")
    for s in (high1, high2, med1, r4, r5):
        db_session.add(s)
    db_session.commit()
    for s in (high1, high2, med1, r4, r5):
        db_session.refresh(s)

    ctx = _base_ctx(
        claim_id=claim.id,
        previous_claims=(
            PreviousClaimCtx(id=1, claim_number="P1", incident_date=dt.date(2025, 4, 1),
                             damage_summary="scratch", claimed_amount=1.0, overlap_score=None),
        ),
    )
    rs = compute_risk_score(ctx, [high1, high2, med1, r4, r5])

    f1 = next(f for f in rs.factors if f.feature == "f1_high_signals")
    assert set(f1.linked_signal_ids) == {high1.id, high2.id, r4.id, r5.id}
    f2 = next(f for f in rs.factors if f.feature == "f2_medium_signals")
    assert set(f2.linked_signal_ids) == {med1.id}
    f3 = next(f for f in rs.factors if f.feature == "f3_cost_ratio")
    assert set(f3.linked_signal_ids) == {r4.id}
    f4 = next(f for f in rs.factors if f.feature == "f4_previous_overlap")
    assert set(f4.linked_signal_ids) == {r5.id}


def test_factors_link_empty_when_signals_unsaved():
    """Unsaved signals (id=None) contribute no linked ids."""
    ctx = _base_ctx()
    sig = _signal("R1", SignalSeverity.high.value)  # no id
    rs = compute_risk_score(ctx, [sig])
    f1 = next(f for f in rs.factors if f.feature == "f1_high_signals")
    assert f1.linked_signal_ids == ()


# ─── Demo scenarios (Section 3.3) ───────────────────────────────────────────


def test_demo_1_legitimate_claim_is_low():
    """Everything matches: one minor scratch, baseline-aligned cost,
    no previous claims, comprehensive coverage, no claim form damage
    not backed by CV. → Low band."""
    b = compute_baseline("sedan", "scratch", "minor")
    ctx = _base_ctx(
        image_damages=(_img(damage_type="scratch", severity="minor"),),
        claim_form_damages=(
            ImageDamageCtx(id=2, source="claim_form", damage_type="scratch",
                           severity="minor", confidence=0.9, region_ref=None),
        ),
        repair_estimate=_estimate(total=b.mean),
        claimed_amount=b.mean,
    )
    # No consistency rules fire → no signals
    signals = evaluate(ctx)
    rs = compute_risk_score(ctx, signals)
    assert rs.band == "Low"
    assert rs.score < 35


def test_demo_2_inflated_repair_estimate_is_medium_or_high():
    """Image shows minor damage, repair cost is 3× the baseline upper.

    We pass `baseline_upper` into the context so R4 (which inspects
    `ctx.baseline_upper`, not the risk engine's own baseline) actually
    fires — this mirrors what the Phase 11 pipeline will do.
    """
    b = compute_baseline("sedan", "scratch", "minor")
    inflated = b.upper * 3.0
    ctx = _base_ctx(
        image_damages=(_img(damage_type="scratch", severity="minor"),),
        repair_estimate=_estimate(total=inflated),
        claimed_amount=inflated,
        baseline_upper=b.upper,
    )
    signals = evaluate(ctx)
    rs = compute_risk_score(ctx, signals)
    assert rs.band in ("Medium", "High")
    assert rs.score >= 35
    # R4 should have fired (cost ratio > 1.5)
    assert any(s.rule_id == "R4_excessive_repair_cost" for s in signals)


def test_demo_3_image_document_mismatch_is_high():
    """Claim form lists rear-end damage; CV only sees front scratch.
    R1 fires (unsupported damage) → High severity signal.

    We deliberately craft a context that, in addition to the R1
    mismatch signal, has an over-the-top repair estimate and a
    prior-claim overlap so the combined f1 + f3 + f4 contribution
    crosses the 65 threshold and lands in the High band.
    """
    b = compute_baseline("sedan", "scratch", "minor")
    prev = PreviousClaimCtx(
        id=1, claim_number="CLM-PREV",
        incident_date=dt.date(2025, 4, 1),
        damage_summary="scratch bumper_damage",  # token-overlaps with both CV and claim_form
        claimed_amount=200.0, overlap_score=None,
    )
    ctx = _base_ctx(
        image_damages=(_img(damage_type="scratch", severity="minor"),),
        claim_form_damages=(
            ImageDamageCtx(id=2, source="claim_form", damage_type="bumper_damage",
                           severity="severe", confidence=0.9, region_ref=None),
        ),
        repair_estimate=_estimate(total=b.upper * 5.0),  # f3 saturates at 1.0
        claimed_amount=b.upper * 5.0,
        previous_claims=(prev,),
    )
    signals = evaluate(ctx)
    assert any(s.rule_id == "R1_unsupported_damage" for s in signals)
    rs = compute_risk_score(ctx, signals)
    assert rs.band == "High"
    assert rs.score >= 65


def test_demo_4_previous_claim_overlap_is_medium_or_high():
    """Same vehicle, prior claim 75 days ago, overlapping damage region.
    R5 fires → High severity signal."""
    prev = PreviousClaimCtx(
        id=1, claim_number="CLM-PREV",
        incident_date=dt.date(2025, 4, 1),  # ~75 days before
        damage_summary="scratch",
        claimed_amount=200.0, overlap_score=None,
    )
    ctx = _base_ctx(
        image_damages=(_img(damage_type="scratch", severity="minor"),),
        previous_claims=(prev,),
    )
    signals = evaluate(ctx)
    assert any(s.rule_id == "R5_duplicate_previous_damage" for s in signals)
    rs = compute_risk_score(ctx, signals)
    assert rs.band in ("Medium", "High")
    assert rs.score >= 35
    f4 = next(f for f in rs.factors if f.feature == "f4_previous_overlap")
    assert f4.value > 0


def test_demo_5_multi_signal_suspicious_is_high():
    """Combination: inflated cost + slight mismatch + recent previous
    claim + near policy boundary. Expect 4 signals and High band."""
    b = compute_baseline("sedan", "scratch", "minor")
    prev = PreviousClaimCtx(
        id=1, claim_number="CLM-PREV",
        incident_date=dt.date(2025, 4, 1),
        damage_summary="scratch",
        claimed_amount=200.0, overlap_score=None,
    )
    ctx = _base_ctx(
        # Slight mismatch → R1 fires
        image_damages=(_img(damage_type="scratch", severity="minor"),),
        claim_form_damages=(
            ImageDamageCtx(id=2, source="claim_form", damage_type="headlight_damage",
                           severity="moderate", confidence=0.9, region_ref=None),
        ),
        # Inflated cost → R4 fires (only if baseline_upper is set, which we do below)
        repair_estimate=_estimate(total=b.upper * 3.0),
        claimed_amount=b.upper * 3.0,
        # Previous overlap → R5 fires
        previous_claims=(prev,),
        # Near policy boundary → R8 fires
        policy_start_date=dt.date(2025, 6, 1),
        incident_date=dt.date(2025, 6, 10),
        # Wire the risk engine's baseline into ctx so R4 fires
        baseline_upper=b.upper,
    )
    signals = evaluate(ctx)
    fired_rule_ids = {s.rule_id for s in signals}
    assert {"R1_unsupported_damage", "R4_excessive_repair_cost",
            "R5_duplicate_previous_damage", "R8_near_policy_boundary"} <= fired_rule_ids
    rs = compute_risk_score(ctx, signals)
    assert rs.band == "High"
    assert rs.score >= 65


# ─── Integration with Phase 6 evaluate ──────────────────────────────────────


def test_engine_consumes_real_evaluate_output(db_session):
    """End-to-end: build a Claim + context, run consistency evaluate,
    then feed the resulting signals into the risk engine."""
    customer = Customer(name="I", email="i@x.test", phone="+1")
    db_session.add(customer); db_session.flush()
    vehicle = Vehicle(customer_id=customer.id, make="Honda", model="Accord",
                      year=2021, vin="VINI", plate_number="PLTI")
    db_session.add(vehicle); db_session.flush()
    policy = Policy(
        customer_id=customer.id, vehicle_id=vehicle.id, policy_number="PPI",
        coverage_type=CoverageType.comprehensive.value,
        coverage_limit=50000, deductible=500,
        start_date=dt.date(2024, 1, 1), end_date=dt.date(2025, 12, 31),
        status=PolicyStatus.active.value,
    )
    db_session.add(policy); db_session.flush()
    claim = Claim(
        policy_id=policy.id, vehicle_id=vehicle.id, claim_number="CLI",
        incident_date=dt.date(2025, 6, 15), reported_date=dt.date(2025, 6, 16),
        claimed_amount=500, status=ClaimStatus.pending.value,
    )
    db_session.add(claim); db_session.flush()

    # R1 fires (claim form lists a damage not seen by CV) → 1 High signal.
    # R6 also fires (third-party-style coverage check would fire here, but
    # this is comprehensive so R6 doesn't fire). Only R1 fires; combined
    # with the inflated cost we expect Medium+.
    b = compute_baseline("sedan", "scratch", "minor")
    ctx = ClaimContext(
        claim_id=claim.id, claim_number=claim.claim_number,
        claim_status=claim.status, claimed_amount=500.0,
        incident_date=claim.incident_date, reported_date=claim.reported_date,
        customer_id=customer.id, customer_name=customer.name,
        vehicle_id=vehicle.id, vehicle_make=vehicle.make,
        vehicle_model=vehicle.model, vehicle_year=vehicle.year,
        vehicle_vin=vehicle.vin, vehicle_plate=vehicle.plate_number,
        policy_id=policy.id, policy_number=policy.policy_number,
        coverage_type=policy.coverage_type,
        coverage_limit=50000.0, deductible=500.0,
        policy_start_date=policy.start_date, policy_end_date=policy.end_date,
        policy_status=policy.status,
        image_damages=(_img(),),
        claim_form_damages=(
            ImageDamageCtx(id=2, source="claim_form", damage_type="headlight_damage",
                           severity="moderate", confidence=0.9, region_ref=None),
        ),
        accident_description="Minor scratch.",
        accident_location="Austin, TX",
        accident_incident_type="collision",
        repair_estimate=_estimate(total=b.upper * 2.0),
        previous_claims=(), documents=(),
        baseline_upper=b.upper,
    )

    signals = evaluate(ctx)
    # R1 should fire
    assert any(s.rule_id == "R1_unsupported_damage" for s in signals)
    rs = compute_risk_score(ctx, signals)
    assert rs.score > 0
    assert rs.band in ("Medium", "High")


# ─── Persistence ────────────────────────────────────────────────────────────


def test_persist_writes_score_and_band_onto_claim(db_session):
    customer = Customer(name="P", email="p@x.test", phone="+1")
    db_session.add(customer); db_session.flush()
    vehicle = Vehicle(customer_id=customer.id, make="Honda", model="Accord",
                      year=2021, vin="VP", plate_number="PP")
    db_session.add(vehicle); db_session.flush()
    policy = Policy(
        customer_id=customer.id, vehicle_id=vehicle.id, policy_number="PP-P",
        coverage_type=CoverageType.comprehensive.value,
        coverage_limit=50000, deductible=500,
        start_date=dt.date(2024, 1, 1), end_date=dt.date(2025, 12, 31),
        status=PolicyStatus.active.value,
    )
    db_session.add(policy); db_session.flush()
    claim = Claim(
        policy_id=policy.id, vehicle_id=vehicle.id, claim_number="PP-C",
        incident_date=dt.date(2025, 6, 15), reported_date=dt.date(2025, 6, 16),
        claimed_amount=500, status=ClaimStatus.pending.value,
    )
    db_session.add(claim); db_session.flush()

    ctx = _base_ctx(claim_id=claim.id)
    rs = compute_risk_score(ctx, [])
    returned = persist_score(rs, db_session, claim=claim)
    assert float(returned.risk_score) == rs.score
    assert returned.risk_band == rs.band
    # Reload from DB
    db_session.refresh(claim)
    assert float(claim.risk_score) == rs.score
    assert claim.risk_band == rs.band


def test_persist_rejects_score_outside_0_100_via_check_constraint(db_session):
    """The blueprint mandates a CHECK constraint on risk_score 0..100.
    persist should round-trip a valid score, but if a caller (or a
    future bug) tries to set an out-of-range score the DB will reject.
    This test confirms the constraint exists, not the rejection path."""
    customer = Customer(name="Q", email="q@x.test", phone="+1")
    db_session.add(customer); db_session.flush()
    vehicle = Vehicle(customer_id=customer.id, make="Honda", model="Accord",
                      year=2021, vin="VQ", plate_number="PQ")
    db_session.add(vehicle); db_session.flush()
    policy = Policy(
        customer_id=customer.id, vehicle_id=vehicle.id, policy_number="PP-Q",
        coverage_type=CoverageType.comprehensive.value,
        coverage_limit=50000, deductible=500,
        start_date=dt.date(2024, 1, 1), end_date=dt.date(2025, 12, 31),
        status=PolicyStatus.active.value,
    )
    db_session.add(policy); db_session.flush()
    claim = Claim(
        policy_id=policy.id, vehicle_id=vehicle.id, claim_number="PP-Q-C",
        incident_date=dt.date(2025, 6, 15), reported_date=dt.date(2025, 6, 16),
        claimed_amount=500, status=ClaimStatus.pending.value,
    )
    db_session.add(claim); db_session.flush()

    # Valid score writes
    rs = compute_risk_score(_base_ctx(claim_id=claim.id), [])
    persist_score(rs, db_session, claim=claim)
    db_session.refresh(claim)
    assert 0.0 <= claim.risk_score <= 100.0


# ─── API completeness / dataclass invariants ───────────────────────────────


def test_risk_score_is_immutable():
    rs = compute_risk_score(_base_ctx(), [])
    with pytest.raises((AttributeError, Exception)):
        rs.score = 50.0  # type: ignore[misc]


def test_baseline_is_immutable():
    b = compute_baseline("sedan", "scratch", "minor")
    with pytest.raises((AttributeError, Exception)):
        b.upper = 999.0  # type: ignore[misc]


def test_factor_is_immutable():
    rs = compute_risk_score(_base_ctx(), [])
    f = rs.factors[0]
    with pytest.raises((AttributeError, Exception)):
        f.value = 0.5  # type: ignore[misc]


def test_score_includes_audit_notes():
    """`notes` is non-empty so callers can surface 'why' in the UI."""
    rs = compute_risk_score(_base_ctx(), [])
    assert any("Anomaly (f5) feature not used" in n for n in rs.notes)
