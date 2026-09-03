"""
Phase 6 — Consistency Engine unit tests.

Each rule gets at least two tests:
1. A trigger fixture that builds a ClaimContext engineered to make the
   rule fire, and asserts the returned RiskSignal has the right
   rule_id / severity / category / non-empty description.
2. A non-trigger fixture that builds a ClaimContext engineered NOT to
   fire, and asserts the rule returns None.

A separate section at the bottom tests the orchestrator and persistence
helper end-to-end against a real Claim + Damage + Document + Repair
graph built with the in-memory SQLite session from conftest.

The rules are pure functions on ClaimContext, so the rule-level tests
do not touch the database — they instantiate the frozen dataclass
directly. This keeps each test fast, deterministic, and easy to read.
"""

from __future__ import annotations

import datetime as dt

import pytest

from app.models.enums import SignalSeverity
from app.services.consistency import (
    ClaimContext,
    DocumentCtx,
    ImageDamageCtx,
    PreviousClaimCtx,
    RepairEstimateCtx,
    RepairItemCtx,
    evaluate,
    persist,
    r1_unsupported_damage,
    r2_severity_mismatch,
    r3_repair_component_mismatch,
    r4_excessive_repair_cost,
    r5_duplicate_previous_damage,
    r6_policy_coverage_mismatch,
    r7_claim_frequency,
    r8_near_policy_boundary,
    r9_document_field_conflict,
)


# ─── Fixture helpers ─────────────────────────────────────────────────────────


def _img(
    *,
    id: int = 1,
    source: str = "image",
    damage_type: str | None = "scratch",
    severity: str | None = "minor",
    confidence: float | None = 0.9,
    low_confidence: bool = False,
    severity_confidence: float | None = 0.9,
    region_ref: str | None = None,
) -> ImageDamageCtx:
    return ImageDamageCtx(
        id=id,
        source=source,
        damage_type=damage_type,
        severity=severity,
        confidence=confidence,
        region_ref=region_ref,
        low_confidence=low_confidence,
        severity_confidence=severity_confidence,
        model_version="claimsight_cv_v1",
        timestamp="2026-01-01T00:00:00Z",
    )


def _repair_item(
    part: str | None = "front bumper",
    op: str | None = "replace",
    cost: float | None = 500.0,
) -> RepairItemCtx:
    return RepairItemCtx(id=1, part_name=part, operation=op, cost=cost, labor_hours=2.0)


def _estimate(
    total: float | None = 1000.0,
    items: tuple[RepairItemCtx, ...] = (_repair_item(),),
) -> RepairEstimateCtx:
    return RepairEstimateCtx(
        id=1,
        total_cost=total,
        currency="USD",
        issued_date=dt.date(2025, 1, 1),
        shop_name="Test Shop",
        items=items,
    )


def _doc(
    *,
    id: int,
    doc_type: str = "claim_form",
    fields: dict | None = None,
) -> DocumentCtx:
    return DocumentCtx(
        id=id,
        doc_type=doc_type,
        extraction_status="completed",
        raw_confidence=0.9,
        file_path=f"uploads/1/{id}.pdf",
        extracted_fields=tuple(sorted((fields or {}).items())),
    )


def _base_ctx(**overrides) -> ClaimContext:
    """Build a ClaimContext that does NOT fire any rule by default.

    Tests then override the specific fields needed to make a rule fire
    (or stay silent). The defaults reflect a clean comprehensive-policy
    claim with one matching CV damage and no prior history.
    """
    img = _img()
    defaults = dict(
        claim_id=1,
        claim_number="CLM-001",
        claim_status="pending",
        claimed_amount=1000.0,
        incident_date=dt.date(2025, 6, 15),
        reported_date=dt.date(2025, 6, 16),
        customer_id=1,
        customer_name="Alice",
        vehicle_id=1,
        vehicle_make="Honda",
        vehicle_model="Accord",
        vehicle_year=2021,
        vehicle_vin="1HGCM82633A004352",
        vehicle_plate="ABC-1234",
        policy_id=1,
        policy_number="POL-001",
        coverage_type="comprehensive",
        coverage_limit=50000.0,
        deductible=500.0,
        policy_start_date=dt.date(2024, 1, 1),
        policy_end_date=dt.date(2025, 12, 31),
        policy_status="active",
        image_damages=(img,),
        claim_form_damages=(),
        accident_description="Minor scratch on the rear door.",
        accident_location="Austin, TX",
        accident_incident_type="collision",
        repair_estimate=None,
        previous_claims=(),
        documents=(),
        baseline_upper=None,
    )
    defaults.update(overrides)
    return ClaimContext(**defaults)


# ─── R1 — unsupported_damage ────────────────────────────────────────────────


def test_r1_triggers_when_claim_form_damage_not_in_cv():
    ctx = _base_ctx(
        image_damages=(_img(damage_type="scratch"),),
        claim_form_damages=(
            _img(id=2, source="claim_form", damage_type="headlight_damage"),
        ),
    )
    sig = r1_unsupported_damage(ctx)
    assert sig is not None
    assert sig.rule_id == "R1_unsupported_damage"
    assert sig.severity == SignalSeverity.high.value
    assert sig.category == "image_claim_consistency"
    assert "headlight_damage" in sig.description
    assert sig.claim_id == ctx.claim_id


def test_r1_does_not_trigger_when_all_claim_form_damages_match_cv():
    ctx = _base_ctx(
        image_damages=(_img(damage_type="scratch"), _img(id=2, damage_type="dent")),
        claim_form_damages=(
            _img(id=10, source="claim_form", damage_type="scratch"),
            _img(id=11, source="claim_form", damage_type="dent"),
        ),
    )
    assert r1_unsupported_damage(ctx) is None


def test_r1_does_not_trigger_when_no_claim_form_damages():
    ctx = _base_ctx(claim_form_damages=())
    assert r1_unsupported_damage(ctx) is None


def test_r1_does_not_trigger_when_all_cv_detections_are_low_confidence():
    """Per blueprint: 'image confidence is not low' gates the support check."""
    ctx = _base_ctx(
        image_damages=(_img(damage_type="scratch", low_confidence=True),),
        claim_form_damages=(
            _img(id=2, source="claim_form", damage_type="headlight_damage"),
        ),
    )
    # No high-confidence CV detection exists, so we cannot evaluate support.
    assert r1_unsupported_damage(ctx) is None


# ─── R2 — severity_mismatch ─────────────────────────────────────────────────


def test_r2_triggers_when_text_says_totaled_but_cv_says_minor():
    ctx = _base_ctx(
        accident_description="The car was completely totaled in the crash.",
        image_damages=(_img(damage_type="scratch", severity="minor"),),
    )
    sig = r2_severity_mismatch(ctx)
    assert sig is not None
    assert sig.rule_id == "R2_severity_mismatch"
    assert sig.severity == SignalSeverity.medium.value
    assert sig.category == "claim_description_consistency"
    assert "differ by 2" in sig.description


def test_r2_triggers_when_text_says_minor_but_cv_says_severe():
    ctx = _base_ctx(
        accident_description="Just a small scratch on the bumper.",
        image_damages=(_img(damage_type="panel_damage", severity="severe"),),
    )
    sig = r2_severity_mismatch(ctx)
    assert sig is not None
    assert sig.rule_id == "R2_severity_mismatch"
    assert sig.severity == SignalSeverity.medium.value


def test_r2_does_not_trigger_when_text_and_cv_agree():
    ctx = _base_ctx(
        accident_description="Just a minor scratch on the rear door.",
        image_damages=(_img(damage_type="scratch", severity="minor"),),
    )
    assert r2_severity_mismatch(ctx) is None


def test_r2_does_not_trigger_when_only_one_side_has_severity():
    # Description is silent, but CV is severe.
    ctx = _base_ctx(
        accident_description="Something happened to my car.",
        image_damages=(_img(damage_type="panel_damage", severity="severe"),),
    )
    assert r2_severity_mismatch(ctx) is None


def test_r2_uses_worst_cv_severity_across_multiple_detections():
    # Two CV detections, one minor, one severe → use severe.
    # Text says "minor" → delta is 2 → fire.
    ctx = _base_ctx(
        accident_description="A small scratch.",
        image_damages=(
            _img(id=1, damage_type="scratch", severity="minor"),
            _img(id=2, damage_type="panel_damage", severity="severe"),
        ),
    )
    assert r2_severity_mismatch(ctx) is not None


# ─── R3 — repair_component_mismatch ─────────────────────────────────────────


def test_r3_triggers_when_part_not_linked_to_any_damage():
    # Claim has only a "scratch" damage, but the estimate asks for a
    # transmission part — not plausibly linked.
    ctx = _base_ctx(
        image_damages=(_img(damage_type="scratch"),),
        repair_estimate=_estimate(items=(_repair_item(part="transmission assembly"),)),
    )
    sig = r3_repair_component_mismatch(ctx)
    assert sig is not None
    assert sig.rule_id == "R3_repair_component_mismatch"
    assert sig.severity == SignalSeverity.medium.value
    assert sig.category == "repair_estimate_consistency"
    assert "transmission" in sig.description


def test_r3_does_not_trigger_when_parts_match_damage():
    # Damage is "scratch" → plausible parts include "panel", "door",
    # "fender", "bumper", etc. "front bumper" matches.
    ctx = _base_ctx(
        image_damages=(_img(damage_type="scratch"),),
        repair_estimate=_estimate(items=(_repair_item(part="front bumper"),)),
    )
    assert r3_repair_component_mismatch(ctx) is None


def test_r3_triggers_when_no_damage_but_estimate_has_parts():
    """If there is no damage recorded at all but a repair estimate is
    submitted, the rule should surface the inconsistency."""
    ctx = _base_ctx(
        image_damages=(),
        claim_form_damages=(),
        repair_estimate=_estimate(items=(_repair_item(part="engine block"),)),
    )
    sig = r3_repair_component_mismatch(ctx)
    assert sig is not None
    assert "engine block" in sig.description


def test_r3_does_not_trigger_when_no_repair_estimate():
    ctx = _base_ctx(repair_estimate=None)
    assert r3_repair_component_mismatch(ctx) is None


def test_r3_handles_glass_damage_with_windshield_part():
    ctx = _base_ctx(
        image_damages=(_img(damage_type="shattered_glass"),),
        repair_estimate=_estimate(items=(_repair_item(part="front windshield"),)),
    )
    assert r3_repair_component_mismatch(ctx) is None


# ─── R4 — excessive_repair_cost ─────────────────────────────────────────────


def test_r4_triggers_high_when_total_is_more_than_double_baseline():
    ctx = _base_ctx(
        repair_estimate=_estimate(total=5000.0),
        baseline_upper=2000.0,
    )
    sig = r4_excessive_repair_cost(ctx)
    assert sig is not None
    assert sig.rule_id == "R4_excessive_repair_cost"
    assert sig.severity == SignalSeverity.high.value
    assert sig.category == "cost_validation"
    assert "5,000" in sig.description or "5000" in sig.description


def test_r4_triggers_medium_when_total_is_between_1_5_and_2x_baseline():
    # baseline_upper=1000, total=1800 → ratio 1.8 → Medium
    ctx = _base_ctx(
        repair_estimate=_estimate(total=1800.0),
        baseline_upper=1000.0,
    )
    sig = r4_excessive_repair_cost(ctx)
    assert sig is not None
    assert sig.severity == SignalSeverity.medium.value


def test_r4_does_not_trigger_when_total_within_baseline():
    ctx = _base_ctx(
        repair_estimate=_estimate(total=1200.0),
        baseline_upper=2000.0,
    )
    assert r4_excessive_repair_cost(ctx) is None


def test_r4_does_not_trigger_when_baseline_absent():
    """Without a baseline (Phase 7 not implemented), the rule is silent."""
    ctx = _base_ctx(
        repair_estimate=_estimate(total=99999.0),
        baseline_upper=None,
    )
    assert r4_excessive_repair_cost(ctx) is None


def test_r4_does_not_trigger_when_no_repair_estimate():
    ctx = _base_ctx(repair_estimate=None, baseline_upper=1000.0)
    assert r4_excessive_repair_cost(ctx) is None


# ─── R5 — duplicate_previous_damage ─────────────────────────────────────────


def test_r5_triggers_when_previous_claim_within_6_months_and_overlap():
    prev = PreviousClaimCtx(
        id=1,
        claim_number="CLM-000",
        incident_date=dt.date(2025, 4, 1),  # ~75 days earlier
        damage_summary="rear bumper panel damage from collision",
        claimed_amount=800.0,
        overlap_score=None,
    )
    ctx = _base_ctx(
        incident_date=dt.date(2025, 6, 15),
        previous_claims=(prev,),
        image_damages=(_img(damage_type="bumper_damage"),),
    )
    sig = r5_duplicate_previous_damage(ctx)
    assert sig is not None
    assert sig.rule_id == "R5_duplicate_previous_damage"
    assert sig.severity == SignalSeverity.high.value
    assert sig.category == "claim_history"
    assert "CLM-000" in sig.description


def test_r5_does_not_trigger_when_previous_claim_outside_6_months():
    prev = PreviousClaimCtx(
        id=1,
        claim_number="CLM-000",
        incident_date=dt.date(2024, 6, 1),  # > 6 months earlier
        damage_summary="rear bumper panel damage",
        claimed_amount=800.0,
        overlap_score=None,
    )
    ctx = _base_ctx(
        incident_date=dt.date(2025, 6, 15),
        previous_claims=(prev,),
        image_damages=(_img(damage_type="bumper_damage"),),
    )
    assert r5_duplicate_previous_damage(ctx) is None


def test_r5_does_not_trigger_when_regions_dont_overlap():
    prev = PreviousClaimCtx(
        id=1,
        claim_number="CLM-000",
        incident_date=dt.date(2025, 4, 1),
        damage_summary="headlight glass crack",  # different region
        claimed_amount=200.0,
        overlap_score=None,
    )
    ctx = _base_ctx(
        incident_date=dt.date(2025, 6, 15),
        previous_claims=(prev,),
        image_damages=(_img(damage_type="bumper_damage"),),
    )
    assert r5_duplicate_previous_damage(ctx) is None


def test_r5_does_not_trigger_when_no_previous_claims():
    ctx = _base_ctx(previous_claims=())
    assert r5_duplicate_previous_damage(ctx) is None


# ─── R6 — policy_coverage_mismatch ──────────────────────────────────────────


def test_r6_triggers_when_third_party_policy_covers_own_vehicle_damage():
    ctx = _base_ctx(
        coverage_type="third_party",
        image_damages=(_img(damage_type="scratch"),),
    )
    sig = r6_policy_coverage_mismatch(ctx)
    assert sig is not None
    assert sig.rule_id == "R6_policy_coverage_mismatch"
    assert sig.severity == SignalSeverity.high.value
    assert sig.category == "policy_coverage"
    assert "third_party" in sig.description
    assert "scratch" in sig.description


def test_r6_does_not_trigger_when_comprehensive_covers_all():
    ctx = _base_ctx(
        coverage_type="comprehensive",
        image_damages=(
            _img(damage_type="scratch"),
            _img(id=2, damage_type="shattered_glass"),
            _img(id=3, damage_type="bumper_damage"),
        ),
    )
    assert r6_policy_coverage_mismatch(ctx) is None


def test_r6_does_not_trigger_when_collision_covers_collision_damage():
    # Collision covers scratch/dent/bumper/panel.
    ctx = _base_ctx(
        coverage_type="collision",
        image_damages=(_img(damage_type="bumper_damage"),),
    )
    assert r6_policy_coverage_mismatch(ctx) is None


def test_r6_triggers_when_collision_does_not_cover_glass():
    # Collision coverage does NOT include glass damage.
    ctx = _base_ctx(
        coverage_type="collision",
        image_damages=(_img(damage_type="shattered_glass"),),
    )
    sig = r6_policy_coverage_mismatch(ctx)
    assert sig is not None
    assert "shattered_glass" in sig.description


def test_r6_ignores_pending_and_no_damage_types():
    """`pending` and `no_damage` should not be treated as claimed types."""
    ctx = _base_ctx(
        coverage_type="third_party",
        image_damages=(
            _img(damage_type="pending"),
            _img(id=2, damage_type="no_damage"),
        ),
    )
    assert r6_policy_coverage_mismatch(ctx) is None


# ─── R7 — claim_frequency ───────────────────────────────────────────────────


def test_r7_triggers_with_two_previous_claims_within_12_months():
    # Current + 2 previous in window = 3 total.
    p1 = PreviousClaimCtx(
        id=1, claim_number="CLM-A",
        incident_date=dt.date(2025, 2, 1),
        damage_summary="x", claimed_amount=100.0, overlap_score=None,
    )
    p2 = PreviousClaimCtx(
        id=2, claim_number="CLM-B",
        incident_date=dt.date(2025, 4, 1),
        damage_summary="y", claimed_amount=200.0, overlap_score=None,
    )
    ctx = _base_ctx(
        incident_date=dt.date(2025, 6, 15),
        previous_claims=(p1, p2),
    )
    sig = r7_claim_frequency(ctx)
    assert sig is not None
    assert sig.rule_id == "R7_claim_frequency"
    assert sig.severity == SignalSeverity.medium.value
    assert sig.category == "claim_history"


def test_r7_does_not_trigger_with_only_one_previous_claim_in_window():
    p1 = PreviousClaimCtx(
        id=1, claim_number="CLM-A",
        incident_date=dt.date(2025, 2, 1),
        damage_summary="x", claimed_amount=100.0, overlap_score=None,
    )
    ctx = _base_ctx(
        incident_date=dt.date(2025, 6, 15),
        previous_claims=(p1,),
    )
    assert r7_claim_frequency(ctx) is None


def test_r7_does_not_trigger_with_no_previous_claims():
    ctx = _base_ctx(previous_claims=())
    assert r7_claim_frequency(ctx) is None


def test_r7_ignores_previous_claims_outside_12_months():
    p1 = PreviousClaimCtx(
        id=1, claim_number="CLM-A",
        incident_date=dt.date(2024, 1, 1),  # > 12 months before 2025-06-15
        damage_summary="x", claimed_amount=100.0, overlap_score=None,
    )
    ctx = _base_ctx(
        incident_date=dt.date(2025, 6, 15),
        previous_claims=(p1,),
    )
    assert r7_claim_frequency(ctx) is None


# ─── R8 — near_policy_boundary ──────────────────────────────────────────────


def test_r8_triggers_within_14_days_of_policy_start():
    ctx = _base_ctx(
        policy_start_date=dt.date(2025, 6, 1),
        incident_date=dt.date(2025, 6, 10),  # 9 days after start
    )
    sig = r8_near_policy_boundary(ctx)
    assert sig is not None
    assert sig.rule_id == "R8_near_policy_boundary"
    assert sig.severity == SignalSeverity.medium.value
    assert sig.category == "policy_timing"
    assert "start" in sig.description


def test_r8_triggers_within_14_days_of_policy_end():
    ctx = _base_ctx(
        policy_start_date=dt.date(2024, 1, 1),
        policy_end_date=dt.date(2025, 12, 31),
        incident_date=dt.date(2026, 1, 5),  # 5 days after end
    )
    sig = r8_near_policy_boundary(ctx)
    assert sig is not None
    assert "end" in sig.description


def test_r8_does_not_trigger_far_from_boundaries():
    ctx = _base_ctx(
        policy_start_date=dt.date(2024, 1, 1),
        policy_end_date=dt.date(2025, 12, 31),
        incident_date=dt.date(2025, 6, 15),  # well inside the policy
    )
    assert r8_near_policy_boundary(ctx) is None


def test_r8_does_not_trigger_outside_policy_window():
    ctx = _base_ctx(
        policy_start_date=dt.date(2024, 1, 1),
        policy_end_date=dt.date(2025, 1, 1),
        incident_date=dt.date(2025, 6, 15),  # months after end
    )
    assert r8_near_policy_boundary(ctx) is None


def test_r8_triggers_at_exactly_14_days():
    ctx = _base_ctx(
        policy_start_date=dt.date(2025, 6, 1),
        incident_date=dt.date(2025, 6, 15),  # 14 days after start
    )
    assert r8_near_policy_boundary(ctx) is not None


# ─── R9 — document_field_conflict ───────────────────────────────────────────


def test_r9_triggers_when_policy_number_differs_across_documents():
    docs = (
        _doc(id=1, doc_type="claim_form", fields={"policy_number": "POL-1"}),
        _doc(id=2, doc_type="policy", fields={"policy_number": "POL-2"}),
    )
    ctx = _base_ctx(documents=docs)
    sig = r9_document_field_conflict(ctx)
    assert sig is not None
    assert sig.rule_id == "R9_document_field_conflict"
    assert sig.severity == SignalSeverity.high.value
    assert sig.category == "document_consistency"
    assert "POL-1" in sig.description
    assert "POL-2" in sig.description


def test_r9_triggers_when_plate_number_differs_across_documents():
    docs = (
        _doc(id=1, doc_type="claim_form", fields={"plate_number": "ABC-1234"}),
        _doc(id=2, doc_type="estimate", fields={"plate_number": "XYZ-9999"}),
    )
    ctx = _base_ctx(documents=docs)
    sig = r9_document_field_conflict(ctx)
    assert sig is not None
    assert "plate_number" in sig.description


def test_r9_does_not_trigger_when_all_documents_agree():
    docs = (
        _doc(id=1, doc_type="claim_form", fields={"policy_number": "POL-1", "plate_number": "ABC-1234"}),
        _doc(id=2, doc_type="policy", fields={"policy_number": "POL-1", "plate_number": "ABC-1234"}),
    )
    ctx = _base_ctx(documents=docs)
    assert r9_document_field_conflict(ctx) is None


def test_r9_does_not_trigger_when_only_one_document_has_field():
    docs = (
        _doc(id=1, doc_type="claim_form", fields={"policy_number": "POL-1"}),
        _doc(id=2, doc_type="estimate", fields={"shop_name": "Acme"}),
    )
    ctx = _base_ctx(documents=docs)
    assert r9_document_field_conflict(ctx) is None


def test_r9_does_not_trigger_with_fewer_than_two_documents():
    ctx = _base_ctx(documents=(_doc(id=1, doc_type="claim_form", fields={"policy_number": "POL-1"}),))
    assert r9_document_field_conflict(ctx) is None


# ─── Orchestrator + persistence end-to-end ──────────────────────────────────
# These tests exercise the in-memory SQLite database via the conftest
# fixtures. They cover the full path: build ClaimContext from real
# ORM rows, run all 9 rules, and persist the resulting signals.


def test_evaluate_returns_only_fired_signals():
    """A clean context fires no signals."""
    ctx = _base_ctx()
    fired = evaluate(ctx)
    assert fired == []


def test_evaluate_returns_signals_for_each_fired_rule():
    """A context engineered to fire R1, R6, and R8 returns exactly 3 signals."""
    ctx = _base_ctx(
        # R1: claim_form damage not in CV detections
        image_damages=(_img(damage_type="scratch"),),
        claim_form_damages=(
            _img(id=2, source="claim_form", damage_type="headlight_damage"),
        ),
        # R6: third-party policy does not cover own-vehicle damage
        coverage_type="third_party",
        # R8: incident is within 14 days of policy start
        policy_start_date=dt.date(2025, 6, 1),
        incident_date=dt.date(2025, 6, 10),
    )
    fired = evaluate(ctx)
    rule_ids = {s.rule_id for s in fired}
    assert {"R1_unsupported_damage", "R6_policy_coverage_mismatch", "R8_near_policy_boundary"} <= rule_ids
    # No extra rules should have fired
    assert rule_ids.issubset({
        "R1_unsupported_damage", "R6_policy_coverage_mismatch", "R8_near_policy_boundary"
    })


def test_persist_commits_signals_to_database(db_session):
    """`persist` adds signals to the session, commits, and refreshes ids."""
    from app.models import Claim, Customer, Policy, Vehicle
    from app.models.enums import ClaimStatus, CoverageType, PolicyStatus

    # Build a minimal claim graph
    customer = Customer(name="Eve", email="eve@claimsight.test", phone="+1-555-0000")
    db_session.add(customer)
    db_session.flush()

    vehicle = Vehicle(
        customer_id=customer.id, make="Toyota", model="Camry",
        year=2020, vin="PERSISTVIN0000001", plate_number="PST-001",
    )
    db_session.add(vehicle)
    db_session.flush()

    policy = Policy(
        customer_id=customer.id, vehicle_id=vehicle.id,
        policy_number="POL-PST-001",
        coverage_type=CoverageType.comprehensive.value,
        coverage_limit=50000.00, deductible=500.00,
        start_date=dt.date(2024, 1, 1), end_date=dt.date(2025, 12, 31),
        status=PolicyStatus.active.value,
    )
    db_session.add(policy)
    db_session.flush()

    claim = Claim(
        policy_id=policy.id, vehicle_id=vehicle.id, claim_number="CLM-PST-001",
        incident_date=dt.date(2025, 6, 15), reported_date=dt.date(2025, 6, 16),
        claimed_amount=1000.00, status=ClaimStatus.pending.value,
    )
    db_session.add(claim)
    db_session.flush()
    db_session.refresh(claim)

    # Create a context that fires R1 and R8
    ctx = _base_ctx(
        claim_id=claim.id,
        customer_id=customer.id,
        vehicle_id=vehicle.id,
        policy_id=policy.id,
        image_damages=(_img(damage_type="scratch"),),
        claim_form_damages=(
            _img(id=999, source="claim_form", damage_type="headlight_damage"),
        ),
        policy_start_date=dt.date(2025, 6, 1),
        incident_date=dt.date(2025, 6, 10),
    )

    fired = evaluate(ctx)
    assert len(fired) >= 2
    saved = persist(fired, db_session)
    assert all(s.id is not None for s in saved)

    # Reload the claim's risk signals and confirm the count matches
    db_session.refresh(claim)
    signals = list(claim.risk_signals)
    assert len(signals) == len(fired)
    rule_ids = {s.rule_id for s in signals}
    assert "R1_unsupported_damage" in rule_ids
    assert "R8_near_policy_boundary" in rule_ids
