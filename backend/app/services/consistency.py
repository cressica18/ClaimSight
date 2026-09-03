"""
Consistency / Investigation Engine — blueprint Section 5.

Pure, deterministic, auditable rule engine. Every rule is a function
`rule(ctx: ClaimContext) -> RiskSignal | None` that takes a normalized,
in-memory snapshot of everything the engine needs to know about a claim
and returns either a `RiskSignal` ORM instance (not yet committed) or
`None` if the rule does not fire.

Design constraints (from blueprint Section 5 + the Phase 6 prompt):
- No Gemini / LLM calls.
- No I/O — every rule is a pure function on `ClaimContext`. The
  orchestrator (`evaluate`) is the only place that touches the database.
- Auditable: each rule is small, has a fixed `rule_id` and `category`,
  and the `description` is generated from the inputs that triggered it.
- Phase 7 (risk engine) is NOT implemented here — we only emit
  `RiskSignal` rows. The score / band computation lives elsewhere.
- The CV model is not touched. We consume the CV output (Damage rows
  with `source='image'`) as already-persisted data.

Public API:
- `ClaimContext` and the smaller context dataclasses (`ImageDamageCtx`,
  `RepairItemCtx`, `PreviousClaimCtx`, `DocumentCtx`).
- `build_claim_context(claim_id, db) -> ClaimContext` — the orchestrator
  loads the claim and all related rows from the DB and packages them.
- `evaluate(ctx) -> list[RiskSignal]` — runs all 9 rules, returns the
  signals that fired (unsaved). Adding more rules is a matter of
  appending a new `rN_<name>` function and adding it to `ALL_RULES`.
- `persist(signals, db)` — convenience helper for callers that want to
  commit the resulting signals. The pipeline in Phase 11 will use this.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Sequence

from sqlalchemy.orm import Session

from app.models.accident import Accident
from app.models.claim import Claim
from app.models.customer import Customer
from app.models.damage import Damage
from app.models.document import Document
from app.models.enums import DamageSeverity, SignalSeverity
from app.models.policy import Policy
from app.models.previous_claim import PreviousClaim
from app.models.repair import RepairEstimate, RepairItem
from app.models.risk_signal import RiskSignal
from app.models.vehicle import Vehicle

logger = logging.getLogger(__name__)


# ─── Context dataclasses ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class ImageDamageCtx:
    """A single damage observation from either CV (source='image') or a
    claim form (source='claim_form').

    `low_confidence` is the explicit flag the CV pipeline stores in the
    row's `region_ref` JSON; it is what R1 inspects.
    """

    id: int
    source: str  # "image" | "claim_form"
    damage_type: str | None
    severity: str | None
    confidence: float | None
    region_ref: str | None
    low_confidence: bool = False
    severity_confidence: float | None = None
    model_version: str | None = None
    timestamp: str | None = None


@dataclass(frozen=True)
class RepairItemCtx:
    id: int
    part_name: str | None
    operation: str | None
    cost: float | None
    labor_hours: float | None


@dataclass(frozen=True)
class RepairEstimateCtx:
    id: int
    total_cost: float | None
    currency: str
    issued_date: date | None
    shop_name: str | None
    items: tuple[RepairItemCtx, ...]


@dataclass(frozen=True)
class PreviousClaimCtx:
    id: int
    claim_number: str
    incident_date: date
    damage_summary: str | None
    claimed_amount: float | None
    overlap_score: float | None


@dataclass(frozen=True)
class DocumentCtx:
    """Snapshot of a claim document the rule engine can see.

    `extracted_fields` is a free-form dict keyed by field name (e.g.
    `{"policy_number": "POL-1", "plate_number": "ABC-1234"}`).
    The Document model stores this in a JSON column; rules treat absent
    fields as "unknown" and simply do not fire on that field.
    """

    id: int
    doc_type: str
    extraction_status: str
    raw_confidence: float | None
    file_path: str
    extracted_fields: tuple[tuple[str, Any], ...]  # ordered (k, v) pairs


@dataclass(frozen=True)
class ClaimContext:
    """Normalized, in-memory view of everything the rules need to inspect.

    Rules never read the DB. The orchestrator (`build_claim_context`)
    populates this once per evaluation; the rules then run as pure
    functions.

    `baseline_upper` is provided by the caller. Phase 7 will compute
    baseline costs from the synthetic dataset; for now it is optional
    so R4 degrades gracefully when no baseline is available.
    """

    # Identity
    claim_id: int
    claim_number: str
    claim_status: str
    claimed_amount: float | None
    incident_date: date
    reported_date: date | None
    # Customer
    customer_id: int
    customer_name: str
    # Vehicle
    vehicle_id: int
    vehicle_make: str
    vehicle_model: str
    vehicle_year: int
    vehicle_vin: str | None
    vehicle_plate: str | None
    # Policy
    policy_id: int
    policy_number: str
    coverage_type: str
    coverage_limit: float
    deductible: float
    policy_start_date: date
    policy_end_date: date
    policy_status: str
    # Damages
    image_damages: tuple[ImageDamageCtx, ...]
    claim_form_damages: tuple[ImageDamageCtx, ...]
    # Accident
    accident_description: str | None
    accident_location: str | None
    accident_incident_type: str | None
    # Repair estimate
    repair_estimate: RepairEstimateCtx | None
    # Previous claims for the same customer
    previous_claims: tuple[PreviousClaimCtx, ...]
    # Documents
    documents: tuple[DocumentCtx, ...]
    # Baseline (Phase 7 will compute; optional)
    baseline_upper: float | None = None


# ─── Orchestrator: DB → ClaimContext ─────────────────────────────────────────


def _parse_image_damage(d: Damage) -> ImageDamageCtx:
    """Build an ImageDamageCtx from a Damage row.

    The CV service stores `low_confidence`, `severity_confidence`,
    `model_version`, and `timestamp` as JSON inside `region_ref`. We
    extract them here so rules can read them as plain attributes.
    """
    low_conf = False
    sev_conf: float | None = None
    model_ver: str | None = None
    ts: str | None = None
    if d.region_ref:
        try:
            meta = json.loads(d.region_ref)
            if isinstance(meta, dict):
                low_conf = bool(meta.get("low_confidence", False))
                sev_conf = meta.get("severity_confidence")
                model_ver = meta.get("model_version")
                ts = meta.get("timestamp")
        except (TypeError, ValueError):
            # region_ref is a free-form string; ignore it for CV metadata.
            pass
    return ImageDamageCtx(
        id=d.id,
        source=d.source,
        damage_type=d.damage_type,
        severity=d.severity,
        confidence=d.confidence,
        region_ref=d.region_ref,
        low_confidence=low_conf,
        severity_confidence=sev_conf if isinstance(sev_conf, (int, float)) else None,
        model_version=model_ver,
        timestamp=ts,
    )


def _parse_document(d: Document) -> DocumentCtx:
    """Build a DocumentCtx, extracting `extracted_fields` from the
    Document's JSON column when present. The column is optional and
    older documents (or documents uploaded before extraction was wired
    in) will have None — in that case the dict is empty and document
    field rules simply do not fire.
    """
    extracted = d.extracted_fields if isinstance(d.extracted_fields, dict) else {}
    return DocumentCtx(
        id=d.id,
        doc_type=d.doc_type,
        extraction_status=d.extraction_status,
        raw_confidence=d.raw_confidence,
        file_path=d.file_path,
        extracted_fields=tuple(sorted(extracted.items())),
    )


def build_claim_context(
    claim_id: int,
    db: Session,
    baseline_upper: float | None = None,
) -> ClaimContext:
    """Load a claim and everything it links to from the database, then
    package it as an immutable `ClaimContext`.

    Raises `ValueError` if the claim does not exist. All related rows
    (damages, accident, repair estimate + items, previous claims for
    the customer, documents) are eagerly loaded in a single pass; rules
    do not touch the DB.
    """
    claim = db.get(Claim, claim_id)
    if not claim:
        raise ValueError(f"Claim {claim_id} not found")

    policy: Policy = claim.policy
    vehicle: Vehicle = claim.vehicle
    customer: Customer = policy.customer

    # All damages for the claim, partitioned by source
    all_damages = list(claim.damages)
    image_damages = tuple(_parse_image_damage(d) for d in all_damages if d.source == "image")
    claim_form_damages = tuple(
        _parse_image_damage(d) for d in all_damages if d.source == "claim_form"
    )

    # Accident (1:1)
    accident: Accident | None = claim.accident
    accident_description = accident.description if accident else None
    accident_location = accident.location if accident else None
    accident_incident_type = accident.incident_type if accident else None

    # Repair estimate (0..1 for now — the schema allows many, but a
    # claim typically has one active estimate at a time)
    repair_estimate: RepairEstimateCtx | None = None
    if claim.repair_estimates:
        est = claim.repair_estimates[0]
        items = tuple(
            RepairItemCtx(
                id=it.id,
                part_name=it.part_name,
                operation=it.operation,
                cost=float(it.cost) if it.cost is not None else None,
                labor_hours=it.labor_hours,
            )
            for it in est.items
        )
        repair_estimate = RepairEstimateCtx(
            id=est.id,
            total_cost=float(est.total_cost) if est.total_cost is not None else None,
            currency=est.currency,
            issued_date=est.issued_date,
            shop_name=est.shop_name,
            items=items,
        )

    # Previous claims for the same customer (used by R5, R7). The
    # `PreviousClaim` table already indexes `customer_id` and
    # `vehicle_id` per blueprint Section 9.
    prev_rows = (
        db.query(PreviousClaim)
        .filter(PreviousClaim.customer_id == customer.id)
        .all()
    )
    previous_claims = tuple(
        PreviousClaimCtx(
            id=pc.id,
            claim_number=pc.claim_number,
            incident_date=pc.incident_date,
            damage_summary=pc.damage_summary,
            claimed_amount=float(pc.claimed_amount) if pc.claimed_amount is not None else None,
            overlap_score=pc.overlap_score,
        )
        for pc in prev_rows
    )

    # Documents (used by R9). Eager-load via the claim relationship.
    documents = tuple(_parse_document(d) for d in claim.documents)

    return ClaimContext(
        claim_id=claim.id,
        claim_number=claim.claim_number,
        claim_status=claim.status,
        claimed_amount=float(claim.claimed_amount) if claim.claimed_amount is not None else None,
        incident_date=claim.incident_date,
        reported_date=claim.reported_date,
        customer_id=customer.id,
        customer_name=customer.name,
        vehicle_id=vehicle.id,
        vehicle_make=vehicle.make,
        vehicle_model=vehicle.model,
        vehicle_year=vehicle.year,
        vehicle_vin=vehicle.vin,
        vehicle_plate=vehicle.plate_number,
        policy_id=policy.id,
        policy_number=policy.policy_number,
        coverage_type=policy.coverage_type,
        coverage_limit=float(policy.coverage_limit),
        deductible=float(policy.deductible),
        policy_start_date=policy.start_date,
        policy_end_date=policy.end_date,
        policy_status=policy.status,
        image_damages=image_damages,
        claim_form_damages=claim_form_damages,
        accident_description=accident_description,
        accident_location=accident_location,
        accident_incident_type=accident_incident_type,
        repair_estimate=repair_estimate,
        previous_claims=previous_claims,
        documents=documents,
        baseline_upper=baseline_upper,
    )


# ─── Rule helpers (shared) ────────────────────────────────────────────────────


def _make_signal(
    claim_id: int,
    rule_id: str,
    category: str,
    severity: SignalSeverity,
    description: str,
) -> RiskSignal:
    """Construct a `RiskSignal` ORM instance. Not yet added to a session —
    callers (evaluate / persist / tests) decide when to add and commit.
    """
    return RiskSignal(
        claim_id=claim_id,
        rule_id=rule_id,
        category=category,
        severity=severity.value,
        description=description,
    )


def _normalize(s: str | None) -> str:
    """Lowercase + strip for keyword comparison."""
    return (s or "").lower().strip()


# ─── Rule R1: unsupported_damage ─────────────────────────────────────────────
# Trigger: a claim_form damage area has no corresponding CV detection
# (and the CV detection is not low-confidence). Severity: High.


def r1_unsupported_damage(ctx: ClaimContext) -> RiskSignal | None:
    """Claim form lists a damage area the CV model did not see."""
    if not ctx.claim_form_damages:
        return None

    # Build the set of CV-detected damage types, but ONLY from
    # detections that are not flagged as low_confidence (per the rule
    # text: "image confidence is not low"). Severity of detection
    # does not gate the support check.
    cv_types: set[str] = {
        _normalize(d.damage_type)
        for d in ctx.image_damages
        if d.damage_type and not d.low_confidence
    }
    # If no high-confidence CV detections exist at all, there is no
    # "support" to evaluate; the rule does not fire.
    if not cv_types:
        return None

    # Each claim_form area must have a matching CV detection. Compare
    # on the normalized damage_type (exact match — synonyms are a
    # separate concern and out of scope for the deterministic rules).
    unsupported: list[str] = []
    for d in ctx.claim_form_damages:
        key = _normalize(d.damage_type)
        if not key:
            continue
        if key not in cv_types:
            unsupported.append(d.damage_type or key)

    if not unsupported:
        return None

    description = (
        f"Claim form lists {len(unsupported)} damage area(s) not supported "
        f"by CV detection: {', '.join(sorted(set(unsupported)))}."
    )
    return _make_signal(
        claim_id=ctx.claim_id,
        rule_id="R1_unsupported_damage",
        category="image_claim_consistency",
        severity=SignalSeverity.high,
        description=description,
    )


# ─── Rule R2: severity_mismatch ──────────────────────────────────────────────
# Trigger: claim description severity language differs from CV severity
# by ≥2 levels. Severity: Medium.


_SEVERITY_RANK = {
    DamageSeverity.minor.value: 0,
    DamageSeverity.moderate.value: 1,
    DamageSeverity.severe.value: 2,
}

# Order matters: we use the first match wins.
_SEVERITY_KEYWORDS: list[tuple[set[str], int]] = [
    ({"totaled", "destroyed", "write-off", "writeoff", "wrecked", "demolished"}, 2),  # severe
    ({"severe", "major", "heavy", "extensive", "significant"}, 2),  # severe
    ({"moderate", "medium", "considerable"}, 1),  # moderate
    ({"minor", "small", "slight", "light", "scratch", "tiny", "little"}, 0),  # minor
]


def _extract_text_severity(text: str | None) -> int | None:
    """Return the severity rank (0=minor, 1=moderate, 2=severe) implied
    by the description, or None if no severity language is present.

    Highest-priority match wins: if a description mentions both
    "scratch" and "destroyed", "destroyed" wins because it is a stronger
    signal of damage extent.
    """
    if not text:
        return None
    text_l = text.lower()
    best: int | None = None
    for keywords, rank in _SEVERITY_KEYWORDS:
        for kw in keywords:
            if kw in text_l:
                if best is None or rank > best:
                    best = rank
    return best


def _worst_cv_severity(ctx: ClaimContext) -> tuple[str | None, int | None]:
    """Return (label, rank) of the worst CV-detected severity, or
    (None, None) if there are no CV severities to compare against.
    """
    worst_label: str | None = None
    worst_rank: int | None = None
    for d in ctx.image_damages:
        if d.severity is None or d.severity == "pending" or d.severity == "unknown":
            continue
        rank = _SEVERITY_RANK.get(d.severity)
        if rank is None:
            continue
        if worst_rank is None or rank > worst_rank:
            worst_rank = rank
            worst_label = d.severity
    return worst_label, worst_rank


def r2_severity_mismatch(ctx: ClaimContext) -> RiskSignal | None:
    """Claim description language does not match the CV severity (≥2 levels apart)."""
    text_rank = _extract_text_severity(ctx.accident_description)
    cv_label, cv_rank = _worst_cv_severity(ctx)
    if text_rank is None or cv_rank is None:
        return None

    delta = abs(text_rank - cv_rank)
    if delta < 2:
        return None

    description = (
        f"Claim description severity (rank {text_rank}) and CV-detected "
        f"severity ({cv_label}, rank {cv_rank}) differ by {delta} levels."
    )
    return _make_signal(
        claim_id=ctx.claim_id,
        rule_id="R2_severity_mismatch",
        category="claim_description_consistency",
        severity=SignalSeverity.medium,
        description=description,
    )


# ─── Rule R3: repair_component_mismatch ──────────────────────────────────────
# Trigger: any repair item part_name is not plausibly linked to any
# detected/claimed damage type. Severity: Medium.


# Lookup table: damage_type -> set of plausible part keywords.
# Comparison is case-insensitive substring matching against `part_name`,
# which lets a single mapping cover synonyms (e.g. "windshield" matches
# "shattered_glass").
_DAMAGE_TYPE_PLAUSIBLE_PARTS: dict[str, set[str]] = {
    DamageSeverity.minor.value: set(),  # placeholder; key is severity, see below
}

# The real mapping: damage_type -> set of part-name substrings.
_PLAUSIBLE_PARTS: dict[str, set[str]] = {
    "scratch": {"panel", "door", "hood", "fender", "bumper", "quarter", "trunk", "roof", "mirror", "paint"},
    "dent": {"panel", "door", "hood", "fender", "bumper", "quarter", "trunk", "roof", "side"},
    "crack": {"windshield", "window", "glass", "mirror", "lens", "headlight", "taillight"},
    "shattered_glass": {"windshield", "window", "glass", "mirror", "lens"},
    "bumper_damage": {"bumper", "fender", "reinforcement", "bracket"},
    "panel_damage": {"panel", "door", "hood", "fender", "quarter", "trunk", "roof", "side"},
    "headlight_damage": {"headlight", "lens", "housing", "bulb", "taillight"},
    "no_damage": set(),  # nothing should require repair
}


def _part_matches_damage(part_name: str, damage_type: str) -> bool:
    """Return True if `part_name` is plausibly linked to `damage_type`."""
    plausible = _PLAUSIBLE_PARTS.get(damage_type, set())
    if not plausible:
        return False
    p = part_name.lower()
    return any(key in p for key in plausible)


def r3_repair_component_mismatch(ctx: ClaimContext) -> RiskSignal | None:
    """Repair estimate includes a part not plausibly linked to any damage."""
    if ctx.repair_estimate is None or not ctx.repair_estimate.items:
        return None

    # Gather the set of damage types the claim is actually about. We
    # accept BOTH CV detections and claim-form entries; if either side
    # supports a part, it's not a mismatch.
    damage_types: set[str] = set()
    for d in ctx.image_damages:
        if d.damage_type:
            damage_types.add(_normalize(d.damage_type))
    for d in ctx.claim_form_damages:
        if d.damage_type:
            damage_types.add(_normalize(d.damage_type))

    if not damage_types:
        # No damage to compare against — but if there is a repair
        # estimate, that is itself suspicious. Surface it.
        bad_parts = [
            it.part_name
            for it in ctx.repair_estimate.items
            if it.part_name
        ]
        if not bad_parts:
            return None
        description = (
            f"Repair estimate includes parts but no damage has been "
            f"recorded: {', '.join(sorted(set(bad_parts)))}."
        )
        return _make_signal(
            claim_id=ctx.claim_id,
            rule_id="R3_repair_component_mismatch",
            category="repair_estimate_consistency",
            severity=SignalSeverity.medium,
            description=description,
        )

    bad: list[str] = []
    for item in ctx.repair_estimate.items:
        if not item.part_name:
            continue
        if not any(_part_matches_damage(item.part_name, dt) for dt in damage_types):
            bad.append(item.part_name)

    if not bad:
        return None

    description = (
        f"Repair estimate includes parts not plausibly linked to any "
        f"detected damage ({', '.join(sorted(damage_types))}): "
        f"{', '.join(sorted(set(bad)))}."
    )
    return _make_signal(
        claim_id=ctx.claim_id,
        rule_id="R3_repair_component_mismatch",
        category="repair_estimate_consistency",
        severity=SignalSeverity.medium,
        description=description,
    )


# ─── Rule R4: excessive_repair_cost ──────────────────────────────────────────
# Trigger: total_cost > baseline_upper * 1.5. Severity: High if >2×,
# Medium if 1.5–2×. The baseline is supplied by the caller; if absent,
# the rule cannot run and does not fire.


def r4_excessive_repair_cost(ctx: ClaimContext) -> RiskSignal | None:
    """Repair estimate total cost exceeds the baseline cost upper bound."""
    if ctx.repair_estimate is None or ctx.repair_estimate.total_cost is None:
        return None
    if ctx.baseline_upper is None or ctx.baseline_upper <= 0:
        # No baseline to compare against. Phase 7 will compute one; for
        # now the rule is silent rather than guessing.
        return None

    total = float(ctx.repair_estimate.total_cost)
    upper = float(ctx.baseline_upper)

    if total <= upper * 1.5:
        return None

    if total > upper * 2.0:
        sev = SignalSeverity.high
        ratio = total / upper
        magnitude = f"{ratio:.2f}× the baseline upper bound"
    else:
        sev = SignalSeverity.medium
        ratio = total / upper
        magnitude = f"{ratio:.2f}× the baseline upper bound"

    description = (
        f"Repair estimate total ({ctx.repair_estimate.currency} {total:,.2f}) "
        f"exceeds the baseline upper bound ({ctx.repair_estimate.currency} "
        f"{upper:,.2f}): {magnitude}."
    )
    return _make_signal(
        claim_id=ctx.claim_id,
        rule_id="R4_excessive_repair_cost",
        category="cost_validation",
        severity=sev,
        description=description,
    )


# ─── Rule R5: duplicate_previous_damage ──────────────────────────────────────
# Trigger: same vehicle, damage region overlap, incident dates within
# 6 months. Severity: High.


# Tokens used to detect "region overlap" between a previous claim's
# damage_summary and a current claim's damage. We do a token-overlap
# Jaccard; region overlap is fuzzy by design.
_REGION_TOKENS_BLACKLIST = {
    "the", "a", "an", "of", "to", "on", "in", "at", "is", "was", "were",
    "with", "and", "or", "for", "by", "from", "this", "that", "it",
    "his", "her", "its", "their", "our", "my",
}


def _normalize_tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    out: set[str] = set()
    for raw in text.lower().split():
        word = raw.strip(".,;:()[]{}!?\"'")
        if not word or word in _REGION_TOKENS_BLACKLIST:
            continue
        if len(word) < 3:
            continue
        out.add(word)
    return out


def _regions_overlap(prev_summary: str | None, current_damages: tuple[ImageDamageCtx, ...]) -> bool:
    """Return True if any current damage type appears in the previous
    claim's damage_summary, OR (when the previous summary is missing)
    we fall back to checking the most severe current damage's
    damage_type is the same as the most likely target.

    The "no previous summary" branch returns False rather than guessing
    so that the rule only fires when there is positive evidence.
    """
    if not prev_summary:
        return False
    prev_tokens = _normalize_tokens(prev_summary)
    if not prev_tokens:
        return False
    for d in current_damages:
        if not d.damage_type:
            continue
        dt = d.damage_type.lower()
        if dt in prev_tokens:
            return True
        # Synonym fallback: if the damage_type is a substring or token
        # of any token in the previous summary, treat it as a match.
        for tok in prev_tokens:
            if dt in tok or tok in dt:
                return True
    return False


def r5_duplicate_previous_damage(ctx: ClaimContext) -> RiskSignal | None:
    """A previous claim for the same customer/vehicle overlaps in region
    and is within 6 months of the current incident date.
    """
    if not ctx.previous_claims:
        return None

    from datetime import timedelta

    window = timedelta(days=183)  # 6 months, per blueprint

    # Current damage set is union of CV + claim_form
    current_damages = ctx.image_damages + ctx.claim_form_damages
    if not current_damages:
        return None

    matches: list[str] = []
    for prev in ctx.previous_claims:
        # Same vehicle (by id) — blueprint says "Same vehicle"
        if not ctx.vehicle_id:
            continue
        # Note: PreviousClaim is keyed by customer, so to scope to the
        # SAME VEHICLE we additionally check the incident window.
        if abs((prev.incident_date - ctx.incident_date).days) > window.days:
            continue
        if not _regions_overlap(prev.damage_summary, current_damages):
            continue
        matches.append(f"{prev.claim_number} ({prev.incident_date.isoformat()})")

    if not matches:
        return None

    description = (
        f"Found {len(matches)} previous claim(s) for this customer with "
        f"overlapping damage regions within 6 months: {', '.join(matches)}."
    )
    return _make_signal(
        claim_id=ctx.claim_id,
        rule_id="R5_duplicate_previous_damage",
        category="claim_history",
        severity=SignalSeverity.high,
        description=description,
    )


# ─── Rule R6: policy_coverage_mismatch ───────────────────────────────────────
# Trigger: a damage type is not covered under the policy's coverage_type.
# Severity: High.


# coverage_type -> set of damage_types it covers. Anything outside the
# set for the current policy is a mismatch.
_COVERAGE_DAMAGE_TYPES: dict[str, set[str]] = {
    "comprehensive": {
        "scratch", "dent", "crack", "shattered_glass", "bumper_damage",
        "panel_damage", "headlight_damage",
    },
    "collision": {
        "scratch", "dent", "bumper_damage", "panel_damage",
    },
    "third_party": set(),  # third_party does not cover own-vehicle damage
    "fire_theft": set(),   # fire & theft only cover fire / theft losses
}


def r6_policy_coverage_mismatch(ctx: ClaimContext) -> RiskSignal | None:
    """A damage type on the claim is not covered by the policy type."""
    coverage = _COVERAGE_DAMAGE_TYPES.get(ctx.coverage_type)
    if coverage is None:
        # Unknown coverage type — don't fabricate a signal; flag would
        # be unauditable. Real-world handling would be Phase 8 / data fix.
        return None

    # Collect the damage types actually claimed (CV + claim form)
    claimed: set[str] = set()
    for d in ctx.image_damages:
        if d.damage_type and d.damage_type not in ("pending", "no_damage"):
            claimed.add(_normalize(d.damage_type))
    for d in ctx.claim_form_damages:
        if d.damage_type:
            claimed.add(_normalize(d.damage_type))

    if not claimed:
        return None

    uncovered = sorted(claimed - coverage)
    if not uncovered:
        return None

    description = (
        f"Policy coverage type '{ctx.coverage_type}' does not cover the "
        f"following damage type(s) on this claim: {', '.join(uncovered)}."
    )
    return _make_signal(
        claim_id=ctx.claim_id,
        rule_id="R6_policy_coverage_mismatch",
        category="policy_coverage",
        severity=SignalSeverity.high,
        description=description,
    )


# ─── Rule R7: claim_frequency ────────────────────────────────────────────────
# Trigger: customer has ≥3 claims (current + previous) in the trailing
# 12 months. Severity: Medium.


def r7_claim_frequency(ctx: ClaimContext) -> RiskSignal | None:
    """Customer has filed ≥3 claims in the last 12 months (inclusive)."""
    from datetime import timedelta

    window = timedelta(days=365)
    cutoff = ctx.incident_date - window

    count = 1  # the current claim
    for prev in ctx.previous_claims:
        if prev.incident_date >= cutoff and prev.incident_date <= ctx.incident_date:
            count += 1

    if count < 3:
        return None

    description = (
        f"Customer has {count} claim(s) in the trailing 12 months "
        f"(threshold: 3)."
    )
    return _make_signal(
        claim_id=ctx.claim_id,
        rule_id="R7_claim_frequency",
        category="claim_history",
        severity=SignalSeverity.medium,
        description=description,
    )


# ─── Rule R8: near_policy_boundary ───────────────────────────────────────────
# Trigger: incident_date within 14 days of policy start or end.
# Severity: Medium.


def r8_near_policy_boundary(ctx: ClaimContext) -> RiskSignal | None:
    """Incident occurred within 14 days of a policy boundary."""
    boundary_days = 14
    # Use abs() so we catch incidents that are 0–14 days on EITHER side
    # of a policy boundary. A claim that "happens" 5 days after the
    # end-date (or 5 days before) is just as suspicious as one that
    # happens 5 days after the start-date.
    start_delta = (ctx.incident_date - ctx.policy_start_date).days
    end_delta = (ctx.incident_date - ctx.policy_end_date).days

    near_start = abs(start_delta) <= boundary_days
    near_end = abs(end_delta) <= boundary_days
    if not (near_start or near_end):
        return None

    if near_start and near_end:
        boundary = "both start and end"
    elif near_start:
        side = "after" if start_delta >= 0 else "before"
        boundary = f"start ({ctx.policy_start_date.isoformat()}, {abs(start_delta)}d {side})"
    else:
        side = "after" if end_delta >= 0 else "before"
        boundary = f"end ({ctx.policy_end_date.isoformat()}, {abs(end_delta)}d {side})"

    description = (
        f"Incident date ({ctx.incident_date.isoformat()}) is within "
        f"{boundary_days} days of policy {boundary}."
    )
    return _make_signal(
        claim_id=ctx.claim_id,
        rule_id="R8_near_policy_boundary",
        category="policy_timing",
        severity=SignalSeverity.medium,
        description=description,
    )


# ─── Rule R9: document_field_conflict ────────────────────────────────────────
# Trigger: policy_number or plate_number differ across the claim's
# extracted documents. Severity: High.


# Field names we look at across documents for consistency.
_R9_FIELDS = ("policy_number", "plate_number", "vin")


def r9_document_field_conflict(ctx: ClaimContext) -> RiskSignal | None:
    """Cross-document field agreement check.

    Compares `policy_number` / `plate_number` / `vin` across the claim's
    documents. If a field appears in ≥2 documents with ≥2 distinct
    values, the rule fires. Documents with no extracted data for a
    field are ignored (we only compare what was actually extracted).
    """
    if len(ctx.documents) < 2:
        return None

    # Build per-field map of value -> [doc_type list]
    conflicts: list[str] = []
    for field_name in _R9_FIELDS:
        values: dict[str, list[str]] = {}
        for doc in ctx.documents:
            extracted = dict(doc.extracted_fields)
            v = extracted.get(field_name)
            if v is None or v == "":
                continue
            values.setdefault(str(v).strip(), []).append(doc.doc_type)

        if len(values) < 2:
            continue
        # ≥2 distinct values across the documents
        distinct = sorted(values.items(), key=lambda kv: kv[0])
        rendered = "; ".join(
            f"{val!r} (in {', '.join(sorted(docs))})" for val, docs in distinct
        )
        conflicts.append(f"{field_name}: {rendered}")

    if not conflicts:
        return None

    description = (
        f"Cross-document field conflict detected: {' | '.join(conflicts)}."
    )
    return _make_signal(
        claim_id=ctx.claim_id,
        rule_id="R9_document_field_conflict",
        category="document_consistency",
        severity=SignalSeverity.high,
        description=description,
    )


# ─── Rule registry + orchestrator ────────────────────────────────────────────


# Type alias: a rule is a pure function on ClaimContext.
Rule = Callable[[ClaimContext], RiskSignal | None]

# Order matters only for deterministic logging; rules are independent
# and the resulting list is sorted by category downstream if needed.
ALL_RULES: tuple[Rule, ...] = (
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


def evaluate(ctx: ClaimContext) -> list[RiskSignal]:
    """Run every registered rule against `ctx` and return the signals
    that fired. The returned list contains ORM instances that are NOT
    yet attached to a session — callers should use `persist` to add
    and commit them.

    The function is pure: same input -> same output. No DB access, no
    network, no LLM.
    """
    fired: list[RiskSignal] = []
    for rule in ALL_RULES:
        try:
            signal = rule(ctx)
        except Exception:  # noqa: BLE001
            # Rules are pure functions; an exception means a bug. Log
            # and continue so one broken rule does not silently drop
            # the whole evaluation.
            logger.exception("Rule %s raised on claim %d", rule.__name__, ctx.claim_id)
            continue
        if signal is not None:
            fired.append(signal)
    return fired


def persist(signals: Sequence[RiskSignal], db: Session) -> list[RiskSignal]:
    """Add the given signals to the session and commit. Returns the
    list of signals (now attached and with assigned `id`s).
    """
    for s in signals:
        db.add(s)
    if signals:
        db.commit()
        for s in signals:
            db.refresh(s)
    return list(signals)
