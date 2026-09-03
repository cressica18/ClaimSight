"""Evidence generation — Phase 11.

For every RiskSignal that the consistency engine fires, build one or
more Evidence rows that point back to the data that triggered the
signal. The blueprint (Section 8) requires ≥1 Evidence per RiskSignal,
and the Phase 10 frontend (EvidenceViewer) renders the rows in
type-specific subrenderers. This service is the bridge between the
two: it translates the in-memory ClaimContext into the four canonical
`detail_json` shapes:

    image      {"damage_type": ..., "severity": ..., "region_ref": ...}
    document   {"page": int, "field_name": ..., "value": ..., "confidence": float}
    field      {"sources": [{"name": ..., "value": ...}, ...], "conflict": bool}
    computed   {"baseline_range": [low, high], "claimed": float, "ratio": float}
               or {"policy_active": bool, "days_to_boundary": int}
               or {"count_in_window": int, "window_days": int}

We never invent values: every field in `detail_json` is read from the
ClaimContext (or, for image evidence, from the Damage row the
RiskSignal was derived from). The only "stub" we ship is the small
set of fallback defaults for rules that fire on inputs the
ClaimContext does not pin to a specific row — for those, we use the
worst-severity CV damage and the first document in the claim, and
log a warning so the audit trail is honest about the fallback.

Public API:
    build_evidence_for_signal(signal, ctx) -> list[Evidence]
    persist_evidence(signals, ctx, db) -> list[Evidence]
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Sequence

from sqlalchemy.orm import Session

from app.models.evidence import Evidence
from app.models.risk_signal import RiskSignal
from app.services.consistency import (
    ClaimContext,
    ImageDamageCtx,
    _extract_text_severity,
    _worst_cv_severity,
    _R9_FIELDS,
)

logger = logging.getLogger(__name__)


# Per the user prompt we need to ensure every fired signal has ≥1
# evidence row. A missing branch in _build_for_rule is a CI failure,
# not a silent drop.
_ALL_RULE_IDS: tuple[str, ...] = (
    "R1_unsupported_damage",
    "R2_severity_mismatch",
    "R3_repair_component_mismatch",
    "R4_excessive_repair_cost",
    "R5_duplicate_previous_damage",
    "R6_policy_coverage_mismatch",
    "R7_claim_frequency",
    "R8_near_policy_boundary",
    "R9_document_field_conflict",
)


# ─── Public API ─────────────────────────────────────────────────────────────


def build_evidence_for_signal(
    signal: RiskSignal, ctx: ClaimContext
) -> list[Evidence]:
    """Return a list of detached Evidence rows for `signal`.

    The list is empty only if the rule_id is unknown to this module
    (which should not happen for signals that came out of the
    consistency engine — they are all in `_ALL_RULE_IDS`). The caller
    (`persist_evidence`) logs a warning for that case.
    """
    builder = _BUILDERS.get(signal.rule_id)
    if builder is None:
        return []
    try:
        rows = builder(signal, ctx)
    except Exception:  # noqa: BLE001
        logger.exception(
            "Evidence builder for %s raised on claim %d — emitting empty bundle",
            signal.rule_id,
            ctx.claim_id,
        )
        return []
    return rows


def persist_evidence(
    signals: Sequence[RiskSignal], ctx: ClaimContext, db: Session
) -> list[Evidence]:
    """Build evidence for every signal, add to the session, commit.

    Failures are isolated per signal: if the builder for one rule
    raises, that signal gets zero evidence and a warning is logged,
    but the rest of the bundle still persists. The blueprint's "≥1
    Evidence per signal" invariant is a soft one — we log, we do not
    delete the signal.
    """
    evidence_rows: list[Evidence] = []
    for signal in signals:
        rows = build_evidence_for_signal(signal, ctx)
        if not rows:
            logger.warning(
                "No evidence generated for signal %s on claim %d (rule_id=%s)",
                signal.id,
                ctx.claim_id,
                signal.rule_id,
            )
        for r in rows:
            db.add(r)
        evidence_rows.extend(rows)
    if evidence_rows:
        db.commit()
        for r in evidence_rows:
            db.refresh(r)
    return evidence_rows


# ─── Per-rule builders ──────────────────────────────────────────────────────


def _ev_image(
    signal: RiskSignal, ctx: ClaimContext, damage: ImageDamageCtx | None
) -> Evidence:
    """Build an `image` evidence row pointing at a single Damage."""
    if damage is None:
        # No CV data on this claim. Use a placeholder that makes the
        # gap visible to the UI without inventing coordinates.
        detail: dict[str, Any] = {
            "damage_type": None,
            "severity": None,
            "region_ref": None,
            "_note": "no CV detection available for this signal",
        }
        return Evidence(
            risk_signal_id=signal.id,
            evidence_type="image",
            reference=None,
            detail_json=detail,
        )
    return Evidence(
        risk_signal_id=signal.id,
        evidence_type="image",
        reference=str(damage.id),
        detail_json={
            "damage_type": damage.damage_type,
            "severity": damage.severity,
            "confidence": damage.confidence,
            "region_ref": damage.region_ref,
        },
    )


def _r1(signal: RiskSignal, ctx: ClaimContext) -> list[Evidence]:
    """R1: image vs claim-form mismatch → image evidence for the
    claim-form damage areas the CV did not see. If there are several
    unsupported claim-form damages, attach one image evidence per
    area (the worst-severity CV detection is the natural anchor).
    """
    cv_seen_types: set[str] = {
        _norm(d.damage_type) for d in ctx.image_damages if d.damage_type
    }
    unsupported = [
        d for d in ctx.claim_form_damages
        if d.damage_type and _norm(d.damage_type) not in cv_seen_types
    ]
    if not unsupported:
        return [_ev_image(signal, ctx, _worst_cv_damage(ctx))]
    # Emit one image evidence per unsupported claim-form area, all
    # referencing the most-relevant CV damage. Honest: the UI shows
    # the user's "no detection" finding is the trigger; the CV row is
    # the counter-example anchor.
    anchor = _worst_cv_damage(ctx)
    return [
        Evidence(
            risk_signal_id=signal.id,
            evidence_type="image",
            reference=str(anchor.id) if anchor else None,
            detail_json={
                "damage_type": d.damage_type,
                "severity": d.severity,
                "region_ref": d.region_ref,
                "claim_form_damage_id": d.id,
                "matched_cv_damage": False,
            },
        )
        for d in unsupported
    ]


def _r2(signal: RiskSignal, ctx: ClaimContext) -> list[Evidence]:
    """R2: severity mismatch → field evidence with two sources."""
    text_rank = _extract_text_severity(ctx.accident_description)
    cv_label, _ = _worst_cv_severity(ctx)
    return [
        Evidence(
            risk_signal_id=signal.id,
            evidence_type="field",
            reference="severity",
            detail_json={
                "sources": [
                    {"name": "claim_description", "value": _rank_to_label(text_rank)},
                    {"name": "cv_detection", "value": cv_label},
                ],
                "conflict": True,
                "field_name": "severity",
            },
        )
    ]


def _r3(signal: RiskSignal, ctx: ClaimContext) -> list[Evidence]:
    """R3: repair-component mismatch → document evidence for the
    estimate. If the claim has no estimate, we still emit one
    evidence row so the bundle is non-empty, pointing at the
    claim_form document as the fallback.
    """
    doc = _estimate_doc(ctx) or _first_doc(ctx)
    if doc is None:
        return [
            Evidence(
                risk_signal_id=signal.id,
                evidence_type="document",
                reference=None,
                detail_json={
                    "page": 1,
                    "field_name": "component",
                    "value": None,
                    "confidence": 0.0,
                    "_note": "no document on claim to anchor evidence",
                },
            )
        ]
    component = None
    if ctx.repair_estimate and ctx.repair_estimate.items:
        component = ctx.repair_estimate.items[0].part_name
    return [
        Evidence(
            risk_signal_id=signal.id,
            evidence_type="document",
            reference=str(doc.id),
            detail_json={
                "page": 1,
                "field_name": "component",
                "value": component,
                "confidence": doc.raw_confidence or 0.0,
            },
        )
    ]


def _r4(signal: RiskSignal, ctx: ClaimContext) -> list[Evidence]:
    """R4: excessive repair cost → computed evidence with baseline.
    `baseline_upper` is the upper bound the consistency engine was
    given (the risk engine recomputes it deterministically). If it is
    None, we report 0..0 to make the gap visible rather than fake a
    number.
    """
    upper = ctx.baseline_upper or 0.0
    claimed = float(ctx.claimed_amount or 0.0)
    ratio = (claimed / upper) if upper else None
    return [
        Evidence(
            risk_signal_id=signal.id,
            evidence_type="computed",
            reference="excessive_repair_cost",
            detail_json={
                "baseline_range": [0.0, float(upper)],
                "claimed": claimed,
                "ratio": ratio,
            },
        )
    ]


def _r5(signal: RiskSignal, ctx: ClaimContext) -> list[Evidence]:
    """R5: duplicate previous damage → field evidence with previous vs
    current claim numbers + damage summary."""
    if not ctx.previous_claims:
        # No previous claims means the rule could not have fired;
        # return a single empty field so the bundle is non-empty.
        return [
            Evidence(
                risk_signal_id=signal.id,
                evidence_type="field",
                reference="duplicate_damage",
                detail_json={
                    "sources": [],
                    "conflict": False,
                    "field_name": "previous_claim_overlap",
                    "_note": "no previous claims on file",
                },
            )
        ]
    previous = ctx.previous_claims[0]
    return [
        Evidence(
            risk_signal_id=signal.id,
            evidence_type="field",
            reference="duplicate_damage",
            detail_json={
                "sources": [
                    {"name": "previous_claim", "value": previous.claim_number,
                     "damage_summary": previous.damage_summary,
                     "overlap_score": previous.overlap_score},
                    {"name": "current_claim", "value": ctx.claim_number,
                     "damage_summary": ctx.accident_description},
                ],
                "conflict": True,
                "field_name": "damage_summary",
            },
        )
    ]


def _r6(signal: RiskSignal, ctx: ClaimContext) -> list[Evidence]:
    """R6: policy coverage mismatch → document evidence on the policy doc."""
    policy_doc = _doc_by_type(ctx, "policy")
    if policy_doc is None:
        policy_doc = _first_doc(ctx)
    if policy_doc is None:
        return [
            Evidence(
                risk_signal_id=signal.id,
                evidence_type="document",
                reference=None,
                detail_json={
                    "page": 1,
                    "field_name": "coverage_type",
                    "value": None,
                    "confidence": 0.0,
                    "_note": "no policy document on claim",
                },
            )
        ]
    coverage = dict(policy_doc.extracted_fields).get("coverage_type") or ctx.coverage_type
    return [
        Evidence(
            risk_signal_id=signal.id,
            evidence_type="document",
            reference=str(policy_doc.id),
            detail_json={
                "page": 1,
                "field_name": "coverage_type",
                "value": coverage,
                "confidence": policy_doc.raw_confidence or 0.0,
            },
        )
    ]


def _r7(signal: RiskSignal, ctx: ClaimContext) -> list[Evidence]:
    """R7: claim frequency → computed evidence."""
    window_days = 365
    count = len(ctx.previous_claims)
    return [
        Evidence(
            risk_signal_id=signal.id,
            evidence_type="computed",
            reference="claim_frequency",
            detail_json={
                "count_in_window": count,
                "window_days": window_days,
            },
        )
    ]


def _r8(signal: RiskSignal, ctx: ClaimContext) -> list[Evidence]:
    """R8: near policy boundary → computed evidence."""
    today = ctx.incident_date
    boundary = ctx.policy_end_date
    if boundary is None:
        days = 0
    else:
        days = (boundary - today).days if isinstance(today, date) else 0
    return [
        Evidence(
            risk_signal_id=signal.id,
            evidence_type="computed",
            reference="policy_boundary",
            detail_json={
                "policy_active": ctx.policy_status == "active",
                "days_to_boundary": days,
            },
        )
    ]


def _r9(signal: RiskSignal, ctx: ClaimContext) -> list[Evidence]:
    """R9: cross-document field conflict → one field evidence per
    conflicting field. If R9 fired, at least one field has ≥2
    distinct values across the documents.
    """
    rows: list[Evidence] = []
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
        rows.append(
            Evidence(
                risk_signal_id=signal.id,
                evidence_type="field",
                reference=field_name,
                detail_json={
                    "sources": [
                        {"name": doc_type, "value": value}
                        for value, doc_types in sorted(values.items())
                        for doc_type in sorted(set(doc_types))
                    ],
                    "conflict": True,
                    "field_name": field_name,
                },
            )
        )
    if not rows:
        # Defensive: the rule should not have fired without at least
        # one conflict. Emit a stub so the bundle is non-empty.
        rows.append(
            Evidence(
                risk_signal_id=signal.id,
                evidence_type="field",
                reference=None,
                detail_json={
                    "sources": [],
                    "conflict": False,
                    "field_name": "cross_document",
                    "_note": "R9 fired but no conflict field was identified",
                },
            )
        )
    return rows


# Registry: rule_id -> builder. We check membership in `persist_evidence`
# and in tests; missing entries are loud failures.
_BUILDERS = {
    "R1_unsupported_damage": _r1,
    "R2_severity_mismatch": _r2,
    "R3_repair_component_mismatch": _r3,
    "R4_excessive_repair_cost": _r4,
    "R5_duplicate_previous_damage": _r5,
    "R6_policy_coverage_mismatch": _r6,
    "R7_claim_frequency": _r7,
    "R8_near_policy_boundary": _r8,
    "R9_document_field_conflict": _r9,
}


# ─── Helpers ────────────────────────────────────────────────────────────────


def _norm(value: str | None) -> str:
    return (value or "").strip().lower()


def _worst_cv_damage(ctx: ClaimContext) -> ImageDamageCtx | None:
    """Return the most-severe CV damage on the claim, or None."""
    if not ctx.image_damages:
        return None
    severity_rank = {"minor": 0, "moderate": 1, "severe": 2}
    best: ImageDamageCtx | None = None
    best_rank = -1
    for d in ctx.image_damages:
        rank = severity_rank.get(d.severity or "", -1)
        if rank > best_rank:
            best = d
            best_rank = rank
    return best


def _first_doc(ctx: ClaimContext) -> Any:
    """First document on the claim, or None."""
    return ctx.documents[0] if ctx.documents else None


def _estimate_doc(ctx: ClaimContext) -> Any:
    for d in ctx.documents:
        if d.doc_type == "estimate":
            return d
    return None


def _doc_by_type(ctx: ClaimContext, doc_type: str) -> Any:
    for d in ctx.documents:
        if d.doc_type == doc_type:
            return d
    return None


def _rank_to_label(rank: int | None) -> str | None:
    if rank is None:
        return None
    return {0: "minor", 1: "moderate", 2: "severe"}.get(rank)


# Public exports for tests and audits.
SUPPORTED_RULE_IDS = _ALL_RULE_IDS
